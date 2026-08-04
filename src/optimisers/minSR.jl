# using Metal
# using Statistics
# using KernelAbstractions
"""
    MomentumBuffer(p::Int, ansatz; β=0.9f0)

Pre-allocated buffer for gradient momentum smoothing (exponential moving average of gradients).

# Arguments
* `p`: number of parameters in `ansatz` model.
* `ansatz`: ansatz that evaluates wave-function. See [`NeuralAnsatz`](@ref).

## Fields
* `v`: velocity vector (running mean of gradients), shape `(p,)`.
* `β`: momentum decay rate (default: `0.9`)

## Notes
The momentum is inspired from [`adam`](@ref) momentum update.
See: [Momentum paper](https://proceedings.neurips.cc/paper/2020/file/d3f5d4de09ea19461dab00590df91e4f-Paper.pdf)

"""
mutable struct MomentumBuffer{T, V <: AbstractArray{T}}
    v :: V      # velocity / running mean
    β :: T
end
function MomentumBuffer(p::Int, ansatz; β=0.9f0)
    T = Float32
    l = last(ansatz.model.layers)
    v = fill!(similar(l.b, p), zero(T))
    V = typeof(v)
    return MomentumBuffer{T, V}(v, β)
end

function momentum_step!(buf, Δθ_SR)
    β = buf.β
    @. buf.v = β * buf.v + (one(β) - β) * Δθ_SR
    return buf.v   # smoothed update
end


"""
    minSRBuffer(N::Int, p::int, ansatz; velocity=false, β=0.9f0)

Structure meant for `minSR` optimiser, see [`compute_minSR_cg!`](@ref). 
It allows allocation-free, batched and GPU friendly evaluation of `minSR`.

# Variables

* `N`: batch size for  array dimensions.
* `p`: total number of parameters in ansatz model, for array dimensions. 
* `ansatz`: ansatz model that evaluates wave-function. See also [`NeuralAnsatz`](@ref).
 
# Keyword Variables

* `velocity`: boolean variable, if `true` that `minSR` optimiser will apply momentum
    smoothening. By default `false`. See also [`MomentumBuffer`](@ref).
* `β`: also used with [`MomentumBuffer`](@ref).
"""
mutable struct minSRBuffer{T, D, M <: AbstractMatrix{T}, V <: AbstractVector{T},
                             M64cpu <: AbstractMatrix{D}, X}
    O_mean::V
    g::V
    K::M
    Δθ::V
    K_cpu::M64cpu
    vel::Bool
    moment::X
    weights::V
    p_cg::V
    Ap::V
end
function minSRBuffer(N::Int, p::Int, ansatz; velocity=false, β=0.9f0)
    T = Float32
    D = Float64
    l = first(ansatz.model.layers) # l.b is Vector type, l.W is Matrix type (CPU or GPU)

    O_mean      = similar(l.b, p)
    g           = similar(l.b, N)
    w           = similar(l.b, N)
    p_cg        = similar(l.b, N)
    Ap          = similar(l.b, N)
    K           = fill!(similar(l.W, N, N), zero(T))
    K_cpu       = Matrix{D}(undef, N, N)
    Δθ          = fill!(similar(l.b, p), zero(T))
    if !velocity
        vel     = false
        moment  = nothing
    else
        vel     = true 
        moment  = MomentumBuffer(p, ansatz; β=β)
    end

    M       = typeof(K)
    V       = typeof(O_mean)
    M64cpu  = typeof(K_cpu)
    X       = typeof(moment)
    return minSRBuffer{T, D, M, V, M64cpu,X}(O_mean, g, K, Δθ, K_cpu, vel, moment, w, p_cg, Ap)
end


"""
    compute_minSR_cg!(E_mean, variance, jacobian_buf, vmc_buf, 
                        minSR_buf, ansatz, mode, λ, weights, raw_norm)

Computes the minSR (Stochastic Reconfiguration) natural gradient step `Δθ` using 
a matrix-free [`cg_solve!`](@ref).

```math
\\Delta\\theta = \\tilde{J} (\\tilde{J}^T \\tilde{J} + \\lambda \\mathbf{I}) \\tilde{g}
```

The `J̃` is weighted centered per-sample jacobian and `g̃` represents weighted loss
function gradient, see [`apply_loss!`](@ref).

# Arguments
* `E_mean`: mean of local energies `⟨E⟩`. (Float64)
* `variance`: variance `σ²` of local energies. (Float64)
* `jacobian_buf`: holds jacobian `J` `(p,N)` and parameter vector `θ`, and
        parameter vector update `Δθ`.
* `vmc_buf`: holds local energies `E_locs` `(N,)`.
* `minSR_buf`: holds all necessary variables for `minSR` calculations and 
        [`cg_solve!`](@ref) calculations. See [`minSRBuffer`](@ref).
* `ansatz`: ansat for evaluating wave-function. See [`NeuralAnsatz`](@ref).
* `mode`: loss function mode for gradient calculations.
* `λ`: Tikhonov regularisation.
* `weights`: VMC sampler weights. See [`vmc_sample!`](@ref).
* `raw_norm`: VMC estimate of normalisation over batched sample.
"""
function compute_minSR_cg!(E_mean::Float64, variance::Float64, jacobian_buf, 
                        vmc_buf, minSR_buf, ansatz, mode, λ, weights, raw_norm)

    J = jacobian_buf.J
    O_mean = minSR_buf.O_mean
    g = minSR_buf.g
    Δθ = minSR_buf.Δθ
    wgpu = minSR_buf.weights

    E_locs = vmc_buf.E_locs
    tmp = vmc_buf.diag_ham

    N = length(E_locs) # number of samples

    if weights === nothing
        w = Float32(sqrt(1/N))     # uniform weights from Metropolis MC
        apply_loss!(tmp, E_locs, w, E_mean, variance, raw_norm, mode)
    else
        weights .= sqrt.(weights)
        w = weights                # weights from CTMC
        apply_loss!(tmp, E_locs, w, E_mean, variance, raw_norm, mode)
        copyto!(wgpu, w)
    end

    mean!(O_mean, J)        # (p,) in-place: calculate mean of J 

    @. J = J - O_mean       # (p,N) - (p,1) column wise
    J_bar = J
    if weights === nothing
        @. J_bar = w * J_bar
    else
        J_bar .*= reshape(wgpu, 1, :)
    end
    copyto!(g, tmp)
    p_cg   = minSR_buf.p_cg
    Ap  = minSR_buf.Ap

    # needed GPU synchronisation for CG solver
    backend = KernelAbstractions.get_backend(J_bar)
    KernelAbstractions.synchronize(backend)
    cg_solve!(wgpu, J_bar, λ, g, p_cg, Ap, Δθ) # wgpu -> solution of CG solver

    mul!(Δθ, J_bar, wgpu, 1f0, 0f0) # Δθ is flat (p,) vector with updated values of NN parameters
end

"""
    minSR(jacobian_buf, vmc_buf, minSR_buf, H, ansatz, addrs_n; kwargs...)
            -> E_mean, variance, last_addr, acceptance

Function that applies `minSR` optimiser to update `ansatz.model` parameters. See
also [`compute_minSR_cg!`](@ref).

# Variables

* `jacobian_buf`: holds all neccessary variables about jacobians and parameters.
        See [`JacobianBuffer`](@ref).
* `vmc_buffer`: holds result from VMC, especially local energies. See [`VMCBuffer`](@ref).
* `minSR_buf`: holds all necessary variables for `minSR` optimisation step.
        See [`minSRBuffer`](@ref).
* `H`: hamiltonian defined in Rimu.
* `ansatz`: ansatz for evaluation of wave-function. See [`NeuralAnsatz`](@ref).
* `addrs_n`: newly proposed addresses from VMC. those values are returned for next 
        iteration.

# Keyword Arguments

* `vmc`: symbol specifying what vmc sampler was chosen (default `:metropolis`). 
        See [`vmc_sample!`](@ref).
* `burnin`: number of thermalisation steps in VMC before update (default `100`). 
        See [`vmc_energy`](@ref).
* `mode`: symbol specifying what loss function is applied (default `:energy`). 
        See [`apply_loss!`](@ref).
* `λ`: Tikhonov regularisation for `minSR` step (default `0.001f0`).
* `η`: learning rate for parameter update (default `0.001f0`).

## Notes

* `norm clamping`: before new parameters are updated with `Δθ`, we do its norm
        clamping. The `minSR`, especially in the beggining, can suggest too
        big moves in some parameter directions. This clamping is safe check, so
        the update happens within `Float32` precision.
""" 
function minSR(jacobian_buf, vmc_buf, minSR_buf, H, ansatz, addrs_n; 
                vmc=:metropolis, burnin=100, mode=:energy, λ=0.001f0, η = 0.001f0)

    E_mean, variance, last_addr, acceptance, weights, raw_norm = 
        vmc_energy(H, ansatz, addrs_n, vmc_buf, jacobian_buf; 
                        vmc_sampler=vmc, burnin=burnin, mode=mode)

    # safe check -> invalid update
    if !all(isfinite, jacobian_buf.J) || !all(isfinite, vmc_buf.E_locs)
        @warn "NaN/Inf in J or g — skipping update"
        return E_mean, variance, last_addr, acceptance
    end

    compute_minSR_cg!(E_mean, variance, jacobian_buf, vmc_buf, minSR_buf, ansatz, mode, λ, weights, raw_norm)
    Δθ = minSR_buf.Δθ
    θ = jacobian_buf.θ

    if minSR_buf.vel && minSR_buf.moment !== nothing
        Δθ = momentum_step!(minSR_buf.moment, Δθ)
    end

    # norm safe check -> keep Float32 precision
    max_norm = 1f3
    current_norm = Float32(norm(Δθ))
    if !isfinite(current_norm)
        @warn "NaN/Inf in Δθ — skipping update"
        return E_mean, variance, last_addr, acceptance
    elseif current_norm > max_norm
        Δθ .= Δθ ./ (current_norm / max_norm)
    end

    @. θ = θ - η*Δθ  # update weights vector in jacobian_buf

    update!(ansatz.model, jacobian_buf, θ)  # update weights in NN model

    return E_mean, variance, last_addr, acceptance
end


