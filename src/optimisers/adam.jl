# using LinearAlgebra
# using Statistics

"""
    AdamBuffer(N::Int, p::Int, ansatz; β1=0.9f0, β2=0.999f0, ε=1f-5)

Pre-allocated buffer for the Adam optimiser.

# Arguments
* `N`: batch number.
* `p`: number of parameters inside ansatz model.
* `ansatz`: ansatz for wave-function evaluation. See [`NeuralAnsatz`](@ref).

## Fields
* `m`: 1st moment estimate (exponential moving average of gradients).
* `v`: 2nd moment estimate (exponential moving average of squared gradients).
* `Δθ`: reusable buffer for the parameter update vector.
* `t`: step counter (used for bias correction).
* `β1`: decay rate for 1st moment (default: `0.9`).
* `β2`: decay rate for 2nd moment (default: `0.999`).
* `ε`: numerical stability term (default: `1e-5`).
* `E_grads`: buffer for energy gradients of length `N` (number of samples).

## Notes
Adam: A Method for Stochastic Optimization: https://arxiv.org/pdf/1412.6980.
"""
mutable struct AdamBuffer{T, V <: AbstractArray{T}}
    m::V        # 1st moment (mean of gradients)
    v::V        # 2nd moment (mean of squared gradients)
    Δθ::V       # update vector (reusable buffer)
    t::Int      # step counter
    β1::T
    β2::T
    ε::T
    E_grads::V
end
function AdamBuffer(N::Int, p::Int, ansatz; β1=0.9f0, β2=0.999f0, ε=1f-5)
    T = Float32
    l = last(ansatz.model.layers)
    m = fill!(similar(l.b, p), zero(T))
    v = fill!(similar(l.b, p), zero(T))
    Δθ = fill!(similar(l.b, p), zero(T))
    E_grads = similar(l.b, N)
    return AdamBuffer{T, typeof(m)}(m, v, Δθ, 0, β1, β2, ε, E_grads)
end

# Single Adam step — all in-place, zero allocations
function adam_step!(buf::AdamBuffer, g)
    buf.t += 1
    β1, β2, ε, t = buf.β1, buf.β2, buf.ε, buf.t

    # bias correction scalars (CPU scalars, no GPU transfer needed)
    bc1 = one(β1) - β1^t
    bc2 = one(β2) - β2^t

    # 1st moment:  m = β1·m + (1-β1)·g
    @. buf.m = β1 * buf.m + (one(β1) - β1) * g

    # 2nd moment:  v = β2·v + (1-β2)·g^2
    @. buf.v = β2 * buf.v + (one(β2) - β2) * g * g

    # bias-corrected update:  Δθ = η · (m/bc1) / (√(v/bc2) + ε)
    @. buf.Δθ = (buf.m / bc1) / (sqrt(buf.v / bc2) + ε)
end

"""
    adam(jacobian_buf, vmc_buf, adam_buf, H, ansatz, addrs_n; kwargs...)
            -> E_mean, variance, last_addr, acceptance

Apply Adam moment updates and performs gradient descent step for parameter optimisation.

# Arguments
* `jacobian_buf`: holds all neccessary variables about jacobians and parameters.
        See [`JacobianBuffer`](@ref).
* `vmc_buffer`: holds result from VMC, especially local energies. See [`VMCBuffer`](@ref).
* `adam_buf`: holds all necessary variables for `adam` optimisation step. 
        See [`AdamBuffer`](@ref).
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
        See [`apply_loss`](@ref).
* `λ`: no use here (default `0.001f0`).
* `η`: learning rate for parameter update (default `0.001f0`).
"""
function adam(jacobian_buf, vmc_buf, adam_buf, H, ansatz, addrs_n; 
            vmc=:metropolis, burnin=100, mode=:energy, λ=0.001f0, η=0.001f0)

    E_mean, variance, last_addr, acceptance, weights, raw_norm = 
        vmc_energy(H, ansatz, addrs_n, vmc_buf, jacobian_buf;
            vmc_sampler=vmc, burnin=burnin, mode=mode)


    E_locs = vmc_buf.E_locs
    tmp = vmc_buf.diag_ham
    N = length(E_locs)

    # WEIGHTS CTMC / METROPOLIS ?
    if weights === nothing
        w = 1.0/N     # uniform weights from Metropolis MC
    else
        w = weights
    end

    # gradients are in tmp 
    apply_loss!(tmp, E_locs, w, E_mean, variance, raw_norm, mode)

    E_grads = adam_buf.E_grads # weighted loss gradients
    copyto!(E_grads, tmp)
    mul!(adam_buf.Δθ, jacobian_buf.J, E_grads, 1f0, 0f0)
    adam_step!(adam_buf, adam_buf.Δθ) 

    θ = jacobian_buf.θ
    @. θ = θ - η*adam_buf.Δθ
    update!(ansatz.model, jacobian_buf, θ)

    return E_mean, variance, last_addr, acceptance
end
