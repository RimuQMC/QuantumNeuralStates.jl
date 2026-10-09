# using LinearAlgebra

"""
    DescentBuffer(N::Int, p::Int, ansatz)

Pre-allocated buffer for basic gradient descent optimisation.

# Arguments
* `N`: batch number.
* `p`: number of parameters inside ansatz model.
* `ansatz`: ansatz for wave-function evaluation. See [`NeuralAnsatz`](@ref).

## Fields
* `Δθ`: parameter gradient vector, shape `(p,)`.
"""
mutable struct DescentBuffer{T, V <: AbstractArray{T}}
    Δθ::V
end
function DescentBuffer(N::Int, p::Int, ansatz)
    T = Float32
    l = last(ansatz.model.layers)
    Δθ = fill!(similar(l.b, p), zero(T))
    V = typeof(Δθ)
    return DescentBuffer{T, V}(Δθ)
end

"""
    descent(jacobian_buf, vmc_buf, descent_buf, H, ansatz, addrs_n; kwargs...)
            -> E_mean, variance, last_addr, acceptance

Apply basic gradient descent step for parameter optimisation.

```math
\\Delta\\theta = J \\dot \\tilde{g}
```
`J` is pre-sample jacobian (see [`JacobianBuffer`](@ref)) and `g̃` are weighted loss 
gradients (see [`apply_loss!`](@ref)). 

# Arguments
* `jacobian_buf`: holds all neccessary variables about jacobians and parameters.
        See [`JacobianBuffer`](@ref).
* `vmc_buffer`: holds result from VMC, especially local energies. See [`VMCBuffer`](@ref).
* `descent_buf`: holds all necessary variables for `descent` optimisation step.
        See [`DescentBuffer`](@ref).
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
* `λ`: no use here (default `0.001f0`).
* `η`: learning rate for parameter update (default `0.001f0`).
"""
function descent(jacobian_buf, vmc_buf, descent_buf, H, ansatz, addrs_n; 
                vmc=:metropolis, burnin=100, mode=:energy, λ=0.001f0, η = 0.001f0)

    E_mean, variance, last_addr, acceptance, weights = 
        vmc_energy(H, ansatz, addrs_n, vmc_buf, jacobian_buf; 
            vmc_sampler=vmc, burnin=burnin, mode=mode)

    E_locs = vmc_buf.E_locs
    g = vmc_buf.ham_diag    # reused buffer for gradients

    apply_loss!(g, E_locs, weights, E_mean, variance, mode)

    mul!(descent_buf.Δθ, jacobian_buf.J, g, 1f0, 0f0) # (p,) = (p,N) x (N,)

    θ = jacobian_buf.θ
    @. θ = θ - η*descent_buf.Δθ
    update!(ansatz, jacobian_buf, θ)

    return E_mean, variance, last_addr, acceptance
end
