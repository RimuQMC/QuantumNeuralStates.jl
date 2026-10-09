"""
    vmc_sample!(vmc_sampler::Symbol, vmc_buf, jacobian_buf, ham, addrs_n, ansatz)

General function that applies correct VMC sampler defined in `vmc_sampler` variable.
See also [`ctmc_sample!`](@ref) and [`metropolis_sample!`](@ref).
"""
function vmc_sample!(vmc_sampler::Symbol, vmc_buf, jacobian_buf, ham, addrs_n, ansatz)
    if vmc_sampler === :metropolis
        new_addrs, E_locs, weights, grads_n, acceptance = 
            metropolis_sample!(vmc_buf, jacobian_buf, ham, addrs_n, ansatz)
    elseif vmc_sampler === :ctmc
        new_addrs, E_locs, weights, grads_n, acceptance = 
            ctmc_sample!(vmc_buf, jacobian_buf, ham, addrs_n, ansatz)
    else
        @error "Invalid vmc sampler! Choose from :metropolis or :ctmc. You have inserted $(vmc_sampler)"
    end

    return new_addrs, E_locs, weights, grads_n, acceptance
end     


"""
    vmc_energy(H, ansatz, addrs_n, vmc_buf, jacobian_buf; kwargs...)
                -> E_mean, variance, addrs_n, acceptance, weights

Variational Monte Carlo (VMC) energy estimator. Functioning with CTMC or Metropolis samplers.
See [`ctmc_sample!`](@ref) and [`metropolis_sample!`](@ref).

# Arguments
* `H`: hamiltonian defined in Rimu.
* `ansatz`: ansatz that evaluates wave-function. See [`NeuralAnsatz`](@ref).
* `addrs_n`: batched input vector (Rimu style Fock-state representation).
* `vmc_buf`: pre-allocated [`VMCBuffer`](@ref).
* `jacobian_buf`: pre-allocated [`JacobianBuffer`](@ref) for per-sample jacobian computation.

# Keyword Arguments
* `vmc_sampler`: sampling method, `:metropolis`, `:ctmc` (default: `:metropolis`).
* `burnin`: number of thermalisation steps before collecting samples (default: `100`).
* `mode`: minimisation mode - loss function - (default: `:energy`).

# Returns
* `E_mean`: mean of local energies ⟨E_loc⟩ over the Markov chain.
* `variance`: variance of local energies σ²(E_loc).
* `addrs_n`: newly proposed addresses after sampling -> reused addrs_n buffer.
* `acceptance`: acceptance rate (for Metropolis, in CTMC acc=1 always).
* `weights`: per-sample importance weights (`nothing` / uniform for Metropolis, computed for CTMC).

# Notes
* Thermalisation: the Markov chain is thermalised for `burnin` steps before any
  statistics are collected, ensuring the chain has reached the stationary distribution.
* Weights: under Metropolis sampling all weights are uniform; under CTMC sampling
  weights are computed, see [`get_ctmc_weights!`](@ref).

# Example
```julia
E, var, addrs, acc, w = vmc_energy(H, ansatz, addrs_n, mbuf, jbuf; vmc_sampler=:metropolis)
```
"""
function vmc_energy(H, ansatz, addrs_n, vmc_buf, jacobian_buf; 
            vmc_sampler=:metropolis, burnin=100, mode=:energy)
    # termalisation of initial state
    vmc_buf.start = false
    count = 0
    while !vmc_buf.start
        count += 1
        new_addrs, E_locs, weights, grads_n, acceptance = vmc_sample!(vmc_sampler, vmc_buf, jacobian_buf, H, addrs_n, ansatz)
        addrs_n = new_addrs
        if count >= burnin
            vmc_buf.start = true
        end
    end

    # after termalisation I just compute vmc every call (batched)
    new_addrs, E_locs, weights, grads_n, acceptance = vmc_sample!(vmc_sampler, vmc_buf, jacobian_buf, H, addrs_n, ansatz)
    addrs_n = new_addrs

    all(isfinite, grads_n) ||
        error("grads_n contains NaN/Inf: extrema = $(extrema(grads_n))")
    all(isfinite, E_locs) ||
        error("E_locs contain NaN/Inf: extrema = $(extrema(E_locs))")


    E_mean, variance = local_energy_stats!(vmc_buf, E_locs, weights, ansatz.model.batch)

    return E_mean, variance, addrs_n, acceptance, weights
end

@kernel function weighted_sum_kernel!(acc, E_locs, weights)
    i = @index(Global)
    Atomix.@atomic acc[1] += weights[i] * E_locs[i]
end

@kernel function weighted_variance_kernel!(acc, E_locs, weights, E_mean)
    i = @index(Global)
    d = E_locs[i] - E_mean[1]   # <-- index inside kernel
    Atomix.@atomic acc[1] += weights[i] * d * d
end

function local_energy_stats!(vmc_buf, E_locs, weights, batch)
    E_mean   = vmc_buf.E_mean
    variance = vmc_buf.variance
    backend  = KernelAbstractions.get_backend(E_locs)

    fill!(E_mean, 0f0)
    weighted_sum_kernel!(backend)(E_mean, E_locs, weights; ndrange = batch)

    fill!(variance, 0f0)
    weighted_variance_kernel!(backend)(variance, E_locs, weights, E_mean; ndrange = batch)

    return E_mean, variance
end
