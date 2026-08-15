"""
    vmc_sample!(vmc_sampler::Symbol, vmc_buf, jacobian_buf, ham, addrs_n, ansatz)

General function that applies correct VMC sampler defined in `vmc_sampler` variable.
See also [`ctmc_sample!`](@ref) and [`metropolis_sample!`](@ref).
"""
function vmc_sample!(vmc_sampler::Symbol, vmc_buf, jacobian_buf, ham, addrs_n, ansatz)
    if vmc_sampler === :metropolis
        new_addrs, E_locs, weights, grads_n, acceptance = 
            metropolis_sample!(vmc_buf, jacobian_buf, ham, addrs_n, ansatz)
    elseif vmc_sampler === :metropolis_heatbath
        new_addrs, E_locs, weights, grads_n, acceptance = 
            metropolis_heatbath_sample!(vmc_buf, jacobian_buf, ham, addrs_n, ansatz)
    elseif vmc_sampler === :ctmc
        new_addrs, E_locs, weights, grads_n, acceptance = 
            ctmc_sample!(vmc_buf, jacobian_buf, ham, addrs_n, ansatz)
    elseif vmc_sampler === :ctmc_heatbath
        new_addrs, E_locs, weights, grads_n, acceptance = 
            ctmc_heatbath_sample!(vmc_buf, jacobian_buf, ham, addrs_n, ansatz)
    else
        @error "Invalid vmc sampler! Choose from :metropolis or :ctmc. You have inserted $(vmc_sampler)"
    end

    return new_addrs, E_locs, weights, grads_n, acceptance
end     


"""
    vmc_energy(H, ansatz, addrs_n, vmc_buf, jacobian_buf; kwargs...)
                     vmc_sampler=:metropolis, burnin=100, mode=:energy)

Variational Monte Carlo (VMC) energy estimator. Functioning with CTMC or Metropolis samplers.
See [`ctmc_sample!`](@ref) and [`metropolis_sample!`](@ref).

# Arguments
* `H`: hamiltonian defined in Rimu.
* `ansatz`: ansatz that evaluates wave-function. See [`NeuralAnsatz`](@ref).
* `addrs_n`: batched input vector (Rimu style Fock-state representation).
* `vmc_buf`: pre-allocated [`VMCBuffer`](@ref).
* `jacobian_buf`: pre-allocated [`JacobianBuffer`](@ref) for per-sample jacobian computation.

# Keyword Arguments
* `vmc_sampler`: sampling method, `:metropolis`, `:ctmc`, `:metropolis_heatbath`, `ctmc_heatbath` 
    (default: `:metropolis`).
* `burnin`: number of thermalisation steps before collecting samples (default: `100`).
* `mode`: minimisation mode - loss function - (default: `:energy`).

# Returns
* `E_mean`: mean of local energies ⟨E_loc⟩ over the Markov chain.
* `variance`: variance of local energies σ²(E_loc).
* `addrs_n`: newly proposed addresses after sampling -> reused addrs_n buffer.
* `acceptance`: acceptance rate (for Metropolis, in CTMC acc=1 always).
* `weights`: per-sample importance weights (`nothing` / uniform for Metropolis, computed for CTMC).

# Notes
* Thermalisation: the Markov chain is burned in for `burnin` steps before any
  statistics are collected, ensuring the chain has reached the stationary distribution.
* Weights: under Metropolis sampling all weights are uniform; under CTMC sampling
  weights are computed, see [`get_ctmc_weights!`](@ref).

# Example
```julia
E, var, addrs, acc, w = vmc_energy_metro(H, ansatz, addrs_n, mbuf, jbuf; vmc_sampler=:metropolis)
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

    if weights === nothing
        E_mean = 0.0
        for k in eachindex(E_locs)
            E_mean += E_locs[k]
        end
        E_mean /= ansatz.model.batch

        variance = 0.0
        for k in eachindex(E_locs)
            variance += (E_locs[k] - E_mean)^2
        end
        variance /= ansatz.model.batch
    else
        E_mean = 0.0
        for k in eachindex(E_locs)
            E_mean += weights[k] * E_locs[k]
        end

        variance = 0.0
        for k in eachindex(E_locs)
            variance += weights[k] * (E_locs[k] - E_mean)^2
        end
    end

    #@show any(isnan, E_locs), any(isinf, E_locs), extrema(E_locs)
    if any(isnan, grads_n) || any(isinf, grads_n)
        error("grads_n contains NaN/Inf: extrema = $(extrema(grads_n))")
    end
      return E_mean, variance, addrs_n, acceptance, weights
end
