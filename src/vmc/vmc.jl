"""
    vmc_sample!(vmc_sampler::Symbol, vmc_buf, jacobian_buf, ham, addrs_n, ansatz)

General function that applies correct VMC sampler defined in `vmc_sampler` variable.
See also [`ctmc_sample!`](@ref) and [`metropolis_sample!`](@ref).
"""
function vmc_sample!(vmc_sampler::Symbol, vmc_buf, jacobian_buf, ham, addrs_n, ansatz; 
                    eclamp::Bool=true)
    if vmc_sampler === :metropolis
        new_addrs, E_locs, weights, grads_n, acceptance = 
            metropolis_sample!(vmc_buf, jacobian_buf, ham, addrs_n, ansatz; eclamp=eclamp)
    elseif vmc_sampler === :ctmc
        new_addrs, E_locs, weights, grads_n, acceptance = 
            ctmc_sample!(vmc_buf, jacobian_buf, ham, addrs_n, ansatz; eclamp=eclamp)
    else
        throw(ArgumentError("invalid vmc_sampler :$vmc_sampler, choose :metropolis or :ctmc"))
    end

    return new_addrs, E_locs, weights, grads_n, acceptance
end     


"""
    vmc_energy(H, ansatz, addrs_n, vmc_buf, jacobian_buf; kwargs...)
        -> E_mean, variance, addrs_n, acceptance, weights

Estimate the variational energy ``\\langle H \\rangle_\\psi`` with Variational Monte Carlo (VMC).

The batch of `B` walkers in `addrs_n` is treated as `B` independent Markov chains.
They are first thermalised for `burnin` steps. Then one more sampling step evaluates
the local energies and per-sample gradients, which are reduced to the weighted mean
and variance. Type of sampling is decided in [`vmc_sample!`](@ref).

# Arguments
* `H`: Hamiltonian defined in Rimu.
* `ansatz`: wave-function ansatz, see [`NeuralAnsatz`](@ref).
* `addrs_n`: batch of walker addresses (Rimu Fock-state representation).
* `vmc_buf`: pre-allocated [`VMCBuffer`](@ref).
* `jacobian_buf`: pre-allocated [`JacobianBuffer`](@ref) for the per-sample Jacobians.
  It holds the gradients of the last sampling step on return.

# Keyword Arguments
* `vmc_sampler::Symbol=:metropolis`: sampler, `:metropolis` or `:ctmc`.
* `burnin::Integer=100`: number of thermalisation steps before samples are collected.
* `mode::Symbol=:energy`: minimisation mode (loss function). Currently unused.
* `eclamp::Bool=true`: clip outlying local energies, see [`elocs_clamping!`](@ref).

# Returns
* `E_mean`: weighted mean of the local energies, ``\\langle E_{loc} \\rangle``.
* `variance`: weighted variance of the local energies, ``\\sigma^2(E_{loc})``.
* `addrs_n`: final walker positions (the same array as the input `addrs_n`).
* `acceptance`: fraction of walkers that moved in the last step, in `[0, 1]`.
  For Metropolis this is the acceptance rate. For lazy CTMC it is about `0.75`.
* `weights`: `(B,)` per-sample weights used in the averages.

# Notes
* **Thermalisation:** during the `burnin` steps only the walkers move. Local energies
  and gradients are not computed, which keeps burn-in cheap.
* **Weights:** Metropolis weights are uniform (`1/B`). CTMC weights are residence times,
  see [`get_ctmc_weights!`](@ref).
* **Sanity checks:** an error is thrown if the local energies or gradients contain
  `NaN`/`Inf`.

# Example
```julia
E, var, addrs, acc, w = vmc_energy(H, ansatz, addrs_n, vmc_buf, jac_buf;
                                   vmc_sampler=:ctmc, burnin=50, eclamp=true)
```
"""
function vmc_energy(H, ansatz, addrs_n, vmc_buf, jacobian_buf;
                    vmc_sampler::Symbol=:metropolis, burnin::Integer=100,
                    mode::Symbol=:energy, eclamp::Bool=true)
    # termalisation of initial state
    vmc_buf.start = false
    count = 0
    while !vmc_buf.start
        count += 1
        new_addrs, E_locs, weights, grads_n, acceptance = 
                    vmc_sample!(vmc_sampler, vmc_buf, jacobian_buf, H, addrs_n, ansatz; 
                                eclamp=eclamp)
        addrs_n = new_addrs
        if count >= burnin
            vmc_buf.start = true
        end
    end

    # after termalisation I just compute vmc every call (batched)
    new_addrs, E_locs, weights, grads_n, acceptance = 
                    vmc_sample!(vmc_sampler, vmc_buf, jacobian_buf, H, addrs_n, ansatz;
                               eclamp=eclamp)
    addrs_n = new_addrs

    all(isfinite, grads_n) ||
        error("grads_n contains NaN/Inf: extrema = $(extrema(grads_n))")
    all(isfinite, E_locs) ||
        error("E_locs contain NaN/Inf: extrema = $(extrema(E_locs))")


    E_mean, variance = local_energy_stats!(vmc_buf, E_locs, weights, ansatz.model.batch)

    return E_mean, variance, addrs_n, acceptance, weights
end

"""
    weighted_sum_kernel!(acc, E_locs, weights)

Kernel that accumulates the weighted sum `acc[1] += Σ_i w_i E_loc,i` with atomic adds.
Launched with `ndrange = batch`, one thread per sample. `acc` must be zeroed beforehand.
"""
@kernel function weighted_sum_kernel!(acc, E_locs, weights)
    i = @index(Global)
    Atomix.@atomic acc[1] += weights[i] * E_locs[i]
end

"""
    weighted_variance_kernel!(acc, E_locs, weights, E_mean)

Kernel that accumulates the weighted variance `acc[1] += Σ_i w_i (E_loc,i − ⟨E⟩)²` with
atomic adds. Launched with `ndrange = batch`, one thread per sample. `acc` must be zeroed
beforehand, and `E_mean` must already hold `⟨E⟩`.
"""
@kernel function weighted_variance_kernel!(acc, E_locs, weights, E_mean)
    i = @index(Global)
    d = E_locs[i] - E_mean[1]
    Atomix.@atomic acc[1] += weights[i] * d * d
end

"""
    local_energy_stats!(vmc_buf, E_locs, weights, batch) -> (E_mean, variance)

Compute the weighted mean and variance of the local energies,

```math
\\langle E \\rangle = \\sum_i w_i\\, E_{\\mathrm{loc},i}, \\qquad
\\sigma^2 = \\sum_i w_i \\left(E_{\\mathrm{loc},i} - \\langle E \\rangle\\right)^2 ,
```

on the backend of `E_locs`. The results are written into `vmc_buf.E_mean` and
`vmc_buf.variance`, both (1,) device arrays.

# Arguments
* `vmc_buf`: holds the output buffers `E_mean` and `variance`, see [`VMCBuffer`](@ref).
* `E_locs`: (batch,) local energies.
* `weights`: (batch,) sampler weights, normalised to `Σw = 1`.
* `batch`: number of samples.
"""
function local_energy_stats!(vmc_buf, E_locs, weights, batch)
    E_mean = vmc_buf.E_mean
    variance = vmc_buf.variance
    backend = KernelAbstractions.get_backend(E_locs)

    fill!(E_mean, 0f0)
    weighted_sum_kernel!(backend)(E_mean, E_locs, weights; ndrange = batch)

    fill!(variance, 0f0)
    weighted_variance_kernel!(backend)(variance, E_locs, weights, E_mean; ndrange = batch)

    return E_mean, variance
end
