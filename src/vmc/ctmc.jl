# using KernelAbstractions

"""
    ctmc_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz; eclamp=true)
        -> new_addrs, E_locs, weights, grads_n, acc

Perform one step of the lazy Continuous-Time Monte Carlo (CTMC) sampler for VMC.

For each walker `n` in the batch, new configurations are proposed only among its
off-diagonal connections ``m`` with ``H_{nm} \\neq 0`` (optionally restricted by the
truncation mask of `ansatz`). The target `m` is selected according to jump rates
built from the wave-function amplitudes `log|ψ(m)|`.

In CTMC, a walker always jumps to the proposed state. To compensate, each sample
carries a *residence time*: the `weight` describing how long the walker effectively
stays on configuration `n` before jumping. See [`get_ctmc_weights!`](@ref).

# Lazy chain
A pure jump chain can be periodic. For example, with particle-number or bipartite
structure it can alternate between two sublattices forever, which breaks ergodicity
of the Markov chain. To make the chain aperiodic, the sampler is *lazy*: each walker
moves to its proposed state with probability `0.75` and stays at `n` with
probability `0.25`. The stationary distribution is unchanged.

# Arguments
* `vmc_buf`: [`VMCBuffer`](@ref) holding intermediate variables for allocation-free
  calculations.
* `jacobian_buf`: [`JacobianBuffer`](@ref) used to compute the per-sample Jacobians.
* `hamiltonian`: Hamiltonian defined in Rimu.
* `addrs_n`: current walker addresses (length = batch size `B`). Updated in place.
* `ansatz`: wave-function ansatz, see [`NeuralAnsatz`](@ref).

# Keywords
* `eclamp::Bool=true`: clip outlying local energies, see
  [`calculate_local_energy!`](@ref) and [`elocs_clamping!`](@ref).

# Returns
* `new_addrs`: `(B,)` next walker positions (the same array as `addrs_n`).
* `E_locs`: `(B,)` local energies of the configurations `n`.
* `weights`: `(B,)` CTMC residence-time weights.
* `grads_n`: `(p, B)` per-sample gradients of `log ψ(n)`, or `nothing` if `vmc_buf.start`
  is `false`.
* `acc`: fraction of walkers that moved, in `[0, 1]`.

All arrays live on the same device (CPU/GPU) as the inputs.

## Note
When `vmc_buf.start` is `false` (e.g. during burn-in), only the walkers are advanced.
In that case `E_locs` and `weights` are **not** computed, and their contents are
meaningless.
"""
function ctmc_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz; eclamp::Bool=true)
    B = length(addrs_n) # batch 

    # --- STEP 0: all needed variables from buffer ---------------------------------
    addrs_m = vmc_buf.addrs_m       
    ham_diag = vmc_buf.ham_diag     
    ham_offdiag_buf = vmc_buf.ham_offdiag_buf
    addrs_offdiag_buf = vmc_buf.addrs_offdiag_buf
    offsets = vmc_buf.offsets
    accepted = vmc_buf.accepted
    E_locs = vmc_buf.E_locs                     
    vals_offdiag_buf = vmc_buf.vals_offdiag_buf
    weights = vmc_buf.weights               

    # --- STEP 1: collect all offdiagonals -----------------------------------------
    mask = ansatz.truncation === nothing ? nothing : ansatz.truncation.mask
    collect_offdiagonals!(addrs_offdiag_buf, ham_offdiag_buf, ham_diag, offsets, 
                          hamiltonian, addrs_n, mask)
    total_offdiag = isempty(offsets) ? 0 : Int(@allowscalar offsets[end])
    total_offdiag == 0 && error("no valid off-diagonals for any walker: the sampler cannot move " *
                                "(check the truncation mask)")

    # --- STEP 2: NN forward on offdiagonals ---------------------------------------
    vals_offdiag = multi_compute_logψ!(ansatz, addrs_offdiag_buf, vals_offdiag_buf, offsets)
    vals_offdiag_valid = view(vals_offdiag, :, 1:total_offdiag)

    # --- STEP 3: Propose new addresses --------------------------------------------
    m_logψ, m_sign = log_psi!(ansatz, vals_offdiag_valid)
    if ansatz.meanfield !== nothing 
        ansatz.meanfield(addrs_offdiag_buf.data, offsets, m_logψ)    # MF on offdiags before proposal
    end

    ctmc_state_proposal!(addrs_m, addrs_offdiag_buf.data, offsets, m_logψ, E_locs) # E_locs buffer reuse

    # --- STEP 4: NN on starting addresses -----------------------------------------
    vals_n = compute_logψ(ansatz, addrs_n)

    Random.rand!(E_locs)
    accepted .= E_locs .< 0.75f0     # move with prob 0.75 → stay with prob 0.25

    # --- STEP 5: E_loc, gradient, and weights calculations ------------------------
    if !vmc_buf.start
        grads_n = nothing
    else
        grads_n = back_jacobian!(ansatz, jacobian_buf)  # (p, B)
        neuron_statistics(ansatz; idx=vmc_buf.block_idx)
        jacobian_statistics(ansatz, jacobian_buf.J; idx=vmc_buf.block_idx)

        n_logψ, n_sign = log_psi!(ansatz, vals_n)
        if ansatz.meanfield !== nothing 
            ansatz.meanfield(addrs_n, n_logψ)    # MF on original addresses
        end

        calculate_local_energy!(ansatz, vmc_buf, n_logψ, n_sign, m_logψ, m_sign; eclamp=eclamp) # saved in E_locs
        get_ctmc_weights!(weights, m_logψ, offsets, n_logψ) # saved in weights
    end

    # --- STEP 6: new sampled addresses --------------------------------------------
    addrs_n .= ifelse.(accepted, addrs_m, addrs_n)
    new_addrs = addrs_n # reference for addrs_n 
    acc = sum(accepted) / B

    # --- RETURNS ------------------------------------------------------------------
    # new_addrs: (B,) next walker positions -> CPU/GPU
    # E_locs:    (B,) local energies        -> CPU/GPU
    # weights:   (B,) sampler weights       -> CPU/GPU
    # grads_n:   (p,B) gradients            -> CPU/GPU
    # acc:       acceptance over batch input ([0,1] -> 0.75 = 75%)
    return new_addrs, E_locs, weights, grads_n, acc
end

