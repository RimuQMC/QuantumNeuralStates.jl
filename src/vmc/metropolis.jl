
"""
    metropolis_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz; eclamp=true)
        -> new_addrs, E_locs, weights, grads_n, acc

Perform one step of the Metropolis–Hastings (MCMC) sampler for VMC.

For each walker `n` in the batch, a new configuration `m` is proposed uniformly at
random among the off-diagonal connections of `n`, i.e. the configurations with
``H_{nm} \\neq 0``, optionally restricted by the truncation mask of `ansatz`.
The proposal is accepted with probability

```math
A(m|n) = \\min\\left(1, \\frac{|\\psi(m)|^2}{|\\psi(n)|^2} \\frac{N_n}{N_m}\\right),
```

where ``N_n`` and ``N_m`` are the numbers of valid off-diagonal connections of `n`
and `m`. The factor ``N_n/N_m`` is the Hastings correction for the uniform proposal
``q(m|n) = 1/N_n``. It ensures detailed balance with respect to ``|\\psi|^2`` even
when configurations have different numbers of connections.

The exponent of the amplitude ratio is clamped to `[-80, 80]` before `exp()` to avoid
`Float32` overflow. Rejected proposals keep the walker in place, so the chain is
aperiodic without an explicit lazy step (compare [`ctmc_sample!`](@ref)).

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
* `E_locs`: `(B,)` local energies of the configurations `n` *before* the move.
* `weights`: `(B,)` uniform weights `1/B`.
* `grads_n`: `(p, B)` per-sample gradients of `log ψ(n)`, or `nothing` if `vmc_buf.start`
  is `false`.
* `acc`: acceptance rate over the batch, in `[0, 1]`.

All arrays live on the same device (CPU/GPU) as the inputs.

## Note
When `vmc_buf.start` is `false` (e.g. during thermalization), only the walkers are
advanced. In that case `E_locs` and `weights` are **not** computed: `E_locs` holds the
random acceptance draws and `weights` keeps its previous contents.
"""
function metropolis_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz; 
                            eclamp::Bool=true)
    B = ansatz.model.batch # batch 

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

    # --- STEP 2: proposal and NN forward on proposal ------------------------------
    metropolis_state_proposal!(addrs_m, addrs_offdiag_buf.data, offsets, E_locs)    # E_locs buffer reuse
    metropolis_count_ratio!(E_locs, addrs_m, offsets, hamiltonian, mask)     # E_locs ← N_n / N_m

    vals_m = compute_logψ(ansatz, addrs_m)
    copyto!(ansatz.model.z_last, vals_m)

    # --- STEP 2.5: NN on off-diagonals (before the addrs_n forward pass!) ---------
    if vmc_buf.start
        vals_offdiag = multi_compute_logψ!(ansatz, addrs_offdiag_buf, vals_offdiag_buf, offsets)
        vals_offdiag_valid = view(vals_offdiag, :, 1:total_offdiag)
        offdiag_logψ, offdiag_sign = log_psi!(ansatz, vals_offdiag_valid)
        if ansatz.meanfield !== nothing
            ansatz.meanfield(addrs_offdiag_buf.data, offsets, offdiag_logψ)   # MF on off-diagonals
        end
    end

    # --- STEP 3: NN on starting addresses -----------------------------------------
    vals_n = compute_logψ(ansatz, addrs_n)

    if !vmc_buf.start
        # no need to calculate gradient during thermalization
        grads_n = nothing
    else
        grads_n = back_jacobian!(ansatz, jacobian_buf)  # (p, B)
        neuron_statistics(ansatz; idx=vmc_buf.block_idx)
        jacobian_statistics(ansatz, jacobian_buf.J; idx=vmc_buf.block_idx)
    end

    # --- STEP 4: acceptance of proposals ------------------------------------------
    m_logψ, m_sign = log_psi!(ansatz, ansatz.model.z_last)
    n_logψ, n_sign = log_psi!(ansatz, vals_n)
    if ansatz.meanfield !== nothing
        ansatz.meanfield(addrs_m, m_logψ)      # MF on proposals
        ansatz.meanfield(addrs_n, n_logψ)      # MF on current walkers
    end

    # A = |ψ_m|²/|ψ_n|² · N_n/N_m   (count ratio stored in E_locs)
    m_logψ .= exp.(clamp.(2f0 .* (m_logψ .- n_logψ), -80f0, 80f0)) .* E_locs # (B,)
    ratios = m_logψ
    Random.rand!(E_locs)            # buffer reuse
    draws = E_locs                  # (B,)
    accepted .= draws .< ratios     # (B,) Bool
    acc = sum(accepted)/B

    # --- STEP 5: new sampled addresses --------------------------------------------
    addrs_n .= ifelse.(accepted, addrs_m, addrs_n)     # (B,) - reuse addrs_n as buffer
    new_addrs = addrs_n     # reference for addrs_n 

    # --- STEP 6: E_loc calculations and weights -----------------------------------
    if vmc_buf.start
        calculate_local_energy!(ansatz, vmc_buf, n_logψ, n_sign, offdiag_logψ, offdiag_sign;
                               eclamp=eclamp) # saved in E_locs
        fill!(weights, 1f0 / B)
    end

    # --- RETURNS ---------------------------------------------------------------------
    # new_addrs: (B,) next walker positions -> CPU/GPU
    # E_locs:    (B,) local energies        -> CPU/GPU
    # weights:   (B,) uniform weights       -> CPU/GPU
    # grads_n:   (p,B) gradients            -> CPU/GPU 
    # acc:       acceptance over batch input ([0,1] -> 0.75 = 75%)
    return new_addrs, E_locs, weights, grads_n, acc
end

