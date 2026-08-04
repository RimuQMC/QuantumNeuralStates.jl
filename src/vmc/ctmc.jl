
"""
    ctmc_heatbath_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz)
        -> new_addrs, E_locs, weights, grads_n, acc, raw_norm

Similar as [`ctmc_sample!`](@ref) but during proposal step, new addresses are proposed 
using random draw weighted by hamiltonian elements - heatbath. Also CTMC weights
calculation is weighted with its hamiltonian values.

## Notes
heatbath inspiration:
https://pubs.acs.org/doi/10.1021/acs.jctc.6b00407
"""
function ctmc_heatbath_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz)
    B = length(addrs_n) # batch 

    # --- STEP 0: all needed variables from buffer ------------------------------------
    addrs_m = vmc_buf.addrs_m         
    diag_ham = vmc_buf.diag_ham     
    flat_addrs_all = vmc_buf.flat_addrs_m
    flat_Hmn_all = vmc_buf.flat_offdiag_ham
    walker_idx_all = vmc_buf.walker_idx
    offsets_all = vmc_buf.offsets
    accepted = vmc_buf.accepted
    total_buf = vmc_buf.total_buf
    vals_n_cpu = vmc_buf.vals_n_cpu
    vec_cpu = vmc_buf.vec_cpu
    E_locs = vmc_buf.E_locs
    flat_vals_m = vmc_buf.flat_vals_m

    # Due to uknown size of all possible offdiagonals I push! dynamically
    empty!(flat_addrs_all)
    empty!(flat_Hmn_all)
    empty!(walker_idx_all)
    empty!(total_buf)

    # --- STEP 1: one random proposal per walker + collect ALL off-diags for E_loc ----
    offsets_all[1] = 0
    for b in 1:B
        col           = hamiltonian * addrs_n[b]
        diag_ham[b]   = diagonal_element(col)

        for (k, (addr_m, H_mn)) in enumerate(offdiagonals(col))
          # collect ALL off-diagonals for E_loc
          if iszero(H_mn) # ignore zero off_diagonals elements
              continue
          end

          # reject any offdiagonal that leaves the truncated subspace
          occ_m = onr(addr_m)
          if ansatz.truncation !== nothing && violates_truncation(occ_m, ansatz.truncation.mask)
              continue
          end

          push!(flat_addrs_all, addr_m)
          push!(flat_Hmn_all,   H_mn)
          push!(walker_idx_all, b)
          push!(total_buf, H_mn)
        end
        offsets_all[b+1] = length(flat_addrs_all) # offsets are lengths of spawned offdiagonals
    end

    # --- STEP 2: NN forward on proposals ---------------------------------------------
    multi_compute_logψ(ansatz, flat_addrs_all, flat_vals_m)

    total_buf .= total_buf .* exp.(clamp.(flat_vals_m, -80f0, 80f0))

    # --- STEP 3: Propose new addresses ---------------------------------------------
    for b in 1:B
        _state_proposal!(offsets_all, flat_addrs_all, total_buf, addrs_m, b) # new addresses are in addrs_m
    end

    # --- STEP 4: NN on starting addresses --------------------------------------------
    vals_n  = compute_logψ(ansatz, addrs_n)
    copyto!(vals_n_cpu, vals_n)
    if ansatz.meanfield !== nothing
        compute_mflogψ!(ansatz, addrs_n, vals_n_cpu)
    end

    if !vmc_buf.start
        grads_n = nothing
    else
        grads_n = back_jacobian!(jacobian_buf)  # (p, B)
        neuron_statistics(ansatz; idx=vmc_buf.block_idx)
        jacobian_statistics(ansatz, jacobian_buf.J; idx=vmc_buf.block_idx)
    end

    # --- STEP 5: new sampled addresses - CPU ONLY ------------------------------------
    addrs_n  .= addrs_m # (B,) - reuse addrs_n as buffer
    new_addrs = addrs_n # reference for addrs_n 

    # --- STEP 6: E_loc calculations --------------------------------------------------
    raw_norm = nothing
    if vmc_buf.start === true
        calculate_local_energy!(ansatz, vmc_buf) # saved in E_locs
        raw_norm = get_ctmc_weights!(total_buf, offsets_all, vals_n_cpu, B) # saved in vals_n_cpu
    end
    acc = 1
    copyto!(vec_cpu, view(vals_n_cpu, 1, :))
    weights = vec_cpu 

    # --- RETURNS ---------------------------------------------------------------------
    # new_addrs: (B,) next walker positions -> CPU
    # E_locs:    (B,) local energies        -> CPU (possibly GPU)
    # weights:   sampler weights 
    # grads_n:   (p,B) gradients            -> GPU 
    # acc:       acceptance over batch input (in %)
    # raw_norm:  norm approximation over sampled batch
    return new_addrs, E_locs, weights, grads_n, acc, raw_norm
end


"""
    ctmc_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz)
        -> new_addrs, E_locs, weights, grads_n, acc, raw_norm

VMC sampler using Continous Time Monte Carlo algorithm (CTMC). New addresses are proposed
from offdiagonal. This algorithm always accept some new proposed state and
introduce so called `residence time`, which represent `weights` of how long the walker 
effectively stays on a configuration before jumping. Also see [`get_ctmc_weights!`](@ref).

# Variables

* `vmc_buf`: [`VMCBuffer`](@ref) intermediate variables for allocation-free calculations.
* `jacobian_buf`: [`JacobianBuffer`](@ref) used during calculation of pre-sample jacobians.
* `hamiltonian`: hamiltonian defined in Rimu.
* `addrs_n`: current sample addresses (of batch size).
* `ansatz`: ansatz for wave-function evaluation. See [`NeuralAnsatz`](@ref).
"""
function ctmc_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz)
    B = length(addrs_n) # batch 

    # --- STEP 0: all needed variables from buffer ------------------------------------
    addrs_m = vmc_buf.addrs_m       
    diag_ham = vmc_buf.diag_ham     
    flat_addrs_all = vmc_buf.flat_addrs_m
    flat_Hmn_all = vmc_buf.flat_offdiag_ham
    walker_idx_all = vmc_buf.walker_idx
    offsets_all = vmc_buf.offsets
    accepted = vmc_buf.accepted
    total_buf = vmc_buf.total_buf
    vals_n_cpu = vmc_buf.vals_n_cpu 
    vec_cpu = vmc_buf.vec_cpu
    E_locs = vmc_buf.E_locs
    flat_vals_m = vmc_buf.flat_vals_m

    # Due to uknown size of all possible offdiagonals I push! dynamically
    empty!(flat_addrs_all)
    empty!(flat_Hmn_all)
    empty!(walker_idx_all)
    empty!(total_buf)

    # --- STEP 1: one random proposal per walker + collect ALL off-diags for E_loc ----
    offsets_all[1] = 0
    for b in 1:B
        col           = hamiltonian * addrs_n[b]
        diag_ham[b]   = diagonal_element(col)

        for (k, (addr_m, H_mn)) in enumerate(offdiagonals(col))
          # collect ALL off-diagonals for E_loc
          if iszero(H_mn) # ignore zero off_diagonals elements
              continue
          end

          # reject any offdiagonal that leaves the truncated subspace
          occ_m = onr(addr_m)
          if ansatz.truncation !== nothing && violates_truncation(occ_m, ansatz.truncation.mask)
              continue
          end

          push!(flat_addrs_all, addr_m)
          push!(flat_Hmn_all,   H_mn)
          push!(walker_idx_all, b)
          push!(total_buf, 1.0)
        end
        offsets_all[b+1] = length(flat_addrs_all) # offsets are lengths of spawned offdiagonals
    end

    # --- STEP 2: NN forward on proposals ---------------------------------------------
    multi_compute_logψ!(ansatz, flat_addrs_all, flat_vals_m)

    total_buf .= exp.(clamp.(flat_vals_m, -80f0, 80f0))

    # --- STEP 3: Propose new addresses ---------------------------------------------
    for b in 1:B
        _state_proposal!(offsets_all, flat_addrs_all, total_buf, addrs_m, b) # new addresses are in addrs_m
    end

    # --- STEP 4: NN on starting addresses --------------------------------------------
    vals_n  = compute_logψ(ansatz, addrs_n)
    copyto!(vals_n_cpu, vals_n)
    if ansatz.meanfield !== nothing
        compute_mflogψ!(ansatz, addrs_n, vals_n_cpu)
    end

    if !vmc_buf.start
        grads_n = nothing
    else
        grads_n = back_jacobian!(jacobian_buf)  # (p, B)
        neuron_statistics(ansatz; idx=vmc_buf.block_idx)
        jacobian_statistics(ansatz, jacobian_buf.J; idx=vmc_buf.block_idx)
    end

    # --- STEP 5: new sampled addresses - CPU ONLY ------------------------------------
    addrs_n  .= addrs_m # (B,) - reuse addrs_n as buffer
    new_addrs = addrs_n # reference for addrs_n 

    # --- STEP 6: E_loc calculations --------------------------------------------------
    raw_norm = nothing
    if vmc_buf.start === true
        calculate_local_energy!(ansatz, vmc_buf) # saved in E_locs
        raw_norm = get_ctmc_weights!(total_buf, offsets_all, vals_n_cpu, B) # saved in vals_n_cpu
    end
    acc = 1
    copyto!(vec_cpu, view(vals_n_cpu, 1, :))
    weights = vec_cpu 

    # --- RETURNS ---------------------------------------------------------------------
    # new_addrs: (B,) next walker positions -> CPU
    # E_locs:    (B,) local energies        -> CPU (possibly GPU)
    # weights:   sampler weights 
    # grads_n:   (p,B) gradients            -> GPU 
    # acc:       acceptance over batch input (in %)
    # raw_norm:  norm approximation over sampled batch
    return new_addrs, E_locs, weights, grads_n, acc, raw_norm
end

