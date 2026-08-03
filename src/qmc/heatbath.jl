
function metropolis_heatbath_sample!(metropolis_buf, jacobian_buf, hamiltonian, addrs_n, ansatz)
    B = length(addrs_n) # batch 

    # --- STEP 0: all needed variables from buffer ------------------------------------
    addrs_m          = metropolis_buf.addrs_m         # one proposed addr_m per walker
    diag_ham         = metropolis_buf.diag_ham        # H_nn diagonal per walker
    flat_addrs_all   = metropolis_buf.flat_addrs_m
    flat_Hmn_all     = metropolis_buf.flat_offdiag_ham
    walker_idx_all   = metropolis_buf.walker_idx
    offsets_all      = metropolis_buf.offsets
    accepted         = metropolis_buf.accepted
    total_buf        = metropolis_buf.total_buf
    E_locs = metropolis_buf.E_locs
    flat_vals_m = metropolis_buf.flat_vals_m
    vals_n_cpu       = metropolis_buf.vals_n_cpu    # used here to save weights of ctmc at the end
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
        # nonzero_count = 0

        for (k, (addr_m, H_mn)) in enumerate(offdiagonals(col))
          # collect ALL off-diagonals for E_loc
          #println(H_mn)
          if iszero(H_mn) # ignore zero off_diagonals elements
              continue
          end
          # nonzero_count += 1
          push!(flat_addrs_all, addr_m)
          push!(flat_Hmn_all,   H_mn)
          push!(walker_idx_all, b)
          push!(total_buf, H_mn) # for Loss norm if needed
        end
        offsets_all[b+1] = length(flat_addrs_all) # offsets are lengths of spawned offdiagonals
        # inspiration: https://pubs.acs.org/doi/10.1021/acs.jctc.6b00407
        _state_proposal!(offsets_all, flat_addrs_all, total_buf, addrs_m, b)
    end



    # --- STEP 2: NN forward on proposals ---------------------------------------------
    vals_m = compute_logψ(ansatz, addrs_m)
    copyto!(ansatz.model.z_last, vals_m)    # save computed values 
    vals_m = ansatz.model.z_last

    # --- STEP 3: NN on starting addresses --------------------------------------------
    vals_n  = compute_logψ(ansatz, addrs_n) #[1,:]

    # --- STEP 4: acceptance of proposed offdiagonals ---------------------------------
    # reuse vals_m as buffer for ratios
    vals_m   .= exp.(clamp.(2f0 .* (vals_m .- vals_n), -80f0, 80f0)) # (B,)
    ratios    = vals_n_cpu    # reuse _cpu buffer for GPU->CPU
    copyto!(ratios, vals_m)                  # (B,) 
    draws     = rand(Float64, B)             # (B,)
    accepted .= draws .< view(ratios, 1, :)              # (B,) Bool
    acc       = sum(accepted)/B

    # --- STEP 5: new sampled addresses - CPU ONLY ------------------------------------
    addrs_n  .= ifelse.(accepted, addrs_m, addrs_n) # (B,) - reuse addrs_n as buffer
    new_addrs = addrs_n       # reference for addrs_n 


    if !metropolis_buf.start
        # no need to calculate gradient during thermalization
        grads_n = nothing
    else
        # jacobian using last NN forward for calculations => vals_n 
        grads_n = back_jacobian!(jacobian_buf)  # (p, B)    
        copyto!(ansatz.model.z_last, vals_n)    # save as fill_flat will rewrite NN
        vals_n = ansatz.model.z_last

        total    = length(flat_addrs_all)
        n_chunks = cld(total, B)        # number of batches run throuh NN
        empty!(flat_vals_m)
        _fill_flat_vals_m!(ansatz, n_chunks, B, total, flat_addrs_all, addrs_m, 
                          vals_n_cpu, flat_vals_m) # using vals_n_cpu as buffer for forward passes 

        total_buf .= total_buf .* exp.(flat_vals_m) # reusing buffers for CTMC proposal/also used in E_locs
    end

    copyto!(vals_n_cpu, vals_n) 

    # --- STEP 6: E_loc calculations --------------------------------------------------
    weights = nothing # uniform weights in Metropolis
    raw_norm = nothing
    if metropolis_buf.start === true
        # fill E_locs from buffer with local energies
        calculate_local_energy!(ansatz, metropolis_buf, :metropolis)
        raw_norm = get_ctmc_weights!(total_buf, offsets_all, vals_n_cpu, B)        # saved in vals_n_cpu
    end

    # --- RETURNS ---------------------------------------------------------------------
    # new_addrs: (B,) next walker positions -> CPU
    # E_locs:    (B,) local energies        -> CPU (possibly GPU)
    # grads_n:   (p,B) gradients            -> GPU 
    # acc:       acceptance over batch input (in %)
    return new_addrs, E_locs, weights, grads_n, acc, raw_norm
end

function ctmc_heatbath_sample!(metropolis_buf, jacobian_buf, hamiltonian, addrs_n, ansatz)
    B = length(addrs_n) # batch 

    # --- STEP 0: all needed variables from buffer ------------------------------------
    addrs_m          = metropolis_buf.addrs_m         # one proposed addr_m per walker
    diag_ham         = metropolis_buf.diag_ham        # H_nn diagonal per walker
    flat_addrs_all   = metropolis_buf.flat_addrs_m
    flat_Hmn_all     = metropolis_buf.flat_offdiag_ham
    walker_idx_all   = metropolis_buf.walker_idx
    offsets_all      = metropolis_buf.offsets
    accepted         = metropolis_buf.accepted
    total_buf        = metropolis_buf.total_buf
    vals_n_cpu       = metropolis_buf.vals_n_cpu    # used here to save weights of ctmc at the end
    vec_cpu          = metropolis_buf.vec_cpu
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
        # nonzero_count = 0

        for (k, (addr_m, H_mn)) in enumerate(offdiagonals(col))
          # collect ALL off-diagonals for E_loc
          #println(H_mn)
          if iszero(H_mn) # ignore zero off_diagonals elements
              continue
          end
          # nonzero_count += 1
          push!(flat_addrs_all, addr_m)
          push!(flat_Hmn_all,   H_mn)
          push!(walker_idx_all, b)
          push!(total_buf, H_mn) # buffer for CTMC proposal
        end
        offsets_all[b+1] = length(flat_addrs_all) # offsets are lengths of spawned offdiagonals
        # inspiration: https://pubs.acs.org/doi/10.1021/acs.jctc.6b00407
    end

    total    = length(flat_addrs_all)
    n_chunks = cld(total, B)        # number of batches run throuh NN
    E_locs = metropolis_buf.E_locs
    flat_vals_m = metropolis_buf.flat_vals_m
    empty!(flat_vals_m)

    # --- STEP 2: NN forward on proposals ---------------------------------------------
    _fill_flat_vals_m!(ansatz, n_chunks, B, total, flat_addrs_all, addrs_m, 
                      vals_n_cpu, flat_vals_m)

    total_buf .= total_buf .* exp.(flat_vals_m) # reusing buffers for CTMC proposal/also used in E_locs

    # --- STEP 2.1: Propose new addresses ---------------------------------------------
    for b in 1:B
        _state_proposal!(offsets_all, flat_addrs_all, total_buf, addrs_m, b) # new addresses are in addrs_m
    end

    # --- STEP 3: NN on starting addresses --------------------------------------------
    vals_n  = compute_logψ(ansatz, addrs_n)
    copyto!(vals_n_cpu, vals_n)

    if !metropolis_buf.start
        grads_n = nothing
    else
        grads_n = back_jacobian!(jacobian_buf)  # (p, B)
    end

    # --- STEP 5: new sampled addresses - CPU ONLY ------------------------------------
    addrs_n  .= addrs_m # (B,) - reuse addrs_n as buffer
    new_addrs = addrs_n       # reference for addrs_n 

    # --- STEP 6: E_loc calculations --------------------------------------------------
    raw_norm = nothing
    if metropolis_buf.start === true
        # fill E_locs from buffer with local energies
        calculate_local_energy!(ansatz, metropolis_buf, :ctmc)      # saved in E_locs
        # saves CTMC weights in vals_n_cpu
        raw_norm = get_ctmc_weights!(total_buf, offsets_all, vals_n_cpu, B)        # saved in vals_n_cpu
    end
    acc = 1
    copyto!(vec_cpu, view(vals_n_cpu, 1, :))
    weights = vec_cpu 
    # --- RETURNS ---------------------------------------------------------------------
    # new_addrs: (B,) next walker positions -> CPU
    # E_locs:    (B,) local energies        -> CPU (possibly GPU)
    # grads_n:   (p,B) gradients            -> GPU 
    # acc:       accpentance is always 1 in ctmc
    return new_addrs, E_locs, weights, grads_n, acc, raw_norm
end
