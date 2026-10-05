
# """
#     ctmc_heatbath_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz)
#         -> new_addrs, E_locs, weights, grads_n, acc
#
# Similar as [`ctmc_sample!`](@ref) but during proposal step, new addresses are proposed 
# using random draw weighted by hamiltonian elements - heatbath. Also CTMC weights
# calculation is weighted with its hamiltonian values.
#
# ## Notes
# heatbath inspiration:
# https://pubs.acs.org/doi/10.1021/acs.jctc.6b00407
# """
# function ctmc_heatbath_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz)
#     B = length(addrs_n) # batch 
#
#     # --- STEP 0: all needed variables from buffer ------------------------------------
#     addrs_m = vmc_buf.addrs_m         
#     diag_ham = vmc_buf.diag_ham     
#     flat_addrs_all = vmc_buf.flat_addrs_m
#     flat_Hmn_all = vmc_buf.flat_offdiag_ham
#     walker_idx_all = vmc_buf.walker_idx
#     offsets_all = vmc_buf.offsets
#     accepted = vmc_buf.accepted
#     total_buf = vmc_buf.total_buf
#     vals_n_cpu = vmc_buf.vals_n_cpu
#     vec_cpu = vmc_buf.vec_cpu
#     E_locs = vmc_buf.E_locs
#     flat_vals_m = vmc_buf.flat_vals_m
#
#     # Due to uknown size of all possible offdiagonals I push! dynamically
#     empty!(flat_addrs_all)
#     empty!(flat_Hmn_all)
#     empty!(walker_idx_all)
#     empty!(total_buf)
#
#     # --- STEP 1: one random proposal per walker + collect ALL off-diags for E_loc ----
#     offsets_all[1] = 0
#     for b in 1:B
#         col           = hamiltonian * addrs_n[b]
#         diag_ham[b]   = diagonal_element(col)
#
#         for (k, (addr_m, H_mn)) in enumerate(offdiagonals(col))
#           # collect ALL off-diagonals for E_loc
#           if iszero(H_mn) # ignore zero off_diagonals elements
#               continue
#           end
#
#           # reject any offdiagonal that leaves the truncated subspace
#           occ_m = onr(addr_m)
#           if ansatz.truncation !== nothing && violates_truncation(occ_m, ansatz.truncation.mask)
#               continue
#           end
#
#           push!(flat_addrs_all, addr_m)
#           push!(flat_Hmn_all,   H_mn)
#           push!(walker_idx_all, b)
#         end
#         offsets_all[b+1] = length(flat_addrs_all) # offsets are lengths of spawned offdiagonals
#     end
#
#     # --- STEP 2: NN forward on offdiagonals ----------------------------------------
#     multi_compute_logψ!(ansatz, flat_addrs_all, flat_vals_m)
#
#     # --- STEP 3: Propose new addresses ---------------------------------------------
#     resize!(total_buf, length(flat_addrs_all))
#     total_buf .= psi(ansatz, flat_vals_m) .* flat_Hmn_all
#
#     for b in 1:B
#         _state_proposal!(offsets_all, flat_addrs_all, total_buf, addrs_m, b) # new addresses are in addrs_m
#     end
#
#     # --- STEP 4: NN on starting addresses --------------------------------------------
#     vals_n  = compute_logψ(ansatz, addrs_n)
#     copyto!(vals_n_cpu, vals_n)
#     if ansatz.meanfield !== nothing
#         compute_mflogψ!(ansatz, addrs_n, vals_n_cpu)
#     end
#
#     if !vmc_buf.start
#         grads_n = nothing
#     else
#         grads_n = back_jacobian!(ansatz, jacobian_buf)  # (p, B)
#         neuron_statistics(ansatz; idx=vmc_buf.block_idx)
#         jacobian_statistics(ansatz, jacobian_buf.J; idx=vmc_buf.block_idx)
#     end
#
#     # --- STEP 5: new sampled addresses - CPU ONLY ------------------------------------
#     addrs_n  .= addrs_m # (B,) - reuse addrs_n as buffer
#     new_addrs = addrs_n # reference for addrs_n 
#
#     # --- STEP 6: E_loc calculations --------------------------------------------------
#     if vmc_buf.start === true
#         n_logψ, n_sign = log_psi!(ansatz.ansatz_type, ansatz, vals_n_cpu)
#         calculate_local_energy!(ansatz, vmc_buf, n_logψ, n_sign) # saved in E_locs
#         get_ctmc_weights!(total_buf, offsets_all, n_logψ, B) # saved in n_logψ
#         copyto!(vec_cpu, n_logψ)
#     end
#     acc = 1
#     weights = vec_cpu 
#
#     # --- RETURNS ---------------------------------------------------------------------
#     # new_addrs: (B,) next walker positions -> CPU
#     # E_locs:    (B,) local energies        -> CPU (possibly GPU)
#     # weights:   sampler weights 
#     # grads_n:   (p,B) gradients            -> GPU 
#     # acc:       acceptance over batch input (in %)
#     return new_addrs, E_locs, weights, grads_n, acc
# end


"""
    ctmc_sample!(vmc_buf, jacobian_buf, hamiltonian, addrs_n, ansatz)
        -> new_addrs, E_locs, weights, grads_n, acc

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
    ham_diag = vmc_buf.ham_diag     
    ham_offdiag_buf = vmc_buf.ham_offdiag_buf
    addrs_offdiag_buf = vmc_buf.addrs_offdiag_buf
    offsets = vmc_buf.offsets
    accepted = vmc_buf.accepted
    E_locs = vmc_buf.E_locs                     
    vals_offdiag_buf = vmc_buf.vals_offdiag_buf
    weights = vmc_buf.weights               


    mask = ansatz.truncation === nothing ? nothing : ansatz.truncation.mask
    collect_offdiagonals!(addrs_offdiag_buf, ham_offdiag_buf, ham_diag, offsets, 
                          hamiltonian, addrs_n, mask)
    total_offdiag = isempty(offsets) ? 0 : Int(@allowscalar offsets[end])
    total_offdiag == 0 && error("no valid off-diagonals for any walker: the sampler cannot move " *
                                "(check the truncation mask)")

    # --- STEP 2: NN forward on offdiagonals ----------------------------------------
    vals_offdiag = multi_compute_logψ!(ansatz, addrs_offdiag_buf, vals_offdiag_buf, offsets)
    vals_offdiag_valid = view(vals_offdiag, :, 1:total_offdiag)

    # --- STEP 3: Propose new addresses ---------------------------------------------
    m_logψ, m_sign = log_psi!(ansatz, vals_offdiag_valid)

    state_proposal!(addrs_m, addrs_offdiag_buf.data, offsets, m_logψ, E_locs) # E_locs buffer reuse

    # --- STEP 4: NN on starting addresses --------------------------------------------
    vals_n = compute_logψ(ansatz, addrs_n)
    n_logψ, n_sign = log_psi!(ansatz, vals_n)
    # copyto!(vals_n_cpu, vals_n)
    # if ansatz.meanfield !== nothing
    #     compute_mflogψ!(ansatz, addrs_n, vals_n_cpu)
    # end

    # --- STEP 5: E_loc, gradient, and weights calculations ---------------------------
    if !vmc_buf.start
        grads_n = nothing
    else
        grads_n = back_jacobian!(ansatz, jacobian_buf)  # (p, B)
        neuron_statistics(ansatz; idx=vmc_buf.block_idx)
        jacobian_statistics(ansatz, jacobian_buf.J; idx=vmc_buf.block_idx)

        calculate_local_energy!(ansatz, vmc_buf, n_logψ, n_sign, m_logψ, m_sign) # saved in E_locs
        get_ctmc_weights!(weights, m_logψ, offsets, n_logψ) # saved in weights
    end

    # --- STEP 6: new sampled addresses -----------------------------------------------
    addrs_n  .= addrs_m # (B,) - reuse addrs_n as buffer
    new_addrs = addrs_n # reference for addrs_n 
    acc = 1

    # --- RETURNS ---------------------------------------------------------------------
    # new_addrs: (B,) next walker positions -> CPU/GPU
    # E_locs:    (B,) local energies        -> CPU/GPU
    # weights:   sampler weights            -> CPU/GPU
    # grads_n:   (p,B) gradients            -> CPU/GPU
    # acc:       acceptance over batch input (in %)
    return new_addrs, E_locs, weights, grads_n, acc
end

"""
    collect_offdiagonals!(addrs_buf, hmn_buf, diag, offsets, hamiltonian, addrs_n, mask)

Collect, on the device, the diagonal element and all valid off-diagonal connections of
every address in `addrs_n`.

Column `b` of `addrs_buf.data` and `hmn_buf.data` receives the off-diagonal addresses and
offdiagonal matrix elements of parent `b`; `diag[b]` its diagonal element; `offsets` the 
cumulative sum of number of offdiagonals. Off-diagonals with a zero matrix element, and 
(if `mask !== nothing`) those violating the truncation, are skipped.

If any parent has more off-diagonals than the current number of rows `K` (in `addrs_buf` or
`hmn_buf`), both row buffers are grown to fit the largest count and the collection is repeated, 
so no connection is ever lost. The grown size is kept for later calls.

# Variables
* `addrs_buf`: [`GPUGrowRowBuffer`](@ref) of addresses, `(K, B)`.
* `hmn_buf`: [`GPUGrowRowBuffer`](@ref) of Float32 matrix elements, `(K, B)`.
* `diag`: (B,) receives the diagonal elements.
* `offsets`: (B,) Int32 device vector, receives `cumsum(num_offdiagonals)`.
* `hamiltonian`: Rimu Hamiltonian; has to be Float32/isbits compatible.
* `addrs_n`: (B,) vector of the current addresses.
* `mask`: truncation mask on the device, or `nothing` for no truncation.
"""
function collect_offdiagonals!(addrs_buf::GPUGrowRowBuffer, hmn_buf::GPUGrowRowBuffer,
                               diag, offsets, hamiltonian, addrs_n, mask)
    B = length(addrs_n)
    B == 0 && return nothing
    backend = KernelAbstractions.get_backend(diag)

    while true
        addrs_offdiag = addrs_buf.data          # (K, B)
        hmn = hmn_buf.data                      # (K, B)
        _collect_offdiagonals_kernel!(backend)(addrs_offdiag, hmn, diag, offsets,
                                               hamiltonian, addrs_n, mask; ndrange = B)
        KernelAbstractions.synchronize(backend)

        kmax = Int(maximum(offsets))                    # largest true count
        kmax <= size(addrs_offdiag, 1) && break         # everything fitted

        ensure_capacity!(addrs_buf, kmax)               # grow K for both, then repeat
        ensure_capacity!(hmn_buf, kmax)
    end

    cumsum!(offsets, offsets)                          # cumulative offsets, on the device
    return nothing
end

# one thread per parent b: iterates the column H * addrs_n[b]
@kernel function _collect_offdiagonals_kernel!(addrs_offdiag, hmn, diag, counts,
                                               hamiltonian, @Const(addrs_n), mask)
    b = @index(Global, Linear)              # parent
    K = size(addrs_offdiag, 1)                    # rows available per parent
    addr = addrs_n[b]
    k = 0                                   # valid off-diagonals found

    col = hamiltonian * addr
    diag[b] = diagonal_element(col)

    for (a2, v) in offdiagonals(col)
        if !iszero(v) && !_skip_truncated(a2, mask)
            k += 1
            if k <= K                       # write only what fits array
                addrs_offdiag[k, b] = a2
                hmn[k, b] = v
            end
        end
    end

    counts[b] = Int32(k)            # always the true count
end

# truncation switch resolved by dispatch: with `nothing`
@inline _skip_truncated(addr, ::Nothing) = false
@inline _skip_truncated(addr, mask) = violates_truncation(onr(addr), mask)
