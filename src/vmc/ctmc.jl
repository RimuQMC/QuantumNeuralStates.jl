# using KernelAbstractions

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

    # --- STEP 0: all needed variables from buffer ---------------------------------
    addrs_m = vmc_buf.addrs_m       
    ham_diag = vmc_buf.ham_diag     
    ham_offdiag_buf = vmc_buf.ham_offdiag_buf
    addrs_offdiag_buf = vmc_buf.addrs_offdiag_buf
    offsets = vmc_buf.offsets
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

    ctmc_state_proposal!(addrs_m, addrs_offdiag_buf.data, offsets, m_logψ, E_locs) # E_locs buffer reuse

    # --- STEP 4: NN on starting addresses -----------------------------------------
    vals_n = compute_logψ(ansatz, addrs_n)
    # copyto!(vals_n_cpu, vals_n)
    # if ansatz.meanfield !== nothing
    #     compute_mflogψ!(ansatz, addrs_n, vals_n_cpu)
    # end

    # --- STEP 5: E_loc, gradient, and weights calculations ------------------------
    if !vmc_buf.start
        grads_n = nothing
    else
        grads_n = back_jacobian!(ansatz, jacobian_buf)  # (p, B)
        neuron_statistics(ansatz; idx=vmc_buf.block_idx)
        jacobian_statistics(ansatz, jacobian_buf.J; idx=vmc_buf.block_idx)

        n_logψ, n_sign = log_psi!(ansatz, vals_n)
        calculate_local_energy!(ansatz, vmc_buf, n_logψ, n_sign, m_logψ, m_sign) # saved in E_locs
        get_ctmc_weights!(weights, m_logψ, offsets, n_logψ) # saved in weights
    end

    # --- STEP 6: new sampled addresses --------------------------------------------
    addrs_n .= addrs_m # (B,) - reuse addrs_n as buffer
    new_addrs = addrs_n # reference for addrs_n 
    acc = 1

    # --- RETURNS ------------------------------------------------------------------
    # new_addrs: (B,) next walker positions -> CPU/GPU
    # E_locs:    (B,) local energies        -> CPU/GPU
    # weights:   (B,) sampler weights            -> CPU/GPU
    # grads_n:   (p,B) gradients            -> CPU/GPU
    # acc:       acceptance over batch input (in %)
    return new_addrs, E_locs, weights, grads_n, acc
end

