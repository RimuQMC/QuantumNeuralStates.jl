# using Rimu
# using NNlib
# using Statistics
# using StatsBase
# using KernelAbstractions

# ---------------------------------------------------------------------------
# Metropolis Monte Carlo sampler (works for both single/batch inputs)
# inspiration: https://arxiv.org/pdf/2402.11014
# ---------------------------------------------------------------------------
"""
    VMCBuffer(ansatz, addr)

This structure holds all necessary intermediate variables for VMC steps. If 
`ansatz` lives on GPU than it can work in CPU/GPU hybrid regime.

# Variables

* `ansatz`: [`NeuralAnsatz`](@ref).
* `addr`: Rimu type of address in Fock-state representation.

"""
mutable struct VMCBuffer{A, VA <: AbstractVector{A}}
    addrs_m::VA                         # (B,)      - spawned and chosen addresses
    flat_addrs_m::VA                    # (total,)  - all spawned addresses
    flat_vals_m::Vector{Float64}        # (total*out_dim,) - outputs NN(flat_addrs_m)
    flat_offdiag_ham::Vector{Float64}   # (total,)  - H_mn values for all spawned addresses
    diag_ham::Vector{Float64}           # (B,)      - H_nn values
    start::Bool                         # when to start E_loc calculations (after termalisation)
    offsets::Vector{Int}                # (B+1,)    - number of offdiagonals from each spawning address
    vals_n_cpu::Matrix{Float64}         # (1, B)    - for transfer GPU -> CPU
    vec_cpu::Vector{Float64}            # (B,)      - for transfer view on CPU
    E_locs::Vector{Float64}             # (B,)      - calculated local energies, and GPU -> CPU usage
    accepted::Vector{Bool}              # (B,)      - boolean vector accept/reject
    total_buf::Vector{Float64}          # (total,)  - helper for CTMC Proposal buffer
    block_idx::Int                      # keeps track of what iteration block I am in
end
function VMCBuffer(ansatz, addr)
    batch = ansatz.model.batch
    out_dim = size(last(ansatz.model.layers).z, 1)
    A = typeof(addr)
    T = Float32

    start = false
    addrs_m = Vector{A}(undef, batch)
    flat_addrs_m = Vector{A}(undef, 0)
    flat_vals_m = Vector{T}(undef, 0)
    flat_offdiag_ham = Vector{T}(undef, 0)
    diag_ham = Vector{T}(undef, batch)
    offset = Vector{Int32}(undef, batch+1)
    vals_n_cpu = Matrix{T}(undef, out_dim, batch)
    vec_cpu = Vector{T}(undef, batch)
    E_locs = Vector{T}(undef, batch)
    accepted = Vector{Bool}(undef, batch)
    total_buf = Vector{T}(undef, 0)
    VA = typeof(addrs_m)
    return VMCBuffer{A, VA}(addrs_m, flat_addrs_m, flat_vals_m, flat_offdiag_ham, diag_ham,
                    start, walker_idx, offset, vals_n_cpu, vec_cpu, E_locs, accepted, total_buf, 1)
end

# @kernel function _state_proposal_kernel!(addrs_m, offsets, addrs_m_all, distro, rand_vals)
#     b = @index(Global)
#     @inbounds begin
#         start = offsets[b]
#         stop  = offsets[b+1]
#
#         total_w = 0f0
#         for i in (start+1):stop
#             total_w += abs(distro[i])
#         end
#
#         target  = rand_vals[b] * total_w
#         cumul   = 0f0
#         k_prop  = stop                 # fallback: last element (numerical drift), same as CPU version
#         for i in (start+1):stop
#             cumul += abs(distro[i])
#             if cumul >= target
#                 k_prop = i
#                 break
#             end
#         end
#
#         addrs_m[b] = addrs_m_all[k_prop]
#     end
# end
# function state_proposal!(addrs_m, offsets, addrs_m_all, distro, rand_vals, batch)
#     Random.rand!(rand_vals)
#     backend = KernelAbstractions.get_backend(distro)
#     _state_proposal_kernel!(backend)(addrs_m, offsets, addrs_m_all, distro, rand_vals; ndrange=batch)
#     KernelAbstractions.synchronize(backend)
#     return addrs_m
# end

"""
    _state_proposal!(offsets, addrs_m_all, distro, addrs_m, b)

This function propose new address in VMC sampler step. From the spawning address
I collect all offdiagonal connections in `addrs_m_all` which are order using 
`offsets` so it is clear what addresses belongs to what spawning source address.
The new proposed address is then randomly picked from distribution `distro`.

# Variables

* `offsets`: vector that holds ordering of spawning address to its offdiagonals.
* `addrs_m_all`: flatten vector that holds all offdiagonals address from current batch.
* `distro`: distribution vector. For example, if filled with `1` then the random draw is
    uniform, if filled with corresponding offdiagonal hamiltonian matrix elements `H_mn` the
    random draw is weighted by the elements.
* `addrs_m`: buffer vector that holds all newly proposed states (size of batch).
* `b`: batch size.
"""
@kernel function _state_proposal_kernel!(k_prop_out, offsets, distro, rand_vals)
    b = @index(Global)
    @inbounds begin
        start = offsets[b]
        stop = offsets[b+1]

        total_w = 0f0
        for i in (start+1):stop
            total_w += abs(distro[i])
        end

        target = rand_vals[b] * total_w
        cumul = 0f0
        k = stop   # fallback
        for i in (start+1):stop
            cumul += abs(distro[i])
            if cumul >= target
                k = i
                break
            end
        end

        k_prop_out[b] = k # proposed indices
    end
end
function state_proposal!(addrs_m, addrs_m_all, offsets, distro, rand_vals, 
                         k_prop_buf, k_prop_cpu, batch)
    Random.rand!(rand_vals)

    backend = KernelAbstractions.get_backend(distro)
    _state_proposal_kernel!(backend)(k_prop_buf, offsets, distro, rand_vals; 
                                     ndrange=batch)
    KernelAbstractions.synchronize(backend)

    copyto!(k_prop_cpu, k_prop_buf)
    @inbounds for b in 1:batch
        addrs_m[b] = addrs_m_all[k_prop_cpu[b]]
    end
    return addrs_m
end
# function _state_proposal!(offsets, addrs_m_all, distro, addrs_m, b)
#     nonzero_count = offsets[b+1] - offsets[b]
#
#     # inverse CDF algorthm
#     range_start = offsets[b]
#     total_w = 0.0
#     @inbounds for i in (offsets[b]+1):offsets[b+1] #range_start:length(distro)
#         total_w += abs(distro[i])
#     end
#     target = rand() * total_w   # uniform in [0, total_w)
#     cumul = 0.0
#     k_prop = nonzero_count + 1   # fallback if numerical drift
#     @inbounds for i in (offsets[b]+1):offsets[b+1] #range_start:length(distro)
#         cumul += abs(distro[i])
#         if cumul >= target
#             k_prop = i - range_start + 1
#             break
#         end
#     end
#     addrs_m[b] = addrs_m_all[range_start + k_prop - 1]
# end

"""
    get_ctmc_weights!(distro, offsets, log_psi, batch)

This function calculates CTMC weights for batched approach.

```math
w_b = \\frac{|\\psi(n_b)|}{\\sum_m \\text{distro}(m_b)}
```
where `distro` is CDF probability distribution also used in VMC [`_state_proposal!`](@ref).

```math
||\\psi||^2 = \\sum_s p(s) \\frac{|\\psi^2|}{p(s)} = \\sum_s p(s)*Z*\\frac{|\\psi(s)|}{R(s)} =
Z * \\mathbf{E}_{s~p} \\big[ \\frac{|\\psi(s)|}{R(s)} \\big]
```

# Variables
* `distro`: distribution same as in [`_state_proposal!`](@ref).
* `offsets`: vector that maps offdiagonal spawns from its spawning source.
* `log_psi`: holds log amplitudes of wave-function calculated using [`log_psi!`](@ref).
* `batch`: batch number.
"""
function get_ctmc_weights!(distro, offsets, log_psi, batch)
    backend = KernelAbstractions.get_backend(distro)
    _ctmc_weights_kernel!(backend)(log_psi, offsets, distro; ndrange=batch)
    KernelAbstractions.synchronize(backend)

    sum_weights = sum(log_psi)
    log_psi ./= sum_weights
end

@kernel function _ctmc_weights_kernel!(log_psi, offsets, distro)
    b = @index(Global)
    @inbounds begin
        total = 0f0
        for i in (offsets[b]+1):offsets[b+1]
            total += abs(distro[i])
        end
        log_psi[b] = exp(clamp(log_psi[b] - log(total + 1f-35), -80f0, 80f0))
    end
end
# function get_ctmc_weights!(distro, walker_idx, log_psi, tmp_vec, batch)
#     NNlib.scatter!(+, tmp_vec, distro, walker_idx) # sum of |ψ(m)| offdiagonals
#     log_psi .= exp.(clamp.(log_psi .- log.(tmp_vec .+ 1f-35), -80f0, 80f0))
#     sum_weights = sum(log_psi)
#     println(sum_weights)
#     log_psi ./= sum_weights
# end
# function get_ctmc_weights!(distro, offsets, log_psi, batch)
#     # inverse CDF algorithm
#     sum_weights = 0.0
#     for b in 1:batch
#         total_w = 0.0
#         @inbounds for i in (offsets[b]+1):offsets[b+1]
#             total_w += abs(distro[i])
#         end
#         total_w = log(total_w + 1f-35)
#         log_psi[b] = exp(clamp(log_psi[b] - total_w, -80f0, 80f0))
#         sum_weights += log_psi[b]
#     end
#     println(sum_weights)
#     # raw_norm = sum_weights / batch
#     log_psi ./= sum_weights
#
#     # return raw_norm # if loss function cares about normalisation
# end
