# using Rimu
# using NNlib
# using Statistics
# using StatsBase

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
    walker_idx::Vector{Int}             # (total,)  - map indexing of addrs_n to spawned addrs_m
    offsets::Vector{Int}                # (B+1,)    - number of offdiagonals from each spawning address
    vals_n_cpu::Matrix{Float64}         # (1, B)    - for transfer GPU -> CPU
    vec_cpu::Vector{Float64}            # (B,)      - for transfer view on CPU
    E_locs::Vector{Float64}             # (B,)      - calculated local energies, and GPU -> CPU usage
    accepted::Vector{Bool}              # (B,)      - boolean vector accept/reject
    total_buf::Vector{Float64}          # (total,)  - helper for CTMC Proposal buffer
    block_idx::Int                      # keeps track of what iteration block I am in
end
function VMCBuffer(ansatz, addr)
    batch     = size(last(ansatz.model.layers).z, 2)
    out_dim = size(last(ansatz.model.layers).z, 1)
    A         = typeof(addr)

    addrs_m = Vector{A}(undef, batch)
    flat_addrs_m = Vector{A}(undef, 0)
    flat_vals_m = Vector{Float64}(undef, 0)
    flat_offdiag_ham = Vector{Float64}(undef, 0)
    diag_ham = Vector{Float64}(undef, batch)
    start = false
    walker_idx = Vector{Int}(undef, 0)
    offset = Vector{Int}(undef, batch+1)
    vals_n_cpu = Matrix{Float64}(undef, out_dim, batch)
    vec_cpu = Vector{Float64}(undef, batch)
    E_locs = Vector{Float64}(undef, batch)
    accepted = Vector{Bool}(undef, batch)
    total_buf = Vector{Float64}(undef, 0)
    VA = typeof(addrs_m)
    return VMCBuffer{A, VA}(addrs_m, flat_addrs_m, flat_vals_m, flat_offdiag_ham, diag_ham,
                    start, walker_idx, offset, vals_n_cpu, vec_cpu, E_locs, accepted, total_buf, 1)
end

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
function _state_proposal!(offsets, addrs_m_all, distro, addrs_m, b)
    nonzero_count = offsets[b+1] - offsets[b]

    # inverse CDF algorthm
    range_start = offsets[b]
    total_w = 0.0
    @inbounds for i in (offsets[b]+1):offsets[b+1] #range_start:length(distro)
        total_w += abs(distro[i])
    end
    target = rand() * total_w   # uniform in [0, total_w)
    cumul = 0.0
    k_prop = nonzero_count + 1   # fallback if numerical drift
    @inbounds for i in (offsets[b]+1):offsets[b+1] #range_start:length(distro)
        cumul += abs(distro[i])
        if cumul >= target
            k_prop = i - range_start + 1
            break
        end
    end
    addrs_m[b] = addrs_m_all[range_start + k_prop - 1]
end

# """
#     calculate_local_energy!(ansatz, vmc_buf)
#
# This function calculates local energies in VMC. 
#
# ```math
# E_{loc}(n) = H_{nn} + \\sum_m H_{nm} * \\frac{\\psi(m)}{\\psi(n)}
# ```
#
# It utilise [`VMCBuffer`](@ref) for all necessary intermediate variables for allocation-free 
# calculations. The wave-function ratio is clamped to `Float32` precision of `exp()`.
#
# ## Note
# The local energies are clamped at the end to counter node-like instabilities. Those can occur
# not only in nodes themselves but also in not pre-trained neural network which can looks like a 
# node. 
# """
# function calculate_local_energy!(ansatz, vmc_buf::VMCBuffer)
#     # Taking neccessary stuff from metropolis buffer
#     flat_vals_m = vmc_buf.flat_vals_m   # (total,)
#     diag_ham = vmc_buf.diag_ham         # (B,)
#     flat_Hmn = vmc_buf.flat_offdiag_ham # (total,)
#     walker_idx = vmc_buf.walker_idx     # (total,)
#     E_locs = vmc_buf.E_locs             # (B,)
#     vals_n_cpu = vmc_buf.vals_n_cpu     # (1, B)
#
#     # E_loc(n) = H_nn + Σ_m H_mn * ψ(m) / ψ(n)
#     vals_n_expanded  = view(view(vals_n_cpu, 1, :), walker_idx) # (total,): mapping (B,) -> (total,)
#     flat_vals_m .= clamp.(flat_vals_m .- vals_n_expanded, -80f0, 80f0)
#     flat_vals_m .= flat_Hmn .* exp.(flat_vals_m) # (total,)
#     offdiag_contribs = flat_vals_m
#     E_locs .= diag_ham .+ scatter(+, offdiag_contribs, walker_idx, dstsize=(ansatz.model.batch,))
#
#     # Carefully clip E_locs spikes for smoothening E_locs
#     med = median(E_locs)
#     spike_window = max(50, 5 * abs(med))
#     upper_bound = med + spike_window
#     lower_bound = med - spike_window
#     E_locs .= clamp.(E_locs, lower_bound, upper_bound)
# end

"""
    get_ctmc_weights!(distro, offsets, vals_n_cpu, batch) -> raw_norm

This function calculates CTMC weights for batched approach.

```math
w_b = \\frac{|\\psi(n_b)|}{R(n_b)}
```
where `R` is transition rate also used in VMC proposal step. The function returns
some normalisation estimate over sampled batch `raw_norm`. It can be used as 
normalisation approximation over sampled batch.

```math
||\\psi||^2 = \\sum_s p(s) \\frac{|\\psi^2|}{p(s)} = \\sum_s p(s)*Z*\\frac{|\\psi(s)|}{R(s)} =
Z * \\mathbf{E}_{s~p} \\big[ \\frac{|\\psi(s)|}{R(s)} \\big]
```

# Variables

* `distro`: distribution same as in [`_state_proposal!`](@ref).
* `offsets`: vector that maps offdiagonal spawns from its spawning source.
* `vals_n_cpu`: holds logψ values of current batched sample, calculated using ansatz.
* `batch`: batch number.
"""
function get_ctmc_weights!(distro, offsets, vals_n_cpu, batch)
    # inverse CDF algorithm
    sum_weights = 0.0
    for b in 1:batch
        total_w = 0.0
        @inbounds for i in (offsets[b]+1):offsets[b+1]
            total_w += abs(distro[i])
        end
        total_w = log(total_w + 1f-35)
        vals_n_cpu[1, b] = exp(clamp(vals_n_cpu[1, b] - total_w, -80f0, 80f0))
        sum_weights += vals_n_cpu[1, b]
    end
    raw_norm = sum_weights / batch
    vals_n_cpu ./= sum_weights

    return raw_norm # if loss function cares about normalisation
end
