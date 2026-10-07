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
    VMCBuffer(ansatz, addr; K_init = nothing)

This structure holds all intermediate variables of a VMC step. All buffers live on the
device of `ansatz.model` (CPU or GPU), except `accepted` and the host scratch `median_cpu`.

# Variables
* `ansatz`: [`NeuralAnsatz`](@ref).
* `addr`: Rimu address, used for the address type and to initialise address buffers.

# Keyword Arguments
* `K_init`: initial number of off-diagonal slots per sample. Defaults to the global
    estimate `num_offdiagonals(ansatz.hamiltonian)` if available, otherwise to
    `num_offdiagonals(ansatz.hamiltonian * addr)`. This is only a starting guess; the
    off-diagonal buffers grow automatically if a sample has more connections.
"""
mutable struct VMCBuffer{A, VA<:AbstractVector{A},
                         GRA<:GPUGrowRowBuffer, GRF<:GPUGrowRowBuffer, GC<:GPUGrowColumnBuffer,
                         BAI<:AbstractArray{Int32}, BAF<:AbstractArray{Float32}}
    addrs_m::VA                     # (B,)    proposed addresses
    addrs_offdiag_buf::GRA          # (K, B)  off-diagonal addresses
    vals_offdiag_buf::GC            # (out_dim, n) network outputs of the off-diagonals
    ham_offdiag_buf::GRF            # (K, B)  off-diagonal matrix elements
    ham_diag::BAF                   # (B,)    diagonal elements H_nn
    start::Bool                     # when to start E_loc calculations (after thermalisation)
    offsets::BAI                    # (B,)    cumulative number of off-diagonals
    E_locs::BAF                     # (B,)    local energies
    accepted::AbstractVector{Bool}  # (B,)    for metropolis acceptance
    median_cpu::Vector{Float32}     # (B,)    host scratch for the E_loc median

    # outside VMC use
    block_idx::Int
    E_mean::BAF                     # (1,)
    variance::BAF                   # (1,)
    weights::BAF                    # (B,)    
end
function VMCBuffer(ansatz, addr; K_init = nothing)
    batch = ansatz.model.batch
    model_z = last(ansatz.model.layers).z
    out_dim = size(model_z, 1)
    A = typeof(addr)
    T = eltype(model_z)
    backend = KernelAbstractions.get_backend(model_z)

    K0 = if K_init !== nothing
        K_init
    else
        try
            num_offdiagonals(ansatz.hamiltonian)             # global estimate, if available
        catch err
            err isa MethodError || rethrow()                 # fall back only if the method is missing
            num_offdiagonals(ansatz.hamiltonian * addr)      # always available: this column's count
        end
    end
    K0 = max(Int(K0), 1)

    # proposed addresses: initialised with a valid address, never garbage
    addrs_m = KernelAbstractions.allocate(backend, A, batch)
    fill!(addrs_m, addr)

    # off-diagonals: (K, B), grow in K
    addrs_offdiag_buf = GPUGrowRowBuffer(backend, A, K0, (batch,))
    ham_offdiag_buf = GPUGrowRowBuffer(backend, T, K0, (batch,))

    # network outputs of the off-diagonals: (out_dim, n), grow in n
    vals_offdiag_buf = GPUGrowColumnBuffer(backend, T, (out_dim,), K0 * batch)

    ham_diag = KernelAbstractions.zeros(backend, T, batch)
    offsets = KernelAbstractions.zeros(backend, Int32, batch)
    E_locs = KernelAbstractions.zeros(backend, T, batch)
    accepted = KernelAbstractions.zeros(backend, Bool, batch)
    median_cpu = Vector{Float32}(undef, batch)

    E_mean = KernelAbstractions.zeros(backend, T, 1)
    variance = KernelAbstractions.zeros(backend, T, 1)
    weights = KernelAbstractions.zeros(backend, T, batch)

    start = false

    VA  = typeof(addrs_m)
    GRA = typeof(addrs_offdiag_buf)
    GRF = typeof(ham_offdiag_buf)
    GC  = typeof(vals_offdiag_buf)
    BAI = typeof(offsets)
    BAF = typeof(ham_diag)

    return VMCBuffer{A,VA,GRA,GRF,GC,BAI,BAF}(
                addrs_m, addrs_offdiag_buf, vals_offdiag_buf, ham_offdiag_buf, ham_diag,
                start, offsets, E_locs, accepted, median_cpu, 1, E_mean, variance, weights)
end

Base.show(io::IO, ::MIME"text/plain", ::VMCBuffer) = print(io, "VMCBuffer")

"""
    ctmc_state_proposal!(addrs_m, addrs_m_all, offsets, distro, rand_vals)

This function proposes a new address for every spawning address in a CTMC sampler
step. From each spawning address `b` all valid off-diagonal connections are collected
in column `b` of `addrs_m_all`; the new address is drawn among them with probability
proportional to `exp(distro)`, by inverse-CDF sampling. The chosen address is written
directly into `addrs_m[b]` on the device.

For numerical stability the maximum of `distro` over each parent's off-diagonals is
subtracted before exponentiating, so every term is `≤ 1` and the largest is exactly `1`.

# Variables
* `addrs_m`: (B,) device vector, receives the proposed address of each parent.
* `addrs_m_all`: (K, B) off-diagonal addresses; column `b` belongs to parent `b`.
* `offsets`: device vector of length `B`, cumulative number of valid off-diagonals per
    parent. Parent `b` owns stream positions `_column_start(offsets, b) + 1 : offsets[b]`.
* `distro`: log-weights of all valid off-diagonals, packed in stream order (length
    `last(offsets)`). For example `logψ` of the off-diagonals for ψ-weighted proposals,
    or zeros for uniform proposals.
* `rand_vals`: (B,) device buffer, refilled with uniform random numbers in `[0, 1)`.

## Note
A parent without off-diagonals leaves `addrs_m[b]` unchanged.
"""
function ctmc_state_proposal!(addrs_m, addrs_m_all, offsets, distro, rand_vals)
    Random.rand!(rand_vals)                     # in place, on the device
    backend = KernelAbstractions.get_backend(rand_vals)
    _ctmc_proposal_kernel!(backend)(addrs_m, addrs_m_all, offsets, distro, rand_vals;
                                    ndrange = length(addrs_m))
    return nothing
end

@kernel function _ctmc_proposal_kernel!(addrs_m, @Const(addrs_m_all), @Const(offsets),
                                        @Const(distro), @Const(rand_vals))
    b     = @index(Global, Linear)              # parent
    start = _column_start(offsets, b)           # stream positions before parent b
    stop  = Int(offsets[b])                     # last stream position of parent b

    if stop > start                             # parent has off-diagonals
        # 1) maximum of this parent's log-weights
        mx = -Inf32
        for g in (start + 1):stop
            mx = max(mx, distro[g])
        end

        # 2) stabilized total: every term ≤ 1
        tot = 0f0
        for g in (start + 1):stop
            tot += exp(distro[g] - mx)
        end

        # 3) inverse CDF on the same shifted terms
        target = rand_vals[b] * tot
        cumul  = 0f0
        k      = stop - start                   # fallback: last off-diagonal (rounding)
        for g in (start + 1):stop
            cumul += exp(distro[g] - mx)
            if cumul >= target
                k = g - start                   # row of addrs_m_all in column b
                break
            end
        end
        addrs_m[b] = addrs_m_all[k, b]
    end
end

"""
    metropolis_state_proposal!(addrs_m, addrs_m_all, offsets, rand_vals)

Propose for every parent `b` one of its valid off-diagonals in column `b` of
`addrs_m_all`, uniformly at random. A parent without off-diagonals leaves `addrs_m[b]`
unchanged (it is rejected later by a zero count ratio).

# Variables
* `addrs_m`: (B,) device vector, receives the proposed address of each parent.
* `addrs_m_all`: (K, B) off-diagonal addresses; column `b` belongs to parent `b`.
* `offsets`: device vector of length `B`, cumulative number of valid off-diagonals per
    parent. Parent `b` owns stream positions `_column_start(offsets, b) + 1 : offsets[b]`.
* `rand_vals`: (B,) device buffer, refilled with uniform random numbers in `[0, 1)`.
"""
function metropolis_state_proposal!(addrs_m, addrs_m_all, offsets, rand_vals)
    Random.rand!(rand_vals)                     # in place, on the device
    backend = KernelAbstractions.get_backend(rand_vals)
    _metropolis_proposal_kernel!(backend)(addrs_m, addrs_m_all, offsets, rand_vals;
                                          ndrange = length(addrs_m))
    return nothing
end

@kernel function _metropolis_proposal_kernel!(addrs_m, @Const(addrs_m_all), @Const(offsets),
                                              @Const(rand_vals))
    b     = @index(Global, Linear)
    start = _column_start(offsets, b)
    n     = Int(offsets[b]) - start                     # valid off-diagonals of parent b
    if n > 0
        k = min(n, floor(Int, rand_vals[b] * n) + 1)    # uniform row 1..n
        addrs_m[b] = addrs_m_all[k, b]
    end
end

"""
    metropolis_count_ratio!(ratio, addrs_m, offsets, hamiltonian, mask)

Write `N_n / N_m` into `ratio[b]`: the number of valid off-diagonals of parent `b`
(from `offsets`) over that of its proposal `addrs_m[b]`. Needed for detailed balance
with uniform proposals. Parents without off-diagonals get `0` (always rejected).
"""
function metropolis_count_ratio!(ratio, addrs_m, offsets, hamiltonian, mask)
    backend = KernelAbstractions.get_backend(ratio)
    _metropolis_count_ratio_kernel!(backend)(ratio, addrs_m, offsets, hamiltonian, mask;
                                             ndrange = length(addrs_m))
    return nothing
end

@kernel function _metropolis_count_ratio_kernel!(ratio, @Const(addrs_m), @Const(offsets),
                                                 hamiltonian, mask)
    b  = @index(Global, Linear)
    Nn = Int(offsets[b]) - _column_start(offsets, b)
    Nm = 0
    for (a2, v) in offdiagonals(hamiltonian * addrs_m[b])
        if !iszero(v) && !_skip_truncated(a2, mask)
            Nm += 1
        end
    end
    ratio[b] = (Nn > 0 && Nm > 0) ? Float32(Nn) / Float32(Nm) : 0f0
end

"""
    get_ctmc_weights!(weights, log_distro, offsets, log_psi) -> weights

This function calculates the normalised CTMC holding-time weights of the batch,

```math
w_b = \\frac{|\\psi(n_b)|}{\\sum_m |\\text{exp(log_distro)}(m_b)|}
```

where the sum runs over the valid off-diagonals `m` of parent `b`, and `log_distro` holds
`log_psi_m`, the same as used in [`state_proposal!`](@ref).

# Variables
* `weights`: (B,) device vector, receives the normalised weights (`Σ_b w_b = 1`).
* `distro`: log amplitudes of all valid off-diagonals, packed in stream order
    (length `last(offsets)`), same as in [`state_proposal!`](@ref).
* `offsets`: device vector of length `B`, cumulative number of valid off-diagonals per
    parent. Parent `b` owns stream positions `_column_start(offsets, b) + 1 : offsets[b]`.
* `log_psi`: (B,) log amplitudes of the current samples, see [`log_psi!`](@ref).

## Note
A parent without off-diagonals gets weight `0`.
"""
function get_ctmc_weights!(weights, log_distro, offsets, log_psi)
    backend = KernelAbstractions.get_backend(weights)
    _ctmc_weights_kernel!(backend)(weights, log_distro, offsets, log_psi;
                                   ndrange = length(weights))

    lw_max = maximum(weights)                   # batch maximum of the log-weights
    weights .= exp.(weights .- lw_max)          # every exponent ≤ 1
    weights ./= sum(weights)
    return weights
end

@kernel function _ctmc_weights_kernel!(weights, @Const(log_distro), @Const(offsets), @Const(log_psi))
    b = @index(Global, Linear)
    start = _column_start(offsets, b)
    stop = Int(offsets[b])

    if stop > start
        # row maximum
        mx = -Inf32
        for g in (start + 1):stop
            mx = max(mx, log_distro[g])
        end
        # stabilized sum: every term ≤ 1
        tot = 0f0
        for g in (start + 1):stop
            tot += exp(log_distro[g] - mx)
        end
        weights[b] = log_psi[b] - (mx + log(tot))     # log w_b
    else
        weights[b] = -Inf32                           # no off-diagonals → weight 0
    end
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
        # KernelAbstractions.synchronize(backend)

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

