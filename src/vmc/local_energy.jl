# using KernelAbstractions
# using Statistics

"""
    calculate_local_energy!(ansatz, vmc_buf, n_logψ, n_sign)

This function calculates local energies in VMC. It is designed for dispatch on
the type of [`AnsatzType`](@ref) as different representations of wave-function
could require different approaches how to evaluate the equation:

```math
E_{loc}(n) = H_{nn} + \\sum_m H_{nm} * \\frac{\\psi(m)}{\\psi(n)}.
```

It utilise `vmc_buf` for all necessary intermediate variables for allocation-free 
calculations. The wave-function ratio is clamped to `Float32` precision for `exp()`.

# Variables
* `ansatz`: ansatz for wave-function, see [`NeuralAnsatz`](@ref).
* `vmc_buf`: buffer for VMC calculations, see [`VMCBuffer`](@ref).
* `n_logψ`: this represent vector of `log|ψ|` wave-function amplitudes.
* `n_sign`: this represent vector/number of signs of `ψ` wave-function.

## Note
The local energies are clamped [`elocs_clamping!`](@ref) at the end to counter node-like 
instabilities. Those can occur not only in nodes themselves but also in not pre-trained 
neural network which can looks like a node. 
"""
function calculate_local_energy!(ansatz, vmc_buf::VMCBuffer, n_logψ, n_sign, m_logψ, m_sign)
    calculate_local_energy!(ansatz.ansatz_type, ansatz, vmc_buf, n_logψ, n_sign, m_logψ, m_sign)
end
function calculate_local_energy!(::AnsatzType, ansatz, vmc_buf::VMCBuffer,
                                 n_logψ, n_sign, m_logψ, m_sign)
    ham_diag = vmc_buf.ham_diag                     # (B,)
    ham_offdiag = vmc_buf.ham_offdiag_buf.data      # (K, B)
    offsets = vmc_buf.offsets                       # (B,) cumulative
    E_locs = vmc_buf.E_locs                         # (B,)  
    median_cpu = vmc_buf.median_cpu                 # (B,) copy of E_locs

    backend = KernelAbstractions.get_backend(E_locs)
    _local_energy_kernel!(backend)(E_locs, ham_diag, offsets, ham_offdiag, m_logψ, m_sign,
                                   n_logψ, n_sign; ndrange = ansatz.model.batch)

    elocs_clamping!(E_locs, median_cpu)
end

@kernel function _local_energy_kernel!(E_locs, ham_diag, offsets, ham_offdiag,
                                m_logψ, m_sign::Number, n_logψ, n_sign::Number)
    b = @index(Global)
    @inbounds begin
        start = _column_start(offsets, b)
        stop  = Int(offsets[b])
        total = 0f0
        for i in (start + 1):stop
            k = i - start                          # row of ham_offdiag in column b
            total += ham_offdiag[k, b] * exp(clamp(m_logψ[i] - n_logψ[b], -40f0, 40f0)) * n_sign * m_sign
        end
        E_locs[b] = ham_diag[b] + total
    end
end
@kernel function _local_energy_kernel!(E_locs, ham_diag, offsets, ham_offdiag,
                                m_logψ, m_sign::AbstractArray, n_logψ, n_sign::AbstractArray)
    b = @index(Global)
    @inbounds begin
        start = _column_start(offsets, b)
        stop  = Int(offsets[b])
        total = 0f0
        for i in (start + 1):stop
            k = i - start                          # row of ham_offdiag in column b
            total += ham_offdiag[k, b] * exp(clamp(m_logψ[i] - n_logψ[b], -40f0, 40f0)) * n_sign[b] * m_sign[i]
        end
        E_locs[b] = ham_diag[b] + total
    end
end

"""
    elocs_clamping!(E_locs, median_cpu) -> E_locs

Clamp outliers of the local energies in place to the window
`median ± n_q · Q_q(|E_loc − median|)`, where `Q_q` is the `q`-quantile of the
distances from the median.

# Arguments
* `E_locs`: (B,) device vector of local energies, overwritten with clipped values.
* `median_cpu`: (B,) host buffer; its contents are overwritten and reordered.

# Keyword Arguments
* `q`: quantile that defines the spread (default `0.99`; use `0.95` for small batches).
* `n_q`: window width in units of that spread (default `10`).

# Note
`median!` and `quantile!` are not GPU-supported and run on the CPU copy, which is
faster than the GPU for arrays of this size.
"""
function elocs_clamping!(E_locs, median_cpu; q::Float32 = 0.99f0, n_q::Float32 = 10f0)
    copyto!(median_cpu, E_locs)
    med = median!(median_cpu)
    median_cpu .= abs.(median_cpu .- med)       # reuse buffer: |E − med| - distance from median
    dq = Float32(quantile!(median_cpu, q))      # 99th percentile, in place
    if !(isfinite(dq) && dq > 0f0)
        @warn "elocs_clamping!: q-quantile spread = $dq degenerate/non-finite — skipping clipping"
        return E_locs
    end
    width = n_q * dq
    n_clipped = count(>(width), median_cpu)
    if n_clipped > 0
        E_locs .= clamp.(E_locs, med - width, med + width)     # always clip
    end
    if n_clipped > 5                                            # only warn if unusually many
        @warn "elocs_clamping!: clipped $n_clipped / $(length(median_cpu)) samples " *
              "beyond ±$(width) (= $(n_q) × q$(q) spread)"
    end
    return E_locs
end
