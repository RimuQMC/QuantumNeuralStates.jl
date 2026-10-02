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
    ham_diag = vmc_buf.ham_diag                 # (B,)   device
    ham_offdiag = vmc_buf.ham_offdiag_buf.data     # (K, B) device
    offsets = vmc_buf.offsets                  # (B,)   device, cumulative
    E_locs = vmc_buf.E_locs
    median_cpu = vmc_buf.median_cpu

    backend = KernelAbstractions.get_backend(E_locs)
    _local_energy_kernel!(backend)(E_locs, ham_diag, offsets, ham_offdiag, m_logψ, m_sign,
                                   n_logψ, n_sign; ndrange = ansatz.model.batch)
    KernelAbstractions.synchronize(backend)

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
            total += ham_offdiag[k, b] * exp(clamp(m_logψ[i] - n_logψ[b], -80f0, 80f0)) * n_sign * m_sign
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
            total += ham_offdiag[k, b] * exp(clamp(m_logψ[i] - n_logψ[b], -80f0, 80f0)) * n_sign[b] * m_sign[i]
        end
        E_locs[b] = ham_diag[b] + total
    end
end

"""
    elocs_clamping!(E_locs, median_cpu) -> E_locs

Function that do median based clamping of local energies. This is needed especially 
when we are near nodes of the wave-function.

# Note
`median` functions are not GPU supported and CPU calculations are around 100x faster
then GPU on arrays of size(E_locs).
"""
function elocs_clamping!(E_locs, median_cpu)
    # Carefully clip E_locs spikes for smoothening E_locs
    copyto!(median_cpu, E_locs)
    med = median!(median_cpu)
    spike_window = max(50, 5 * abs(med))
    upper_bound = med + spike_window
    lower_bound = med - spike_window
    E_locs .= clamp.(E_locs, lower_bound, upper_bound)
    return E_locs
end
