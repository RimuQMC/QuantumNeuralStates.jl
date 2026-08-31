# using KernelAbstractions
# using Statistics

@kernel function _local_energy_kernel!(E_locs, diag_ham, offsets, flat_Hmn, 
                                m_logψ, m_sign::Number, n_logψ, n_sign::Number)
    b = @index(Global)
    @inbounds begin
        total = 0f0
        for i in (offsets[b]+1):offsets[b+1]
            total += flat_Hmn[i] * exp(clamp(m_logψ[i] - n_logψ[b], -80f0, 80f0)) * n_sign * m_sign
        end
        E_locs[b] = diag_ham[b] + total
    end
end
@kernel function _local_energy_kernel!(E_locs, diag_ham, offsets, flat_Hmn, 
                                m_logψ, m_sign::AbstractArray, n_logψ, n_sign::AbstractArray)
    b = @index(Global)
    @inbounds begin
        total = 0f0
        for i in (offsets[b]+1):offsets[b+1]
            total += flat_Hmn[i] * exp(clamp(m_logψ[i] - n_logψ[b], -80f0, 80f0)) * n_sign[b] * m_sign[i]
        end
        E_locs[b] = diag_ham[b] + total
    end
end

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
function calculate_local_energy!(ansatz, vmc_buf::VMCBuffer, n_logψ, n_sign)
    calculate_local_energy!(ansatz.ansatz_type, ansatz, vmc_buf, n_logψ, n_sign)
end
function calculate_local_energy!(::AnsatzType, ansatz, vmc_buf::VMCBuffer, n_logψ, n_sign)
    flat_vals_m = vmc_buf.flat_vals_m
    diag_ham    = vmc_buf.diag_ham
    flat_Hmn    = vmc_buf.flat_offdiag_ham
    offsets     = vmc_buf.offsets
    E_locs      = vmc_buf.E_locs

    vals_m = reshape(flat_vals_m, ansatz.ansatz_type.num_outputs, :)
    m_logψ, m_sign = log_psi!(ansatz.ansatz_type, ansatz, vals_m)

    backend = KernelAbstractions.get_backend(E_locs)
    _local_energy_kernel!(backend)(E_locs, diag_ham, offsets, flat_Hmn, m_logψ, m_sign, 
                                   n_logψ, n_sign; ndrange=ansatz.model.batch)
    KernelAbstractions.synchronize(backend)

    elocs_clamping!(E_locs)
end
# function calculate_local_energy!(::AnsatzType, ansatz, vmc_buf::VMCBuffer, n_logψ, n_sign)
#     # Taking neccessary stuff from metropolis buffer
#     flat_vals_m = vmc_buf.flat_vals_m   # (total,)
#     diag_ham = vmc_buf.diag_ham         # (B,)
#     flat_Hmn = vmc_buf.flat_offdiag_ham # (total,)
#     walker_idx = vmc_buf.walker_idx     # (total,)
#     E_locs = vmc_buf.E_locs             # (B,)
#     vals_n_cpu = vmc_buf.vals_n_cpu     # (1, B)
#
#     vals_m = reshape(flat_vals_m, ansatz.ansatz_type.num_outputs, :)
#     m_logψ, m_sign = log_psi!(ansatz.ansatz_type, ansatz, vals_m)
#
#     # E_loc(n) = H_nn + Σ_m H_mn * ψ(m) / ψ(n)
#     n_logψ_expanded  = view(n_logψ, walker_idx) # (total,): mapping (B,) -> (total,)
#     if n_sign isa Number 
#         n_sign_expanded  = n_sign # if there is no sign for chosen ansatz
#     else
#         n_sign_expanded  = view(n_sign, walker_idx) # (total,): mapping (B,) -> (total,)
#     end
#     m_logψ .= flat_Hmn .* exp.(clamp.(m_logψ .- n_logψ_expanded, -80f0, 80f0)) .* n_sign_expanded .* m_sign # (total,)
#     # m_logψ .= clamp.(m_logψ .- n_logψ_expanded, -80f0, 80f0)
#     # m_logψ .= flat_Hmn .* exp.(m_logψ) .* n_sign_expanded .* m_sign # (total,)
#     offdiag_contribs = m_logψ
#     E_locs .= diag_ham .+ scatter(+, offdiag_contribs, walker_idx, dstsize=(ansatz.model.batch,))
#
#     elocs_clamping!(E_locs)
# end


"""
    elocs_clamping!(E_locs) -> E_locs

Function that do median based clamping of local energies. This is needed especially 
when we are near nodes of the wave-function.
"""
function elocs_clamping!(E_locs)
    # Carefully clip E_locs spikes for smoothening E_locs
    med = median(E_locs)
    spike_window = max(50, 5 * abs(med))
    upper_bound = med + spike_window
    lower_bound = med - spike_window
    E_locs .= clamp.(E_locs, lower_bound, upper_bound)
    return E_locs
end
