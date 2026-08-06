"""
    calculate_local_energy!(ansatz, vmc_buf)

This function calculates local energies in VMC. It is designed for dispatch on
the type of [`AnsatzType`](@ref) as different representations of wave-function
require different approaches how to evaluate the equation:

```math
E_{loc}(n) = H_{nn} + \\sum_m H_{nm} * \\frac{\\psi(m)}{\\psi(n)}.
```

It utilise [`VMCBuffer`](@ref) for all necessary intermediate variables for allocation-free 
calculations. The wave-function ratio is clamped to `Float32` precision of `exp()`.

## Note
The local energies are clamped [`elocs_clamping!`](@ref) at the end to counter node-like 
instabilities. Those can occur not only in nodes themselves but also in not pre-trained 
neural network which can looks like a node. 
"""
function calculate_local_energy!(ansatz, vmc_buf::VMCBuffer)
    calculate_local_energy!(ansatz.ansatz_type, ansatz, vmc_buf)
end
function calculate_local_energy!(::LogPsi, ansatz, vmc_buf::VMCBuffer)
    # Taking neccessary stuff from metropolis buffer
    flat_vals_m = vmc_buf.flat_vals_m   # (total,)
    diag_ham = vmc_buf.diag_ham         # (B,)
    flat_Hmn = vmc_buf.flat_offdiag_ham # (total,)
    walker_idx = vmc_buf.walker_idx     # (total,)
    E_locs = vmc_buf.E_locs             # (B,)
    vals_n_cpu = vmc_buf.vals_n_cpu     # (1, B)

    # E_loc(n) = H_nn + Σ_m H_mn * ψ(m) / ψ(n)
    vals_n_expanded  = view(view(vals_n_cpu, 1, :), walker_idx) # (total,): mapping (B,) -> (total,)
    flat_vals_m .= clamp.(flat_vals_m .- vals_n_expanded, -80f0, 80f0)
    flat_vals_m .= flat_Hmn .* exp.(flat_vals_m) # (total,)
    offdiag_contribs = flat_vals_m
    E_locs .= diag_ham .+ scatter(+, offdiag_contribs, walker_idx, dstsize=(ansatz.model.batch,))

    elocs_clamping!(E_locs)
end
function calculate_local_energy!(::LogPsiSignTanh, ansatz, vmc_buf::VMCBuffer)
    # Taking neccessary stuff from metropolis buffer
    flat_vals_m = vmc_buf.flat_vals_m   # (total,)
    diag_ham = vmc_buf.diag_ham         # (B,)
    flat_Hmn = vmc_buf.flat_offdiag_ham # (total,)
    walker_idx = vmc_buf.walker_idx     # (total,)
    E_locs = vmc_buf.E_locs             # (B,)
    vals_n_cpu = vmc_buf.vals_n_cpu     # (out_dim, B)

    out_dim = size(last(ansatz.model.layers).z, 1)
    vals_m = reshape(flat_vals_m, out_dim, :)

    # E_loc(n) = H_nn + Σ_m H_mn * ψ(m) / ψ(n)
    # ψ(n) = exp(NeuralNetwork(n)[1])*tanh(NeuralNetwork(n)[2])
    vals_n_cpu[1, :] .+= log.(abs.(vals_n_cpu[2,:])) # log(ψ) = NeuralNetwork(n)[1]*log(abs(NeuralNetwork(n)[2]))
    vals_n_cpu[2, :] .= sign.(vals_n_cpu[2,:]) # sign(ψ) = sing(NeuralNetwork(n)[2])
    vals_m[1, :] .+= log.(abs.(vals_m[2,:])) 
    vals_m[2, :] .= sign.(vals_m[2,:])

    vals_n_expanded1 = view(view(vals_n_cpu, 1, :), walker_idx) # (total,): mapping (B,) -> (total,)
    vals_n_expanded2 = view(view(vals_n_cpu, 2, :), walker_idx) # (total,): mapping (B,) -> (total,)
    vals_m1 = view(vals_m, 1, :)
    vals_m2 = view(vals_m, 2, :)

    vals_m1 .= flat_Hmn .* exp.(clamp.(vals_m1 .- vals_n_expanded1, -80f0, 80f0)) .* vals_m2 .* vals_n_expanded2 # adding sign (total,)
    offdiag_contribs = vals_m1
    E_locs .= diag_ham .+ scatter(+, offdiag_contribs, walker_idx, dstsize=(ansatz.model.batch,))

    elocs_clamping!(E_locs)
end

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
