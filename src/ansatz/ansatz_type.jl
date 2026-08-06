
"""
    AnsatzType

Abstract type for specifying what wave-function ansatz should [`NeuralAnsatz`](@ref) and
neural network represent. The representation should reflect how many outputs neural 
network predicts.

See also [`LogPsi`](@ref) and [`LogPsiSignTanh`](@ref).
"""
abstract type AnsatzType end

"""
    psi_from_output

Function that returns `ψ` wave-function value from [`NeuralAnsatz`](@ref). For each
[`AnsatzType`](@ref) there needs to be defined evaluation function which is used in 
[`vmc_sample!`](@ref).

## Note
Important details is that `flat_vals` is a type of `::Vector` holding the output
values from neural network (in case of multiple outputs the result is flatten and
needs to be properly resized for correct evaluation).
See [`LogPsiSignTanh`](@ref) as an example of multi-output ansatz.
"""
function psi_from_output(ansatz, flat_vals::AbstractVector)
    return psi_from_output(ansatz.ansatz_type, ansatz, flat_vals)
end

"""
    LogPsi

Type of [`AnsatzType`](@ref), using neural network with one continous output. This 
output (noted as `a1`) should represent `a1 = log(ψ)`.

```math
\\psi = \\exp(a1)
```
"""
struct LogPsi <: AnsatzType end

function psi_from_output(::LogPsi, ansatz, flat_vals::AbstractVector)
    return exp.(clamp.(flat_vals, -80f0, 80f0))
end

"""
    LogPsiSignTanh

Type of [`AnsatzType`](@ref) allowing representing sign phase `+-1, 0`. It uses 
two neural networks outputs. First output (noted as `a1`) is continous number and
second output (noted `a2`) is using `tanh` (or `tanh_fast`) activation. The final 
wave-function is then represented as:

```math
\\psi = \\exp(a1) \\text{tanh}(a2) = \\sigma \\exp(a1) \\text{abs}(\\text{tanh}(a2))
```
where `σ` represents sign of the wave-function and `exp(a1) abs(tanh(a2))` represents
amplitude of wave-function.
"""
struct LogPsiSignTanh <: AnsatzType end

function psi_from_output(::LogPsiSignTanh, ansatz, flat_vals::AbstractVector)
    out_dim = size(last(ansatz.model.layers).z, 1)
    vals = reshape(flat_vals, out_dim, :)
    a1 = view(vals, 1, :)
    a2 = view(vals, 2, :)
    return exp.(clamp.(a1, -80f0, 80f0)) .* a2
end
