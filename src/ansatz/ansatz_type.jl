
"""
    AnsatzType

Abstract type for specifying how the wave-function [`NeuralAnsatz`](@ref) is defined with respect to
neural network representation. The representation should reflect how many outputs neural 
network predicts.

Each defined `AnsatzType` also needs to have defined dispatched functions of 
[`psi`](@ref), [`log_psi!`](@ref), and [`init_gradient_seed`](@ref)!

See also [`LogPsi`](@ref) and [`LogPsiSignTanh`](@ref).
"""
abstract type AnsatzType end

"""
    psi(ansatz, flat_vals) -> ψ

Function that returns `ψ` wave-function value from [`NeuralAnsatz`](@ref). For each
[`AnsatzType`](@ref) there needs to be defined this function (dispatch design).

This function is used in [`ctmc_sample!`](@ref), used for distribution sum for new
proposals and ctmc weights calculation.

## Note
Important details is that `flat_vals` is a type of `::Vector` holding the output
values from neural network (in case of multiple outputs the result is flatten and
needs to be properly resized for correct evaluation).
See [`LogPsiSignTanh`](@ref) as an example of multi-output ansatz.
"""
function psi(ansatz, flat_vals::AbstractVector)
    return psi(ansatz.ansatz_type, ansatz, flat_vals)
end

"""
    log_psi!(ansatz, vals) -> log|ψ|, sign(ψ)

Function that returns `log|ψ|` amplitude of wave-function and `sign(ψ)`. For each
[`AnsatzType`](@ref) there needs to be defined this function (dispatch design).
This function takes `vals::AbstracArray` neural network outputs and evaluates 
in-place return values.

This function is used inside [`vmc_sample!`](@ref) and [`calculate_local_energy!`](@ref)
for wave-function ratios calculations. It should be called only ONCE as it mutates 
`vals` in-place.

## Note
See also definition in [`LogPsiSignTanh`](@ref) for multi-output example and in
[`LogPsi`](@ref) for single-output example.
"""
function log_psi!(ansatz, vals)
    return log_psi!(ansatz.ansatz_type, ansatz, vals)
end

"""
    init_gradient_seed(ansatz) -> init_gradient_seed(ansatz.ansatz_type, ansatz)

This function ensures correct gradient seed values in backpropagation. It returns correct
[`AnsatzType`](@ref) dispatch function that returns the gradient seed. It is important
that for every `ansatz_type` this function is defined for correct backpropagation.

## Notes
In the [`apply_loss_mode!`] function, all derivatives of loss functions w.r.t.
`log|ψ|` are defined here. This is typical procedure in neural networks field.

```math
\\frac{\\partial \\mathcal{L}}{\\partial \\text{log}(\\text{abs}(\\psi))} = \\frac{\\partial
\\mathcal{L}}{\\partial \\psi}*\\psi
```
In cases where wave-function ansatz is more complicated (example [`LogPsiSignTanh`](@ref))
the derivatives can have extra terms w.r.t. neural network outputs. From the example we
can define those derivatives as

```math
\\begin{aligned}
\\frac{\\partial \\mathcal{L}}{\\partial \\text{out}[1]} &= \\frac{\\partial \\mathcal{L}}{\\partial
\\psi} * \\frac{\\partial \\psi}{\\partial \\text{out}[1]} = \\frac{\\partial \\mathcal{L}}{
\\partial \\psi} * \\psi \\\\

\\frac{\\partial \\mathcal{L}}{\\partial \\text{out}[2]} &= \\frac{\\partial \\mathcal{L}}{\\partial
\\psi} * \\frac{\\partial \\psi}{\\partial \\text{out}[2]} = \\frac{\\partial \\mathcal{L}}{
\\partial \\psi} * e^{\\text{out}[1]} = \\frac{\\partial \\mathcal{L}}{\\partial \\psi} * 
\\frac{\\psi}{\\psi} * e^{\\text{out}[1]} = \\frac{\\partial \\mathcal{L}}{\\partial \\psi} * 
\\psi / \\text{out}[2]
\\end{aligned}
```
On this example we can see that if the derivative of loss function w.r.t. neural network
outputs are equal to `∂L/∂ψ * ψ`, than everything is accounted in [`apply_loss_mode!`](@ref) 
already (the seed for such output can be just filled with `ones`). Nevertheless, if there are 
some extra terms as `1/out[2]` in the second equation of the example, this extra term needs to 
be accounted in `init_gradient_seed`.
"""
function init_gradient_seed(ansatz)
    return init_gradient_seed(ansatz.ansatz_type, ansatz)
end

"""
    LogPsi

Type of [`AnsatzType`](@ref), using neural network with one continous output (`identity`
activation). This output (noted as `out[1]`) should represent `out[1] = log(ψ)`.

```math
\\psi = \\exp(\\text{out}[1])
```
# Variables
* `num_outputs`: how many outputs this ansatz expect.
"""
struct LogPsi <: AnsatzType 
    num_outputs::Int
end
LogPsi() = LogPsi(1)

function log_psi!(::LogPsi, ansatz, vals::AbstractArray{T}) where {T}
    out1 = view(vals, 1, :) 
    out1 .= out1 .- ansatz.logψ_centering
    return out1, one(T) # log|ψ|, sign
end

function psi(::LogPsi, ansatz, flat_vals::AbstractVector)
    return exp.(clamp.(flat_vals .- ansatz.logψ_centering, -80f0, 80f0))
end

function init_gradient_seed(::LogPsi, ansatz)
    batch = ansatz.model.batch
    last_l = last(ansatz.model.layers)
    out_dim = size(last_l.z, 1) # should be 1 
    T = eltype(last_l.z)

    init_seed = fill!(similar(last_l.z, out_dim, batch), one(T))
    return init_seed
end

"""
    LogPsiSignTanh

Type of [`AnsatzType`](@ref) allowing representing sign phase `+-1, 0`. It uses 
two neural networks outputs. First output (noted as `out[1]`) is continous number (
`identity` activation) and second output (noted `out[2]`) is using `tanh` (or 
`tanh_fast`) activation. The final wave-function is then represented as:

```math
\\psi = \\exp(\\text{out}[1]) \\text{out}[2] = \\sigma \\exp(\\text{out}[1]) \\text{abs}(\\text{out}[2])
```
where `σ` represents sign of the wave-function and `exp(out[1]) abs(out[2])` represents
amplitude of wave-function.

# Variables
* `num_outputs`: how many outputs this ansatz expect.
"""
struct LogPsiSignTanh <: AnsatzType 
    num_outputs::Int
end
LogPsiSignTanh() = LogPsiSignTanh(2)

function log_psi!(::LogPsiSignTanh, ansatz, vals::AbstractArray)
    out1 = view(vals, 1, :)
    out2 = view(vals, 2, :)

    out1 .= (out1 .+ log.(abs.(out2))) .- ansatz.logψ_centering
    out2 .= sign.(out2)
    return out1, out2 # view as: logψ, sign
end

function psi(::LogPsiSignTanh, ansatz, flat_vals::AbstractVector)
    out_dim = size(last(ansatz.model.layers).z, 1)
    vals = reshape(flat_vals, out_dim, :)
    a1 = view(vals, 1, :)
    a2 = view(vals, 2, :)
    return exp.(clamp.(a1 .- ansatz.logψ_centering, -80f0, 80f0)) .* a2
end

function init_gradient_seed(::LogPsiSignTanh, ansatz)
    # ψ = exp(out[1])*out[2] -> out[2] = tanh(a[2])
    # dL/dout[1] = dL/dψ * dψ/dout[1] = dL/dψ * (ψ + 0) = dL/dψ * ψ
    #                                     = dL/dlogψ ---> so far loss gradients
    # dL/dout[2] = dL/dψ * dψ/dout[2] = dL/dψ * (0 + exp(out[1])) = dL/dψ * exp(out[1])
    #                                     = dL_dψ * ψ * exp(out[1])/ψ 
    #                                     = dL/dlogψ / out[2]
    
    batch = ansatz.model.batch
    last_l = last(ansatz.model.layers)
    out_dim = size(last_l.z, 1) # should be 2
    T = eltype(last_l.z)
    ϵ = T(1e-6) # safe check for division around zero

    init_seed = similar(last_l.z, out_dim, batch) # correct device
    init_seed[1, :] .= one(T)
    # the 1/out[2] factor now lives in the init gradient seed (as upper derivative)
    init_seed[2, :] .= one(T) ./ safe_denom.(view(last_l.z, 2, :), ϵ)

    return init_seed
end





