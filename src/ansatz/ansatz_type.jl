
"""
    AnsatzType

Abstract type for specifying how the wave-function [`NeuralAnsatz`](@ref) is defined with respect to
neural network representation. The representation should reflect how many outputs neural 
network predicts.

Each defined `AnsatzType` also needs to have defined dispatched functions of 
[`psi!`](@ref), [`log_psi!`](@ref), and [`init_gradient_seed!`](@ref)!

See also [`LogPsi`](@ref) and [`LogPsiSignTanh`](@ref).
"""
abstract type AnsatzType end

"""
    psi!(ansatz, vals) -> ψ
    psi!(ansatz, logψ, signψ) -> ψ

Compute the wave-function values `ψ`, in-place. Each [`AnsatzType`](@ref) must
define its own `psi!(::AnsatzType, ansatz, logψ, signψ)`.

The first form takes the raw network outputs `vals` and calls
[`log_psi!`](@ref) internally. The second form takes `logψ` and `signψ` exactly
as returned by [`log_psi!`](@ref), for when it has already been called.

Both forms overwrite their input with `ψ`, so the result must not be passed to
`psi!` or `log_psi!` again.

See [`LogPsiSignTanh`](@ref) for a multi-output and [`LogPsi`](@ref) for a
single-output example.
"""
function psi!(ansatz, vals)
    logψ, signψ = log_psi!(ansatz, vals)
    return psi!(ansatz.ansatz_type, ansatz, logψ, signψ)
end
function psi!(ansatz, logψ, signψ)
    return psi!(ansatz.ansatz_type, ansatz, logψ, signψ)
end

"""
    log_psi!(ansatz, vals) -> log|ψ|, sign(ψ)

Compute `log|ψ|` and `sign(ψ)` from the network outputs `vals`, in-place.
Each [`AnsatzType`](@ref) must define its own method.

Used in [`vmc_sample!`](@ref) and [`calculate_local_energy!`](@ref) for
wave-function ratios. Call it only once on the same `vals`: it overwrites the
raw outputs, so a second call would process already-processed values. To get
`ψ` afterwards, pass the results to [`psi!`](@ref) as `psi!(ansatz, logψ, signψ)`.

See [`LogPsiSignTanh`](@ref) for a multi-output and [`LogPsi`](@ref) for a
single-output example.
"""
function log_psi!(ansatz, vals)
    return log_psi!(ansatz.ansatz_type, ansatz, vals)
end

"""
    init_gradient_seed!(ansatz, seed) -> seed

This function ensures correct gradient seed values in backpropagation. It returns correct
[`AnsatzType`](@ref) dispatch function that returns the gradient seed. It is important
that for every `ansatz_type` this function is defined for correct backpropagation.

# Notes
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
be accounted in `init_gradient_seed!`.
"""
function init_gradient_seed!(ansatz, seed)
    return init_gradient_seed!(ansatz.ansatz_type, ansatz, seed)
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

function psi!(::LogPsi, ansatz, logψ::AbstractArray{T}, signψ) where {T}
    logψ .= exp.(clamp.(logψ, T(-80), T(80)))
    return logψ # final ψ - wave-function
end

function init_gradient_seed!(::LogPsi, ansatz, seed)
    # fill!(seed, one(T)) - seed filled with ones() in initialisation in JacobianBuffer
    return seed
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

    out1 .= (out1 .+ log.(max.(abs.(out2), floatmin(T)))) .- ansatz.logψ_centering
    out2 .= sign.(out2)
    return out1, out2 # view as: logψ, sign
end

function psi!(::LogPsiSignTanh, ansatz, logψ::AbstractArray{T}, signψ) where {T}
    logψ .= exp.(clamp.(logψ, T(-80), T(80))) .* signψ
    return logψ # final ψ - wave-function
end

function init_gradient_seed!(::LogPsiSignTanh, ansatz, seed)
    # ψ = exp(out[1])*out[2] -> out[2] = tanh(a[2])
    # dL/dout[1] = dL/dψ * dψ/dout[1] = dL/dψ * (ψ + 0) = dL/dψ * ψ
    #                                     = dL/dlogψ ---> so far loss gradients
    # dL/dout[2] = dL/dψ * dψ/dout[2] = dL/dψ * (0 + exp(out[1])) = dL/dψ * exp(out[1])
    #                                     = dL_dψ * ψ * exp(out[1])/ψ 
    #                                     = dL/dlogψ / out[2]
    
    T = eltype(seed)
    z = last(ansatz.model.layers).z
    view(seed, 1, :) .= one(T)
    view(seed, 2, :) .= one(T) ./ safe_denom.(view(z, 2, :), T(1e-6))

    return seed
end





