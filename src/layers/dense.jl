# using LinearAlgebra
"""
    Dense(in, out, act; kwargs...)

A fully-connected (dense) neural network layer, computing:

```math
\\begin{aligned}
a &= W x + b \\\\
z &= \\text{act\\_func}(a)
\\end{aligned}
```

Allocation-free at runtime — all intermediate results are stored in pre-allocated buffers.

# Arguments

* `in`: input dimension.
* `out`: output dimension.
* `act`: activation function — must be registered in `ACT_DERIV` (`activations.jl`).

# Keyword Arguments
* `batch`: batch size.
* `device`: device function deciding where the layer lives, e.g. `identity` for CPU 
            ,`cu` for GPU (CUDA), or `mtl` for GPU (Metal).
* `Layer_Norm`: if `true`, attaches a [`LayerNorm`](@ref) layer after the activation.

## Weight Initialisation
* `He` initialisation (`sqrt(2/in)`) for `relu` and `gelu`
* `Glorot/Xavier` initialisation (`sqrt(2/(in+out))`) for all other activations
* Biases initialised to zero

# Example
```julia
layer = Dense(64, 32, tanh)
layer_gpu = Dense(64, 32, relu; batch=16, device=cu)
layer_ln  = Dense(64, 32, gelu; Layer_Norm=true)
```
"""
mutable struct Dense{T, M <: AbstractMatrix{T}, V <: AbstractVector{T},
                     F <: Function, G <: Function, B <: AbstractArray{T}}
    W        ::M
    b        ::V
    act_func ::F
    act_deriv::G
    a        ::B
    z        ::B

    # applying NormLayer or not
    layer_norm::Union{LayerNorm, Nothing}
end
function Dense(in::Int, out::Int, act::Function;
        batch::Int = 1, device::Function = identity, Layer_Norm=false)
    T   = Float32
    if act === relu || act === gelu
        std = T(sqrt(2.0 / in))         # He
    else
        std = T(sqrt(2.0 / (in + out))) # Glorot/Xavier
    end
    W   = device(randn(T, out, in) .* std)
    b   = device(zeros(T, out))
    act_d = get(ACT_DERIV, act, nothing)
    if act_d === nothing 
        error("No derivative registered for $act. Use only function define in activations.jl or define yours there.")
    end
    if batch == 1
        a = device(zeros(T, out))
        z = device(zeros(T, out))
    else
        a = device(zeros(T, out, batch))
        z = device(zeros(T, out, batch))
    end

    layer_norm = if Layer_Norm===false
        nothing
    else
        LayerNorm(out, batch, device)
    end

    M, V, F, G, B = typeof(W), typeof(b), typeof(act), typeof(act_d), typeof(z)
    return Dense{T, M, V, F, G, B}(W, b, act, act_d, a, z, layer_norm)
end

"""
    forward(layer::Dense, x::AbstractArray) -> layer.z

Allocation-free forward pass through a [`Dense`](@ref) layer. It 
calculates forward pass in-place of `Dense` structure, but also returns
its layer output for easier chaining.

# Arguments
* `layer`: a `Dense` layer
* `x`: input array, shape `(in,)` or `(in, batch)`

## Notes
Bias `b` is skipped when `act_func === identity` as last output layer of 
Neural Network represents log(ψ) and wave-function is invariant with multiplication
of scalar.
"""
function forward(layer::Dense, x::AbstractArray)
    mul!(layer.a, layer.W, x, 1f0, 0f0)
    # layer.a .+= layer.b
    if layer.act_func !== identity
        layer.a .+= layer.b
    end
    if layer.layer_norm !== nothing
        ln_forward!(layer.layer_norm, layer.a)
    end
    layer.z  .= layer.act_func.(layer.a)
    return layer.z
end

"""
    forward(layer::Dense, x::AbstractArray, layerMulti) -> layerMulti.a

Similar as [`forward`](@ref), but it allows multi batched forward pass. In
this case new `layerMulti` holds input and output of this pass.
"""
function forward(layer::Dense, x::AbstractArray, layerMulti)
    mul!(layerMulti.a, layer.W, x, 1f0, 0f0)
    # layer.a .+= layer.b
    if layer.act_func !== identity
        layerMulti.a .+= layer.b
    end
    if layer.layer_norm !== nothing
        ln_forward!(layer.layer_norm, layerMulti.a, layerMulti.layer_norm)
    end
    layerMulti.a  .= layer.act_func.(layerMulti.a)
    return layerMulti.a
end

"""
    n_params(layer::Dense) -> num 

This functions returns total number of learnable parameters inside layer `num`.
"""
function n_params(layer::Dense)
    if layer.layer_norm !== nothing
        return length(layer.W) + length(layer.b) + 2
    else
        return length(layer.W) + length(layer.b)
    end
end

