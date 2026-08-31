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
struct Dense{T,M<:AbstractMatrix{T},V<:AbstractVector{T},F<:Union{Function,Tuple},
                     G<:Union{Function,Tuple},B<:AbstractArray{T},R<:Union{Nothing,Tuple},
                     LN<:Union{LayerNorm, Nothing}}
    W::M
    b::V
    act_func::F
    act_deriv::G
    a::B
    z::B
    act_ranges::R

    # applying NormLayer or not
    layer_norm::LN
end
# single-activation constructor (R=Nothing)
function Dense(in::Int, out::Int, act::Function;
               batch::Int=1, device::Function=identity, Layer_Norm=false)
    T, W, b, a, z, layer_norm = _dense_alloc(in, out, act; batch, device, Layer_Norm)
    act_d = _lookup_deriv(act)

    M, V, F, G, B, LN = typeof(W), typeof(b), typeof(act), typeof(act_d), typeof(z), typeof(layer_norm)
    return Dense{T,M,V,F,G,B,Nothing,LN}(W, b, act, act_d, a, z, nothing, layer_norm)
end
# multi-activation constructor (R=NTuple{N,UnitRange})
function Dense(in::Int, out::Int, acts::NTuple{N,Function};
               batch::Int=1, device::Function=identity, Layer_Norm=false) where N
    T, W, b, a, z, layer_norm = _dense_alloc(in, out, acts; batch, device, Layer_Norm)
    act_ds  = map(_lookup_deriv, acts) # tuple-map for activations and its derivatives
    ranges  = ntuple(i -> i:i, N)

    M, V, F, G, B, R, LN = typeof(W), typeof(b), typeof(acts), typeof(act_ds), typeof(z), typeof(ranges), typeof(layer_norm)
    return Dense{T,M,V,F,G,B,R,LN}(W, b, acts, act_ds, a, z, ranges, layer_norm)
end

"""
    apply_act!(layer::Dense, a, z)

This function applyies activation function in [`Dense`](@ref) layer. It is designed in
dispatched way so of I have multiple outputs in layer I can define multiple 
activation function to be applied (it is used mostly for final neural network outputs
to allow many wave-function representations. See also [`AnsatzType`](@ref)).
"""
function apply_act!(layer::Dense{T,M,V,F,G,B,Nothing,LN}, a, z) where {T,M,V,F,G,B,LN}
    # single activations
    z .= layer.act_func.(a)
    return z
end
function apply_act!(layer::Dense{T,M,V,F,G,B,R,LN}, a, z) where {T,M,V,F,G,B,R<:Tuple,LN}
    # multi activations
    map(layer.act_func, layer.act_ranges) do f, r
        @views z[r,:] .= f.(a[r,:])
    end
    return z
end

"""
    forward(layer::Dense, x::AbstractArray) -> layer.z

Allocation-free forward pass through a [`Dense`](@ref) layer. It 
calculates forward pass in-place of `Dense` structure, but also returns
its layer output for easier chaining.

# Arguments
* `layer`: a `Dense` layer
* `x`: input array, shape `(in,)` or `(in, batch)`

"""
function forward(layer::Dense, x::AbstractArray)
    mul!(layer.a, layer.W, x, 1f0, 0f0)
    layer.a .+= layer.b
    if layer.layer_norm !== nothing
        ln_forward!(layer.layer_norm, layer.a)
    end

    apply_act!(layer, layer.a, layer.z)
    return layer.z
end

"""
    forward(layer::Dense, x::AbstractArray, layerMulti) -> layerMulti.a

Similar as [`forward`](@ref), but it allows multi batched forward pass. In
this case new `layerMulti` holds input and output of this pass.
"""
function forward(layer::Dense, x::AbstractArray, layerMulti)
    mul!(layerMulti.a, layer.W, x, 1f0, 0f0)
    layerMulti.a .+= layer.b
    if layer.layer_norm !== nothing
        ln_forward!(layer.layer_norm, layerMulti.a, layerMulti.layer_norm)
    end

    apply_act!(layer, layerMulti.a, layerMulti.a)
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

