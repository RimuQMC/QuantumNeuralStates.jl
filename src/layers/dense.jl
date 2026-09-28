# using LinearAlgebra

"""
    Dense(in => out, act; kwargs...) <: ParametricLayer

A fully-connected (dense) neural network layer, computing:

```math
\\begin{aligned}
a &= W x + b \\\\
z &= \\text{act\\_func}(a)
\\end{aligned}
```

Allocation-free at runtime — all intermediate results are stored in pre-allocated buffers.

# Arguments
* `in => out`: input and output dimensions.
* `act`: activation function — must be registered in [`ACT_DERIV`](@ref).

# Keyword Arguments
* `batch`: batch size.
* `device`: device function deciding where the layer lives, e.g. `identity` for CPU 
            ,`cu` for GPU (CUDA), or `mtl` for GPU (Metal).
* `layer_norm`: if `true`, attaches a [`LayerNorm`](@ref) layer after the activation.

## Weight Initialisation
* `He` initialisation (`sqrt(2/in)`) for `relu` and `gelu`
* `Glorot/Xavier` initialisation (`sqrt(2/(in+out))`) for all other activations
* Biases initialised to zero

# Example
```julia
layer = Dense(64=>32, tanh)
layer_gpu = Dense(64=>32, relu; batch=16, device=cu)
layer_ln  = Dense(64=>32, gelu; layer_norm=true)
```
"""
struct Dense{T,M<:AbstractMatrix{T},V<:AbstractVector{T},F<:Union{Function,Tuple},
                     G<:Union{Function,Tuple},B<:AbstractArray{T},R<:Union{Nothing,Tuple},
                     LN<:Union{LayerNorm, Nothing}} <: ParametricLayer
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
function Dense(channels::Pair{Int,Int}, act::Function;
               batch::Int=1, device::Function=identity, layer_norm=false)
    in, out = channels.first, channels.second
    T = Float32
    std = _init_std(T, in, out, act)     # see note below re: He/Glorot dispatch
    W = device(randn(T, out, in) .* std)
    b = device(zeros(T, out))
    a = device(zeros(T, out, batch))
    z = device(zeros(T, out, batch))

    Layer_Norm = layer_norm === false ? nothing : LayerNorm(out, batch, device)

    act_d = _lookup_deriv(act)

    M, V, F, G, B, LN = typeof(W), typeof(b), typeof(act), typeof(act_d), typeof(z), typeof(Layer_Norm)
    return Dense{T,M,V,F,G,B,Nothing,LN}(W, b, act, act_d, a, z, nothing, Layer_Norm)
end
# multi-activation constructor (R=NTuple{N,UnitRange})
function Dense(channels::Pair{Int,Int}, acts::NTuple{N,Function};
               batch::Int=1, device::Function=identity, layer_norm=false) where N
    in, out = channels.first, channels.second
    T = Float32
    std = _init_std(T, in, out, acts)     # see note below re: He/Glorot dispatch
    W = device(randn(T, out, in) .* std)
    b = device(zeros(T, out))
    a = device(zeros(T, out, batch))
    z = device(zeros(T, out, batch))

    Layer_Norm = layer_norm === false ? nothing : LayerNorm(out, batch, device)

    act_ds  = map(_lookup_deriv, acts) # tuple-map for activations and its derivatives
    ranges  = ntuple(i -> i:i, N)

    M, V, F, G, B, R, LN = typeof(W), typeof(b), typeof(acts), typeof(act_ds), typeof(z), typeof(ranges), typeof(Layer_Norm)
    return Dense{T,M,V,F,G,B,R,LN}(W, b, acts, act_ds, a, z, ranges, Layer_Norm)
end

function Base.show(io::IO, l::Dense)
    print(io, "Dense(", size(l.W, 2), "=>", size(l.W, 1), ", ", _actname(l.act_func))
    l.layer_norm === nothing || print(io, ", LayerNorm")
    print(io, ")")
end

"""
    _signature(l::Dense) -> String

Architecture-signature string for one `Dense` layer: input/output size,
activation, and whether `LayerNorm` is present. Used by [`chain_signature`](@ref).
"""
_signature(l::Dense) =
    "Dense($(size(l.W,2))=>$(size(l.W,1)),$(_actname(l.act_func)),LN=$(l.layer_norm !== nothing))"

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

function apply_act!(layer::Dense{T,M,V,F,G,B,R,LN}, a, z) where {T,M,V,F,G,B,R<:Tuple,LN}
    # multi activations in Dense layer
    map(layer.act_func, layer.act_ranges) do f, r
        @views z[r,:] .= f.(a[r,:])
    end
    return z
end

