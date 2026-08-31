# using LinearAlgebra
# using Statistics

"""
    LayerRange(W, b)

Index ranges for one layer's weights and biases inside the flat parameter
vector.

Possibly used with [`LayerNorm`](@ref) if needed (otherwise dummy variables).
"""
struct LayerRange
    W::UnitRange{Int}
    b::UnitRange{Int}
    γ::Union{Nothing, UnitRange{Int}}   
    β::Union{Nothing, UnitRange{Int}}
end
LayerRange(W, b) = LayerRange(W, b, nothing, nothing)

"""
    JacobianBuffer(ansatz, buffers::Tuple)

Pre-allocated storage for a per-sample Jacobian over a [`Chain`](@ref). 
Two storages live side by side:

* `J_layers`: a tuple of `(J_W, J_b)` per layer, with `J_W` and `J_b` as 
    contiguous arrays of shape `(out, in, batch)` and `(out, batch)`. 
    [`back!`](@ref) writes here.
* `J`: the flatten `(p, batch)` Jacobian. Filled by [`flatten_jacobian!`] 
    after the reverse pass. This is the array consumed downstream by optimisers.

# Arguments

* `ansatz`: [`NeuralAnsatz`](@ref) that holds Neural Network structure. See [`Chain`](@ref)
* `buffers`: Buffers holdin each layers input/output gradients. See [`DenseBuffer`](@ref).

# Futher Arguments:

* `ranges`: tuple of [`LayerRange`].
* `δ_init`: output-side seed gradient
* `θ`: flatten parameter vector mirroring the chain's current weights.
* `zipped`: precomputed `(layer, buf, (J_W, J_b), x)` tuples, one per
  layer, fed into the recursive [`_backprop!`].
* `ln_zipped`: See [`LayerNorm`](@ref). J_γ/J_β live inside.

## Notes

θ layout per layer:  W | b | γ | β   (γ/β only when `LayerNorm` is present)
"""
struct JacobianBuffer{JC <: AbstractArray, R <: Tuple, DI <: AbstractArray,
                      V <: AbstractVector, JL <: Tuple, ZP <: Tuple, LN}
    J        ::JC
    J_layers ::JL
    ranges   ::R
    δ_init   ::DI
    θ        ::V
    zipped   ::ZP
    ln_zipped::LN
end
function JacobianBuffer(ansatz, buffers::Tuple)
    chain = ansatz.model
    refW  = first(chain.layers).W
    refb  = first(chain.layers).b
    T     = eltype(refW)
    batch = chain.batch

    # Build ranges and contiguous per-layer Jacobian storage in one pass
    rs        = ()
    J_layers  = ()
    ln_zipped = ()
    offset    = 0
    for layer in chain.layers
        out_dim, in_dim = size(layer.W)
        nW = length(layer.W)
        nb = length(layer.b)

        r_W = (offset+1):(offset+nW)
        offset += nW
        r_b = (offset+1):(offset+nb)  
        offset += nb

        # Contiguous arrays — reshapes inside back!
        J_W = similar(refW, out_dim, in_dim, batch)
        J_b = similar(refb, out_dim, batch)
        J_layers = (J_layers..., (J_W, J_b))

        if layer.layer_norm !== nothing
            H   = out_dim
            r_γ = (offset+1):(offset+H)
            offset += H
            r_β = (offset+1):(offset+H)
            offset += H
            rs        = (rs..., LayerRange(r_W, r_b, r_γ, r_β))
            ln_zipped = (ln_zipped..., layer.layer_norm)   # J_γ/J_β already inside
        else
            rs        = (rs..., LayerRange(r_W, r_b))
            ln_zipped = (ln_zipped..., nothing)
        end
    end
    p = offset

    J = similar(refW, p, batch) # (p, batch)

    # Flat parameter vector, initialised to the chain's current weights
    θ = similar(refb, p)
    for (layer, r) in zip(chain.layers, rs)
        view(θ, r.W) .= reshape(layer.W, :)
        view(θ, r.b) .= layer.b
        if !isnothing(r.γ)
            view(θ, r.γ) .= reshape(layer.layer_norm.γ, :)
            view(θ, r.β) .= reshape(layer.layer_norm.β, :)
        end
    end

    # Seed gradient at the output: ones (times batch when batched)
    δ_init = init_gradient_seed(ansatz)

    # Each layer's input: chain input first, then previous layers' outputs
    layer_inputs = (chain.x, Base.front(map(l -> l.z, chain.layers))...)

    # Per-layer bundle consumed by _backprop!
    zipped = map(tuple, chain.layers, buffers, J_layers, layer_inputs)

    return JacobianBuffer(J, J_layers, rs, δ_init, θ, zipped, ln_zipped)
end

"""
    _backprop!(δ, zipped) -> δ

Walk the per-layer `zipped` tuple from last to first, applying [`back!`](@ref) 
recursivelly. 

## Notes
Contains `_ln_dispatch!` functions for `LayerNorm` dispatch.
"""
@inline _backprop!(δ, ::Tuple{}, ::Tuple{}) = δ
@inline function _backprop!(δ, zipped, ln_zipped)
    δ = _backprop!(δ, Base.tail(zipped), Base.tail(ln_zipped))
    (layer, buf, (J_W, J_b), x) = first(zipped)
    return _ln_dispatch!(layer, buf, J_W, J_b, δ, x, first(ln_zipped))
end

@inline _ln_dispatch!(layer, buf, J_W, J_b, δ, x, ::Nothing) =
    back!(layer, buf, J_W, J_b, δ, x)

@inline _ln_dispatch!(layer, buf, J_W, J_b, δ, x, ln::LayerNorm) =
    back!(layer, buf, J_W, J_b, δ, x, ln)

"""
    flatten_jacobian!(jac) -> jac.J

Creates `@view` of each layer's gradients `(J_W, J_b)` (`(J_W, J_b, J_γ, J_β)`) 
into the flatten Jacobian `jac.J`. See [`JacobianBuffer`](@ref).
"""
function flatten_jacobian!(jac::JacobianBuffer)
    batch = size(jac.J, 2)
    for (r, (J_W, J_b), ln_e) in zip(jac.ranges, jac.J_layers, jac.ln_zipped)
        view(jac.J, r.W, :) .= reshape(J_W, :, batch)
        view(jac.J, r.b, :) .= J_b
        if !isnothing(ln_e)
            view(jac.J, r.γ, :) .= reshape(ln_e.J_γ, :, batch)
            view(jac.J, r.β, :) .= reshape(ln_e.J_β, :, batch)
        end
    end
    return jac.J
end

"""
    back_jacobian!(ansatz, jac) -> jac.J

Run the full reverse pass over all layers and assemble the flat Jacobian
`jac.J`. Returns `jac.J` of shape `(p, batch)` (or `(p,)` for a single
input): each column is one sample's full gradient with respect to the
flat parameter vector `θ`.
"""
function back_jacobian!(ansatz, jac::JacobianBuffer)
    jac.δ_init .= init_gradient_seed(ansatz)
    _backprop!(jac.δ_init, jac.zipped, jac.ln_zipped)
    flatten_jacobian!(jac)
    return jac.J
end
