# using LinearAlgebra
# using Statistics


"""
    layer_inputs(chain) -> Tuple

The input each layer received during the forward pass: `chain.x` for the first
layer, then each preceding layer's `z` output (previous layer output `z` is new
layer input `x`). 

## Note
Requires a completed forward pass.
"""
layer_inputs(chain) = (chain.x, Base.front(map(l -> l.z, chain.layers))...)

"""
    make_buffers(chain) -> Tuple

Build the backpropagation buffer for every layer in [`Chain`](@ref), dispatching on
layer type via [`make_buffer`](@ref). Requires a completed forward pass, since
buffer shapes are derived from each layer's input (see [`layer_inputs`](@ref)).
"""
make_buffers(chain) = map(make_buffer, chain.layers, layer_inputs(chain))

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

Base.show(io::IO, ::MIME"text/plain", ::LayerRange) = print(io, "LayerRange")

"""
    JacobianBuffer(ansatz, buffers::Tuple)

Pre-allocated storage for a per-sample Jacobian over a [`Chain`](@ref). 
Two storages live side by side:

* `J_layers`: a tuple with one `(J_W, J_b)` pair per layer, where `J_W` has
    shape `(size(W)..., batch)` and `J_b` has shape `(length(b), batch)`. Layers 
    without parameters [`FreeLayer`](@ref), get `(nothing, nothing)`.
    [`back!`](@ref) writes here.
* `J`: the flatten `(p, batch)` Jacobian. Filled by [`flatten_jacobian!`] 
    after the reverse pass. This is the array consumed downstream by optimisers.

# Arguments

* `ansatz`: [`NeuralAnsatz`](@ref) that holds Neural Network structure. See [`Chain`](@ref)
* `buffers`: Buffers holdin each layers input/output gradients. See in [`DenseBuffer`](@ref),
    [`ConvBuffer`](@ref), [`PoolBuffer`](@ref).

# Fields
* `J`, `J_layers`: see above.
* `ranges`: tuple of [`LayerRange`](@ref). Empty for layers without parameters.
* `δ_init`: output-side seed gradient
* `θ`: flatten parameter vector mirroring the chain's current weights.
* `zipped`: precomputed `(layer, buf, (J_W, J_b), x)` tuples, one per
  layer, fed into the recursive [`_backprop!`](@ref).
* `ln_zipped`: tuple with the [`LayerNorm`](@ref) of each layer, or `nothing`
  if the layer has none. J_γ/J_β live inside.

# Notes
Per layer, the parameters are laid out in `θ` as

    W | b | γ | β

with `W` flattened column-major (`reshape(W, :)`), and `γ`/`β` present only
when the layer has a `LayerNorm`. Layers without parameters take up no space.
"""
struct JacobianBuffer{JC <: AbstractArray, R <: Tuple, DI <: AbstractArray,
                      V <: AbstractVector, JL <: Tuple, ZP <: Tuple, LN}
    J::JC
    J_layers::JL
    ranges::R
    δ_init::DI
    θ::V
    zipped::ZP
    ln_zipped::LN
end
function JacobianBuffer(ansatz, buffers::Tuple)
    chain = ansatz.model
    ref = first(filter(hasparams, chain.layers))
    refW, refb = ref.W, ref.b
    batch = chain.batch

    rs = (); J_layers = (); ln_zipped = (); offset = 0
    for layer in chain.layers
        if !hasparams(layer)        # for FreeLayers skip
            rs        = (rs..., LayerRange((offset+1):offset, (offset+1):offset))  # empty ranges
            J_layers  = (J_layers..., (nothing, nothing))
            ln_zipped = (ln_zipped..., nothing)
            continue
        end

        nW = length(layer.W)
        nb = length(layer.b)        # out_dim (Dense) or C_out (Conv)
        r_W = (offset+1):(offset+nW);  offset += nW
        r_b = (offset+1):(offset+nb);  offset += nb

        J_W = similar(refW, size(layer.W)..., batch)      # any rank of W
        J_b = similar(refb, nb, batch)
        J_layers = (J_layers..., (J_W, J_b))

        ln = layer.layer_norm
        if ln !== nothing
            r_γ = (offset+1):(offset+nb);  offset += nb
            r_β = (offset+1):(offset+nb);  offset += nb
            rs        = (rs..., LayerRange(r_W, r_b, r_γ, r_β))
            ln_zipped = (ln_zipped..., ln)
        else
            rs        = (rs..., LayerRange(r_W, r_b))
            ln_zipped = (ln_zipped..., nothing)
        end
    end
    p = offset

    J = similar(refW, p, batch)
    θ = similar(refb, p)
    for (layer, r) in zip(chain.layers, rs)
        hasparams(layer) || continue
        view(θ, r.W) .= reshape(layer.W, :)
        view(θ, r.b) .= layer.b
        if !isnothing(r.γ)
            view(θ, r.γ) .= reshape(layer.layer_norm.γ, :)
            view(θ, r.β) .= reshape(layer.layer_norm.β, :)
        end
    end

    δ_init = similar(last(ansatz.model.layers).z)
    fill!(δ_init, one(eltype(δ_init)))

    zipped = map(tuple, chain.layers, buffers, J_layers, layer_inputs(chain))
    return JacobianBuffer(J, J_layers, rs, δ_init, θ, zipped, ln_zipped)
end
Base.show(io::IO, ::MIME"text/plain", ::JacobianBuffer) = print(io, "JacobianBuffer")


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
    flatten_jacobian!(jac::JacobianBuffer) -> jac.J

Assemble the per-layer gradients stored in `jac.J_layers` (and `jac.ln_zipped`
for any LayerNorm layers) into the flat Jacobian `jac.J`, using the ranges in
`jac.ranges`. Parameter-free layers (e.g. `Pool`) are skipped. See [`JacobianBuffer`](@ref).
"""
function flatten_jacobian!(jac::JacobianBuffer)
    batch = size(jac.J, 2)
    map(jac.ranges, jac.J_layers, jac.ln_zipped) do r, (J_W, J_b), ln
        _flatten!(jac.J, r, J_W, J_b, ln, batch)
    end
    return jac.J
end

"""
    _flatten!(J, r, J_W, J_b, ln, batch)

Copy one layer's `J_W`/`J_b` into its slice of `J`, at the ranges in `r`, then
delegate the LayerNorm slice to [`_flatten_ln!`](@ref). No-op when `J_W`/`J_b`
are `nothing` (parameter-free layer).
"""
_flatten!(J, r, ::Nothing, ::Nothing, ln, batch) = nothing
function _flatten!(J, r, J_W, J_b, ln, batch)
    view(J, r.W, :) .= reshape(J_W, :, batch)
    view(J, r.b, :) .= J_b
    _flatten_ln!(J, r, ln, batch)
    return nothing
end

"""
    _flatten_ln!(J, r, ln, batch)

Copy `ln.J_γ`/`ln.J_β` into their slice of `J`, at the ranges in `r`. No-op
when `ln` is `nothing` (layer has no LayerNorm).
"""
_flatten_ln!(J, r, ::Nothing, batch) = nothing
function _flatten_ln!(J, r, ln, batch)
    view(J, r.γ, :) .= reshape(ln.J_γ, :, batch)
    view(J, r.β, :) .= reshape(ln.J_β, :, batch)
    return nothing
end

"""
    back_jacobian!(ansatz, jac::JacobianBuffer) -> jac.J

Run the reverse pass over every layer (via [`_backprop!`](@ref)) and flatten the
result into `jac.J` (via [`flatten_jacobian!`](@ref)). `jac.J` has shape `(p, batch)`: 
each column is one sample's full gradient with respect to the flat parameter vector `θ`.
"""
function back_jacobian!(ansatz, jac::JacobianBuffer)
    δ_init = init_gradient_seed!(ansatz, jac.δ_init)
    _backprop!(δ_init, jac.zipped, jac.ln_zipped)
    flatten_jacobian!(jac)
    return jac.J
end
