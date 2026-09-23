
"""
    PoolBuffer(δ::AbstractArray)

Pre-allocated buffer for backpropagation through a [`Pool`](@ref) layer.

# Fields
* `δ`: gradient sent to the layer below, shape `(L..., C, batch)`, the same
  as the pool's input.
"""
struct PoolBuffer{D <: AbstractArray}
    δ::D      # (L..., C, batch), same shape as the pool's input
end

make_buffer(l::Pool,  x) = PoolBuffer(x)

# sum / mean: every position gets δ[c,b] * scale
@kernel function _pool_back_uniform_kernel!(δx, δ, scale)
    idx = @index(Global, NTuple)                   # (p..., c, b)
    Nsp = ndims(δx) - 2
    @inbounds δx[idx...] = δ[idx[Nsp+1], idx[Nsp+2]] * scale
end

# max / min: only the first extremal position gets δ[c,b]; δx must be pre-zeroed
@kernel function _pool_back_extremum_kernel!(δx, δ, x, ::Val{op}) where op
    c, b = @index(Global, NTuple)
    Nsp  = ndims(x) - 2
    spatial = CartesianIndices(size(x)[1:Nsp])

    best_p = first(spatial)
    best_v = @inbounds x[best_p, c, b]
    @inbounds for p in spatial
        v = x[p, c, b]
        if (op === :max && v > best_v) || (op === :min && v < best_v)
            best_v = v
            best_p = p
        end
    end
    @inbounds δx[best_p, c, b] = δ[c, b]
end

function _pool_uniform!(δx, δ, scale)
    backend = KernelAbstractions.get_backend(δx)
    _pool_back_uniform_kernel!(backend)(δx, δ, eltype(δx)(scale); ndrange = size(δx))
    KernelAbstractions.synchronize(backend)
end
function _pool_extremum!(op::Val, δx, δ, x)
    backend = KernelAbstractions.get_backend(δx)
    fill!(δx, zero(eltype(δx)))
    _pool_back_extremum_kernel!(backend)(δx, δ, x, op;
        ndrange = (size(x, ndims(x)-1), size(x, ndims(x))))
    KernelAbstractions.synchronize(backend)
end

# Val{op} dispatches
_pool_backward!(::Val{:sum},   δx, δ, x) = _pool_uniform!(δx, δ, 1)
_pool_backward!(::Val{:mean},  δx, δ, x) = _pool_uniform!(δx, δ, 1 / prod(size(x)[1:end-2]))
_pool_backward!(op::Val{:max}, δx, δ, x) = _pool_extremum!(op, δx, δ, x)
_pool_backward!(op::Val{:min}, δx, δ, x) = _pool_extremum!(op, δx, δ, x)

"""
    back!(layer::Pool, buf::PoolBuffer, ::Nothing, ::Nothing, δ, x) -> buf.δ

Reverse pass through one [`Pool`](@ref) layer. The layer has no parameters, so
no Jacobian is written and the `J_W`, `J_b` slots are `nothing`. Returns the
gradient `buf.δ` to be passed on to the layer below.

# Arguments
* `layer`: `Pool` layer.
* `buf`: [`PoolBuffer`](@ref) holding the outgoing gradient.
* `δ`: gradient incoming from the layer above, shape `(C, batch)`.
* `x`: input array of the layer, shape `(L..., C, batch)`.
"""
function back!(layer::Pool, buf::PoolBuffer, ::Nothing, ::Nothing,
               δ::AbstractArray, x::AbstractArray)
    _pool_backward!(layer.op, buf.δ, δ, x)
    return buf.δ
end
