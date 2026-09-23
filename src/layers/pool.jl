# using Statistics
# using KernelAbstractions

"""
    Pool(op::Symbol; device=identity)

A global pooling layer, reducing each channel over all spatial sites:

```math
z_{c} = \\operatorname*{op}_{\\mathbf{i}} \\; x_{c}[\\mathbf{i}],
\\qquad \\text{op} \\in \\{\\max, \\min, \\text{sum}, \\text{mean}\\}
```

where ``\\mathbf{i}`` runs over all spatial sites and ``c`` is the channel. For
`:mean`, this is ``z_c = \\frac{1}{N} \\sum_{\\mathbf{i}} x_c[\\mathbf{i}]``, with
``N`` the number of spatial sites.

An input of shape `(L..., C, batch)` is reduced to an output of shape `(C, batch)`,
so a `Pool` typically sits between [`Conv`](@ref) layer and a [`Dense`](@ref) layer.
The layer has no trainable parameters!

# Arguments
* `op`: pooling operation, one of `:max`, `:min`, `:sum`, `:mean`.

# Keyword Arguments
* `device`: device function deciding where the layer lives, e.g. `identity` for CPU,
            `cu` for GPU (CUDA), or `mtl` for GPU (Metal).

# Example
```julia
pool = Pool(:mean)
pool_gpu = Pool(:max; device=cu)
```
"""
mutable struct Pool{T, A<:AbstractArray{T}, O} <: FreeLayer
    op::Val{O}
    z::Union{Nothing, A}
end
function Pool(op::Symbol; device::Function=identity)
    op in (:max, :min, :sum, :mean) ||
        error("Pooling `op` must be one of :max, :min, :sum, :mean")
    T = Float32
    A = typeof(device(zeros(T, 1, 1)))  # array type only, nothing is kept
    return Pool{T, A, op}(Val(op), nothing)
end

_signature(::Pool{T,A,O}) where {T,A,O} = "Pool($O)"


"""
    forward(layer::Pool, x::AbstractArray) -> layer.z

Forward pass through a [`Pool`](@ref) layer. It calculates the forward pass
in-place of the `Pool` structure and returns its layer output. It utilise
`@kernel` call function that automatically runs on `Pool` device.

On the first call, the buffer `layer.z` is allocated with shape `(C, batch)`.
All subsequent calls are allocation-free.

# Arguments
* `layer`: a `Pool` layer
* `x`: input array, shape `(L..., C, batch)`.
"""
function forward(layer::Pool{T}, x::AbstractArray{T}) where T
    C_in  = size(x, ndims(x) - 1)
    batch = size(x, ndims(x))

    # First pass = initialize `z`
    z0 = layer.z
    if z0 === nothing
        layer.z = similar(x, C_in, batch)
    end

    z = something(layer.z)

    backend = KernelAbstractions.get_backend(x)
    _pool_forward_kernel!(backend)(z, x, layer.op; ndrange = (C_in, batch))
    KernelAbstractions.synchronize(backend)
    return z
end

"""
    forward(layer::Pool, x::AbstractArray, layerMulti) -> layerMulti.a

Similar as [`forward`](@ref), but it allows multi batched forward pass. `layerMulti`
holds the (pre-allocated, fixed `buffer_size`) output of this pass.
"""
function forward(layer::Pool{T}, x::AbstractArray{T}, layerMulti) where T
    C_in  = size(x, ndims(x) - 1)
    batch = size(x, ndims(x))
    backend = KernelAbstractions.get_backend(x)
    _pool_forward_kernel!(backend)(layerMulti.a, x, layer.op; ndrange = (C_in, batch))
    KernelAbstractions.synchronize(backend)
    return layerMulti.a
end

@kernel function _pool_forward_kernel!(z, x, ::Val{op}) where op
    c, b = @index(Global, NTuple)
    Nsp = ndims(x) - 2
    T = eltype(z)

    spatial = CartesianIndices(size(x)[1:Nsp])

    # initial value depends on op 
    # (branches resolve at compile time since op is a type parameter)
    acc = if op === :max
        typemin(T)
    elseif op === :min
        typemax(T)
    else
        zero(T)
    end

    @inbounds for p in spatial
        v = x[p, c, b]
        if op === :max
            acc = max(acc, v)
        elseif op === :min
            acc = min(acc, v)
        else # :sum and :mean
            acc += v
        end
    end

    if op === :mean
        acc /= T(length(spatial))
    end

    @inbounds z[c, b] = acc
end
