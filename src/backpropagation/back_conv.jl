
"""
    ConvBuffer(layer::Conv, x::AbstractArray)

Pre-allocated buffer for backpropagation and Jacobian calculations through a
[`Conv`](@ref) layer.

# Arguments
* `layer`: a `Conv` layer.
* `x`: input array of the layer, shape `(L_in..., C_in, batch)`, used as the
  shape template for the outgoing gradient.

# Fields
* `δz`: gradient at the pre-activation, shape `(L_out..., C_out, batch)`.
* `δ`: gradient sent to the layer below, shape `(L_in..., C_in, batch)`.
"""
struct ConvBuffer{DZ <: AbstractArray, D <: AbstractArray}
    δz::DZ    # (L_out..., C_out, batch)  δ_a, the gradient at the pre-activation
    δ ::D     # (L_in...,  C_in,  batch)  gradient sent to the layer below
end
function ConvBuffer(layer::Conv, x::AbstractArray)
    layer.a === nothing && error("Run a forward pass before building backprop buffers.")
    return ConvBuffer(similar(layer.a), similar(x))
end
Base.show(io::IO, ::MIME"text/plain", ::ConvBuffer) = print(io, "ConvBuffer")

"""
    make_buffer(l::Conv, x) -> ConvBuffer

Build the [`ConvBuffer`](@ref) for `l`, sized from `x`.
"""
make_buffer(l::Conv,  x) = ConvBuffer(l, x)

"""
    _conv_JW_kernel!(J_W, δz, x, stride, pad, ::Val{Nsp})

```math
J_W[k..., c_{in}, c_{out}, b] = \\sum_{p} \\delta z[p, c_{out}, b] \\cdot x[\\mathrm{src}(i), c_{in}, b],
\\qquad i = (p-1)s + k - p_l
```

where `p_l = _padleft(pad, K)` and `src` is padding-dependent.
"""
@kernel function _conv_JW_kernel!(J_W, δz, x, stride::Int, pad, ::Val{Nsp}) where Nsp
    idx = @index(Global, NTuple)  # (k..., c_in, c_out, b)
    c_in = idx[Nsp+1]
    c_out = idx[Nsp+2]
    b = idx[Nsp+3]

    s = zero(eltype(J_W))
    @inbounds for p in CartesianIndices(ntuple(d -> size(δz, d), Val(Nsp)))
        src = ntuple(d -> _src(pad, 
                     (p[d] - 1) * stride + idx[d] - _padleft(pad, size(J_W, d)),
                     size(x, d)), Val(Nsp))
        if all(i -> i > 0, src) 
            s += δz[p, c_out, b] * x[CartesianIndex(src), c_in, b]
        end
    end
    @inbounds J_W[idx...] = s
end

"""
    _conv_Jb_kernel!(J_b, δz, ::Val{Nsp})

```math
J_b[c_{out}, b] = \\sum_{p} \\delta z[p, c_{out}, b]
```
"""
@kernel function _conv_Jb_kernel!(J_b, δz, ::Val{Nsp}) where Nsp
    o, n = @index(Global, NTuple)
    s = zero(eltype(J_b))
    @inbounds for p in CartesianIndices(ntuple(i -> size(δz, i), Val(Nsp)))
        s += δz[p, o, n]
    end
    @inbounds J_b[o, n] = s
end

"""
    _conv_dx_kernel!(δx, δz, W, stride, pad, ::Val{Nsp})

```math
\\delta x[q..., c_{in}, b] = \\sum_{o} \\sum_{k} W[k, c_{in}, o] \\cdot \\delta z[(q-k)/s + 1, o, b]
```

where `p_l = _padleft(pad, K)` and `tap_inv` is padding-dependent (the inverse of `src`).
"""
@kernel function _conv_dx_kernel!(δx, δz, W, stride::Int, pad, ::Val{Nsp}) where Nsp
    idx = @index(Global, NTuple)      # (q..., c_in, b)
    c_in = idx[Nsp+1]
    b = idx[Nsp+2]
    C_out = size(W, Nsp+2)

    s = zero(eltype(δx))
    @inbounds for k in CartesianIndices(ntuple(d -> size(W, d), Val(Nsp)))
        t  = ntuple(d -> _tap_inv(pad,
                     idx[d] - k[d] + _padleft(pad, size(W, d)), size(δx, d)), Val(Nsp))
        ok = all(ntuple(d -> (t[d] >= 0) & (t[d] % stride == 0) &
                             (t[d] ÷ stride < size(δz, d)), Val(Nsp)))
        if ok
            p = CartesianIndex(ntuple(d -> t[d] ÷ stride + 1, Val(Nsp)))
            for o in 1:C_out
                s += W[k, c_in, o] * δz[p, o, b]
            end
        end
    end
    @inbounds δx[idx...] = s
end

"""
    back!(layer::Conv, buf::ConvBuffer, J_W, J_b, δ, x) -> buf.δ

Reverse pass through one [`Conv`](@ref) layer, writing the per-sample Jacobian
into the contiguous arrays `J_W` and `J_b`. Returns the gradient `buf.δ` to be
passed on to the layer below.

`J_W` and `J_b` are computed by [`_conv_JW_kernel!`](@ref) and
[`_conv_Jb_kernel!`](@ref); `buf.δ` is computed by [`_conv_dx_kernel!`](@ref).

# Arguments
* `layer`: a `Conv` layer.
* `buf`: [`ConvBuffer`](@ref) holding the input and output gradients of
    backpropagation.
* `J_W`: Jacobian of the kernel, shape `(kernel_size..., C_in, C_out, batch)`.
* `J_b`: Jacobian of the bias, shape `(C_out, batch)`.
* `δ`: gradient incoming from the layer above, shape `(L_out..., C_out, batch)`.
* `x`: input array of the layer, shape `(L_in..., C_in, batch)`.
"""
function back!(layer::Conv, buf::ConvBuffer, J_W, J_b,
               δ::AbstractArray, x::AbstractArray)
    backend = KernelAbstractions.get_backend(x)

    a = something(layer.a)
    apply_act_deriv!(buf.δz, layer, a)
    buf.δz .*= δ

    Nsp = Val(ndims(x) - 2)
    _conv_JW_kernel!(backend)(J_W, buf.δz, x, layer.stride, layer.pad, Nsp; ndrange = size(J_W))
    _conv_Jb_kernel!(backend)(J_b, buf.δz, Nsp; ndrange = size(J_b))
    _conv_dx_kernel!(backend)(buf.δ, buf.δz, layer.W, layer.stride, layer.pad, Nsp; ndrange = size(buf.δ))
    KernelAbstractions.synchronize(backend)
    return buf.δ
end
