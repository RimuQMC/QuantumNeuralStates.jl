# using LinearAlgebra
# KernelAbstractions

"""
    PadMode

This abstract type is used for different padding types in 
[`Conv`](@ref) layers. See also different padding types
[`NoPad`](@ref), [`Zeros`](@ref), [`Periodic`](@ref).
"""
abstract type PadMode end

"""
    NoPad <: PadMode

This structure does not apply padding to [`Conv`](@ref) layer (default).
It also means that layer output spatial dimensions will shrink by `K-1` factor,
where `K` is convolution kernel window dimension.

# Examples
```text
input  (length 5):  x1  x2  x3  x4  x5
kernel (K = 3):     windows [x1 x2 x3], [x2 x3 x4], [x3 x4 x5]
output (length 3):  y1  y2  y3
```
"""
struct NoPad <: PadMode end # no padding, output shrinks

"""
    Periodic <: PadMode

This padding struct applies periodic padding to [`Conv`](@ref) layer. The 
layer output spatial dimensions stay same as input spatial dimensions.

# Examples
```text
1D, K = 3:

    x1  x2  x3   →   x3 | x1  x2  x3 | x1

2D, 3×3 kernel (padded border shown outside the box):

         x33  x31  x32  x33  x31
             ┌──────────────┐
         x13 │ x11  x12  x13│ x11
         x23 │ x21  x22  x23│ x21
         x33 │ x31  x32  x33│ x31
             └──────────────┘
         x13  x11  x12  x13  x11
```
"""
struct Periodic <: PadMode end # periodic boundary

"""
    Zeros <: PadMode

This padding struct applies zeros padding to [`Conv`](@ref) input dimension 
borders. Therefore, output spatial dimension does not shrink (same as input 
spatial dimension) and it creates open-like border system behaviour.

# Examples
```text
1D, K = 3:

    x1  x2  x3   →   0 | x1  x2  x3 | 0

2D, 3×3 kernel (padded border shown outside the box):

          0    0    0    0    0
             ┌──────────────┐
          0  │ x11  x12  x13│  0
          0  │ x21  x22  x23│  0
          0  │ x31  x32  x33│  0
             └──────────────┘
          0    0    0    0    0
```
"""
struct Zeros <: PadMode end  # open boundary, missing neighbours contribute 0

"""
    _padleft(pad, K) -> Int

Left padding for a kernel of size `K`: `0` for `NoPad`, `(K-1)÷2` otherwise
(right side gets `K÷2`, so even `K` also works).
"""
@inline _padleft(::NoPad, K) = 0
@inline _padleft(::PadMode, K) = (K - 1) ÷ 2

"""
    _lout(pad, L, K, s) -> Int

Output length of one spatial dimension, given input length `L`, kernel size
`K`, and stride `s`.
"""
@inline _lout(::NoPad, L, K, s) = fld(L - K, s) + 1
@inline _lout(::PadMode, L, K, s) = (L - 1) ÷ s + 1     # = L for s = 1

"""
    _src(pad, i, L) -> Int

Map a raw (possibly padded) input index `i` to a real index in `1:L`, per
`pad`. `0` means "skip this tap" (`Zeros`, out of range).
"""
@inline _src(::NoPad, i, L) = i
@inline _src(::Periodic, i, L) = mod1(i, L)
@inline _src(::Zeros, i, L) = (1 <= i <= L) ? i : 0

"""
    _tap_inv(pad, t, L) -> Int

Inverse of [`_src`](@ref) for the `dx` kernel: given `t = (p-1)*stride`,
returns the candidate offset (wrapped into `0:L-1` for `Periodic`, unchanged
otherwise).
"""
@inline _tap_inv(::PadMode, t, L) = t
@inline _tap_inv(::Periodic, t, L) = mod(t, L)      # 0-based, always in 0:L-1


"""
    Conv(kernel_size, C_in => C_out, act; kwargs...) <: ParametricLayer

An N-dimensional convolutional neural network layer, computing:

```math
\\begin{aligned}
a_{c}[\\mathbf{p}] &= \\sum_{c'=1}^{C_\\text{in}} \\sum_{\\mathbf{j}} W[\\mathbf{j}, c', c] \\; x_{c'}[s(\\mathbf{p}-1) + \\mathbf{j}] + b_c \\\\
z_{c}[\\mathbf{p}] &= \\text{act\\_func}(a_{c}[\\mathbf{p}])
\\end{aligned}
```

where ``\\mathbf{p}`` is the (multi-)index of the output site, ``\\mathbf{j}`` runs over the
kernel window, ``c'`` and ``c`` are the input and output channels, and ``s`` is the stride.
Without padding, each spatial dimension of the output has length
``L_\\text{out} = \\lfloor (L_\\text{in} - k)/s \\rfloor + 1``. With padding 
``L_\\text{out} = L_\\text{in}`` - spatial dimensions are preserved. 

Buffers `a` and `z` are allocated on the first forward pass; afterwards the layer is
allocation-free at runtime.

# Arguments
* `kernel_size`: tuple of kernel sizes, one per spatial dimension, e.g. `(3,)` for 1D
                 or `(3, 3)` for 2D.
* `C_in => C_out`: number of input and output channels.
* `act`: activation function — must be registered in [`ACT_DERIV`](@ref).

# Keyword Arguments
* `stride`: step between neighbouring kernel windows (default `1`).
* `pad`: padding mode, a subtype of [`PadMode`](@ref) (default `NoPad()`).
* `batch`: batch size.
* `device`: device function deciding where the layer lives, e.g. `identity` for CPU,
            `cu` for GPU (CUDA), or `mtl` for GPU (Metal).
* `Layer_Norm`: no need for it in this layer type (default=nothing).

# Notes
The kernel `W` has shape `(kernel_size..., C_in, C_out)`. With
`fan_in = prod(kernel_size) * C_in` and `fan_out = prod(kernel_size) * C_out` (
number of input/output connections).

## Weight Initialisation
* `He` initialisation (`sqrt(2/fan_in)`) for `relu` and `gelu`
* `Glorot/Xavier` initialisation (`sqrt(2/(fan_in+fan_out))`) for all other activations
* Biases initialised to zero

# Example
```julia
layer = Conv((3,), 1 => 8, tanh)
layer_2d = Conv((3, 3), 1 => 8, relu; stride=2, device=cu)
layer_2d_periodic = Conv((3, 3), 1 => 8, relu; stride=2, device=cu, pad=Periodic())
```
"""
mutable struct Conv{T, K<:AbstractArray{T}, V<:AbstractVector{T},
                    F<:Function, G<:Function, Z<:AbstractArray{T}, PM<:PadMode} <: ParametricLayer
    W::K        # kernel (dimension of window, C_in, C_out = number of kernels/features)
    b::V
    act_func::F
    act_deriv::G
    a::Union{Nothing, Z}    # will be initialised in first forward pass
    z::Union{Nothing, Z}    # will be initialised in first forward pass

    stride::Int
    pad::PM

    layer_norm::Union{LayerNorm, Nothing}
end
function Conv(kernel_size::NTuple{N, Int}, channels::Pair{Int,Int}, act::Function;
              stride::Int=1, device::Function=identity, Layer_Norm=false, 
              batch::Int=1, pad::PadMode=NoPad()) where N
    # L_in is spatial dimension of input 
    T = Float32
    C_in, C_out = channels.first, channels.second
    fan_in  = prod(kernel_size) * C_in
    fan_out = prod(kernel_size) * C_out
    std = _init_std(T, fan_in, fan_out, act)

    W = device(randn(T, (kernel_size..., C_in, C_out)) .* std)
    b = device(zeros(T, C_out))

    act_d = _lookup_deriv(act)

    layer_norm = if Layer_Norm===false
        nothing
    else
        @warn "This type of layer does not support layer normalisation! Defaults to nothing."
        nothing
    end

    K,V,F,G,PM = typeof(W), typeof(b), typeof(act), typeof(act_d), typeof(pad)
    return Conv{T,K,V,F,G,K,PM}(W, b, act, act_d, nothing, nothing, 
                                   stride, pad, layer_norm)
end

function Base.show(io::IO, l::Conv)
    N = ndims(l.W)                       # W: (K..., C_in, C_out)
    print(io, "Conv(", size(l.W)[1:N-2], ", ", size(l.W, N-1), "=>", size(l.W, N),
          ", ", _actname(l.act_func), "; pad=", nameof(typeof(l.pad)), "()")
    l.stride == 1            || print(io, ", stride=", l.stride)
    l.layer_norm === nothing || print(io, ", LayerNorm")
    print(io, ")")
end

"""
    _signature(l::Conv) -> String

Architecture-signature string for one `Conv` layer: kernel size, channels,
activation, stride, padding mode, and whether `LayerNorm` is present. Used
by [`chain_signature`](@ref).
"""
_signature(l::Conv{T,K,V,F,G,Z,PM}) where {T,K,V,F,G,Z,PM} =
    "Conv($(size(l.W)[1:end-2]),$(size(l.W,ndims(l.W)-1))=>$(size(l.W,ndims(l.W))),"*
    "$(_actname(l.act_func)),stride=$(l.stride),pad=$(PM),LN=$(l.layer_norm !== nothing))"

"""
    forward(layer::Conv, x::AbstractArray) -> layer.z

Forward pass through a [`Conv`](@ref) layer. It calculates the forward pass
in-place of the `Conv` structure and also returns its layer output. It utilise
`@kernel` call function that automatically runs on `Conv` device.

# Arguments
* `layer`: a `Conv` layer
* `x`: input array, shape `(L_in..., C_in, batch)`.

# Note
*On the first call, the buffers `layer.a` and `layer.z` are allocated with shape
`(L_out..., C_out, batch)`, where `L_out` is determined by the input grid,
kernel size, stride and padding mode. All subsequent calls are allocation-free.
"""
function forward(layer::Conv, x::AbstractArray)
    # First pass = initialize `a` and `z`
    Nsp = Val(ndims(x) - 2)
    a0 = layer.a
    if a0 === nothing
        C_out = size(layer.W)[end]
        batch = size(x)[end]

        L_out = ntuple(d -> _lout(layer.pad, size(x, d), size(layer.W, d), layer.stride), Nsp)
        any(<(1), L_out) &&
            error("Grid $(size(x)[1:end-2]) is too small for kernel $(size(layer.W)[1:end-2])")

        layer.z = fill!(similar(layer.W, L_out..., C_out, batch), zero(eltype(layer.W)))
        layer.a = fill!(similar(layer.W, L_out..., C_out, batch), zero(eltype(layer.W)))
    end

    a = something(layer.a)  # narrowed to Z type
    z = something(layer.z)

    backend = KernelAbstractions.get_backend(x)
    _conv_forward_kernel!(backend)(a, x, layer.W, layer.b, layer.stride, layer.pad, Nsp; 
                          ndrange = size(a))
    KernelAbstractions.synchronize(backend)
    apply_act!(layer, a, z)
    return z
end

"""
    forward(layer::Conv, x::AbstractArray, layerMulti) -> layerMulti.a

Similar as [`forward`](@ref), but it allows multi batched forward pass. `layerMulti`
holds the (pre-allocated, fixed `buffer_size`) output of this pass.
"""
function forward(layer::Conv, x::AbstractArray, layerMulti)
    Nsp = Val(ndims(x) - 2)
    backend = KernelAbstractions.get_backend(x)
    _conv_forward_kernel!(backend)(layerMulti.a, x, layer.W, layer.b, layer.stride, layer.pad, Nsp;
                                   ndrange = size(layerMulti.a))
    KernelAbstractions.synchronize(backend)
    apply_act!(layer, layerMulti.a, layerMulti.a)
    return layerMulti.a
end

@kernel function _conv_forward_kernel!(a, x, W, b, stride::Int, pad, ::Val{Nsp}) where Nsp
    idx = @index(Global, NTuple)  # (p..., c_out, n)
    c_out = idx[Nsp+1]
    n = idx[Nsp+2]

    s = zero(eltype(a))
    @inbounds for c_in in 1:size(x, Nsp+1)
        for k in CartesianIndices(ntuple(d -> size(W, d), Val(Nsp)))
            src = ntuple(d -> _src(pad, 
                         (idx[d] - 1) * stride + k[d] - _padleft(pad, size(W, d)),
                         size(x, d)), Val(Nsp))
            if all(i -> i > 0, src)                 
                s += x[CartesianIndex(src), c_in, n] * W[k, c_in, c_out]
            end
        end
    end
    @inbounds a[idx...] = s + b[c_out]
end


