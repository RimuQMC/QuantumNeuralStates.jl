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
input (length 5):  x1  x2  x3  x4  x5
kernel (K = 3): windows [x1 x2 x3], [x2 x3 x4], [x3 x4 x5]
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
@inline _src(::Periodic, i, L) = i < 1 ? i + L : (i > L ? i - L : i)
@inline _src(::Zeros, i, L) = (1 <= i <= L) ? i : 0

"""
    _tap_inv(pad, t, L) -> Int

Inverse of [`_src`](@ref) for the `dx` kernel: given `t = (p-1)*stride`,
returns the candidate offset (wrapped into `0:L-1` for `Periodic`, unchanged
otherwise).
"""
@inline _tap_inv(::PadMode, t, L) = t
@inline _tap_inv(::Periodic, t, L) = t < 0 ? t + L : (t >= L ? t - L : t)      # 0-based, always in 0:L-1


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
* `Layer_Norm`: no need for it in this layer type (default=nothing). DOES NOT EXIST HERE.
* `init_scale`: allows rescaling of layer's weights during initialisation (`weights .*= init_scale`).

# Notes
The kernel `W` has shape `(kernel_size..., C_in, C_out)`. With
`fan_in = prod(kernel_size) * C_in` and `fan_out = prod(kernel_size) * C_out` (
number of input/output connections).

## Weight Initialisation
* `He` initialisation (`sqrt(2/fan_in)`) for `relu` and `gelu`.
* `Glorot/Xavier` initialisation (`sqrt(2/(fan_in+fan_out))`) for all other activations.
* Biases initialised to zero.

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
              batch::Int=1, pad::PadMode=NoPad(), init_scale::Real=1) where N
    # L_in is spatial dimension of input 
    T = Float32
    C_in, C_out = channels.first, channels.second
    fan_in  = prod(kernel_size) * C_in
    fan_out = prod(kernel_size) * C_out
    std = _init_std(T, fan_in, fan_out, act)

    W = device(randn(T, (kernel_size..., C_in, C_out)) .* std .* T(init_scale))
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
    _check_input(pad, input_size, K)

Check that the input grid is at least as large as the kernel `K` in every spatial
dimension.
"""
_check_input(::PadMode, input_size, K) =
    all(input_size .>= K) || error("input grid $input_size is smaller than kernel $K")

"""
    forward(layer::Conv, x::AbstractArray) -> layer.z

Forward pass through a [`Conv`](@ref) layer, using kernel function of
[`conv_tiled!`](@ref):

```math
\\begin{aligned}
a_{\\mathbf{p},o,n} &= b_o + \\sum_{\\mathbf{k}} \\sum_{c=1}^{C_\\text{in}}
    W_{\\mathbf{k},c,o} \\; x_{\\mathbf{s}(\\mathbf{p},\\mathbf{k}),c,n} \\\\
z_{\\mathbf{p},o,n} &= \\text{act\\_func}(a_{\\mathbf{p},o,n})
\\end{aligned}
```

with output site ``\\mathbf{p}``, kernel tap ``\\mathbf{k}``, input channel
``c``, output channel ``o``, sample ``n``, and source index per spatial
dimension ``d``

```math
s_d(\\mathbf{p}, \\mathbf{k}) = (p_d - 1)\\,\\mathrm{stride} + k_d - \\mathrm{pad}_d,
```

mapped into the input grid according to the padding mode ([`NoPad`](@ref),
[`Zeros`](@ref), [`Periodic`](@ref)). Taps falling outside the input under
`Zeros` padding are skipped.

# Arguments
* `layer`: a `Conv` layer.
* `x`: input array, shape `(L_in..., C_in, batch)`.

# Layouts
- `x`: `(L_in..., C_in, batch)`
- `layer.W`: `(K..., C_in, C_out)`
- `layer.a`, `layer.z`: `(L_out..., C_out, batch)`

# How the computation is organised
The pass runs as a single device kernel ([`conv_tiled!`](@ref)):

* **Work item = one output site, `TILE` output channels.** Each work item
  keeps `TILE = min(32, C_out)` accumulators in registers. Every input value
  `x[s, c, n]` is loaded once and used for `TILE` multiply-adds, which keeps
  the number of memory loads per arithmetic operation low.
* **Fused bias and activation.** `a = acc + b` and `z = act_func(a)` are
  written directly; no separate activation pass is needed.
* **Weights in shared memory.** If `W` fits into the shared-memory budget
  ([`SMEM_BUDGET`](@ref), 32 KB), each workgroup copies it once into fast
  on-chip memory; otherwise it is read from global memory.
* **Compile-time sizes.** Grid, kernel and channel sizes are passed to the
  kernel as type parameters, so loops are unrolled and index arithmetic is
  constant-folded. A specialised kernel is compiled once per layer shape.

The kernel is dimension-agnostic (1D, 2D, 3D, ...), supports any stride and
padding mode, and runs on all KernelAbstractions backends (CUDA, Metal, CPU).

# Notes
* On the first call, the buffers `layer.a` and `layer.z` are allocated with
  shape `(L_out..., C_out, batch)`, where `L_out` follows from the input grid,
  kernel size, stride and padding mode. All subsequent calls are
  allocation-free.
* `layer.a` holds the pre-activation required for backpropagation.
"""
function forward(layer::Conv, x::AbstractArray)
    Nsp = ndims(x) - 2
    if layer.a === nothing
        C_out = size(layer.W)[end]
        batch = size(x)[end]
        L_out = ntuple(d -> _lout(layer.pad, size(x, d), size(layer.W, d), layer.stride), Nsp)
        any(<(1), L_out) &&
            error("Grid $(size(x)[1:end-2]) is too small for kernel $(size(layer.W)[1:end-2])")
        layer.z = fill!(similar(layer.W, L_out..., C_out, batch), zero(eltype(layer.W)))
        layer.a = fill!(similar(layer.W, L_out..., C_out, batch), zero(eltype(layer.W)))
    end
    a = something(layer.a)
    z = something(layer.z)
    conv_tiled!(a, z, x, layer)
    return z
end

"""
    forward(layer::Conv, x::AbstractArray, layerMulti) -> layerMulti.a

Multi-batched forward pass through a [`Conv`](@ref) layer into the
pre-allocated buffer `layerMulti.a` (fixed `buffer_size`). Computes the same
quantity as [`forward`](@ref):

```math
z_{\\mathbf{p},o,n} = \\text{act\\_func}\\Big(b_o + \\sum_{\\mathbf{k}} \\sum_{c=1}^{C_\\text{in}}
    W_{\\mathbf{k},c,o} \\; x_{\\mathbf{s}(\\mathbf{p},\\mathbf{k}),c,n}\\Big)
```

using the same kernel ([`conv_tiled!`](@ref)) and computational scheme. The
result is written in place: `layerMulti.a` holds the activated output, since
the pre-activation is not needed in inference-only passes.

# Arguments
* `layer`: a `Conv` layer.
* `x`: input array, shape `(L_in..., C_in, buffer_size)`.
* `layerMulti`: holds the output buffer `a` of shape `(L_out..., C_out, buffer_size)`.
"""
function forward(layer::Conv, x::AbstractArray, layerMulti)
    conv_tiled!(layerMulti.a, layerMulti.a, x, layer)
    return layerMulti.a
end

"""
    conv_tiled!(a, z, x, layer::Conv; tile=32, wg=256, smem_budget) -> z

Convolution of `layer::Conv` applied to `x` with fused bias and activation: `a`
receives the pre-activation, `z` the activated output. `a === z` is allowed.
This function launches kernel function [`_conv_tiled_kernel!`](@ref).

Works for any number of spatial dimensions, any stride and any
[`PadMode`](@ref), on every KernelAbstractions backend. See [`forward`](@ref)
for the full description.

All sizes are read from the arrays and passed to the kernel as `Val`s, so a
specialised kernel is compiled once per layer shape on first use.

# Keyword Arguments
* `tile`: maximum number of output channels per work item (default `32`); the
  actual tile is `min(tile, C_out)`.
* `wg`: work items per workgroup (default `256`).
* `smem_budget`: bytes of shared memory allowed for staging `W` (default
  [`_default_smem_budget`](@ref) of the backend of `x`).
"""
function conv_tiled!(a, z, x, layer::Conv; tile::Int = 32, wg::Int = 256,
                     smem_budget::Int = _default_smem_budget(KernelAbstractions.get_backend(x)))
    Nsp = ndims(x) - 2
    Ls = size(x)[1:Nsp]
    Ps = size(a)[1:Nsp]
    Ks = size(layer.W)[1:Nsp]
    C_in = size(x, Nsp + 1)
    C_out = size(layer.W, Nsp + 2)
    N = size(x, Nsp + 2)

    TILE = min(tile, C_out)
    total = prod(Ps) * cld(C_out, TILE) * N

    WS = length(layer.W) * sizeof(eltype(a)) <= smem_budget
    WN = WS ? length(layer.W) : 1

    backend = KernelAbstractions.get_backend(x)
    _conv_tiled_kernel!(backend, wg)(a, z, x, layer.W, layer.b, layer.act_func, layer.pad,
        layer.stride, N,
        Val(Ls), Val(Ps), Val(Ks), Val(C_in), Val(C_out),
        Val(TILE), Val(WS), Val(WN);
        ndrange = cld(total, wg) * wg)
    return z
end

"""
    _conv_tiled_kernel!(backend, wg)(a, z, x, W, b, act, pad, stride, N,
                                     Val(Ls), Val(Ps), Val(Ks), Val(C_in), Val(C_out),
                                     Val(TILE), Val(WS), Val(WN); ndrange)

Device kernel of [`conv_tiled!`](@ref): N-dimensional convolution with fused
bias and activation. Each work item computes `TILE` output channels of one
output site of one sample; see [`forward`](@ref) for the full description.

# Launch layout
1D ndrange, rounded up to a multiple of the workgroup size `wg`. All work items
of a workgroup first copy `W` into shared memory (if `WS`) and synchronise;
work items with a linear index beyond the real work then return. Each
remaining work item recovers its index `(p..., t, n)` (site, channel tile,
sample) from its linear index.

# Layouts
- `x`: `(L..., C_in, N)`
- `W`: `(K..., C_in, C_out)`
- `a`, `z`: `(P..., C_out, N)`

# Type parameters
* `Ls`, `Ps`, `Ks`: input grid, output grid and kernel window.
* `C_in`, `C_out`: number of input and output channels.
* `TILE`: output channels per work item.
* `WS`: whether `W` is staged in shared memory.
* `WN`: length of the shared buffer (`length(W)` if `WS`, otherwise `1`).
"""
@kernel function _conv_tiled_kernel!(a, z, @Const(x), @Const(W), @Const(b), act, pad,
        stride::Int, N::Int,
        ::Val{Ls}, ::Val{Ps}, ::Val{Ks}, ::Val{C_in}, ::Val{C_out},
        ::Val{TILE}, ::Val{WS}, ::Val{WN}) where {Ls,Ps,Ks,C_in,C_out,TILE,WS,WN}

    tid = @index(Local, Linear)     # position inside the workgroup
    WG = prod(@groupsize())         # work items per workgroup

    # stage W in shared memory (one copy per workgroup, filled cooperatively)
    ws = @localmem eltype(a) (WN,)
    if WS
        for i in tid:WG:WN
            @inbounds ws[i] = W[i]
        end
    end
    @synchronize()

    lin = @index(Global, Linear)    # position in the whole launch

    N_t = cld(C_out, TILE)          # number of channel tiles
    plane = prod(Ls)                # x: distance in memory between input channels
    wk = prod(Ks)                   # W: distance in memory between input channels (size of one kernel window)
    NP = prod(Ps)                   # a: distance in memory between output channels
    total = NP * N_t * N            # number of real work items

    if lin <= total
        idx = Tuple(CartesianIndices((Ps..., N_t, N))[lin])   # (p..., t, n)
        t = idx[end-1]                                # channel tile
        n = idx[end]                                  # sample

        x_n = (n - 1) * plane * C_in        # x: offset of sample n
        o0 = (t - 1) * TILE                 # first output channel of tile - 1
        wstride = wk * C_in                 # W: stride between output channels
        wsrc = WS ? ws : W                  # resolved at compile time

        acc = _zero_acc(Val(TILE), eltype(a))   # TILE accumulators in registers
        for k in CartesianIndices(Ks)           # kernel taps
            src = _src_tuple(pad, idx, k, stride, Val(Ls), Val(Ks))
            if all(>(0), src)                         # skip taps outside the input
                xi = x_n + LinearIndices(Ls)[src...]  # x[src..., 1, n]
                wi = LinearIndices(Ks)[k]             # W[k, 1, 1]
                for _ in 1:C_in                       # input channels
                    acc = _acc_fma(acc, @inbounds(x[xi]), wsrc, wi, wstride, o0, Val(C_out))
                    xi += plane
                    wi += wk
                end
            end
        end

        aoff = _site_linear(idx, Val(Ps)) + (n - 1) * NP * C_out   # a[p..., 1, n]
        _acc_store!(a, z, act, b, acc, aoff, NP, o0, Val(C_out))
    end
end

"""
    SMEM_BUDGET

Maximum number of bytes of shared (workgroup-local) memory used for staging
the kernel weights `W` in [`conv_tiled!`](@ref). This memory is asigned with
[`_default_smem_budget`](@ref) function.

Hardware limits per workgroup:
* CUDA:  48 KB of static shared memory (`@localmem`)
* Metal: 32 KB of threadgroup memory

32 KB is the largest value valid on both backends. If `W` does not fit into
this budget, the weights are read directly from global memory.
"""
const SMEM_BUDGET = 32 * 1024

"""
    _default_smem_budget(backend) -> Int

Default shared-memory budget for [`conv_tiled!`](@ref): [`SMEM_BUDGET`](@ref)
on GPU backends, `0` on the CPU (where staging `W` brings no benefit).
"""
_default_smem_budget(::KernelAbstractions.CPU) = 0
_default_smem_budget(::KernelAbstractions.Backend) = SMEM_BUDGET

"""
    _zero_acc(Val(TILE), T) -> NTuple{TILE,T}

Tuple of `TILE` zeros of type `T`, used as per-thread accumulators for `TILE`
output channels. `TILE` is a compile-time constant, so the tuple lives in
registers.
"""
@inline _zero_acc(::Val{TILE}, ::Type{T}) where {TILE,T} = ntuple(_ -> zero(T), Val(TILE))

"""
    _acc_fma(acc, xv, w, wi, wstride, o0, Val(C_out)) -> NTuple{TILE}

Multiplies one input value `xv` with the weights of the `TILE` output channels
`o0+1, ..., o0+TILE` and adds the products to the accumulators:

```math
\\mathrm{acc}_j \\leftarrow \\mathrm{acc}_j + x_v \\, W[k, c, o_0 + j], \\qquad j = 1, \\dots, \\mathrm{TILE}
```

# Arguments
* `acc`: accumulators, `NTuple{TILE}`.
* `xv`: input value `x[src, c, n]`.
* `w`: weights, either shared memory or the global array `W`.
* `wi`: linear index of `W[k, c, 1]`.
* `wstride`: distance in `W` between two output channels, `prod(K) * C_in`.
* `o0`: first output channel of the tile minus one.
* `Val(C_out)`: total number of output channels. Entries with `o0 + j > C_out`
  (ragged last tile) are left unchanged.
"""
@inline function _acc_fma(acc::NTuple{TILE,T}, xv::T, w, wi::Int, wstride::Int,
                          o0::Int, ::Val{C_out}) where {TILE,T,C_out}
    ntuple(Val(TILE)) do j
        o = o0 + j
        o <= C_out ? muladd(xv, @inbounds(w[wi + (o - 1) * wstride]), acc[j]) : acc[j]
    end
end

"""
    _acc_store!(a, z, act, b, acc, aoff, ostride, o0, Val(C_out))

Adds the bias, applies the activation and writes the results of one output
site for the `TILE` output channels of the tile:

```math
a_{p,o,n} = \\mathrm{acc}_{o - o_0} + b_o, \\qquad z_{p,o,n} = \\mathrm{act}(a_{p,o,n})
```

`a === z` is allowed (in-place).

# Arguments
* `aoff`: linear index of `a[p..., 1, n]`.
* `ostride`: distance in `a` between two output channels, `prod(P)`.
* `o0`, `Val(C_out)`: as in [`_acc_fma`](@ref).
"""
@inline function _acc_store!(a, z, act, b, acc::NTuple{TILE,T}, aoff::Int, ostride::Int,
                             o0::Int, ::Val{C_out}) where {TILE,T,C_out}
    ntuple(Val(TILE)) do j
        o = o0 + j
        if o <= C_out
            v = acc[j] + @inbounds(b[o])
            i = aoff + (o - 1) * ostride
            @inbounds a[i] = v
            @inbounds z[i] = act(v)
        end
        nothing
    end
    nothing
end

"""
    _src_tuple(pad, idx, k, stride, Val(Ls), Val(Ks)) -> NTuple{Nsp,Int}

Input source index of kernel tap `k` for output site `p = idx[1:Nsp]`, per
spatial dimension `d`:

```math
s_d = \\mathrm{src}\\big(\\mathrm{pad},\\; (p_d - 1)\\,\\mathrm{stride} + k_d - \\mathrm{padleft}_d,\\; L_d\\big)
```

An entry `0` marks a tap outside the input (e.g. `Zeros` padding) that is
skipped. Any [`PadMode`](@ref) works for which `_src` and `_padleft` are
defined.
"""
@inline function _src_tuple(pad, idx, k, stride, ::Val{Ls}, ::Val{Ks}) where {Ls,Ks}
    ntuple(d -> _src(pad, (idx[d] - 1) * stride + k[d] - _padleft(pad, Ks[d]), Ls[d]),
           Val(length(Ls)))
end

"""
    _site_linear(idx, Val(Ps)) -> Int

Linear index of the output site `p = idx[1:Nsp]` in the output grid `Ps`.
"""
@inline _site_linear(idx, ::Val{Ps}) where Ps =
    LinearIndices(Ps)[ntuple(d -> idx[d], Val(length(Ps)))...]
