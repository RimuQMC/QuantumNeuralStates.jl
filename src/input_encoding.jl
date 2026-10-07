# using KernelAbstractions

"""
    InputEncoding

Abstract supertype of all input encodings. An encoding describes how a Fock
configuration is turned into the network input: the grid size of the input and
which channels (feature maps) it has.

Every encoding provides:
* `size`: spatial grid, e.g. `(12,)` or `(4,4)`
* [`nsites`](@ref): number of sites M = prod(size)
* [`nchannels`](@ref): number of input channels C
* [`occupation_channel`](@ref): channel into which the raw occupations are written
* [`encode!`](@ref): in-place construction of the derived channels on the device

Internally all encodings work on the input viewed as `x[m, c, b]`
(site, channel, sample), which is the same memory as the `Chain` input
`(size..., C, batch)` for Conv, or `(M*C, batch)` for Dense.
"""
abstract type InputEncoding end

"""
    OccupationEncoding(size, H: device=identity)

Plain occupation-number input. One channel: `x[m, 1, b] = n_m` of sample `b`.

`size` is the spatial grid of the input, e.g. `(12,)` for a 1D model, `(4,4)` for a 2D model. 
It must contain as many sites as `H` has modes.

# Note
`device` variable do nothing (it is just here for same notation).

# Example
```julia
enc   = OccupationEncoding((12,), H)
model = Chain(enc, Dense(12=>100, relu; batch=batch, device=device), …;
              batch=batch, device=device)
```
"""
struct OccupationEncoding{D} <: InputEncoding
    size::NTuple{D,Int}
    OccupationEncoding{D}(size::NTuple{D,Int}) where {D} = new{D}(size)
end
function OccupationEncoding(sz::NTuple{D,Int}, H; device::Function=identity) where {D}
    M = num_modes(starting_address(H))
    prod(sz) == M ||
        error("input size $sz has $(prod(sz)) sites but H has $M modes")
    return OccupationEncoding{D}(sz)
end

Base.show(io::IO, e::OccupationEncoding) =
    print(io, "OccupationEncoding(", e.size, ", C = 1)")

"""
    MomentumEncoding(size, H; device = identity, kscale = nothing, occupation = true)

Occupation input combined with the momentum of each mode, for momentum-space
Hamiltonians (e.g. `FroehlichPolaron`). With `D` spatial dimensions:

`occupation = true` (C = 1 + D channels)
* `x[m, 1,   b] = n_m`
* `x[m, 1+d, b] = n_m · k_{m,d}`      for d = 1 … D

`occupation = false` (C = D channels)
* `x[m, d, b]   = n_m · k_{m,d}`      for d = 1 … D
  (note: occupation of a mode with k = 0 is then invisible to the network)

Momenta can divided by `kscale`. By default this scaling is set to 1. `device` must 
be the same device function as used for the `Chain`.

The `n` (occupation) channel is filled by [`prepare_input!`](@ref) (including `input_scale_func`
and normalisation); the `n·k` (momenta) channels are then built on the device by
[`encode!`](@ref) from that n channel.

# Example
```julia
enc   = MomentumEncoding((4,4), H; device = device)          # C = 3: n, n·kx, n·ky
model = Chain(enc, Conv((3,3), nchannels(enc)=>16, relu; batch=batch, device=device, pad=Zeros()), …;
              batch=batch, device=device)
```
"""
struct MomentumEncoding{D,A,WN} <: InputEncoding     # WN::Bool: has plain n channel
    size::NTuple{D,Int}
    K::A           # (M, D) scaled momenta on device: K[m, d] = k_d of mode m
    MomentumEncoding{D,A,WN}(size::NTuple{D,Int}, K::A) where {D,A,WN} =
        new{D,A,WN}(size, K)
end
function MomentumEncoding(sz::NTuple{D,Int}, H; device::Function = identity,
                          kscale = nothing, occupation::Bool = true) where {D}
    hasproperty(H, :ks) ||
        error("MomentumEncoding needs a momentum-space Hamiltonian with `ks`, got $(typeof(H))")
    ks = H.ks
    M  = length(ks)
    length(first(ks)) == D ||
        error("input size $sz is $D-D but H has $(length(first(ks)))-D momenta")
    prod(sz) == M ||
        error("input size $sz has $(prod(sz)) sites but H has $M modes")

    s = kscale === nothing ? 1f0 : Float32(kscale)    # default: raw H.ks values

    # K[m, d]: mode m is the same index as in onr(addr) and as site m of the grid
    K = Matrix{Float32}(undef, M, D)
    for m in 1:M, d in 1:D
        K[m, d] = Float32(ks[m][d]) / s
    end
    Kd = device(K)
    return MomentumEncoding{D,typeof(Kd),occupation}(sz, Kd)
end

Base.show(io::IO, e::MomentumEncoding{D,A,WN}) where {D,A,WN} =
    print(io, "MomentumEncoding(", e.size, ", C = ", nchannels(e),
          WN ? ": n, n·k)" : ": n·k)")

"""
    NoEncoding(size; channels = 1)
    NoEncoding(n::Int)

No physics encoding: the network takes a plain array of the given shape, so the
`Chain` behaves as an ordinary neural network. No Hamiltonian is needed.

* Conv input : `size` is the spatial grid, `channels` the number of input channels,
               the model input has shape `(size..., channels, batch)`.
* Dense input: `NoEncoding(n)` gives a flat input of width `n`, shape `(n, batch)`.

The array passed to the model is copied into the input buffer unchanged.
`NoEncoding` cannot be used with [`NeuralAnsatz`](@ref), which needs an encoding
that turns Fock states into network inputs.

# Example
```julia
model = Chain(NoEncoding(10),
              Dense(10=>32, relu; batch=batch),
              Dense(32=>1, identity; batch=batch); batch=batch)
y = model(rand(Float32, 10, batch))

model = Chain(NoEncoding((8,8); channels=3),
              Conv((3,3), 3=>16, relu; batch=batch, pad=Zeros()),
              Pool(:sum),
              Dense(16=>1, identity; batch=batch); batch=batch)
y = model(rand(Float32, 8, 8, 3, batch))
```
"""
struct NoEncoding{D} <: InputEncoding
    size::NTuple{D,Int}
    channels::Int
    NoEncoding{D}(size::NTuple{D,Int}, channels::Int) where {D} = new{D}(size, channels)
end
function NoEncoding(sz::NTuple{D,Int}; channels::Int = 1) where {D}
    all(>(0), sz) || error("NoEncoding size must be positive, got $sz")
    channels >= 1 || error("NoEncoding needs at least one channel, got $channels")
    return NoEncoding{D}(sz, channels)
end
NoEncoding(n::Int) = NoEncoding((n,))

Base.show(io::IO, e::NoEncoding) =
    print(io, "NoEncoding(", e.size, ", C = ", e.channels, ")")

"""
    _signature(e::InputEncoding) -> String

Architecture-signature string for an input encoding: encoding type, input grid
size, and number of channels (plus the occupation flag for `MomentumEncoding`).
Used by [`chain_signature`](@ref).
"""
_signature(e::OccupationEncoding) =
    "OccupationEncoding($(e.size),C=1)"

_signature(e::MomentumEncoding{D,A,WN}) where {D,A,WN} =
    "MomentumEncoding($(e.size),C=$(nchannels(e)),occupation=$(WN))"

_signature(e::NoEncoding) =
    "NoEncoding($(e.size),C=$(e.channels))"

"""
    nsites(enc) -> Int

Number of sites (modes) of the input grid, `prod(enc.size)`.
"""
nsites(e::InputEncoding) = prod(e.size)

"""
    nchannels(enc) -> Int

Number of input channels the encoding produces. Use it for the first layer,
e.g. `Conv((3,3), nchannels(enc)=>16, …)`.
"""
nchannels(::OccupationEncoding) = 1
nchannels(::MomentumEncoding{D,A,WN}) where {D,A,WN} = WN ? D + 1 : D
nchannels(e::NoEncoding) = e.channels

"""
    occupation_channel(enc) -> Int

Channel into which [`prepare_input!`](@ref) writes the (scaled) occupations.
For `MomentumEncoding(…; occupation=false)` this is the last channel, which
[`encode!`](@ref) afterwards turns into n·k_D.
"""
occupation_channel(::OccupationEncoding) = 1
occupation_channel(::MomentumEncoding{D,A,WN}) where {D,A,WN} = WN ? 1 : D
occupation_channel(::NoEncoding) =
    error("NoEncoding has no occupation channel; it is not tied to a Hamiltonian")

"""
    _check_device(e, x)

Check if arrays (now used for momentum arrays `e.K`) live on the same device as the 
chain input `x`.
"""
_check_device(::InputEncoding, x) = nothing
_check_device(e::MomentumEncoding, x) =
    KernelAbstractions.get_backend(e.K) == KernelAbstractions.get_backend(x) ||
        error("MomentumEncoding and Chain live on different devices; " *
              "pass the same `device` to both")

"""
    encode!(xe, enc) -> xe

Build the derived input channels in place on the device. `xe` is the chain
input viewed as `xe[m, c, b]` (site, channel, sample). It expects the
occupations to be already present in `xe[:, occupation_channel(enc), :]`.

* `OccupationEncoding`: nothing to do.
* `MomentumEncoding`  : writes `xe[m, off+d, b] = xe[m, nch, b] * K[m, d]`.

## Notes
- `x`: input of shape `(M, C, B)`, indexed as `x[site, channel, sample]`
- `K`: `(M, D)` momentum table, `K[m, d]` = component `d` of mode `m`
- `nch`: channel holding the occupation `n`
- `off`: output channel offset (`n·k_d` goes to channel `off + d`)
"""
encode!(xe, ::InputEncoding) = xe

function encode!(xe, e::MomentumEncoding{D,A,WN}) where {D,A,WN}
    M, _, B = size(xe)
    nch     = occupation_channel(e)      # where n is
    off     = WN ? 1 : 0                 # n·k_d goes to channel off + d
    backend = KernelAbstractions.get_backend(xe)
    _momentum_encode_kernel!(backend)(xe, e.K, nch, off; ndrange = (M, B))
    # KernelAbstractions.synchronize(backend)
    return xe
end

@kernel function _momentum_encode_kernel!(x, @Const(K), nch::Int, off::Int)
    m, b = @index(Global, NTuple)        # this thread's site m and sample b
    D    = size(K, 2)                    # number of momentum components (2 for 2D)

    v = x[m, nch, b]                     # occupation n of site m in sample b
    for d in 1:D                         # d = 1 → k_x, d = 2 → k_y, …
        x[m, off + d, b] = v * K[m, d]   # write n · k_d of site m into its channel
    end
end

