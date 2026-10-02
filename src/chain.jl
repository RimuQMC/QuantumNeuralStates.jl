# using LinearAlgebra
# using Statistics

"""
    _input_shape(l, enc::InputEncoding, batch) -> Tuple

Shape of the `Chain` input buffer for first layer `l` and encoding `enc`:
`(M*C, batch)` for `Dense`, `(size..., C, batch)` for `Conv`. Errors if the layer
doesn't match the encoding's sites, channels, or spatial dimensions.
"""
function _input_shape(l::Dense, enc::InputEncoding, batch)
    nin  = size(l.W, 2)
    need = nsites(enc) * nchannels(enc)
    nin == need ||
        error("Dense first layer expects $nin inputs but $(enc) provides $need " *
              "(sites × channels). Use Dense($need=>…)")
    return (nin, batch)
end
function _input_shape(l::Conv, enc::InputEncoding, batch)
    Nsp = ndims(l.W) - 2
    sz  = enc.size
    length(sz) == Nsp ||
        error("Conv with $Nsp spatial dims needs a $Nsp-D encoding, got size $sz")
    Cin = size(l.W, Nsp + 1)
    Cin == nchannels(enc) ||
        error("first layer has $Cin input channels but $(enc) provides " *
              "$(nchannels(enc)). Use Conv(…, nchannels(enc)=>…)")
    _check_input(l.pad, sz, size(l.W)[1:Nsp])
    return (sz..., Cin, batch)
end


"""
    Chain(enc::InputEncoding, layers...; device=identity, batch=1)

A Neural Network (NN) container that chains layers into a single forward-pass
model. This design is inspired by `Flux.jl` notation.

# Arguments
* `enc`: input encoding; defines the input grid and its channels.
         See [`OccupationEncoding`](@ref), [`MomentumEncoding`](@ref).
* `layers...`: layers to chain. The first layer must fit the encoding:
         `Conv` input channels = `nchannels(enc)`, or
         `Dense` input width   = `nsites(enc) * nchannels(enc)`.

# Keyword Arguments
* `device`: function determining whether the NN lives on CPU or GPU.
* `batch`: batch size.

# Example
```julia
enc   = MomentumEncoding((4,4), H; device = device)
model = Chain(enc,
              Conv((3,3), nchannels(enc)=>16, relu; batch=batch, device=device, pad=Zeros()),
              Pool(:mean; device=device),
              Dense(16=>1, identity; batch=batch, device=device);
              batch=batch, device=device)
```

## Fields
* `x` : device input buffer, shape `(size..., C, batch)` (Conv) or `(M*C, batch)` (Dense).
* `xe`: the same memory viewed as `xe[m, c, b]` (site, channel, sample), used by
        [`encode!`](@ref).

## Notes
Layers are allocation-free at runtime. `z_last` is useful when two forward passes
are needed simultaneously — save the first output into `z_last` before running the second.
"""
mutable struct Chain{L<:Tuple,X<:AbstractArray,XE<:AbstractArray,U<:AbstractArray,
                     F<:Function,E<:InputEncoding}
    layers::L
    x::X
    xe::XE
    z_last::U

    device::F
    batch::Int
    enc::E
end
function Chain(enc::InputEncoding, layers...; device::Function = identity, batch::Int = 1)
    l  = first(layers)
    x  = fill!(similar(l.W, _input_shape(l, enc, batch)...), 0f0)
    xe = reshape(x, nsites(enc), nchannels(enc), batch)       # same memory, [m, c, b]
    _check_device(enc, x)

    z_out  = _forward_layers(layers, x)       # forward pass for all layers initialisation
    z_last = similar(z_out)

    L, X, XE, U, F, E = typeof(layers), typeof(x), typeof(xe), typeof(z_last),
                        typeof(device), typeof(enc)
    return Chain{L,X,XE,U,F,E}(layers, x, xe, z_last, device, batch, enc)
end

Base.show(io::IO, c::Chain) =
    print(io, "Chain(", length(c.layers), " layers, ", _group(n_params(c)), " params)")

function Base.show(io::IO, ::MIME"text/plain", c::Chain)
    lines = [sprint(show, l) * "," for l in c.layers]
    ptxt  = [n_params(l) == 0 ? "" : _group(n_params(l)) * " params" for l in c.layers]
    wl    = maximum(length, lines)

    println(io, "Chain(")
    for i in eachindex(c.layers)
        print(io, "  ", rpad(lines[i], wl))
        isempty(ptxt[i]) || print(io, "  # ", ptxt[i])
        println(io)
    end
    println(io, ")")

    println(io, "  encoding: ", c.enc)
    println(io, "  input: ", size(c.x), ", batch: ", c.batch, ", device: ",
                _devname(c.device), " (", eltype(c.x), ")")
    println(io, "  parameters: ", _group(n_params(c)))
    print(io,   "  memory estimate: ", _fmt_bytes(memory_estimate(c)))
end

"""
    MultiForwardBuffer(model, addr, buffer_size)

Allows forward passes through the Neural Network with an arbitrary (usually
larger) batch size. Especially useful on GPU: all off-diagonal evaluations of
VMC and energy calculations can be done in one big forward pass.

It mimics [`Chain`](@ref), with its own input buffers (`x`, `xe`, `x_cpu`) of
batch size `buffer_size` and the same input encoding as `model`.

# Arguments
* `model`: `Chain` model of the Neural Network that is mimicked.
* `addr`: an address, used to type and fill the address buffer.
* `buffer_size`: batch size of the multi-forward passes.

# Example
Recommended way to choose `buffer_size`:
julia> addr = OccupationNumberFS{4}()
julia> H = FroehlichPolaron(addr)
julia> col = H*addr
julia> num_offdiagonals(col)
8
A good value is `num_offdiagonals(col) + 10%`, so that all off-diagonals need
only one forward pass.
"""
mutable struct MultiForwardBuffer{L,A,X<:AbstractArray,XE<:AbstractArray,
                                  CX<:AbstractArray,CZ<:AbstractArray}
    layers::L
    x::X
    xe::XE                 # x viewed as [m, c, b]
    buffer_size::Int

    # buffer for raw input
    addrs::A

    # CPU buffers
    x_cpu::CX
    z_cpu::CZ
end

function MultiForwardBuffer(model, addr, buffer_size)
    layers = Tuple(
        MultiForwardLayer(
            similar(l.z, size(l.z)[1:end-1]..., buffer_size),
            l isa ParametricLayer && l.layer_norm !== nothing ?
                LayerNorm_multiforward(size(l.z, ndims(l.z)-1), buffer_size, model.device) : nothing
        ) for l in model.layers
    )

    enc   = model.enc
    x     = fill!(similar(model.x, size(model.x)[1:end-1]..., buffer_size), 0f0)
    xe    = reshape(x, nsites(enc), nchannels(enc), buffer_size)
    x_cpu = zeros(Float32, size(model.x)[1:end-1]..., buffer_size)
    z_cpu = Matrix{Float64}(undef, size(last(model.layers).z, 1), buffer_size)

    addrs = fill(addr, buffer_size)

    L=typeof(layers); A=typeof(addrs); X=typeof(x); XE=typeof(xe)
    CX=typeof(x_cpu); CZ=typeof(z_cpu)
    return MultiForwardBuffer{L,A,X,XE,CX,CZ}(layers, x, xe, buffer_size, addrs, x_cpu, z_cpu)
end

"""
    prepare_chain_input!(chain, x)
    prepare_chain_input!(chain, x, multi_forward_buffer)

Loads the input `x` (occupations already written by [`prepare_input!`](@ref))
into the device buffer of `chain`, then builds the remaining input channels on
the device with [`encode!`](@ref).

If dispatched with [`MultiForwardBuffer`](@ref), it loads into that buffer
instead, for bigger batched passes.
"""
function prepare_chain_input!(chain::Chain, x::AbstractArray)
    if x !== chain.x                                    # external input: copy it in
        size(x) == size(chain.x) ||
            error("input has size $(size(x)), model expects $(size(chain.x))")
        copyto!(chain.x, x)
    end
    encode!(chain.xe, chain.enc)
    return chain.x
end

function prepare_chain_input!(chain::Chain, x::AbstractArray, multi_forward_buffer)
    if x !== multi_forward_buffer.x                     # external input: copy it in
        size(x) == size(multi_forward_buffer.x) ||
            error("input has size $(size(x)), buffer expects $(size(multi_forward_buffer.x))")
        copyto!(multi_forward_buffer.x, x)
    end
    encode!(multi_forward_buffer.xe, chain.enc)
    return multi_forward_buffer.x
end
# function prepare_chain_input!(chain::Chain, x::AbstractArray)
#     # size(x) == size(chain.x) ||
#     #     error("input has size $(size(x)), model expects $(size(chain.x))")
#     # copyto!(chain.x, x)
#     encode!(chain.xe, chain.enc)
#     return chain.x
# end
# function prepare_chain_input!(chain::Chain, x::AbstractArray, multi_forward_buffer)
#     # size(x) == size(multi_forward_buffer.x) ||
#     #     error("input has size $(size(x)), buffer expects $(size(multi_forward_buffer.x))")
#     # copyto!(multi_forward_buffer.x, x)
#     encode!(multi_forward_buffer.xe, chain.enc)
#     return multi_forward_buffer.x
# end

"""
    forward(chain, x)
    forward(chain, x, multi_forward_buffer)

Forward pass through all layers of the Neural Network. The input is loaded into 
model by [`prepare_chain_input!`](@ref) and returns final Neural Network output. 

If dispatched with [`MultiForwardBuffer`](@ref) then input and output lives within
this buffer for bigger batch pass.
"""
function forward(chain::Chain, x::AbstractArray)
    prepare_chain_input!(chain, x)
    return _forward_layers(chain.layers, chain.x)
end
@inline _forward_layers(::Tuple{}, x) = x
@inline _forward_layers(layers::Tuple, x) =
    _forward_layers(Base.tail(layers), forward(first(layers), x))

function forward(chain::Chain, x::AbstractArray, multi_forward_buffer)
    prepare_chain_input!(chain, x, multi_forward_buffer)
    return _forward_layers(chain.layers, multi_forward_buffer.layers, multi_forward_buffer.x)
end
@inline _forward_layers(::Tuple{}, ::Tuple{}, x) = x
@inline _forward_layers(layers::Tuple, multi::Tuple, x) =
    _forward_layers(Base.tail(layers), Base.tail(multi),
                    forward(first(layers), x, first(multi)))

"""
    (model)(x)

Wrapper over [`forward`](@ref) function for maybe simplier calling of Neural Network
forward pass.
"""
function (model::Chain)(x)
    return forward(model, x)
end
function (model::Chain)(x, multi_forward_buffer)
    return forward(model, x, multi_forward_buffer)
end

"""
    n_params(chain::Chain)

This function calls `n_params(layers...)` on each layer and sum the result. Therefore it 
returns total number of learnable parameters inside neural network.
"""
n_params(chain::Chain) = sum(n_params, chain.layers) # total number in model::Chain

