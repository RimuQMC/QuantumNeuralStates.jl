# using LinearAlgebra
# using Statistics

# shape of the chain input, decided by the first layer
_input_shape(l::Dense, input_size, batch) = (size(l.W, 2), batch)

function _input_shape(l::Conv, input_size, batch)
    Nsp = ndims(l.W) - 2
    length(input_size) == Nsp ||
        error("Conv with $Nsp spatial dims needs input_size of length $Nsp, got $input_size")
    _check_input(l.pad, input_size, size(l.W)[1:Nsp])                                  # CHANGED
    return (input_size..., size(l.W, Nsp + 1), batch)
end

_check_input(::NoPad, input_size, K) =
    all(input_size .>= K) || error("input_size $input_size is smaller than kernel $K")
_check_input(::PadMode, input_size, K) = nothing


"""
    Chain(layers...; device=identity, batch=1, input_size=())

A Neural Network (NN) container that chains layers into a single forward-pass model. This
design is inspired by `Flux.jl` notation (general ML julia library).

# Arguments
* `layers...`: represent Tuple of layers from which the NN should be chained.
               For different layer types see also: [`Dense`](@ref).

# Keyword Arguments
* `device`: function which determine if the NN lives and computes on CPU or GPU
* `batch`: determine what is the batch size.
* `input_size`: Tuple specifying input dimensions. Needed for layers independent
    of input size (as [`Conv`](@ref)) for buffer initialisations.

# Example
```julia
model = Chain(layer1, layer2, layer3)
model_batched = Chain(layer1, layer2, layer3; batch=32)
```

## Notes
Layers are allocation-free at runtime. `z_last` is useful when two forward passes
are needed simultaneously — save the first output into `z_last` before running the second.

"""
mutable struct Chain{L<:Tuple,X<:AbstractArray,U<:AbstractArray,F<:Function}
    layers::L
    x::X
    z_last::U

    device::F
    batch::Int
end
function Chain(layers...; device::Function = identity, batch::Int = 1,
               input_size::Tuple = ())
    l = first(layers)
    x = fill!(similar(l.W, _input_shape(l, input_size, batch)...), 0f0)

    z_out = _forward_layers(layers, x) # forward pass for all layers initialisation

    z_last = similar(z_out)

    L, X, U, F = typeof(layers), typeof(x), typeof(z_last), typeof(device)
    return Chain{L,X,U,F}(layers, x, z_last, device, batch)
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

    println(io, "  input: ", size(c.x), ", batch: ", c.batch, ", device: ",
                _devname(c.device), " (", eltype(c.x), ")")
    println(io, "  parameters: ", _group(n_params(c)))
    print(io,   "  memory estimate: ", _fmt_bytes(memory_estimate(c)))
end

"""
    MultiForwardBuffer

This struct allows to do forward pass through Neural Network in arbitrary large batch size.
It is especially useful for GPU use as it allows to put all forward passes from VMC and
Energy calculations generated from all offdiagonal elements.

It mimics [`Chain`](@ref) struct just with custom batch size variables meant for forward pass.

# Arguments

* `model`: Chain model of Neural Network that would be mimic 
* `addrs`: buffer that can hold address input in right length
* `buffer_size`: New custom size of Multi batch passes

# Example

If my batch is 1024, than during VMC all offdiagonals needs to be evaluated with Neural 
Network. Therefore number of offdiagonals effects how many times the forward 
pass would be called. Advantage of this `MultiForwardBuffer` is that all those offdiagonal 
passes can be done in one big forward pass -> more GPU friedly.

Recommended way to evaluate `buffer_size`:
```
julia> addr = OccupationNumberFS{4}()
julia> H = FroehlichPolaron(addr)
julia> col = H*addr
julia> num_offdiagonals(col)
8
```
So optional value would be `num_offdiagonals(col) + 10%`. This way only 1 forward pass would
be needed for all offdiagonals evaluation.
"""
mutable struct MultiForwardBuffer{L,A,X<:AbstractArray,CX<:AbstractArray,CZ<:AbstractArray}
    layers::L
    x::X
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

    x = similar(model.x, size(model.x)[1:end-1]..., buffer_size)
    x_cpu = zeros(Float32, size(model.x)[1:end-1]..., buffer_size)
    z_cpu = Matrix{Float64}(undef, size(last(model.layers).z, 1), buffer_size)

    addrs = fill(addr, buffer_size)

    L=typeof(layers); A=typeof(addrs); X=typeof(x); CX=typeof(x_cpu); CZ=typeof(z_cpu)
    return MultiForwardBuffer{L,A,X,CX,CZ}(layers, x, buffer_size, addrs, x_cpu, z_cpu)
end

"""
    prepare_chain_input!(chain, x)
    prepare_chain_input!(chain, x, multi_forward_buffer)

Loads input `x::AbstractArray` into Neural Network `chain`. This 
ensures that the input lives on same device (GPU/CPU) as the Neural Network.

If dispatched with [`MultiForwardBuffer`](@ref) then it loads input into
this buffer for bigger batched passes.
"""
function prepare_chain_input!(chain::Chain, x::AbstractArray)
    copyto!(chain.x, x)
end
function prepare_chain_input!(chain::Chain, x::AbstractArray, multi_forward_buffer)
    copyto!(multi_forward_buffer.x, x)
end

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

