# using LinearAlgebra
# using Statistics

"""
    Chain(layers...; device=identity, batch=1)

A Neural Network (NN) container that chains layers into a single forward-pass model. This
design is inspired by `Flux.jl` notation (general ML julia library).

# Arguments

* `layers...`: represent Tuple of layers from which the NN should be chained.
               For different layer types see also: [`Dense`](@ref).

# Keyword Arguments

* `device`: function which determine if the NN lives and computes on CPU or GPU
* `batch`: determine what is the batch size.

# Example
```julia
model = Chain(layer1, layer2, layer3)
model_batched = Chain(layer1, layer2, layer3; batch=32)
```

## Notes
Layers are allocation-free at runtime. `z_last` is useful when two forward passes
are needed simultaneously — save the first output into `z_last` before running the second.

"""
mutable struct Chain{L <: Tuple, X <: AbstractArray, U <: AbstractArray,F<:Function}
    layers  ::L
    x       ::X
    z_last  ::U

    device::F
    batch::Int
end
function Chain(layers...; device::Function = identity, batch::Int = 1)
    # Define input
    l      = first(layers)
    in_dim = size(l.W, 2)
    x = similar(l.W, in_dim, batch)

    # Define last computed output of NN
    lout    = last(layers)
    out_dim = size(lout.z, 1)
    z_last = similar(lout.z, out_dim, batch)

    L, X, U, F = typeof(layers), typeof(x), typeof(z_last), typeof(device)
    return Chain{L, X, U, F}(layers, x, z_last, device, batch)
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
                similar(l.a, size(l.a, 1), buffer_size),
                l.layer_norm !== nothing ? LayerNorm_multiforward(size(l.a, 1), buffer_size, model.device) : nothing
        ) for l in model.layers
    )

    x = similar(model.x, size(model.x, 1), buffer_size)

    x_cpu = zeros(Float32, size(model.x, 1), buffer_size)
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
    input = chain.x

    for (i, layer) in enumerate(chain.layers)
        input = forward(layer, input)
    end

    return input
end
function forward(chain::Chain, x::AbstractArray, multi_forward_buffer)
    prepare_chain_input!(chain, x, multi_forward_buffer)
    input = multi_forward_buffer.x

    for (layerNN, layerMulti) in zip(chain.layers, multi_forward_buffer.layers)
        input = forward(layerNN, input, layerMulti)
    end

    return input
end

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

