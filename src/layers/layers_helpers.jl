
"""
    _init_std(T, in, out, act)

This function do weights initialisation in neural network layers. There are
two types of initialisation depending on chosen activation function in layers.
It uses `He` initialisation for `relu / gelu` and `Glorot` initialisation for 
the rest of activation functions.

## Note
If layer has multiple activation functions the `Glorot` is chosen for inirialisation.
"""
_init_std(T, in, out, act::Function) =
    (act === relu || act === gelu) ? T(sqrt(2.0/in)) : T(sqrt(2.0/(in+out)))
_init_std(T, in, out, ::Tuple) = T(sqrt(2.0/(in+out))) 

"""
    _lookup_deriv(act)

This function is doing dictionary look up for corresponding derivative functions.
It uses `ACT_DERIV` dictionary which connects `act => act_deriv`.
"""
_lookup_deriv(act::Function) = get(ACT_DERIV, act) do
    error("No derivative registered for $act. Use only functions defined in activations.jl or define yours there.")
end

"""
    _dense_alloc(in, out, act; batch, device, Layer_Norm)

This function define properly sized buffers used in [`Dense`](@ref) layer. This is
done in batched approach and it is also `device` agnostics (CPU/GPU). Also allows
defining [`LayerNorm`](@ref) for current layer.
"""
function _dense_alloc(in::Int, out::Int, act;
                       batch::Int, device::Function, Layer_Norm)
    T = Float32
    std = _init_std(T, in, out, act)     # see note below re: He/Glorot dispatch
    W  = device(randn(T, out, in) .* std)
    b  = device(zeros(T, out))

    a = device(zeros(T, out, batch))
    z = device(zeros(T, out, batch))

    layer_norm = Layer_Norm === false ? nothing : LayerNorm(out, batch, device)
    return T, W, b, a, z, layer_norm
end

"""
    MultiForwardLayer

This struct mimic each [`Chain`](@ref) layer output variables made for customized batch size
forward passes. See also [`MultiForwardBuffer`](@ref).

# Arguments

* `a`: is buffer for pre-actiovation and post-activation variable in each Chain layer.
* `layer_norm`: is buffer for layer normalisation struct with new batch size.
    
"""
mutable struct MultiForwardLayer{A<:AbstractArray,L}
    a::A
    layer_norm::Union{L, Nothing}

    function MultiForwardLayer(a::A, ln) where {A<:AbstractArray}
        L = typeof(ln)
        return new{A,L}(a, ln)
    end
end
            

