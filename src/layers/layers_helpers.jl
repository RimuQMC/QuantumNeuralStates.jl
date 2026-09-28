
"""
    AbstractLayer

Abstract type for different types of neural network layer architectures.
"""
abstract type AbstractLayer end

"""
    ParametricLayer

This type of [`AbstractLayer`](@ref) is meant for neural network layer
which contain learnable parameters, for example [`Dense`](@ref) or 
[`Conv`](@ref).
"""
abstract type ParametricLayer <: AbstractLayer end # has W, b, (layer_norm)

"""
    FreeLayer

This type of [`AbstractLayer`](@ref) is meant for neural network layer
which DOES NOT contain learnable parameters, for example [`Pool`](@ref).
Such layers are usually meant only for some dimension reduction or special
operations rather than for neural network training.
"""
abstract type FreeLayer <: AbstractLayer end # no parameters

"""
    hasparams(::ParametricLayer) = true
    hasparams(::FreeLayer) = false

Helper functions used in backpropagation to distinguish layers with trainable weights 
(ignore [`FreeLayer`](@ref), collect [`ParametricLayer`](@ref)).
"""
hasparams(::ParametricLayer) = true
hasparams(::FreeLayer) = false

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
    apply_act!(layer::ParametricLayer, a, z)

This function applies activation function on [`ParametricLayer`](@ref). 

This function is dispatched on [`Dense`](@ref) if the layer has multiple activations.
It is mostly used in final neural network `Dense` output layers, to allow many output 
wave-function representations. See also [`AnsatzType`](@ref).
"""
function apply_act!(layer::ParametricLayer, a, z)
    # single activations in any layer
    z .= layer.act_func.(a)
    return z
end

"""
    apply_act_deriv!(δz, layer::ParametricLayer, a)

This function applies derivative of activation function on [`ParametricLayer`](@ref). 

This function is dispatched on [`Dense`](@ref) if the layer has multiple activations.
It is mostly used in final neural network `Dense` output layers, to allow many output 
wave-function representations. See also [`AnsatzType`](@ref).
"""
function apply_act_deriv!(δz, layer::ParametricLayer, a)
    δz .= layer.act_deriv.(a)
    return δz
end

"""
    n_params(layer) -> Int

Return the total number of learnable parameters of `layer`
(weights, biases, and layer-normalization parameters, if present).
"""
n_params(layer::ParametricLayer) =
    length(layer.W) + length(layer.b) + n_params(layer.layer_norm)

n_params(::FreeLayer) = 0   # no learnable parameters
n_params(::Nothing)   = 0

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
            

