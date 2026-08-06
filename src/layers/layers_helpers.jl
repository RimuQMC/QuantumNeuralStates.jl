
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
    (act === relu || act === gelu) ? T(sqrt(2.0/in)) : T(sqrt(2.0/(in+out))) # init-std selection also dispatched rather than `if act === relu || act === gelu`
_init_std(T, in, out, ::Tuple) = T(sqrt(2.0/(in+out))) # for a tuple of acts, just use Glorot (safe default) unless you want per-head logic

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

    if batch == 1
        a = device(zeros(T, out))
        z = device(zeros(T, out))
    else
        a = device(zeros(T, out, batch))
        z = device(zeros(T, out, batch))
    end

    layer_norm = Layer_Norm === false ? nothing : LayerNorm(out, batch, device)
    return T, W, b, a, z, layer_norm
end
