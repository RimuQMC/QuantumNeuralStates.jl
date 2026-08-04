
"""
    DenseBuffer(layer::Dense)

Pre-allocated buffer for backpropagation and Jacobian calculations through a `Dense` layer.
"""
struct DenseBuffer{DZ <: AbstractArray, D <: AbstractArray}
    δz::DZ # new sending output (out,) single | (out, N) batch
    δ ::D  # incoming previous output (in,)  single | (in,  N) batch
end
function DenseBuffer(layer::Dense)
    out_dim, in_dim = size(layer.W)
    batch = ndims(layer.a) == 1 ? 1 : size(layer.a, 2)
    δz = similar(layer.a)
    δ  = batch == 1 ? similar(layer.b, in_dim) : similar(layer.W, in_dim, batch)
    return DenseBuffer(δz, δ)
end


@inline function _fill_JW_Jb!(J_W, J_b, W, δz, x)
    if ndims(x) == 1
        J_W .= δz .* reshape(x, 1, :)
        J_b .= δz
    else
        out_dim, in_dim = size(W)
        batch   = size(δz, 2)
        J_W .= reshape(δz, out_dim, 1, batch) .* reshape(x, 1, in_dim, batch)
        J_b .= δz
    end
end


"""
    back!(layer::Dense, buf, J_W, J_b, δ, x) -> δ_pass
    back!(layer::Dense, buf, J_W, J_b, δ, x, ln) -> δ_pass

Reverse pass through one `Dense` layer, writing the per-sample Jacobian
into the contiguous arrays `J_W` and `J_b`. 
Returns the accumulated gradient `buf.δ` for next layer calculations layer.

# Arguments

* `layer`: layer of Neural Network.
* `buf`: custom layer buffer that holds input and output gradients of 
    backpropagation.
* `δ`: gradient incomming from previous layer.
* `x`: input array for that layer.

If dispatched with `ln` ([`LayerNorm`](@ref)) then the gradient accumulation
needs to account for extra normalisation step of pre-activation output.
"""
function back!(layer::Dense, buf::DenseBuffer, J_W, J_b,
               δ::AbstractArray, x::AbstractArray)
    buf.δz .= δ .* layer.act_deriv.(layer.a)
    _fill_JW_Jb!(J_W, J_b, layer.W, buf.δz, x)
    mul!(buf.δ, layer.W', buf.δz, 1f0, 0f0)
    return buf.δ
end
function back!(layer::Dense, buf::DenseBuffer, J_W, J_b,
               δ::AbstractArray, x::AbstractArray, ln::LayerNorm)
    buf.δz .= δ .* layer.act_deriv.(layer.a)
    ln.J_γ .= buf.δz .* ln.x̂          
    ln.J_β .= buf.δz                    
    ln_backward!(ln, buf.δz)            
    _fill_JW_Jb!(J_W, J_b, layer.W, buf.δz, x)
    mul!(buf.δ, layer.W', buf.δz, 1f0, 0f0)
    return buf.δ
end
