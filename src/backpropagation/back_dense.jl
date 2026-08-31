
"""
    DenseBuffer(layer::Dense)

Pre-allocated buffer for backpropagation and Jacobian calculations through a `Dense` layer.
"""
struct DenseBuffer{DZ <: AbstractArray, D <: AbstractArray}
    δz::DZ # new sending output (out, batch)
    δ ::D  # incoming previous output (in,  batch)
end
function DenseBuffer(layer::Dense)
    out_dim, in_dim = size(layer.W)
    batch = size(layer.a, 2)
    δz = similar(layer.a)
    δ  = similar(layer.W, in_dim, batch)
    return DenseBuffer(δz, δ)
end


@inline function _fill_JW_Jb!(J_W, J_b, W, δz, x)
    out_dim, in_dim = size(W)
    batch   = size(δz, 2)
    J_W .= reshape(δz, out_dim, 1, batch) .* reshape(x, 1, in_dim, batch)
    J_b .= δz
end


function apply_act_deriv!(δz, layer::Dense{T,M,V,F,G,B,Nothing,LN}, a) where {T,M,V,F,G,B,LN}
    δz .= layer.act_deriv.(a)
    return δz
end
function apply_act_deriv!(δz, layer::Dense{T,M,V,F,G,B,R,LN}, a) where {T,M,V,F,G,B,R<:Tuple,LN}
    map(layer.act_deriv, layer.act_ranges) do f, r
        @views δz[r,:] .= f.(a[r,:])
    end
    return δz
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
    apply_act_deriv!(buf.δz, layer, layer.a)
    buf.δz .*= δ
    _fill_JW_Jb!(J_W, J_b, layer.W, buf.δz, x)
    mul!(buf.δ, layer.W', buf.δz, 1f0, 0f0)
    return buf.δ
end
function back!(layer::Dense, buf::DenseBuffer, J_W, J_b,
               δ::AbstractArray, x::AbstractArray, ln::LayerNorm)
    apply_act_deriv!(buf.δz, layer, layer.a)
    buf.δz .*= δ
    ln.J_γ .= buf.δz .* ln.x̂          
    ln.J_β .= buf.δz                    
    ln_backward!(ln, buf.δz)            
    _fill_JW_Jb!(J_W, J_b, layer.W, buf.δz, x)
    mul!(buf.δ, layer.W', buf.δz, 1f0, 0f0)
    return buf.δ
end
