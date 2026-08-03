
"""
    LayerNorm(out, batch, device=identity; T=Float32, ϵ=T(1f-3))

LayerNorm for any NN architecture. Norm of pre-activation `a` used for stabilizing
input of activation function

```math
\\tilde{a} = ((a - μ)/σ) * γ + β = \\hat{x} * γ + β
```

`γ, β`: learnable — updated by optimiser, live in the parameter vector.
`μ, σ`: NOT learnable — deterministic functions of the current input a.

# Variables

* `out`: output dimension of Neural Network layer.
* `batch`: batch dimension.
* `device`: should correspond to Neural Network choice of device (CPU/GPU).

# Keywords

* `T`: determine precision type. By default `Float32` (recommended for GPU).
* `ϵ`: stabiliser, prevents zero division.
"""
mutable struct LayerNorm{T, M <: AbstractMatrix{T}}
    # -- learnable ----------------------------------------
    γ::M    # (out,1)   scale  — (out,1) broadcasts over batch
    β::M    # (out,1)   shift
    ε::T    # stabiliser

    # -- forward cache (needed by backward) ---------------
    x̂::M    # (out,batch)  (a−μ)/σ         — used in ∂γ and LN grad
    σ::M    # (1,  batch)  std per sample   — divides the LN grad

    # -- gradient accumulators (standard optimiser) -------
    J_γ::M    # (out,1)
    J_β::M    # (out,1)

    # -- scratch ------------------------------------------
    μ::M        # μ     (1,  batch)  forward scratch  : mean, overwritten each call
    Hbuf::M     # Hbuf  (out,batch)  dual-use         : (a−μ)² in fwd  /  δ'=δ⊙γ in bwd
    sbuf::M     # sbuf  (1,  batch)  backward only    : s₁=mean(δ', dims=1)
    sbuf2::M    # sbuf2 (1,  batch)  backward only    : s₂=mean(δ'⊙x̂, dims=1)

    ones_H::M    # contraction of out dimension for sum   (1, out)
end
function LayerNorm(out::Int, batch::Int, device=identity;
                   T::Type{<:AbstractFloat}=Float32, ε=T(1f-3))
    γ = device(ones(T, out, 1))
    β = device(zeros(T, out, 1))   
    M = typeof(γ)
    return LayerNorm{T, M}(
        γ, β, T(ε),
        device(zeros(T, out, batch)),   # x̂
        device(zeros(T, 1, batch)),     # σ
        device(zeros(T, out, batch)),   # J_γ    (bwd)
        device(zeros(T, out, batch)),   # J_β    (bwd)
        device(zeros(T, 1, batch)),     # μ      (fwd scratch)
        device(zeros(T, out, batch)),   # Hbuf   (δ⊙γ bwd / (a−μ)² fwd)
        device(zeros(T, 1, batch)),     # sbuf   (s₁ bwd)
        device(zeros(T, 1, batch)),     # sbuf2  (s₂ bwd)
        device(ones(T, 1, out))         # contraction of out dim
       )
end
"""
    LayerNorm_multiforward(out, batch, device=identity; T=Float32, ϵ=T(1f-3))

Similar to [`LayerNorm`](@ref), but meant for custom batch size only in forward passes.
"""
function LayerNorm_multiforward(out::Int, batch::Int, device=identity;
                   T::Type{<:AbstractFloat}=Float32, ε=T(1f-3))
    γ = device(ones(T, 1, 1))
    β = device(zeros(T, 1, 1))   
    M = typeof(γ)
    return LayerNorm{T, M}(
        γ, β, T(ε),
        device(zeros(T, out, batch)),   # x̂
        device(zeros(T, 1, batch)),     # σ
        device(zeros(T, 1, 1)),         # dummy
        device(zeros(T, 1, 1)),         # dummy
        device(zeros(T, 1, batch)),     # μ      (fwd scratch)
        device(zeros(T, out, batch)),   # Hbuf   ((a−μ)² fwd)
        device(zeros(T, 1, 1)),         # dummy
        device(zeros(T, 1, 1)),         # dummy
        device(ones(T, 1, out))         # contraction of out dim
       )
end

"""
    ln_forward!(ln, a)

Forward pass function for Layer Normalisation of pre-activation `a` and 
write γ*x̂+β inplace into `a`. See [`LayerNorm`](@ref).
"""
function ln_forward!(ln::LayerNorm{T}, a::AbstractMatrix{T}) where T
    H = size(a, 1)
    # ones_H (1,H) * a (H,B) → μ (1,B) : sum over features = left-mul by row-of-ones
    mul!(ln.μ, ln.ones_H, a)
    ln.μ   .*= T(1)/T(H)
    ln.Hbuf .= (a .- ln.μ) .^ 2
    mul!(ln.σ, ln.ones_H, ln.Hbuf)

    # ln.σ .= sqrt.(ln.σ .* (T(1)/T(H)) .+ ln.ε)
    ln.σ .= ln.σ .* (T(1)/T(H))
    ln.σ .= sqrt.(ln.σ .+ ln.ε)

    ln.x̂ .= (a .- ln.μ) ./ ln.σ
    a .= ln.γ .* ln.x̂ .+ ln.β
end

"""
    ln_forward!(ln, a, ln_multi)

Similar as [`ln_forward!`](@ref). Dispatch function which is used if multi batched 
forward pass is used.
"""
function ln_forward!(ln::LayerNorm{T}, a::AbstractMatrix{T}, ln_multi::LayerNorm{T}
    ) where T
    H = size(a, 1)

    mul!(ln_multi.μ, ln_multi.ones_H, a)
    ln_multi.μ   .*= T(1)/T(H)
    ln_multi.Hbuf .= (a .- ln_multi.μ) .^ 2
    mul!(ln_multi.σ, ln_multi.ones_H, ln_multi.Hbuf)

    ln_multi.σ .= ln_multi.σ .* (T(1)/T(H))

    ln_multi.σ .= sqrt.(ln_multi.σ .+ ln.ε)

    ln_multi.x̂ .= (a .- ln_multi.μ) ./ ln_multi.σ

    a .= ln.γ .* ln_multi.x̂ .+ ln.β
end

# https://robotchinwag.com/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/
"""
    ln_backward!(ln, δ) → ∂L/∂a

Backpropagation function which calculates derivatives of Loss function w.r.t.
pre-activation variable `a` modified with layer normalisation.

## Appendix

Full chain through μ(a) and σ(a):
  ∂aᵢ = (δγᵢ − s₁ − x̂ᵢ·s₂) / σ       where  δγ = δ⊙γ
  s₁  = mean(δγ,   over features)    corrects ∂μ/∂aⱼ = 1/H
  s₂  = mean(δγ⊙x̂, over features)    corrects ∂σ/∂aⱼ = x̂ⱼ/H
"""
function ln_backward!(ln::LayerNorm{T}, δ::AbstractMatrix{T}) where T
    H    = size(δ, 1)
    invH = T(1)/T(H)

    δγx̂ = ln.Hbuf    # (out,batch)
    s₁  = ln.sbuf    # (1,  batch)   feature-mean of δγ
    s₂  = ln.sbuf2   # (1,  batch)   feature-mean of δγ⊙x̂

    # --- s₁ = mean(δγ), s₂ = mean(δγ⊙x̂) -------------------------------
    δ .= δ .* ln.γ
    mul!(s₁, ln.ones_H, δ)
    s₁  .*= invH
    δγx̂ .= δ .* ln.x̂
    mul!(s₂, ln.ones_H, δγx̂)
    s₂  .*= invH

    # --- ∂a = (δγ − s₁ − x̂·s₂) / σ ------------------------------------
    δ .= (δ .- s₁ .- ln.x̂ .* s₂) ./ ln.σ # broadcast (1,batch) → (out,batch)

    return δ
end



