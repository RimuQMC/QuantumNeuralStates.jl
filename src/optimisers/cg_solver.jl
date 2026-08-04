
"""
    cg_matvec!(Ap, J_bar, p, λ::Float32, tmp)

In-place matrix-vector product for the [`cg_solve!`](@ref) linear system solver.

```math
Ap = (\\tilde{J}^T \\dot \\tilde{J} + \\lambda \\mathbf{I}) \\dot p
```

# Arguments
* `Ap`: intermediate vector `(N,)`.
* `J_bar`: weighted, centred Jacobian `J̄`, shape `(p, N)`.
* `p`: input (gradient) vector `(N,)`.
* `λ`: Tikhonov regularisation scalar - default `1f-3`.
* `tmp`: scratch buffer `(p,)` — holds `J̄·p` on the way through
"""
function cg_matvec!(Ap, J_bar, p, λ::Float32, tmp)
    mul!(tmp, J_bar,  p,   1f0, 0f0)    # tmp = J̄  · p  (p,) - J_bar is (p,N)
    mul!(Ap,  J_bar', tmp, 1f0, 0f0)    # Ap  = J̄ᵀ · tmp = J̃ᵀJ̃·p (N,)
    @. Ap = Ap + λ * p                  # Ap += λ·p   (fused, single GPU kernel)
end

"""
    cg_solve!(x, J_bar, λ, r, p_vec, Ap, tmp;
              max_iter=100, tol=√eps, null_tol=√eps, max_null_hits=3)

Conjugate Gradient solver for the minSR linear system. Written for batched 
approach and GPU friendly calculations.

```math
(\\tilde{J}^T \\tilde{J} + \\lambda \\mathbf{I}) x = b
```

where `b` is passed in `r` (overwritten during solve) and the initial guess is `x = 0`.

# Arguments
* `x`: solution vector `(N,)` — initialised to zero, result written here.
* `J_bar`: weighted centred Jacobian `(p, N)`.
* `λ`: regularisation scalar.
* `r`: RHS vector `b` on input, residual on output `(N,)`.
* `p_vec`: CG search direction buffer `(N,)`.
* `Ap`: matvec output buffer `(N,)`.
* `tmp`: scratch buffer `(p,)`.

# Keyword Arguments
* `max_iter`: maximum CG iterations (default: `100`).
* `tol`: relative residual tolerance `‖r‖/‖b‖ ≤ tol` (default: `√eps(Float32)`).
* `null_tol`: relative null-space threshold (default: `√eps(Float32)`)
* `max_null_hits`: max null-space restarts before early exit (default: `3`)

## Null-space guard
When `J̄` has many duplicate columns (few unique configs in batch), `K = J̄ᵀ·J̄` is
rank-deficient. CG reveals this as `pᵀAp ≈ 0` (step size α diverges).
Strategy: skip the update, restart `p` from the current residual, and count
how many times this happens. After `max_null_hits` the residual truly lives
in the null space — break cleanly.
`null_tol` is relative: `pᵀAp < null_tol · ‖r‖²` — scale-free regardless
of problem magnitude.

## Notes
inspiration: https://people.eecs.berkeley.edu/~jrs/papers/cg.pdf (p.32)
"""
function cg_solve!(x, J_bar, λ::Float32, r, p_vec, Ap, tmp;
                   max_iter::Int      = 100,
                   tol::Float32       = sqrt(eps(Float32)),
                   null_tol::Float32  = sqrt(eps(Float32)),
                   max_null_hits::Int = 3)

    # Initial configuration
    fill!(x, 0)
    # copyto!(r,     b) # r = b    (x₀ = 0  →  r₀ = b − A·0 = b)
    copyto!(p_vec, r) # p = r₀

    rr = dot(r, r)
    bnrm = sqrt(rr)
    null_hits = 0

    if bnrm < 1f-6  # trivial rhs - safe check
        return nothing
    end

    for iter in 1:max_iter

        cg_matvec!(Ap, J_bar, p_vec, λ, tmp)
        pAp = dot(p_vec, Ap)

        # --- null-space check ----------------------------------------------
        if pAp < null_tol * rr
            null_hits += 1
            if null_hits >= max_null_hits
                # @warn "CG: null-space direction hit $null_hits times — " *
                #       "stopping at iter $iter"
                break
            end
            # Restart search direction from current residual and skip update.
            # The component of p in the null space contributes nothing to x;
            # restarting from r re-seeds CG in the column space.
            copyto!(p_vec, r)
            continue
        end

        # --- standard CG update --------------------------------------------
        α = rr / pAp
        if !isfinite(α)
            @warn "CG solver, α is Inf/NaN - break"
            break
        end

        @. x = x + α * p_vec     # x  ← x + α p
        @. r = r - α * Ap        # r  ← r − α Ap

        rr_new = dot(r, r)
        if !isfinite(rr_new)
            @warn "CG solver, rr_new is Inf/NaN - break"
            break
        end

        if sqrt(rr_new) ≤ tol * bnrm
            break
        end

        β = rr_new / rr # β ensures that p-update is in A-conjugate space (orthogonal space)
        @. p_vec = r + β * p_vec
        rr = rr_new
    end
    return nothing
end
