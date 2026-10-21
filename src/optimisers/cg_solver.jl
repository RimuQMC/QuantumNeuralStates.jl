
"""
    cg_matvec!(Ap, J_bar, p, λ, tmp) -> (pKp, pp)

In-place matrix–vector product for [`cg_solve!`](@ref),

```math
A p = (\\bar{J}^T \\bar{J} + \\lambda \\mathbf{I})\\, p ,
```

returning the curvature `pᵀKp = ‖J̄p‖²` (with `K = J̄ᵀJ̄`) and `pᵀp`, used by the
null-space check and to assemble `pᵀAp = pᵀKp + λ·pᵀp` without extra cancellation.

# Arguments
* `Ap`: (N,) output, overwritten.
* `J_bar`: (n_params, N) weighted, centred Jacobian.
* `p`: (N,) search direction.
* `λ`: Tikhonov shift.
* `tmp`: (n_params,) scratch; holds `J̄p` on exit.
"""
function cg_matvec!(Ap, J_bar, p, λ::Float32, tmp)
    mul!(tmp, J_bar,  p,   1f0, 0f0)    # tmp = J̄ p         (N,)
    mul!(Ap,  J_bar', tmp, 1f0, 0f0)    # Ap  = J̄ᵀ J̄ p      (N,)
    @. Ap = Ap + λ * p                  # Ap += λ p
    pKp = dot(tmp, tmp)                 # pᵀKp = ‖J̄ p‖²
    pp  = dot(p, p)                     # pᵀp
    return pKp, pp
end

"""
    cg_solve!(x, J_bar, λ, r, p_vec, Ap, tmp; kwargs...) -> (ok, iters, relres)

Conjugate-gradient solver for the minSR system, GPU-friendly and allocation-free,

```math
(\\bar{J}^T \\bar{J} + \\lambda \\mathbf{I})\\, x = b,
```

with `b` passed in `r` and initial guess `x = 0`. Expects the system pre-scaled by the
caller (mean eigenvalue of `J̄ᵀJ̄` ≈ 1, `‖b‖ = 1`), so `λ` and `null_tol` are relative.

# Arguments
* `x`: (N,) solution, written in place.
* `J_bar`: (p, N) weighted, centred, rescaled Jacobian.
* `λ`: relative Tikhonov shift.
* `r`: (N,) RHS `b` on input, residual on output.
* `p_vec`, `Ap`: (N,) CG buffers.
* `tmp`: (p,) scratch buffer.

# Keyword Arguments
* `max_iter`: maximum iterations (default `300`).
* `tol`: relative residual `‖r‖/‖b‖` for convergence (default `1f-2`).
* `null_tol`: null-space threshold on `pᵀKp / pᵀp` (default `1f-6`).
* `max_null_hits`: null-space restarts before stopping (default `3`).

# Returns
* `ok`: `true` if `x` is usable (converged, `max_iter`, or null-space stop);
  `false` on numerical breakdown (Inf/NaN), in which case the step must be skipped.
* `iters`: iterations performed.
* `relres`: `‖r‖/‖b‖` at exit.

# Null-space guard
`K = J̄ᵀJ̄` is singular by construction (centring) and rank-deficient for duplicate
configurations. Directions with `pᵀKp < null_tol · pᵀp` (with `pᵀKp = ‖J̄p‖²` from
the matvec) are skipped by restarting `p` from the residual; after `max_null_hits`
the solve stops with the current partial solution.

# Notes
Reference: J. R. Shewchuk, *An Introduction to the Conjugate Gradient Method Without
the Agonizing Pain*, https://people.eecs.berkeley.edu/~jrs/papers/cg.pdf (p. 32).
"""
function cg_solve!(x, J_bar, λ::Float32, r, p_vec, Ap, tmp;
                   max_iter::Int = 500, tol::Float32 = 1f-2, 
                   null_tol::Float32 = 1f-6, max_null_hits::Int = 3)

    fill!(x, 0f0)
    copyto!(p_vec, r)                   # p₀ = r₀ = b

    rr0 = dot(r, r)                     # ‖b‖² (≈ 1 after caller's scaling)
    if !isfinite(rr0)
        @warn "cg_solve!: ‖b‖² = $rr0 is Inf/NaN on entry — RHS corrupted, aborting solve"
        return false, 0, Inf32
    end
    if rr0 < 1f-12
        @warn "cg_solve!: ‖b‖² = $rr0 ≈ 0 on entry — trivial RHS, returning x = 0"
        return true, 0, 0f0
    end

    rr = rr0
    null_hits = 0

    for iter in 1:max_iter

        pKp, pp = cg_matvec!(Ap, J_bar, p_vec, λ, tmp)

        if !(isfinite(pKp) && isfinite(pp))
            @warn "cg_solve!: matvec overflow at iter $iter (pᵀKp = $pKp, pᵀp = $pp) — " *
                  "J̄ not properly scaled, aborting before residual update"
            return false, iter, sqrt(rr / rr0)
        end

        # --- null-space check: curvature of K = J̄ᵀJ̄ alone along p ------------
        # relative to the mean eigenvalue (= 1 after scaling); λ excluded
        if pKp < null_tol * pp
            null_hits += 1
            if null_hits >= max_null_hits
                @warn "cg_solve!: null-space direction hit $null_hits× at iter $iter — " *
                      "stopping with partial solution (relres = $(sqrt(rr / rr0)))"
                return true, iter, sqrt(rr / rr0)
            end
            copyto!(p_vec, r)           # restart search direction, skip update
            continue
        end

        # --- standard CG update ----------------------------------------------
        pAp = pKp + λ * pp              # pᵀAp, assembled without extra dot
        α = rr / pAp
        if !isfinite(α)
            @warn "cg_solve!: step size α = $α is Inf/NaN at iter $iter " *
                  "(rr = $rr, pᵀAp = $pAp) — breakdown before x update"
            return false, iter, sqrt(rr / rr0)
        end

        @. x = x + α * p_vec            # x ← x + α p
        @. r = r - α * Ap               # r ← r − α Ap

        rr_new = dot(r, r)
        if !isfinite(rr_new)
            @warn "cg_solve!: residual ‖r‖² = $rr_new is Inf/NaN after update at iter $iter " *
                  "(α = $α) — x possibly corrupted, aborting"
            return false, iter, sqrt(rr / rr0)
        end

        relres = sqrt(rr_new / rr0)
        if relres ≤ tol
            return true, iter, relres
        end

        β = rr_new / rr                 # keeps p A-conjugate
        @. p_vec = r + β * p_vec
        rr = rr_new
    end

    if sqrt(rr / rr0) > 10f0*tol
        @warn "cg_solve!: max_iter = $max_iter reached (relres = $(sqrt(rr / rr0)), tol = $tol) — " *
              "using partial solution"
    end
    return true, max_iter, sqrt(rr / rr0)
end
