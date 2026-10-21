# using Metal
# using Statistics
# using KernelAbstractions

"""
    MomentumBuffer(p::Int, ansatz; β=0.9f0)

Pre-allocated buffer for gradient momentum smoothing (exponential moving average of gradients).

# Arguments
* `p`: number of parameters in `ansatz` model.
* `ansatz`: ansatz that evaluates wave-function. See [`NeuralAnsatz`](@ref).

## Fields
* `v`: velocity vector (running mean of gradients), shape `(p,)`.
* `β`: momentum decay rate (default: `0.9`)

## Notes
The momentum is inspired from [`adam`](@ref) momentum update.
See: [Momentum paper](https://proceedings.neurips.cc/paper/2020/file/d3f5d4de09ea19461dab00590df91e4f-Paper.pdf)

"""
mutable struct MomentumBuffer{T, V <: AbstractArray{T}}
    v :: V      # velocity / running mean
    β :: T
end
function MomentumBuffer(p::Int, ansatz; β=0.9f0)
    T = Float32
    l = last(ansatz.model.layers)
    v = fill!(similar(l.b, p), zero(T))
    V = typeof(v)
    return MomentumBuffer{T, V}(v, β)
end

function momentum_step!(buf, Δθ_SR)
    β = buf.β
    @. buf.v = β * buf.v + (one(β) - β) * Δθ_SR
    return buf.v   # smoothed update
end


"""
    minSRBuffer(N::Int, p::Int, ansatz; kwargs...)

Structure meant for `minSR` optimiser, see [`compute_minSR_cg!`](@ref) and [`minSR`](@ref).
It allows allocation-free, batched and GPU friendly evaluation of `minSR` and holds the
step-control settings of the optimiser.

# Variables
* `N`: batch size for array dimensions.
* `p`: total number of parameters in ansatz model, for array dimensions.
* `ansatz`: ansatz model that evaluates wave-function. See also [`NeuralAnsatz`](@ref).

# Keyword Variables
* `velocity`: boolean variable, if `true` the `minSR` optimiser will apply momentum
    smoothening. By default `false`. See also [`MomentumBuffer`](@ref).
* `β`: momentum coefficient, used with [`MomentumBuffer`](@ref).
* `var_track`: running mean of the energy variance, used to detect variance spikes.
* `s_track`: running mean of the Jacobian scale `s = tr(J̄ᵀJ̄)/N`, used to detect
    sudden drops of `s` (sample collapse or saturation).
* `d_loose`: maximal Fubini–Study distance per step in normal operation (default `1f-1`,
    ≈ 5.7°).
* `d_tight`: maximal Fubini–Study distance per step after a variance spike (default `1f-2`,
    ≈ 0.57°). If `d_loose < d_tight`, `d_loose` is set to `d_tight`.
* `ρ`: maximal relative weight change `η‖Δθ‖/‖θ‖` per step (default `5f-2`). During a
    drop of `s` the limit is tightened to `0.1*ρ`.

# Note
* The Fubini–Study distance measures the overlap of the wave function before and after
    the update step, `cos²(d) = |⟨ψ|ψ'⟩|²`, so the exact distance lies in `[0, π/2]`.
    The value `d = η‖J̄ᵀΔθ‖` used here is its linearised estimate on the samples and can
    exceed `π/2` for large steps.
* The minSR step is derived to first order in the parameter change and is accurate only
    for `d ≪ 1`. Sensible values of `d_max` therefore lie in `(0f0, 1f-1]`.
"""
mutable struct minSRBuffer{T,V<:AbstractVector{T},X}
    O_mean::V
    Δθ::V
    vel::Bool
    moment::X
    p_cg::V
    Ap::V

    var_track::RunningMean{T}    # spikes of the energy variance
    s_track::RunningMean{T}      # drops of the Jacobian scale s (saturation)

    d_loose::Float32    # adaptation
    d_tight::Float32    # protection on variance spikes / s drops
    ρ::Float32          # max relative weight change per step
end
function minSRBuffer(N::Int, p::Int, ansatz; 
                     velocity=false, β=0.9f0,
                     var_track::RunningMean{Float32}, s_track::RunningMean{Float32},
                     d_loose::Float32=1f-1, d_tight::Float32=1f-2, ρ::Float32=5f-2)
    T = Float32
    l = first(ansatz.model.layers) # l.b is Vector type, l.W is Matrix type (CPU or GPU)

    O_mean = similar(l.b, p)
    p_cg = similar(l.b, N)
    Ap = similar(l.b, N)
    Δθ = fill!(similar(l.b, p), zero(T))
    if !velocity
        vel = false
        moment = nothing
    else
        vel = true 
        moment = MomentumBuffer(p, ansatz; β=β)
    end

    if d_loose < d_tight
        @warn "d_loose=$(d_loose) value cannot be smaller than d_tight=$(d_tight) value. " *
              "Setting d_loose=d_tight"
        d_loose = d_tight
    end

    V = typeof(O_mean)
    X = typeof(moment)
    return minSRBuffer{T,V,X}(O_mean, Δθ, vel, moment, p_cg, Ap, var_track, s_track,
                              d_loose, d_tight, ρ)
end

"""
    compute_minSR_cg!(E_mean, variance, jacobian_buf, vmc_buf, minSR_buf,
                      ansatz, mode, λ, weights) -> (Δθ, sq, ok)

Compute the minSR natural-gradient direction with the matrix-free [`cg_solve!`](@ref),

```math
\\Delta\\theta = \\bar{J}\\, (\\bar{J}^T \\bar{J} + \\lambda_{\\mathrm{abs}} \\mathbf{I})^{-1}\\, b ,
\\qquad \\lambda_{\\mathrm{abs}} = s\\,\\lambda ,
```

where `J̄[:, b] = √w_b (O_b − Ō)` is the weighted, centred Jacobian and `b` the weighted
loss gradient vector from [`apply_loss!`](@ref). The system is solved in rescaled form
(`J' = J̄/√s`, `b' = b/‖b‖`, `s = tr(J̄ᵀJ̄)/N`), so `λ` is relative to the mean eigenvalue
and all CG quantities are `O(1)`.

where `J̄[:, b] = √w_b (O_b − Ō)` is the weighted, centred Jacobian and `b` the weighted
loss vector from [`apply_loss!`](@ref). The system is solved in rescaled form
(`J' = J̄/√s`, `b' = b/‖b‖`, `s = tr(J̄ᵀJ̄)/N`), `λ` is relative to the mean eigenvalue of `J̄ᵀJ̄`, 
and all CG quantities are `O(1)`. The solution is mapped back as `Δθ = (‖b‖/√s) J'x'`.

# Arguments
* `E_mean`: (1,) device arrays with `⟨E_loc⟩`.
* `variance`: (1,) device arrays with `σ²`.
* `jacobian_buf`: holds the Jacobian `J` (n_params, N).
* `vmc_buf`: holds `E_locs` (N,); `ham_diag` is reused for `b` gradients.
* `minSR_buf`: CG and update buffers, see [`minSRBuffer`](@ref).
* `ansatz`: see [`NeuralAnsatz`](@ref).
* `mode`: loss mode for [`apply_loss!`](@ref).
* `λ`: relative Tikhonov shift.
* `weights`: (N,) normalised sampler weights (`Σw = 1`), see [`vmc_sample!`](@ref).

# Returns
* `Δθ`: (n_params,) update direction.
* `sq`: `√s`, needed by the [`minSR`](@ref) for the Fubini–Study step length and the
    detection of drops of `s`.
* `ok`: `false` if `s`, `‖b‖` or the CG solve is invalid; the [`minSR`](@ref) must skip the step.
"""
function compute_minSR_cg!(E_mean, variance, jacobian_buf, vmc_buf, minSR_buf,
                           ansatz, mode, λ::Float32, weights)

    J      = jacobian_buf.J             # (p, N)
    O_mean = minSR_buf.O_mean           # (p,)
    Δθ     = minSR_buf.Δθ               # (p,) — also CG scratch `tmp`
    p_cg   = minSR_buf.p_cg             # (N,)
    Ap     = minSR_buf.Ap               # (N,)
    E_locs = vmc_buf.E_locs             # (N,) e_locs vector
    g      = vmc_buf.ham_diag           # (N,) loss vector b

    # --- centre & weight -------------------------------------------------------
    mul!(O_mean, J, weights)            # Ō = Σ_b w_b J[:, b]
    weights .= sqrt.(weights)           # w ← √w
    apply_loss!(g, E_locs, weights, E_mean, variance, mode)   # b = √w (E_loc − Ē)
    J .-= O_mean
    J .*= reshape(weights, 1, :)        # J̄[:, b] = √w_b (O_b − Ō)

    # --- rescale: mean eigenvalue of J̄ᵀJ̄ → 1 -----------------------------------
    N = size(J, 2)
    s = dot(J, J) / N                   # tr(J̄ᵀJ̄)/N
    if !(isfinite(s) && s > 0f0)
        @warn "compute_minSR_cg!: Jacobian scale s = tr(J̄ᵀJ̄)/N = $s invalid — skipping update"
        return Δθ, 0f0, false
    end
    sq = sqrt(s)
    J ./= sq                            # J' = J̄/√s

    # --- rescale: ‖b‖ → 1 --------------------------------------------------------
    bn = norm(g)
    if !(isfinite(bn) && bn > 0f0)
        @warn "compute_minSR_cg!: ‖b‖ = $bn invalid (zero or Inf/NaN) — skipping update"
        return Δθ, sq, false
    end
    g ./= bn                            # b' = b/‖b‖

    # --- solve (K' + λI) x' = b'   (λ relative to mean eigenvalue) --------------
    ok, iters, relres = cg_solve!(weights, J, λ, g, p_cg, Ap, Δθ)
    if !ok
        @warn "compute_minSR_cg!: CG breakdown (iters = $iters, relres = $relres) — skipping update"
        return Δθ, sq, false
    end

    # --- map back:  Δθ = J̄ x = (‖b‖/√s) J' x' -----------------------------------
    mul!(Δθ, J, weights, bn / sq, 0f0)
    return Δθ, sq, true
end

"""
    minSR(jacobian_buf, vmc_buf, minSR_buf, H, ansatz, addrs_n; kwargs...)
            -> E_mean, variance, last_addr, acceptance, eta_eff

One minSR optimisation step: sample, compute the direction `Δθ` with
[`compute_minSR_cg!`](@ref), and update the parameters with

```math
\\theta \\leftarrow \\theta - \\eta_{\\mathrm{eff}}\\,\\Delta\\theta, \\qquad
\\eta_{\\mathrm{eff}} = \\eta \\cdot \\min\\!\\left(1,\\ \\frac{d_{\\max}}{d},\\
\\frac{\\mathrm{max\\_norm}}{\\lVert\\Delta\\theta\\rVert},\\
\\frac{\\rho_{\\mathrm{eff}}\\,\\lVert\\theta\\rVert}{\\eta\\,\\lVert\\Delta\\theta\\rVert}\\right),
```

where `d = η ‖J̄ᵀΔθ‖` is the linearised Fubini–Study length of the step on the sampled
configurations. The three limits are:

1. Fubini–Study cap `d ≤ d_max`: limits the change of the state on the samples.
   `d_max = d_tight` after a variance spike, otherwise `d_loose`.
2. Euclidean backstop `‖Δθ‖ ≤ max_norm`: guard against non-finite or divergent updates.
3. Relative weight cap `η‖Δθ‖ ≤ ρ_eff ‖θ‖`: limits the change of the network as a whole,
   including configurations outside the sample set (e.g. the neighbours entering
   `E_loc`). `ρ_eff = 0.1*ρ` during a drop of `s`, otherwise `ρ`.

The Fubini–Study cap and the relative cap are complementary: when the samples respond
only weakly to the parameters (small `s`), `d` underestimates the size of the step and
the relative cap binds; when they respond strongly (large `s`), the Fubini–Study cap binds.

# Arguments
* `jacobian_buf`: Jacobian and parameter vector `θ`, see [`JacobianBuffer`](@ref).
* `vmc_buf`: VMC results (local energies), see [`VMCBuffer`](@ref).
* `minSR_buf`: minSR and CG buffers, see [`minSRBuffer`](@ref).
* `H`: Rimu Hamiltonian.
* `ansatz`: see [`NeuralAnsatz`](@ref).
* `addrs_n`: current walker addresses; the updated ones are returned.

# Keyword Arguments
* `vmc`: sampler, `:metropolis` or `:ctmc` (default `:metropolis`), see [`vmc_sample!`](@ref).
* `burnin`: thermalisation steps before the update (default `100`), see [`vmc_energy`](@ref).
* `mode`: loss mode (default `:energy`), see [`apply_loss!`](@ref).
* `λ`: Tikhonov shift, relative to the mean eigenvalue of `J̄ᵀJ̄` (default `1f-3`).
* `η`: learning rate / imaginary-time step (default `1f-3`).
* `max_norm`: Euclidean backstop on `‖Δθ‖` (default `1f3`).

# Notes
* Variance spike: the variance exceeds 10× its running mean (`var_track`). The
    Fubini–Study cap is tightened to `d_tight`.
* Drop of `s`: `s` falls below 1/100 of its running mean (`s_track`), tolerated for up to
    `max_spike = 20` consecutive steps. This indicates that the walkers concentrate on few
    configurations, so the update direction carries little information while the weights
    can still change considerably; the relative cap is tightened to `0.1 ρ`.
* The Fubini–Study cap follows the natural norm constraint, see DOI:
    10.1016/j.jcp.2024.113351 (Sec. 3.1). The relative cap follows layer-wise trust
    ratios (arXiv:1708.03888) and deep relative trust (arXiv:2002.03432).
* The step is skipped (parameters unchanged) if the solve fails or `d` or `‖Δθ‖` is
    non-finite.
* Momentum, if enabled, is applied before `d` is measured.
"""
function minSR(jacobian_buf, vmc_buf, minSR_buf, H, ansatz, addrs_n;
               vmc = :metropolis, burnin = 100, mode = :energy,
               λ::Float32 = 1f-3, η::Float32 = 1f-3, max_norm::Float32 = 1f3)

    E_mean, variance, last_addr, acceptance, weights =
        vmc_energy(H, ansatz, addrs_n, vmc_buf, jacobian_buf;
                   vmc_sampler=vmc, burnin=burnin, mode=mode)

    # --- variance spike check -------------------------------------------------
    var = Array(variance)[1]
    var_spike = is_outlier!(minSR_buf.var_track, var; up = true, factor = 10f0)

    # --- natural-gradient direction -------------------------------------------
    # after this call: jacobian_buf.J holds J' = J̄/√s,  sq = √s
    Δθ, sq, ok = compute_minSR_cg!(E_mean, variance, jacobian_buf, vmc_buf, minSR_buf,
                                   ansatz, mode, λ, weights)
    if !ok
        @warn "minSR: computation of update failed — skipping update"
        return E_mean, variance, last_addr, acceptance, 0f0
    end

    # --- saturation check: s dropped more than 100× below its running mean ----
    s_drop = is_outlier!(minSR_buf.s_track, sq^2; up = false, factor = 100f0, max_spike=20)
    s_drop && @warn "minSR: Jacobian scale s = $(sq^2) dropped more then 100× " * 
                    "— possible saturation, tight relative weight cap (ρ × 0.1)"

    # --- momentum -------------------------------------------------------------
    if minSR_buf.vel && minSR_buf.moment !== nothing
        Δθ = momentum_step!(minSR_buf.moment, Δθ)
    end

    # --- 1. Fubini–Study step length ------------------------------------------
    δlogψ = minSR_buf.Ap                     # (N,) buffer, change in logψ of that caused step
    mul!(δlogψ, jacobian_buf.J', Δθ)         # J'ᵀΔθ  (J must still be J' from this step)
    v = sq * norm(δlogψ)                     # speed in state space: ‖J̄ᵀΔθ‖
    d = η * v                                # F-S distance of the plain SR step

    if !isfinite(d)
        @warn "minSR: Fubini–Study distance d = $d is Inf/NaN — skipping update"
        return E_mean, variance, last_addr, acceptance, 0f0
    end

    d_loose = minSR_buf.d_loose
    d_tight = minSR_buf.d_tight
    d_max = var_spike ? d_tight : d_loose
    scale_fs = d > d_max ? d_max / d : 1f0 # d_max = max Fubini–Study distance per step

    # --- 2. Euclidean backstop ------------------------------------------------
    nrm = norm(Δθ)
    if !isfinite(nrm)
        @warn "minSR: ‖Δθ‖ = $nrm is Inf/NaN — skipping update"
        return E_mean, variance, last_addr, acceptance, 0f0
    end

    scale_eu = nrm > max_norm ? max_norm / nrm : 1f0

    # --- 3. relative parameter-space guard ------------------------------------
    ρ = minSR_buf.ρ
    θn = norm(jacobian_buf.θ)
    ρ_eff = s_drop ? 0.1f0 * ρ : ρ
    scale_rel = min(1f0, ρ_eff * θn / (η * nrm + eps(Float32)))

    # --- effective step -------------------------------------------------------
    η_eff = η * min(scale_fs, scale_eu, scale_rel)

    # --- diagnostics ----------------------------------------------------------
    # s_val = sq^2
    # scales = (scale_fs, scale_eu, scale_rel)
    # limiter = minimum(scales) == 1f0 ? "-" : ("FS", "EU", "REL")[argmin(scales)]
    # @printf("minSR |   η_eff=%.2e  (FS=%.3e, EU=%.3e, REL=%.3e)  s=%.3e  d=%.2e  d_taken=%.2e  lim=%s\n",
    #         η_eff, scale_fs, scale_eu, scale_rel, s_val, d, η_eff * v, limiter)

    θ = jacobian_buf.θ
    @. θ = θ - η_eff * Δθ
    update!(ansatz, jacobian_buf, θ)          # push weights into the NN model

    return E_mean, variance, last_addr, acceptance, η_eff
end

