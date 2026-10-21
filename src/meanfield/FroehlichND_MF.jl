# https://journals.aps.org/pr/pdf/10.1103/PhysRev.90.297
# https://arxiv.org/pdf/1510.04934

"""
    calc_omega_denominator(h::Rimu.FroehlichPolaron, kidx::Int, eta) -> ε_k

Energy denominator of the Lee–Low–Pines mean-field for phonon mode `kidx`,

```math
\\varepsilon_k = \\omega + \\frac{k^2}{2m} - \\frac{\\mathbf{k}\\cdot\\mathbf{P}}{m}\\,(1 - \\eta),
\\qquad m = \\frac{\\texttt{two\\_m}}{2},
```

where `P = h.p` is the total momentum and `η` the fraction of `P` carried by the
phonons, `P_ph = η P` (see [`solve_self_consistent_eta`](@ref)).

# Arguments
* `h`: Froehlich polaron Hamiltonian.
* `kidx`: index of the phonon mode, `h.ks[kidx]`.
* `eta`: phonon momentum fraction `η`.

See also [`calc_f`](@ref).
"""
function calc_omega_denominator(h::Rimu.FroehlichPolaron{T,M,D}, kidx::Int, eta::T) where {T,M,D}
    k = h.ks[kidx]
    k_dot_P = zero(T)
    k_squared = zero(T)
    for d in 1:D
        k_dot_P += T(k[d]) * T(h.p[d])
        k_squared += T(k[d])^2
    end

    two_m = T(h.two_m)
    # ε_k = ω + k²/2m − (k·p/m)(1 − η),   m = two_m/2
    return T(h.omega) + k_squared / two_m - T(2) * k_dot_P * (one(T) - eta) / two_m
end

"""
    calc_f(h::Rimu.FroehlichPolaron, eta) -> f

Coherent-state displacements of the Lee–Low–Pines mean-field,

```math
f_k = \\frac{v_k}{\\varepsilon_k},
```

with the coupling `v_k` from `Rimu.Hamiltonians.calc_vk` and the denominator `ε_k` from
[`calc_omega_denominator`](@ref). The sign follows the convention
`H = … − Σ_k v_k (a_k^† + a_k)`.

# Arguments
* `h`: Fröhlich polaron Hamiltonian.
* `eta`: phonon momentum fraction `η`.

# Returns
* `f`: (M,) one `f_i` per phonon mode; `f_k = 0` for modes with `v_k = 0`
    (the `k = 0` mode).

# Notes
* Throws an error if `ε_k ≤ 0` for any mode. The mean-field is then beyond its range of
    validity (the total momentum is too large for a bound polaron within this ansatz).
"""
function calc_f(h::Rimu.FroehlichPolaron{T,M,D}, eta::T) where {T,M,D}
    f = zeros(T, M)
    for i in 1:M
        vk = T(Rimu.Hamiltonians.calc_vk(h, i))
        iszero(vk) && continue      # k = 0 mode
        denom = calc_omega_denominator(h, i, eta)
        denom > zero(T) || error("calc_f: ε_k = $denom ≤ 0 for mode $i — beyond the mean-field threshold")
        f[i] = vk / denom           # + sign for H = … − Σ v_k (a† + a)
    end
    return f
end

"""
    solve_self_consistent_eta(h::Rimu.FroehlichPolaron; tol = sqrt(eps(T)), max_iter = 500)
            -> (eta, f)

Solve the Lee–Low–Pines self-consistency for the phonon momentum fraction `η` by fixed-point
iteration. The phonon momentum of the coherent state must equal the fraction `η` of the
total momentum,

```math
\\eta\\,\\mathbf{P} = \\sum_k \\mathbf{k}\\, f_k^2
\\quad\\Longrightarrow\\quad
\\eta = \\frac{1}{P^2}\\sum_k (\\mathbf{k}\\cdot\\mathbf{P})\\, f_k^2(\\eta) ,
```

with `f_k(η)` from [`calc_f`](@ref).

# Arguments
* `h`: Froehlich polaron Hamiltonian.

# Keyword Arguments
* `tol`: relative convergence tolerance on `η` (default `sqrt(eps(T))`, ≈ 3.5e-4 for
    `Float32` and ≈ 1.5e-8 for `Float64`).
* `max_iter`: maximal number of fixed-point iterations (default `500`).

# Returns
* `eta`: converged phonon momentum fraction.
* `f`: (M,) `f_k(η)` at the converged `η`.

# Notes
* For `P = 0` there is no preferred direction, so `η = 0` by symmetry and no iteration
    is performed.
* If the iteration does not converge within `max_iter`, a warning is issued and the last
    iterate is returned.

# References
* T. D. Lee, F. E. Low, D. Pines, Phys. Rev. 90, 297 (1953).
* arXiv:1510.04934.
"""
function solve_self_consistent_eta(h::Rimu.FroehlichPolaron{T,M,D};
                                     tol::T = sqrt(eps(T)), # ≈ 3.5e-4 (Float32), 1.5e-8 (Float64)
                                     max_iter::Int = 500) where {T,M,D}

    P_squared = zero(T)
    for d in 1:D
        P_squared += T(h.p[d])^2
    end
    # No total momentum -> no recoil direction -> eta trivially 0
    iszero(P_squared) && return zero(T), calc_f(h, zero(T))    # p = 0 → P_f = 0 by symmetry

    eta_old = zero(T)
    f = calc_f(h, eta_old)

    for iter in 1:max_iter
        numerator = zero(T)
        for i in 1:M
            k = h.ks[i]
            k_dot_P = zero(T)
            for d in 1:D
                k_dot_P += T(k[d]) * T(h.p[d])
            end
            numerator += k_dot_P * f[i]^2
        end

        eta_new = numerator / P_squared

        if abs(eta_new - eta_old) < tol * max(one(T), abs(eta_new))   # relative criterion
            return eta_new, calc_f(h, eta_new)
        end

        eta_old = eta_new
        f = calc_f(h, eta_old)
    end

    @warn "eta did not converge within max_iter"
    return eta_old, f
end

"""
    strong_coupling_amplitudes(h::Rimu.FroehlichPolaron) -> β

Strong-coupling coherent amplitudes, Eq. (4.6) - Yosef:

```math
\\beta_k = 2 V_k \\left(\\frac{\\lambda^2}{\\lambda^2 + k^2}\\right)^{3/2},
\\qquad \\lambda = \\frac{3\\pi\\alpha}{8} .
```

# Arguments
* `h`: Froehlich polaron Hamiltonian, with coupling constant `h.alpha`.

# Returns
* `β`: (M,) amplitudes, one per phonon mode; `β_k = 0` for modes with `V_k = 0`
    (the `k = 0` mode).

# Notes
* Not used by [`MeanField`](@ref) at the moment; the weak-coupling Lee–Low–Pines
    displacements from [`solve_self_consistent_eta`](@ref) are used instead.
"""
function strong_coupling_amplitudes(h::Rimu.FroehlichPolaron{T,M,D}) where {T,M,D}
    λ  = T(3π * h.alpha / 8)
    λ2 = λ^2
    β  = zeros(T, M)
    for i in 1:M
        vk = T(Rimu.Hamiltonians.calc_vk(h, i))
        iszero(vk) && continue                      # k = 0 mode
        k2 = zero(T)
        for d in 1:D
            k2 += T(h.ks[i][d])^2
        end
        β[i] = 2 * vk * (λ2 / (λ2 + k2))^(T(3) / 2)
    end
    return β
end

"""
    _mf_func_froehlich(o, params, T) -> T

Log-amplitude of the Froehlich coherent-state mean-field for occupation numbers `o`,

```math
\\log\\psi_{\\mathrm{MF}}(n) = \\sum_i n_i \\log|f_i| - \\tfrac{1}{2}\\sum_i \\log(n_i!)
                              - \\tfrac{1}{2}\\sum_i f_i^2 .
```

# Arguments
* `o`: occupation numbers `onr(addr)` of one configuration.
* `params`: `(log_f, lgf, const_term)` from [`_choose_meanfield_function`](@ref), where
    `lgf[n+1] = log(n!)` is a look-up table and `const_term = -0.5 Σ f²`.
* `T`: element type of the result.

# Notes
* Occupations with `n ≥ length(lgf)` use the Stirling formula
    `log(n!) ≈ n log n − n + ½ log(2πn) + 1/(12n)`.
* Called inside [`_mf_addrs_kernel!`](@ref) and [`_mf_offdiag_kernel!`](@ref).
"""
@inline function _mf_func_froehlich(o, params, ::Type{T}) where T
    log_f, lgf, const_term = params
    NT  = Int32(length(lgf))
    acc = zero(T)
    @inbounds for i in 1:length(o)
        ni = Int32(o[i])
        x  = T(ni)
        acc += x * log_f[i]                         # Σ n_i log f_i
        lg = ni < NT ? lgf[ni + Int32(1)] :         # log(n_i!) from table
             x * log(x) - x + T(0.5) * log(T(2π) * x) + inv(T(12) * x)    # Stirling
        acc -= T(0.5) * lg
    end
    return acc + T(const_term)
end
