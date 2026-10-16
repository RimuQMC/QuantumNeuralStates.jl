# https://journals.aps.org/pr/pdf/10.1103/PhysRev.90.297
# https://arxiv.org/pdf/1510.04934

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
    strong_coupling_amplitudes(h) -> β

Strong-coupling coherent amplitudes, Eq. (4.6) - Yosef:
β_k = 2 V_k (λ²/(λ² + k²))^{3/2},  λ = 3πα/8.
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


function forward_logMF_FroehlichND!(result::AbstractArray, input::AbstractArray, params)
    log_f_row, const_term = params
    M, B = size(input)

    # Term 2: result (1,B) = log_f_row (1,M) * input (M,B)
    mul!(result, log_f_row, input, 1.0, 1.0)

    # Term 3: subtract 0.5 * sum_i loggamma(n_i + 1) per column, no temp array
    @inbounds for b in 1:B
        s = 0.0
        for i in 1:M
            s += loggamma(input[i, b] + 1)
        end
        result[1, b] -= 0.5 * s
    end

    # Add the fixed constant to every column
    result .+= const_term

    return result
end
