# https://journals.aps.org/pr/pdf/10.1103/PhysRev.90.297
# https://arxiv.org/pdf/1510.04934

# function calc_omega_denominator(h::Rimu.FroehlichPolaronND{T,M,D}, kidx::Int, eta::Float64) where {T,M,D}
function calc_omega_denominator(h::Rimu.FroehlichPolaron{T,M}, kidx::Int, eta::Float64) where {T,M}
    k = h.ks[kidx]
    k_dot_P = 0.0
    k_squared = 0.0
    for d in 1:D
        k_dot_P += k[d] * h.p[d]
        k_squared += k[d]^2
    end

    denom = h.omega - (k_dot_P / h.mass) * (1.0 - eta) + k_squared / (2.0 * h.mass)
    return denom
end

# function calc_f(h::Rimu.FroehlichPolaronND{T,M,D}, eta::Float64) where {T,M,D}
function calc_f(h::Rimu.FroehlichPolaron{T,M}, eta::Float64) where {T,M}
    f = zeros(Float64, M)
    for i in 1:M
        vk = Rimu.Hamiltonians.calc_vk(h, i)
        if vk == 0.0
            f[i] = 0.0
        else
            denom = calc_omega_denominator(h, i, eta)
            f[i] = -vk / denom
        end
    end
    return f
end

# function solve_self_consistent_eta(h::Rimu.FroehlichPolaronND{T,M,D};
function solve_self_consistent_eta(h::Rimu.FroehlichPolaron{T,M};
                                     tol::Float64 = 1e-10,
                                     max_iter::Int = 500) where {T,M}
    D = 1 # EXTRA!

    P_squared = 0.0
    for d in 1:D
        P_squared += h.p[d]^2
    end

    if P_squared == 0.0
        # No total momentum -> no recoil direction -> eta trivially 0
        f = calc_f(h, 0.0)
        return 0.0, f
    end

    eta_old = 0.0
    f = calc_f(h, eta_old)

    for iter in 1:max_iter
        numerator = 0.0
        for i in 1:M
            k = h.ks[i]
            k_dot_P = 0.0
            for d in 1:D
                k_dot_P += k[d] * h.p[d]
            end
            numerator += k_dot_P * f[i]^2
        end

        eta_new = numerator / P_squared

        if abs(eta_new - eta_old) < tol
            f = calc_f(h, eta_new)
            return eta_new, f
        end

        eta_old = eta_new
        f = calc_f(h, eta_old)
    end

    @warn "eta did not converge within max_iter"
    return eta_old, f
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
