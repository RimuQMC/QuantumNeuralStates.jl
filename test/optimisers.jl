
using LinearAlgebra
using Statistics
using QuantumNeuralStates
using Rimu
using Test

@testset "CG solver" begin
    T = Float32
    N, p = 5, 10
    λ = 1f-3

    # same regime as in compute_minSR_cg!: mean eigenvalue 1, ‖b‖ = 1
    J = randn(T, p, N)
    J ./= sqrt(dot(J, J) / N)
    g = randn(T, N)
    g ./= norm(g)
    J1, g1 = copy(J), copy(g)

    x = Vector{T}(undef, N)     # CG solution
    p_cg = Vector{T}(undef, N)
    Ap = Vector{T}(undef, N)
    Δθ = zeros(T, p)           # scratch (Δθ in production)

    ok, iters, relres = cg_solve!(x, J, λ, g, p_cg, Ap, Δθ)

    A = J1' * J1 + λ * I(N)
    @test ok
    @test norm(A * x - g1) / norm(g1) ≤ relres + 1f-5     # reported residual is honest
    x1 = A \ g1
    @test isapprox(x, x1; rtol = 1e-2)      # within the CG tolerance
end
@testset "Optimisers" begin
    T = Float32
    M = 3
    N = 3
    batch = 3
    addr = BoseFS{missing, M}()
    H = FroehlichPolaron{Float32}(addr; D = 1, alpha = 2, l = 5, mode_cutoff=N)

    enc = OccupationEncoding((M,), H)
    model  = build_model("FCNN", enc, [M, 10, 10, 10, 1], tanh_fast; batch=batch)
    ansatz = NeuralAnsatz(LogPsi(), H, model, batch)

    buffers = make_buffers(ansatz.model)
    jac_buf = JacobianBuffer(ansatz, buffers)
    vmc_buf = VMCBuffer(ansatz, addr)
    addrs_n = fill(addr, batch)
    n_params = length(jac_buf.θ)

    @testset "Descent" begin
        descent_buf = DescentBuffer(batch, n_params, ansatz)
        @test_nowarn descent(jac_buf, vmc_buf, descent_buf, H, ansatz, addrs_n; burnin=3)
    end
    @testset "Adam" begin
        adam_buf = AdamBuffer(batch, n_params, ansatz)
        @test_nowarn adam(jac_buf, vmc_buf, adam_buf, H, ansatz, addrs_n; burnin=3)
    end
    @testset "minSR" begin
        minSR_buf = minSRBuffer(batch, n_params, ansatz)
        @test_nowarn minSR(jac_buf, vmc_buf, minSR_buf, H, ansatz, addrs_n; burnin=3)
    end
    @testset "minSR + momentum" begin
        minSR_buf = minSRBuffer(batch, n_params, ansatz; velocity=true)
        @test_nowarn minSR(jac_buf, vmc_buf, minSR_buf, H, ansatz, addrs_n; burnin=3)
    end
end
