
using LinearAlgebra
using Statistics
using QuantumNeuralStates
using Rimu
using Test

@testset "CG solver" begin
    T = Float32
    N = 5
    p = 10
    x = Vector{T}(undef, N) # result of CG
    J = randn(T, p, N)
    J1 = copy(J)
    λ = 1f-3
    g = randn(T, N)
    g1 = copy(g)
    p_cg = Vector{T}(undef, N)
    Ap = Vector{T}(undef, N)
    Δθ = zeros(T, p)

    cg_solve!(x, J, λ, g, p_cg, Ap, Δθ)
    
    S = J1' * J1 + λ*I(N)
    x1 = S \ g1

    @test isapprox(x, x1; atol=1e-5)
end
@testset "Optimisers" begin
    T = Float32
    M = 3
    N = 3
    batch = 3
    model  = build_model("FCNN", [M, 10, 10, 10, 1], tanh_fast; batch=batch)
    addr = near_uniform(BoseFS{N,M});
    H = HubbardReal1D(addr; u=0.1);
    ansatz = NeuralAnsatz(LogPsi(), H, model, batch)

    buffers = map(DenseBuffer, ansatz.model.layers)
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
