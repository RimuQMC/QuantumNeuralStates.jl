
using LinearAlgebra
using Statistics
using QuantumNeuralStates
using Rimu
using Test

@testset "Helpers" begin
    T = Float32
    M = 3
    N = 3
    batch = 3
    addr = near_uniform(BoseFS{N,M});
    H = HubbardReal1D(addr; u=0.1);
    addrs_n = fill(addr, batch)
    addrs_m = Vector{typeof(addr)}(undef, batch)

    flat_addrs = typeof(addr)[]
    distro = []
    col = H*addrs_n[1]
    for (addr_m, H_mn) in offdiagonals(col)
        push!(flat_addrs, addr_m)
        push!(distro, 1.0)
    end
    offsets_all = [0, num_offdiagonals(col)]

    bool = Bool[]
    for i in 1:100
        QuantumNeuralStates._state_proposal!(offsets_all, flat_addrs, distro, addrs_m, 1)
        if addrs_m[1] in flat_addrs
            push!(bool, true)
        else
            push!(bool, false)
        end
    end
    @test all(bool) === true

    psi_n = [0.2, 10.2, 30.9]
    weights = []
    for i in 1:length(psi_n)
        QuantumNeuralStates.get_ctmc_weights!(distro, offsets_all, psi_n, 1)
        push!(weights, psi_n[i])
    end

    @test weights[3] > weights[2] > weights[1]
end
@testset "Samplers and VMC energy" begin
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

    @test_nowarn vmc_sample!(:metropolis, vmc_buf, jac_buf, H, addrs_n, ansatz);
    @test_nowarn vmc_sample!(:metropolis_heatbath, vmc_buf, jac_buf, H, addrs_n, ansatz);
    @test_nowarn vmc_sample!(:ctmc, vmc_buf, jac_buf, H, addrs_n, ansatz);
    @test_nowarn vmc_sample!(:ctmc_heatbath, vmc_buf, jac_buf, H, addrs_n, ansatz);

    @test_nowarn vmc_energy(H, ansatz, addrs_n, vmc_buf, jac_buf; 
                            vmc_sampler=:metropolis, burnin=3, mode=:energy);
    @test_nowarn vmc_energy(H, ansatz, addrs_n, vmc_buf, jac_buf; 
                            vmc_sampler=:metropolis_heatbath, burnin=3, mode=:variance);
    @test_nowarn vmc_energy(H, ansatz, addrs_n, vmc_buf, jac_buf; 
                            vmc_sampler=:ctmc, burnin=3, mode=:std);
    @test_nowarn vmc_energy(H, ansatz, addrs_n, vmc_buf, jac_buf; 
                            vmc_sampler=:ctmc_heatbath, burnin=3, mode=:logvar);
end

