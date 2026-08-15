
using LinearAlgebra
using Statistics
using QuantumNeuralStates
using Rimu
using Test


@testset "NeuralAnsatz" begin
    T = Float32
    M = 3
    N = 1
    batch = 3
    model  = build_model("FCNN", [M, 10, 10, 10, 1], tanh_fast; batch=batch)
    addr = near_uniform(BoseFS{N,M});
    H = HubbardReal1D(addr; u=0.1);
    ansatz  = NeuralAnsatz(LogPsi(), H, model, batch)

    v1, v2, v3 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    x = zeros(T, 3, 3)
    x[:, 1] .= T.(v1); x[:, 2] .= T.(v2); x[:, 3] = T.(v3)
    addrs = [BoseFS(v1), BoseFS(v2), BoseFS(v3)]

    prepare_input!(ansatz, addrs, ansatz.x_cpu_buffer)
    @test ansatz.x_cpu_buffer == x
    prepare_input_occ!(ansatz, addrs, ansatz.x_cpu_buffer)
    @test ansatz.x_cpu_buffer == x

    output = ansatz.model(x)
    output_old = copy(output)
    output = compute_logψ(ansatz, addrs)
    @test output == output_old

    push!(addrs, BoseFS(v1))
    multi_output = T[]
    multi_compute_logψ!(ansatz, addrs, multi_output)
    @test view(multi_output, 1:3) == view(output, 1, :)

    @test_nowarn psi(ansatz, multi_output)
    @test_nowarn log_psi!(ansatz, output)

    init_grad_seed = QuantumNeuralStates.init_gradient_seed(ansatz)
    seed_check = ones(T, 1, batch)
    @test seed_check == init_grad_seed
end
# @testset "MeanField" begin
#     # MeanFields is defined for FroehlichPolaronND only, cannot be tested now
#
#     # T = Float32
#     # M = 3
#     # N = 1
#     # batch = 3
#     # model  = build_model("FCNN", [M, 10, 10, 10, 1], tanh_fast; batch=batch)
#     #
#     # addr    = BoseFS{missing}{M}()
#     # H       = FroehlichPolaron(addr; l=3.0, v=1.155, mode_cutoff=N)
#     # ansatz  = NeuralAnsatz(LogPsi(), H, model, batch; mean_field=true) 
#     #
#     # @test ansatz.mean_field isa MeanField
# end
@testset "Truncation" begin
    T = Float32
    M = 3
    N = 2
    batch = 3
    model  = build_model("FCNN", [M, 10, 10, 10, 1], tanh_fast; batch=batch)

    addr    = BoseFS{missing}{M}()
    H       = FroehlichPolaron(addr; l=3.0, v=1.155, mode_cutoff=N)
    ansatz  = NeuralAnsatz(LogPsi(), H, model, batch; truncation=1) 

    @test ansatz.truncation isa TruncationBuffer

    addrs1 = BoseFS{missing}((0, 1, 0))
    addrs2 = BoseFS{missing}((0, 2, 0))
    addrs3 = BoseFS{missing}((1, 1, 0))
    addrs4 = BoseFS{missing}((1, 0, 2))

    @test !violates_truncation(onr(addrs1), ansatz.truncation.mask)
    @test !violates_truncation(onr(addrs2), ansatz.truncation.mask)
    @test violates_truncation(onr(addrs3), ansatz.truncation.mask)
    @test violates_truncation(onr(addrs4), ansatz.truncation.mask)
end
