
using LinearAlgebra
using Statistics
using QuantumNeuralStates
using Rimu
using Test

@testset "NeuralAnsatz" begin
    T = Float32
    M, N, batch = 3, 1, 3
    addr  = near_uniform(BoseFS{N,M})
    H     = HubbardReal1D(addr; u=0.1)
    enc   = OccupationEncoding((M,), H)
    model = build_model("FCNN", enc, [10, 10, 10, 1], tanh_fast; batch=batch)
    ansatz = NeuralAnsatz(LogPsi(), H, model, batch)

    v1, v2, v3 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    x     = T[v1 v2 v3]                                  # (M, batch), one column per address
    addrs = [BoseFS(v1), BoseFS(v2), BoseFS(v3)]

    @testset "construction" begin
        @test ansatz.hamiltonian === H
        @test ansatz.model === model
        @test size(ansatz.x_cpu_buffer) == (M, batch)
        @test size(ansatz.z_cpu) == (1, batch)
        @test eltype(ansatz.z_cpu) == Float64
        @test isfinite(ansatz.logψ_centering)
        @test ansatz.normalisation == 1f0                # no max_norm
        @test ansatz.multi_forward_buffer === nothing
        @test ansatz.meanfield === nothing
        @test isempty(ansatz.addrs_buffer) && isempty(ansatz.result_buffer)
        @test ansatz.first_iter
        @test occursin("NeuralAnsatz", sprint(show, MIME"text/plain"(), ansatz))
    end

    @testset "construction errors" begin
        two_out = build_model("FCNN", enc, [10, 2], tanh_fast; batch=batch)
        @test_throws AssertionError NeuralAnsatz(LogPsi(), H, two_out, batch)   # output count
        @test_throws ErrorException NeuralAnsatz(LogPsi(), H, model, batch; max_norm=0)
        @test_throws ErrorException NeuralAnsatz(LogPsi(), H, model, batch; max_norm=-1)
        plain = Chain(NoEncoding((M,)), Dense(M=>1, identity; batch=batch); batch=batch)
        @test_throws ErrorException NeuralAnsatz(LogPsi(), H, plain, batch)
    end

    @testset "input preparation" begin
        prepare_input!(ansatz, addrs, ansatz.x_cpu_buffer)
        @test ansatz.x_cpu_buffer == x

        prepare_input!(ansatz, addrs[2], ansatz.x_cpu_buffer)          # single address
        @test all(ansatz.x_cpu_buffer[:, b] == T.(v2) for b in 1:batch)

        prepare_input_occ!(ansatz, addrs, ansatz.x_cpu_buffer)
        @test ansatz.x_cpu_buffer == x
    end

    @testset "input scaling and normalisation" begin
        a_sqrt = NeuralAnsatz(LogPsi(), H, model, batch; input_scale_func=sqrt)
        prepare_input!(a_sqrt, [BoseFS([2, 0, 0])], a_sqrt.x_cpu_buffer)
        @test a_sqrt.x_cpu_buffer[1, 1] ≈ sqrt(2f0)

        a_norm = NeuralAnsatz(LogPsi(), H, model, batch; max_norm=4)
        @test a_norm.normalisation ≈ 1/4
        prepare_input!(a_norm, [BoseFS([2, 0, 0])], a_norm.x_cpu_buffer)
        @test a_norm.x_cpu_buffer[1, 1] ≈ 0.5f0
    end

    @testset "compute_logψ" begin
        output_old = copy(ansatz.model(x))
        output     = compute_logψ(ansatz, addrs)
        @test output == output_old
        @test size(output) == (1, batch)

        single = Array(compute_logψ(ansatz, addrs[1]))                 # same address in all columns
        @test all(single[1, b] == single[1, 1] for b in 2:batch)
        @test single[1, 1] ≈ output_old[1, 1]
    end

    # @testset "compute_ψ_64 and call operator" begin
    #     logψ = Float64.(Array(compute_logψ(ansatz, addrs)))
    #     ψ    = compute_ψ_64(ansatz, addrs)
    #     @test eltype(ψ) == Float64
    #     @test ψ ≈ exp.(logψ)
    #     @test ansatz(addrs) ≈ exp.(logψ)
    # end

    @testset "multi_compute_logψ!" begin
        ref   = Array(compute_logψ(ansatz, addrs))
        many  = vcat(addrs, addrs, [addrs[1]])                         # 7 addresses, 3 chunks

        out = T[]
        multi_compute_logψ!(ansatz, many, out)
        @test length(out) == length(many)
        @test out ≈ vcat(ref[1, :], ref[1, :], ref[1, 1:1])            # order preserved

        # same result through a MultiForwardBuffer with a different chunk size
        a_mfb = NeuralAnsatz(LogPsi(), H, model, batch; multiforward_buffer=2)
        @test a_mfb.multi_forward_buffer !== nothing
        @test a_mfb.multi_forward_buffer.buffer_size == 2
        out_mfb = T[]
        multi_compute_logψ!(a_mfb, many, out_mfb)
        @test out_mfb ≈ out
    end

    @testset "wave-function helpers" begin
        output = copy(compute_logψ(ansatz, addrs))
        multi_output = T[]
        multi_compute_logψ!(ansatz, addrs, multi_output)
        @test_nowarn psi(ansatz, multi_output)
        @test_nowarn log_psi!(ansatz, output)
        @test QuantumNeuralStates.init_gradient_seed(ansatz) == ones(T, 1, batch)
    end
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
    addr = BoseFS{missing}{M}()
    H = FroehlichPolaron(addr; l=3.0, alpha=2, mode_cutoff=N)
    enc = OccupationEncoding((M,), H)
    model = build_model("FCNN", enc, [M, 10, 10, 10, 1], tanh_fast; batch=batch)

    ansatz = NeuralAnsatz(LogPsi(), H, model, batch; truncation=1) 

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
