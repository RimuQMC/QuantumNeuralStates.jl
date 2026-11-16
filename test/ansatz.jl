
using LinearAlgebra
using Statistics
using QuantumNeuralStates
using Rimu
using Test

@testset "NeuralAnsatz" begin
    T = Float32
    M, N, batch = 3, 3, 3
    addr = BoseFS{missing}{M}()
    H = FroehlichPolaron{Float32}(addr; D=1, l=3.0, alpha=2, mode_cutoff=N)
    enc = OccupationEncoding((M,), H)
    model = build_model("FCNN", enc, [M, 10, 10, 10, 1], tanh; batch=batch)
    ansatz = NeuralAnsatz(LogPsi(), H, model, batch)

    v1, v2, v3 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    x = T[v1 v2 v3]                                  # (M, batch), one column per address
    addrs = [BoseFS(v1), BoseFS(v2), BoseFS(v3)]

    @testset "construction" begin
        @test ansatz.hamiltonian === H
        @test ansatz.model === model
        @test size(ansatz.model.x) == (M, batch)
        @test size(ansatz.model.z_last) == (1, batch)
        @test isfinite(ansatz.logψ_centering)
        @test ansatz.normalisation == 1f0                # no max_norm
        @test ansatz.multi_forward_buffer === nothing
        @test ansatz.meanfield === nothing
        @test isempty(ansatz.addrs_buffer) && isempty(ansatz.result_buffer)
        @test ansatz.first_iter
        @test occursin("NeuralAnsatz", sprint(show, MIME"text/plain"(), ansatz))
    end

    @testset "construction errors" begin
        two_out = build_model("FCNN", enc, [M, 10, 2], tanh; batch=batch)
        @test_throws AssertionError NeuralAnsatz(LogPsi(), H, two_out, batch)   # output count
        @test_throws ErrorException NeuralAnsatz(LogPsi(), H, model, batch; max_norm=0)
        @test_throws ErrorException NeuralAnsatz(LogPsi(), H, model, batch; max_norm=-1)
        plain = Chain(NoEncoding((M,)), Dense(M=>1, identity; batch=batch); batch=batch)
        @test_throws ErrorException NeuralAnsatz(LogPsi(), H, plain, batch)
    end

    @testset "input preparation" begin
        prepare_input!(ansatz, addrs, ansatz.model.x)
        @test ansatz.model.x == x

        prepare_input!(ansatz, addrs[2], ansatz.model.x)          # single address
        @test all(ansatz.model.x[:, b] == T.(v2) for b in 1:batch)
    end

    @testset "input scaling and normalisation" begin
        a_sqrt = NeuralAnsatz(LogPsi(), H, model, batch; input_scale_func=sqrt)
        prepare_input!(a_sqrt, [BoseFS([2, 0, 0])], a_sqrt.model.x)
        @test a_sqrt.model.x[1, 1] ≈ sqrt(2f0)

        a_norm = NeuralAnsatz(LogPsi(), H, model, batch; max_norm=4)
        @test a_norm.normalisation ≈ 1/4
        prepare_input!(a_norm, [BoseFS([2, 0, 0])], a_norm.model.x)
        @test a_norm.model.x[1, 1] ≈ 0.5f0
    end

    @testset "compute_logψ" begin
        output = copy(compute_logψ(ansatz, addrs))
        output_old = copy(ansatz.model(x))
        @test output == output_old
        @test size(output) == (1, batch)

        single = Array(compute_logψ(ansatz, addrs[1]))                 # same address in all columns
        @test all(single[1, b] == single[1, 1] for b in 2:batch)
        @test single[1, 1] ≈ output_old[1, 1]
    end

    @testset "multi_compute_logψ!" begin
        KA = QuantumNeuralStates.KernelAbstractions
        backend = KA.get_backend(ansatz.model.x)

        counts = [2, 3, 2]                           # valid off-diagonals per column
        K, B = maximum(counts), length(counts)
        offsets = cumsum(counts)                      # [2, 5, 7]
        total = offsets[end]                        # fwd = 3 → 3 passes, last one partial

        addrs_mat = reshape([addrs[mod1(i, length(addrs))] for i in 1:K*B], K, B)
        stream = [addrs_mat[k, b] for b in 1:B for k in 1:counts[b]]
        ref = [Array(compute_logψ(ansatz, a))[1, 1] for a in stream]

        addrs_buf = GPUGrowRowBuffer(backend, eltype(addrs_mat), K, (B,))
        copyto!(addrs_buf.data, addrs_mat)
        vals_buf = GPUGrowColumnBuffer(backend, T, (1,), 0)

        vals = multi_compute_logψ!(ansatz, addrs_buf, vals_buf, offsets)
        @test vals_buf.capacity >= total              # grown
        @test Array(view(vals, 1, 1:total)) ≈ ref     # packed, order preserved

        # MultiForwardBuffer with a different chunk size (fwd = 2 → 4 passes)
        a_mfb = NeuralAnsatz(LogPsi(), H, model, batch; multi_forward_buffer=2)
        @test a_mfb.multi_forward_buffer.buffer_size == 2
        vals = multi_compute_logψ!(a_mfb, addrs_buf, vals_buf, offsets)
        @test Array(view(vals, 1, 1:total)) ≈ ref

        # no valid off-diagonals
        @test_nowarn multi_compute_logψ!(ansatz, addrs_buf, vals_buf, zeros(Int, B))
    end

    @testset "wave-function helpers" begin
        KA = QuantumNeuralStates.KernelAbstractions
        backend = KA.get_backend(ansatz.model.x)

        output = copy(compute_logψ(ansatz, addrs))

        B = length(addrs)
        addrs_buf = GPUGrowRowBuffer(backend, eltype(addrs), 1, (B,))   # K = 1
        copyto!(addrs_buf.data, reshape(addrs, 1, B))
        vals_buf = GPUGrowColumnBuffer(backend, T, (1,), 0)

        vals = multi_compute_logψ!(ansatz, addrs_buf, vals_buf, collect(1:B))
        multi_output = copy(view(vals, :, 1:B))
        @test multi_output == output                   # stream path == direct pass

        lg, s = @test_nowarn log_psi!(ansatz, output)
        @test all(isfinite, lg)
        ps1 = @test_nowarn psi!(ansatz, lg, s)
        @test all(isfinite, ps1)
        ps2 = psi!(ansatz, multi_output)
        @test ps1 == ps2

        seed = ones(T, 1, batch)
        @test QuantumNeuralStates.init_gradient_seed!(ansatz, seed) == ones(T, 1, batch)
    end
end

@testset "MeanField" begin
    T = Float32
    M = 3
    N = 10
    batch = 3
    addr = BoseFS{missing, M}() 
    H = FroehlichPolaron{T}(addr; D = 1, alpha = 1, l = 5, mode_cutoff=N)

    enc = OccupationEncoding((3,), H)
    model = build_model("FCNN", enc, [M, 10, 10, 10, 1], tanh; batch=batch)

    ansatz = NeuralAnsatz(LogPsi(), H, model, batch; mean_field=true) 

    @test ansatz.meanfield isa MeanField

    addrs = fill(addr, batch)
    out = compute_logψ(ansatz, addrs)
    lg, s = log_psi!(ansatz, out)
    lg0 = copy(lg)
    ansatz.meanfield(addrs, lg)
    @test all(isfinite, lg)
    @test lg != lg0          # mean field actually added something

end

@testset "Truncation" begin
    T = Float32
    M = 3
    N = 2
    batch = 3
    addr = BoseFS{missing}{M}()
    H = FroehlichPolaron{T}(addr; D=1, l=3.0, alpha=2, mode_cutoff=N)
    enc = OccupationEncoding((M,), H)
    model = build_model("FCNN", enc, [M, 10, 10, 10, 1], tanh; batch=batch)

    ansatz = NeuralAnsatz(LogPsi(), H, model, batch; truncation=1) 

    @test ansatz.truncation isa TruncationBuffer

    addr1 = BoseFS{missing}((0, 1, 0))
    addr2 = BoseFS{missing}((0, 2, 0))
    addr3 = BoseFS{missing}((1, 1, 0))
    addr4 = BoseFS{missing}((1, 0, 2))

    @test !violates_truncation(onr(addr1), ansatz.truncation.mask)
    @test !violates_truncation(onr(addr2), ansatz.truncation.mask)
    @test violates_truncation(onr(addr3), ansatz.truncation.mask)
    @test violates_truncation(onr(addr4), ansatz.truncation.mask)
end
