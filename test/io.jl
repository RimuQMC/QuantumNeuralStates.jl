
using LinearAlgebra
using Statistics
using QuantumNeuralStates
using Rimu
using Test


const tmpdir = mktempdir()

@testset "Saves / Loads" begin
    T = Float32
    M = 3
    N = 3
    batch = 3

    @testset "log_markov_chain" begin
        addr = near_uniform(BoseFS{N,M})
        addrs = fill(addr, batch)
        expected_line = join([QuantumNeuralStates._format_addr(Int.(onr(addr)); w=3) for _ in 1:batch], " | ")

        file = joinpath(tmpdir, "chain.txt")
        rm(file; force=true)   # ensure a clean start regardless of test order/reruns

        @testset "start=true: creates the file, writes one line" begin
            log_markov_chain(file, addrs; start=true)
            @test isfile(file)
            lines = readlines(file)
            @test length(lines) == 1
            @test lines[1] == expected_line
        end

        @testset "start=false: appends a second line, doesn't overwrite the first" begin
            log_markov_chain(file, addrs; start=false)
            lines = readlines(file)
            @test length(lines) == 2
            @test lines[1] == expected_line   # first line untouched
            @test lines[2] == expected_line
        end

        @testset "start=true again: truncates back to one line" begin
            log_markov_chain(file, addrs; start=true)
            lines = readlines(file)
            @test length(lines) == 1   # "w" mode wiped the previous appended line
        end

        rm(file; force=true)
    end

    @testset "Save/Load all at once -> master save/load" begin
        model = build_model("FCNN", [M, 10, 10, 10, 1], tanh_fast; batch=batch)
        addr = near_uniform(BoseFS{N,M})
        H = HubbardReal1D(addr; u=0.1)
        ansatz = NeuralAnsatz(LogPsi(), H, model, batch)
        addrs_n = fill(addr, batch)

        file = joinpath(tmpdir, "all.txt")
        rm(file; force=true)

        buffers = make_buffers(ansatz.model)
        jac_buf = JacobianBuffer(ansatz, buffers)
        x = prepare_input!(ansatz, addrs_n, ansatz.x_cpu_buffer)

        save_master(file, jac_buf.θ, addrs_n, ansatz)
        @test isfile(file)

        # create new network with new (random-init) parameters
        model2 = build_model("FCNN", [M, 10, 10, 10, 1], tanh_fast; batch=batch)
        ansatz2 = NeuralAnsatz(LogPsi(), H, model2, batch)
        buffers2 = make_buffers(ansatz2.model)

        x2 = load_master(ansatz2, file)
        jac_buf2 = JacobianBuffer(ansatz2, buffers2)

        @test QuantumNeuralStates.chain_signature(ansatz.model) == QuantumNeuralStates.chain_signature(ansatz2.model)
        @test x == x2
        @test jac_buf.θ == jac_buf2.θ
        @test ansatz.input_scale_func == ansatz2.input_scale_func
        @test ansatz.max_norm == ansatz2.max_norm
        @test ansatz.normalisation == ansatz2.normalisation

        @testset "architecture mismatch is caught" begin
            model3 = build_model("FCNN", [M, 20, 20, 1], tanh_fast; batch=batch)   # different hidden widths
            ansatz3 = NeuralAnsatz(LogPsi(), H, model3, batch)
            @test_throws ErrorException load_master(ansatz3, file)
        end

        rm(file; force=true)
    end
end
@testset "Neural Network Statistics" begin
    T = Float32
    M = 3
    N = 3
    batch = 3
    model = build_model("FCNN", [M, 10, 10, 10, 1], tanh_fast; batch=batch)
    addr = near_uniform(BoseFS{N,M});
    H = HubbardReal1D(addr; u=0.1);
    file_neuron = joinpath(tmpdir, "neuron.txt")
    file_jacobian = joinpath(tmpdir, "jacobian.txt")
    ansatz = NeuralAnsatz(LogPsi(), H, model, batch; 
                          neuron_statistics=file_neuron, jacobian_statistics=file_jacobian)
    addrs_n = fill(addr, batch)

    compute_logψ(ansatz, addrs_n) # fill neural network with 1 forward pass

    @testset "Neuron statistics" begin
        @test_nowarn neuron_statistics(ansatz)
    end
    @testset "Jacobian statistics" begin
        buffers = make_buffers(ansatz.model)
        jac_buf = JacobianBuffer(ansatz, buffers)
        grads_n = back_jacobian!(ansatz, jac_buf)  # do 1 backpropagation pass
        @test_nowarn jacobian_statistics(ansatz, jac_buf.J)
    end

    rm(file_neuron; force=true)
    rm(file_jacobian; force=true)
end




