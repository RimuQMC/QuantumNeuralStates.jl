
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
    model = build_model("FCNN", [M, 10, 10, 10, 1], tanh_fast; batch=batch)
    addr = near_uniform(BoseFS{N,M});
    H = HubbardReal1D(addr; u=0.1);
    ansatz = NeuralAnsatz(LogPsi(), H, model, batch)

    addrs_n = fill(addr, batch)

    @testset "Save weights" begin
        file = joinpath(tmpdir, "weights.txt")
        θ = randn(T, 100)
        @test_nowarn QuantumNeuralStates.save_weights(file, θ)
        rm(file; force=true)
    end
    @testset "Save addrs" begin
        file = joinpath(tmpdir, "addrs.txt")
        @test_nowarn QuantumNeuralStates.save_addrs(file, addrs_n)
        rm(file; force=true)
    end
    @testset "Save input scaling" begin
        file = joinpath(tmpdir, "input_scale.txt")
        @test_nowarn QuantumNeuralStates.save_input_scale(file, ansatz)
        rm(file; force=true)
    end
    @testset "Save/Load all at once -> master save/load" begin
        file = joinpath(tmpdir, "all.txt")
        
        buffers = map(DenseBuffer, ansatz.model.layers)
        jac_buf = JacobianBuffer(ansatz, buffers)
        x = prepare_input!(ansatz, addrs_n, ansatz.x_cpu_buffer)

        save_master(file, jac_buf.θ, addrs_n, ansatz)

        # create new network with new parameters
        model2 = build_model("FCNN", [M, 10, 10, 10, 1], tanh_fast; batch=batch)
        ansatz2 = NeuralAnsatz(LogPsi(), H, model2, batch)
        buffers2 = map(DenseBuffer, ansatz2.model.layers)
        
        x2 = load_master(ansatz2, file)
        jac_buf2 = JacobianBuffer(ansatz2, buffers2)

        
        @test x == x2
        @test jac_buf.θ == jac_buf2.θ
        @test ansatz.input_scale_func == ansatz2.input_scale_func
        @test ansatz.max_norm == ansatz2.max_norm
        @test ansatz.normalisation == ansatz2.normalisation

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
        buffers = map(DenseBuffer, ansatz.model.layers)
        jac_buf = JacobianBuffer(ansatz, buffers)
        grads_n = back_jacobian!(ansatz, jac_buf)  # do 1 backpropagation pass
        @test_nowarn jacobian_statistics(ansatz, jac_buf.J)
    end

    rm(file_neuron; force=true)
    rm(file_jacobian; force=true)
end




