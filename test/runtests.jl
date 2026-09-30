
using LinearAlgebra
using Rimu
using QuantumNeuralStates
using SafeTestsets
using Test
using ExplicitImports: check_no_implicit_imports


# @safetestset "ExplicitImports" begin
#     using QuantumNeuralStates
#     using ExplicitImports
#     @test check_no_implicit_imports(
#         QuantumNeuralStates; skip=(QuantumNeuralStates, Base, Core)
#     ) === nothing
#     # If this test fails, make your import statements explicit.
#     # For example, replace `using Foo` with `using Foo: bar, baz`.
# end

@safetestset "NeuralNetwork" begin
    include("neuralnetwork.jl")
end

@safetestset "InputEncoding" begin
    include("input_encoding.jl")
end

@safetestset "Ansatz" begin
    include("ansatz.jl")
end

@safetestset "VMC" begin
    include("vmc.jl")
end

@safetestset "Optimisers" begin
    include("optimisers.jl")
end

@safetestset "IO" begin
    include("io.jl")
end

@testset "Main training loop" begin
    N = 5 # number of particles
    M = 5 # number of sites
    addr = near_uniform(BoseFS{N,M})
    H = HubbardReal1D(addr; u=0.1)
    batch  = 10

    conv1 = Conv((3,), 1=>8, relu; batch=batch, pad=Periodic())
    conv2 = Conv((3,), 8=>8, gelu; batch=batch, pad=Periodic())
    pool  = Pool(:mean)
    dense = Dense(8=>8, tanh; batch=batch)
    dense = Dense(8=>1, identity; batch=batch)
    enc = OccupationEncoding((M,), H)
    model = Chain(enc, conv1, conv2, pool, dense; batch=batch)

    phases = [
        TrainingPhase(
            mode       = :energy,
            optimiser  = :adam,
            vmc_sampler= :metropolis,
            stop       = StopBuffer(var_thr=1000),
            η          = 0.001f0,
            skip       = [(1, 3)],  # (epoch, B) → burnin B
            block_size = 10, 
            block_min  = 6, 
            patience   = 3,
            max_epochs = 10,
           )
    ]

    ansatz = NeuralAnsatz(LogPsi(), H, model, batch)

    run_training_loop(H, ansatz, addr, phases; save=false)
    @test true
end
