using LinearAlgebra
using Statistics
using QuantumNeuralStates
using Rimu
using Test

@testset "Layers" begin
    T = Float32
    @testset "Initialisation" begin
        in_dim = 10
        out_dim = 3
        act_func1 = tanh
        act_func2 = relu
        act_func3 = (identity, tanh)

        @test QuantumNeuralStates._init_std(T, in_dim, out_dim, act_func1) == T(sqrt(2.0/(in_dim+out_dim))) # Glorot
        @test QuantumNeuralStates._init_std(T, in_dim, out_dim, act_func2) == T(sqrt(2.0/(in_dim))) # He
        @test QuantumNeuralStates._init_std(T, in_dim, out_dim, act_func3) == T(sqrt(2.0/(in_dim+out_dim)))

        @test QuantumNeuralStates._lookup_deriv(act_func1) == tanh_deriv 
    end
    @testset "Dense" begin
        in_dim = 10
        out_dim = 3
        batch = 5
        act = tanh

        T,W,b,a,z,ln = QuantumNeuralStates._dense_alloc(in_dim, out_dim, act; batch=batch, 
                                                        device=identity, Layer_Norm=false)
        @test size(W) == (out_dim, in_dim)
        @test size(b) == (out_dim,)
        @test size(a) == (out_dim, batch)
        @test size(z) == size(a)
        @test ln === nothing

        layer = Dense(in_dim, out_dim, act; batch=batch)
        @test layer.act_func == act
        @test layer.act_deriv == tanh_deriv
        layer_multi = Dense(in_dim, out_dim, (identity, act, relu); batch=batch)
        @test layer_multi.act_func[2] == act
        @test layer_multi.act_deriv[2] == tanh_deriv

        x = rand(T, in_dim, batch)

        mul!(layer.a, layer.W, x, 1f0, 0f0)
        layer.a .+= layer.b
        QuantumNeuralStates.apply_act!(layer, layer.a, layer.z)
        @test !iszero(layer.z)
        @test layer.z[1, 1] == layer.act_func(layer.a[1, 1])
        previous_z = copy(layer.z)

        zero!(layer.a)
        zero!(layer.z)
        forward(layer, x)
        @test layer.z == previous_z

        @test QuantumNeuralStates.n_params(layer) == length(layer.W) + length(layer.b)

        layerMulti = QuantumNeuralStates.MultiForwardLayer(similar(layer.a, size(layer.a, 1), 
                                                            batch), nothing)
        forward(layer, x, layerMulti)
        @test layerMulti.a == layer.z
    end
    @testset "LayerNorm" begin
        in_dim = 100
        out_dim = 3
        batch = 5
        act = tanh

        layer = Dense(in_dim, out_dim, act; batch=batch, Layer_Norm=false)
        layer_ln = Dense(in_dim, out_dim, act; batch=batch, Layer_Norm=true)

        x = randn(T, in_dim, batch) .* 3
        
        forward(layer, x)
        forward(layer_ln, x)

        @test !isapprox(var(layer.a), 1, atol=0.2)
        @test isapprox(var(layer_ln.a), 1, atol=0.2)
    end
end

@testset "Chain" begin
    T = Float32
    in_dim = 10
    out_dim = 3
    batch = 5
    act = tanh
    layer1 = Dense(in_dim, out_dim, act; batch=batch)
    layer2 = Dense(out_dim, 1, identity; batch=batch)
    model = Chain(layer1, layer2; batch=batch)

    addr = near_uniform(BoseFS{5, in_dim})
    model_multi = QuantumNeuralStates.MultiForwardBuffer(model, addr, batch)

    x = randn(T, in_dim, batch)
    QuantumNeuralStates.prepare_chain_input!(model, x)
    QuantumNeuralStates.prepare_chain_input!(model, x, model_multi)

    @test model.x == model_multi.x

    output = forward(model, x)
    output_multi = forward(model, x, model_multi)
    @test output == output_multi 
end
@testset "Backpropagation" begin
    T = Float32
    in_dim = 10
    out_dim = 3
    batch = 5
    act = tanh
    layer1 = Dense(in_dim, out_dim, act; batch=batch)
    layer2 = Dense(out_dim, 1, identity; batch=batch)
    model = Chain(layer1, layer2; batch=batch)
    x = randn(T, in_dim, batch)
    output = forward(model, x)

    addr = BoseFS{missing}{in_dim}()
    H = FroehlichPolaron(addr; l=3.0, v=1.155, mode_cutoff=5)
    ansatz = NeuralAnsatz(LogPsi(), H, model, batch) 

    buffers = map(DenseBuffer, model.layers)
    jac_buf = JacobianBuffer(ansatz, buffers)

    @test_nowarn grads = back_jacobian!(ansatz, jac_buf)
    @test size(jac_buf.J) == (QuantumNeuralStates.n_params(model.layers[1]) + 
                              QuantumNeuralStates.n_params(model.layers[2]), batch)
end
