using LinearAlgebra
using Statistics
using QuantumNeuralStates
using Rimu
using Test
using Random

@testset "Layers" begin
    T = Float32

    @testset "Initialisation" begin
        in_dim = 10
        out_dim = 3
        act_func1 = tanh
        act_func2 = relu
        act_func3 = (identity, tanh)

        @test QuantumNeuralStates._init_std(T, in_dim, out_dim, act_func1) == T(sqrt(2.0/(in_dim+out_dim))) # Glorot
        @test QuantumNeuralStates._init_std(T, in_dim, out_dim, act_func2) == T(sqrt(2.0/(in_dim)))          # He
        @test QuantumNeuralStates._init_std(T, in_dim, out_dim, act_func3) == T(sqrt(2.0/(in_dim+out_dim)))

        @test QuantumNeuralStates._lookup_deriv(act_func1) == tanh_deriv
    end

    @testset "Dense" begin
        in_dim, out_dim, batch = 10, 3, 5
        act = tanh
        layer = Dense(in_dim=>out_dim, act; batch=batch)
        x = rand(T, in_dim, batch)

        @testset "construction" begin
            @test layer.act_func == act
            @test layer.act_deriv == tanh_deriv
            layer_multi = Dense(in_dim=>out_dim, (identity, act, relu); batch=batch)
            @test layer_multi.act_func[2] == act
            @test layer_multi.act_deriv[2] == tanh_deriv
            @test QuantumNeuralStates.hasparams(layer) == true
            @test QuantumNeuralStates.n_params(layer) == length(layer.W) + length(layer.b)
        end

        @testset "forward" begin
            mul!(layer.a, layer.W, x, 1f0, 0f0)
            layer.a .+= layer.b
            QuantumNeuralStates.apply_act!(layer, layer.a, layer.z)
            @test !iszero(layer.z)
            @test layer.z[1, 1] == layer.act_func(layer.a[1, 1])
            previous_z = copy(layer.z)

            layer.a .= 0
            layer.z .= 0
            forward(layer, x)
            @test layer.z == previous_z

            layerMulti = QuantumNeuralStates.MultiForwardLayer(similar(layer.a, size(layer.a, 1), batch), nothing)
            forward(layer, x, layerMulti)
            @test layerMulti.a == layer.z
        end

        @testset "back!" begin
            buf = QuantumNeuralStates.DenseBuffer(layer)
            δ = ones(T, out_dim, batch)
            J_W = similar(layer.W, out_dim, in_dim, batch)
            J_b = similar(layer.b, out_dim, batch)

            δx = back!(layer, buf, J_W, J_b, δ, x)
            @test size(δx) == (in_dim, batch)
            @test size(J_W) == (out_dim, in_dim, batch)
            @test size(J_b) == (out_dim, batch)

            @test J_b ≈ layer.act_deriv.(layer.a)               
            @test J_W[1, 2, 3] ≈ J_b[1, 3] * x[2, 3]            
            @test δx[2, 3] ≈ sum(layer.W[:, 2] .* J_b[:, 3])        
        end
    end

    @testset "Conv" begin
        K, C_in, C_out, L, batch = 3, 2, 4, 8, 5
        act = tanh
        pad = NoPad()
        layer = Conv((K,), C_in=>C_out, act; batch=batch, device=identity, pad=pad)
        x = rand(T, L, C_in, batch)

        @testset "construction" begin
            @test size(layer.W) == (K, C_in, C_out)
            @test size(layer.b) == (C_out,)
            @test layer.act_func == act
            @test layer.act_deriv == tanh_deriv
            @test layer.a === nothing
            @test layer.z === nothing
            @test QuantumNeuralStates.hasparams(layer) == true
            @test QuantumNeuralStates.n_params(layer) == length(layer.W) + length(layer.b)
        end

        @testset "forward" begin
            forward(layer, x)
            L_out = QuantumNeuralStates._lout(pad, L, K, layer.stride)
            @test size(layer.a) == (L_out, C_out, batch)
            @test size(layer.z) == size(layer.a)
            @test !iszero(layer.z)
            @test layer.z[1, 1, 1] == layer.act_func(layer.a[1, 1, 1])

            previous_z = copy(layer.z)
            layer.a .= 0
            layer.z .= 0
            forward(layer, x)
            @test layer.z == previous_z

            layerMulti = QuantumNeuralStates.MultiForwardLayer(similar(layer.a), nothing)
            forward(layer, x, layerMulti)
            @test layerMulti.a == layer.z

            @testset "hand-checked: padding modes" begin
                x_hand = reshape(Float32.(1:5), 5, 1, 1)
                expected = Dict(
                    NoPad()    => Float32[14, 20, 26],
                    Zeros()    => Float32[8, 14, 20, 26, 14],
                    Periodic() => Float32[13, 14, 20, 26, 17],
                )
                for (p, exp) in expected
                    conv = Conv((3,), 1=>1, relu; pad=p)
                    copyto!(conv.W, reshape(Float32[1, 2, 3], 3, 1, 1))
                    forward(conv, x_hand)
                    @test Array(conv.a)[:, 1, 1] ≈ exp
                end
            end
        end

        @testset "back!" begin
            forward(layer, x)   

            @testset "NoPad check" begin
                buf = QuantumNeuralStates.ConvBuffer(layer, x)
                L_out = size(layer.a, 1)
                J_W = similar(layer.W, K, C_in, C_out, batch)
                J_b = similar(layer.b, C_out, batch)
                δ = ones(T, L_out, C_out, batch)

                δx = back!(layer, buf, J_W, J_b, δ, x)
                @test size(δx) == (L, C_in, batch)
                @test size(J_W) == (K, C_in, C_out, batch)
                @test size(J_b) == (C_out, batch)

                δa = layer.act_deriv.(layer.a)
                @test J_b ≈ dropdims(sum(δa; dims=1); dims=1)

                k, c, o, n = 2, 1, 3, 4
                @test J_W[k, c, o, n] ≈ sum(δa[p, o, n] * x[p + k - 1, c, n] for p in 1:L_out)

                q, c2, n2 = 4, 2, 3
                expected_dx = sum(layer.W[k, c2, o] * δa[q - k + 1, o, n2]
                                   for o in 1:C_out, k in 1:K if 1 <= q - k + 1 <= L_out)
                @test δx[q, c2, n2] ≈ expected_dx
            end

            @testset "Periodic/Zeros checks" begin
                for p in (Zeros(), Periodic())
                    conv = Conv((K,), C_in=>C_out, act; batch=batch, pad=p)
                    xx = randn(T, L, C_in, batch)
                    forward(conv, xx)
                    buf = QuantumNeuralStates.ConvBuffer(conv, xx)
                    J_W = similar(conv.W, size(conv.W)..., batch)
                    J_b = similar(conv.b, C_out, batch)
                    δ = ones(T, size(conv.a)...)
                    back!(conv, buf, J_W, J_b, δ, xx)
                    δa = conv.act_deriv.(conv.a)
                    @test J_b ≈ dropdims(sum(δa; dims=1); dims=1)
                end
            end

            @testset "all pad modes" begin
                for pad2 in (NoPad(), Zeros(), Periodic()), (Lg, Kg, s) in ((10, 3, 1), (7, 3, 2), (4, 3, 1))
                    Random.seed!(1234)
                    pad2 isa NoPad && Lg < Kg && continue
                    conv = Conv((Kg,), 3=>4, identity; stride=s, pad=pad2)
                    fill!(conv.b, 0f0)
                    u = randn(T, Lg, 3, 1)
                    forward(conv, u)
                    v = randn(T, size(conv.a)...)

                    buf = QuantumNeuralStates.ConvBuffer(conv, u)
                    J_W = similar(conv.W, size(conv.W)..., 1)
                    J_b = similar(conv.b, size(conv.b, 1), 1)
                    δx = back!(conv, buf, J_W, J_b, v, u)

                    lhs = sum(conv.a .* v)
                    rhs = sum(u .* δx)
                    scale = sum(abs, conv.a .* v)
                    @test isapprox(lhs, rhs; rtol = 1f-3, atol = 1f-5 * scale)
                end
            end
        end
    end

    @testset "Pool" begin
        C, L, batch = 4, 6, 5
        op = :mean
        layer = Pool(op; device=identity)
        x = rand(T, L, C, batch)

        @testset "construction" begin
            @test layer.op == Val(op)
            @test layer.z === nothing
            @test QuantumNeuralStates.hasparams(layer) == false
            @test QuantumNeuralStates.n_params(layer) == 0
        end

        @testset "forward" begin
            forward(layer, x)
            @test size(layer.z) == (C, batch)
            @test !iszero(layer.z)
            @test layer.z[1, 1] ≈ sum(x[:, 1, 1]) / L

            previous_z = copy(layer.z)
            layer.z .= 0
            forward(layer, x)
            @test layer.z == previous_z

            layerMulti = QuantumNeuralStates.MultiForwardLayer(similar(layer.z), nothing)
            forward(layer, x, layerMulti)
            @test layerMulti.a == layer.z

            @testset "all four ops, hand-checked" begin
                x_hand = reshape(Float32.(1:16), 2, 2, 2, 2)
                expected = Dict(
                    :max  => Float32[4 12; 8 16],
                    :min  => Float32[1 9; 5 13],
                    :sum  => Float32[10 42; 26 58],
                    :mean => Float32[2.5 10.5; 6.5 14.5],
                )
                for (op2, exp) in expected
                    pool = Pool(op2)
                    forward(pool, x_hand)
                    @test Array(pool.z) ≈ exp
                end
            end
        end

        @testset "back!" begin
            x_hand = reshape(Float32.(1:16), 2, 2, 2, 2)

            @testset "uniform ops (mean/sum)" begin
                for (op2, scale) in ((:mean, 1/4), (:sum, 1.0))
                    pool = Pool(op2)
                    forward(pool, x_hand)
                    buf = QuantumNeuralStates.PoolBuffer(pool, x_hand)
                    δ = ones(T, size(pool.z)...)
                    δx = back!(pool, buf, nothing, nothing, δ, x_hand)
                    @test all(δx .≈ scale)
                end
            end

            @testset "extremum ops (max/min)" begin
                for (op2, finder) in ((:max, argmax), (:min, argmin))
                    pool = Pool(op2)
                    forward(pool, x_hand)
                    buf = QuantumNeuralStates.PoolBuffer(pool, x_hand)
                    δ = ones(T, size(pool.z)...)
                    δx = back!(pool, buf, nothing, nothing, δ, x_hand)
                    for c in 1:2, n in 1:2
                        p_star = finder(x_hand[:, :, c, n])
                        @test δx[p_star, c, n] == 1f0
                        @test count(!iszero, δx[:, :, c, n]) == 1
                    end
                end
            end
        end
    end

    @testset "LayerNorm" begin
        in_dim, out_dim, batch = 100, 3, 5
        act = tanh
        layer    = Dense(in_dim=>out_dim, act; batch=batch, layer_norm=false)
        layer_ln = Dense(in_dim=>out_dim, act; batch=batch, layer_norm=true)
        x = randn(T, in_dim, batch) .* 3

        @testset "forward" begin
            forward(layer, x)
            forward(layer_ln, x)
            @test !isapprox(var(layer.a), 1, atol=0.2)
            @test isapprox(var(layer_ln.a), 1, atol=0.2)
        end
    end
end

@testset "Chain" begin
    device = select_device()
    @test device === identity

    T = Float32
    batch = 5
    act = tanh
    M = 10
    conv1 = Conv((3,), 1=>8, act; batch=batch, pad=Periodic())
    conv2 = Conv((3,), 8=>8, act; batch=batch, pad=Periodic())
    pool  = Pool(:mean)
    dense = Dense(8=>1, identity; batch=batch)
    model = Chain(conv1, conv2, pool, dense; batch=batch, input_size=(M,))

    addr = near_uniform(BoseFS{5, M})
    model_multi = QuantumNeuralStates.MultiForwardBuffer(model, addr, batch)

    x = randn(T, M, 1, batch)
    QuantumNeuralStates.prepare_chain_input!(model, x)
    QuantumNeuralStates.prepare_chain_input!(model, x, model_multi)
    @test model.x == model_multi.x

    output = forward(model, x)
    output_multi = forward(model, x, model_multi)
    @test output == output_multi

    @test size(conv1.z) == (M, 8, batch)
    @test size(conv2.z) == (M, 8, batch)
    @test size(pool.z)  == (8, batch)
    @test size(output)  == (1, batch)

    expected_params = sum(QuantumNeuralStates.n_params(l) for l in model.layers)
    @test QuantumNeuralStates.n_params(model) == expected_params
end

@testset "Backpropagation" begin
    T = Float32
    batch = 5

    @testset "JacobianBuffer check (Dense-only)" begin
        in_dim, out_dim = 10, 3
        act = tanh
        layer1 = Dense(in_dim=>out_dim, act; batch=batch)
        layer2 = Dense(out_dim=>1, identity; batch=batch)
        model = Chain(layer1, layer2; batch=batch)
        x = randn(T, in_dim, batch)
        forward(model, x)

        addr = BoseFS{missing}{in_dim}()
        H = FroehlichPolaron(addr; l=3.0, alpha=2, mode_cutoff=5)
        ansatz = NeuralAnsatz(LogPsi(), H, model, batch)

        buffers = make_buffers(ansatz.model)
        jac_buf = JacobianBuffer(ansatz, buffers)

        forward(model, x)
        @test_nowarn back_jacobian!(ansatz, jac_buf)
        n_expected = QuantumNeuralStates.n_params(model)
        @test size(jac_buf.J) == (n_expected, batch)

        θ0 = copy(Array(jac_buf.θ))
        forward(model, x)
        J  = Array(back_jacobian!(ansatz, jac_buf))
        ε  = 1f-3
        f(θ) = (QuantumNeuralStates.update!(ansatz, jac_buf, θ); forward(model, x); Array(forward(model, x))[1, :])

        for i in (1, n_expected ÷ 2, n_expected)
            e  = zeros(T, n_expected); e[i] = ε
            fd = (f(θ0 .+ e) .- f(θ0 .- e)) ./ (2ε)
            @test isapprox(fd, J[i, :]; rtol=1f-2, atol=1f-4)
        end
        QuantumNeuralStates.update!(ansatz, jac_buf, θ0)
    end

    @testset "Conv+Pool+Dense: full-chain check" begin
        M = 8
        conv = Conv((3,), 1=>4, tanh; batch=batch, pad=Periodic())
        pool = Pool(:mean)
        dense = Dense(4=>1, identity; batch=batch)
        model = Chain(conv, pool, dense; batch=batch, input_size=(M,))
        x = randn(T, M, 1, batch)
        forward(model, x)

        addr = BoseFS{missing}{M}()
        H = FroehlichPolaron(addr; l=3.0, alpha=2, mode_cutoff=5)
        ansatz = NeuralAnsatz(LogPsi(), H, model, batch)

        buffers = make_buffers(ansatz.model)
        jac_buf = JacobianBuffer(ansatz, buffers)
        n_expected = QuantumNeuralStates.n_params(model)

        forward(model, x)
        @test_nowarn back_jacobian!(ansatz, jac_buf)
        @test size(jac_buf.J) == (n_expected, batch)

        θ0 = copy(Array(jac_buf.θ))
        forward(model, x)
        J  = Array(back_jacobian!(ansatz, jac_buf))
        ε  = 1f-2
        f(θ) = (QuantumNeuralStates.update!(ansatz, jac_buf, θ); forward(model, x); Array(forward(model, x))[1, :])

        r = jac_buf.ranges[1]
        for i in (first(r.W), first(r.b))
            e  = zeros(T, n_expected); e[i] = ε
            fd = (f(θ0 .+ e) .- f(θ0 .- e)) ./ (2ε)
            @test isapprox(fd, J[i, :]; rtol=5f-2, atol=1f-3)
        end
        QuantumNeuralStates.update!(ansatz, jac_buf, θ0)
    end
end
