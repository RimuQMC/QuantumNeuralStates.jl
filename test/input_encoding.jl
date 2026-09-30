
using LinearAlgebra
using Statistics
using QuantumNeuralStates
using Rimu
using Test
using Random

@testset "Input Encodings" begin
    T  = Float32
    H  = FroehlichPolaron(BoseFS{missing,16}(); D=2, alpha=1, l=6)
    sz = (4, 4)
    M, D = 16, 2
    ks = H.ks

    # configurations taken from H itself: vacuum, one-phonon and two-phonon states
    vac    = starting_address(H)
    one_ph = filter(a -> sum(onr(a)) == 1, unique(first.(collect(offdiagonals(H * vac)))))
    two_ph = filter(a -> sum(onr(a)) == 2, unique(first.(collect(offdiagonals(H * first(one_ph))))))
    addrs  = vcat([vac], one_ph, two_ph)
    batch  = length(addrs)
    ph1    = one_ph[findfirst(a -> onr(a)[1] == 1, one_ph)]   # one phonon in mode 1, k = (-2π/6, -2π/6)
    b_ph1  = findfirst(==(ph1), addrs)

    @test !isempty(one_ph)                                    # setup sanity checks
    @test !isempty(two_ph)
    @test all(sum(onr(a)) <= 2 for a in addrs)
    @test length(unique(addrs)) == batch


    @testset "NoEncoding" begin
        nb = 4

        @testset "construction" begin
            e = NoEncoding(10)
            @test e.size == (10,)
            @test nchannels(e) == 1
            @test sprint(show, e) == "NoEncoding((10,), C = 1)"

            ec = NoEncoding((8, 8); channels=3)
            @test nchannels(ec) == 3
            @test QuantumNeuralStates.nsites(ec) == 64

            @test_throws ErrorException NoEncoding((0, 4))
            @test_throws ErrorException NoEncoding((4, 4); channels=0)
            @test_throws ErrorException QuantumNeuralStates.occupation_channel(e)
        end

        @testset "Dense model uses input as given" begin
            model = Chain(NoEncoding(10),
                          Dense(10=>3, identity; batch=nb); batch=nb)
            x = rand(Float32, 10, nb)
            y = model(x)
            @test model.x == x
            @test Array(y) ≈ model.layers[1].W * x .+ model.layers[1].b
        end

        @testset "Conv model shapes" begin
            model = Chain(NoEncoding((8, 8); channels=3),
                          Conv((3,3), 3=>4, relu; batch=nb, pad=Zeros()),
                          Pool(:sum),
                          Dense(4=>1, identity; batch=nb); batch=nb)
            @test size(model.x) == (8, 8, 3, nb)
            x = rand(Float32, 8, 8, 3, nb)
            model(x)
            @test model.x == x
        end

        @testset "errors" begin
            model = Chain(NoEncoding(10), Dense(10=>3, identity; batch=nb); batch=nb)
            @test_throws ErrorException model(rand(Float32, 12, nb))        # wrong shape
            @test_throws ErrorException Chain(NoEncoding(10),                  # wrong width
                                              Dense(12=>3, identity; batch=nb); batch=nb)
            H = FroehlichPolaron(BoseFS{missing,16}(); D=2, alpha=1, l=6)
            @test_throws ErrorException NeuralAnsatz(LogPsi(), H,
                                        Chain(NoEncoding(16), Dense(16=>1, identity; batch=nb);
                                              batch=nb), nb)
        end
    end

    @testset "OccupationEncoding" begin
        enc = OccupationEncoding(sz, H)
        @test enc.size == sz
        @test nsites(enc) == M
        @test nchannels(enc) == 1
        @test QuantumNeuralStates.occupation_channel(enc) == 1
        @test sprint(show, enc) == "OccupationEncoding((4, 4), C = 1)"

        @test_throws ErrorException OccupationEncoding((3, 3), H)     # 9 sites ≠ 16 modes
        @test_throws ErrorException OccupationEncoding((17,), H)
    end

    @testset "MomentumEncoding" begin
        enc  = MomentumEncoding(sz, H)
        encN = MomentumEncoding(sz, H; occupation=false)

        @testset "construction" begin
            @test enc.size == sz
            @test nchannels(enc)  == 3                       # n, n·kx, n·ky
            @test nchannels(encN) == 2                       # n·kx, n·ky
            @test QuantumNeuralStates.occupation_channel(enc)  == 1
            @test QuantumNeuralStates.occupation_channel(encN) == 2
            @test sprint(show, enc)  == "MomentumEncoding((4, 4), C = 3: n, n·k)"
            @test sprint(show, encN) == "MomentumEncoding((4, 4), C = 2: n·k)"
        end

        @testset "momenta" begin
            K = Array(enc.K)
            @test size(K) == (M, D)
            @test K[1, :] ≈ T[-2π/6, -2π/6]                  # hand-checked: first mode
            @test K[6, :] ≈ T[0, 0]                          # k = 0 mode
            @test all(K[m, d] ≈ ks[m][d] for m in 1:M, d in 1:D)

            encS = MomentumEncoding(sz, H; kscale=2.0)
            @test Array(encS.K) ≈ K ./ 2
        end

        @testset "errors" begin
            @test_throws ErrorException MomentumEncoding((3, 3), H)   # wrong site count
            @test_throws ErrorException MomentumEncoding((16,), H)    # 1D size for 2D momenta
        end
    end

    @testset "Chain with encoding" begin
        enc = MomentumEncoding(sz, H)

        @testset "Conv input" begin
            model = Chain(enc,
                          Conv((3,3), nchannels(enc)=>4, relu; batch=batch, pad=Zeros()),
                          Pool(:sum),
                          Dense(4=>1, identity; batch=batch); batch=batch)
            @test size(model.x)  == (4, 4, 3, batch)
            @test size(model.xe) == (M, 3, batch)

            model.xe[7, 2, 1] = 5f0                          # site m = 7  →  (i, j) = (3, 2)
            @test model.x[3, 2, 2, 1] == 5f0                 # same memory
        end

        @testset "Dense input" begin
            model = Chain(enc,
                          Dense(M*nchannels(enc)=>4, relu; batch=batch),
                          Dense(4=>1, identity; batch=batch); batch=batch)
            @test size(model.x)  == (48, batch)
            @test size(model.xe) == (M, 3, batch)

            model.xe[7, 2, 1] = 5f0                          # row = m + M(c-1) = 7 + 16 = 23
            @test model.x[23, 1] == 5f0
        end

        @testset "errors" begin
            @test_throws ErrorException Chain(enc,                     # Conv expects 1 channel
                Conv((3,3), 1=>4, relu; batch=batch, pad=Zeros()); batch=batch)
            @test_throws ErrorException Chain(enc,                     # Dense width 16 ≠ 48
                Dense(M=>4, relu; batch=batch); batch=batch)
        end
    end

    @testset "encode! (hand-checked)" begin
        @testset "with occupation channel" begin
            enc = MomentumEncoding(sz, H)
            xe  = zeros(T, M, 3, 2)
            xe[1, 1, 1] = 2f0                                # 2 phonons in mode 1, sample 1
            xe[6, 1, 2] = 3f0                                # 3 phonons in mode 6 (k = 0), sample 2
            QuantumNeuralStates.encode!(xe, enc)

            @test xe[1, :, 1] ≈ T[2, 2ks[1][1], 2ks[1][2]]
            @test xe[6, :, 2] ≈ T[3, 0, 0]
            @test count(!iszero, xe) == 4                    # nothing written elsewhere
        end

        @testset "without occupation channel" begin
            enc = MomentumEncoding(sz, H; occupation=false)
            xe  = zeros(T, M, 2, 1)
            xe[1, 2, 1] = 2f0                                # occupation sits in the last channel
            QuantumNeuralStates.encode!(xe, enc)

            @test xe[1, :, 1] ≈ T[2ks[1][1], 2ks[1][2]]      # turned into n·kx, n·ky in place
        end

        @testset "OccupationEncoding is a no-op" begin
            enc = OccupationEncoding(sz, H)
            xe  = rand(T, M, 1, 2)
            xe0 = copy(xe)
            QuantumNeuralStates.encode!(xe, enc)
            @test xe == xe0
        end
    end

    @testset "OccupationEncoding input" begin
        enc = OccupationEncoding(sz, H)
        for model in (Chain(enc,
                            Conv((3,3), 1=>4, relu; batch=batch, pad=Zeros()),
                            Pool(:sum),
                            Dense(4=>1, identity; batch=batch); batch=batch),
                      Chain(enc,
                            Dense(M=>4, relu; batch=batch),
                            Dense(4=>1, identity; batch=batch); batch=batch))
            na = NeuralAnsatz(LogPsi(), H, model, batch)
            x  = QuantumNeuralStates.prepare_input!(na, addrs, na.x_cpu_buffer)
            QuantumNeuralStates.prepare_chain_input!(na.model, x)
            XE = Array(na.model.xe)
            @test all(XE[:, 1, b] == T.(onr(addrs[b])) for b in 1:batch)
        end
    end

    @testset "MultiForwardBuffer with encoding" begin
        enc   = MomentumEncoding(sz, H)
        model = Chain(enc,
                      Conv((3,3), nchannels(enc)=>4, relu; batch=batch, pad=Zeros()),
                      Pool(:sum),
                      Dense(4=>1, identity; batch=batch); batch=batch)
        na  = NeuralAnsatz(LogPsi(), H, model, batch; multiforward_buffer=2batch)
        mfb = na.multi_forward_buffer
        @test size(mfb.xe) == (M, 3, 2batch)

        xm = QuantumNeuralStates.prepare_input!(na, addrs, mfb.x_cpu)
        QuantumNeuralStates.prepare_chain_input!(na.model, xm, mfb)
        XEm = Array(mfb.xe)

        x = QuantumNeuralStates.prepare_input!(na, addrs, na.x_cpu_buffer)
        QuantumNeuralStates.prepare_chain_input!(na.model, x)
        XE = Array(na.model.xe)

        @test XEm[:, :, 1:batch] ≈ XE

        # out_main = Array(na.model(QuantumNeuralStates.prepare_input!(na, addrs, na.x_cpu_buffer)))
        # out_mfb  = Array(na.model(QuantumNeuralStates.prepare_input!(na, addrs, mfb.x_cpu), mfb))
        # @test out_mfb[:, 1:batch] ≈ out_main
    end

    @testset "allocations" begin
        enc = MomentumEncoding(sz, H)
        model = Chain(enc,
                      Conv((3,3), nchannels(enc)=>4, relu; batch=batch, pad=Zeros()),
                      Pool(:sum),
                      Dense(4=>1, identity; batch=batch); batch=batch)
        na = NeuralAnsatz(LogPsi(), H, model, batch)

        x = QuantumNeuralStates.prepare_input!(na, addrs, na.x_cpu_buffer)    # warm-up
        QuantumNeuralStates.prepare_chain_input!(na.model, x)

        @test @allocated(QuantumNeuralStates.prepare_input!(na, addrs, na.x_cpu_buffer)) <= 64
        @test @allocated(QuantumNeuralStates.encode!(na.model.xe, enc)) <= 256        # launch only
        @test @allocated(QuantumNeuralStates.prepare_chain_input!(na.model, x)) <= 256
    end

    @testset "1D grid" begin
        H1  = FroehlichPolaron(BoseFS{missing,8}(); D=1, alpha=1, l=6)
        enc = MomentumEncoding((8,), H1)
        @test nchannels(enc) == 2                                              # n, n·k

        vac1 = starting_address(H1)
        a1   = first(filter(a -> sum(onr(a)) == 1, unique(first.(collect(offdiagonals(H1*vac1))))))
        m1   = findfirst(>(0), onr(a1))

        model = Chain(enc,
                      Conv((3,), nchannels(enc)=>4, relu; batch=2, pad=Zeros()),
                      Pool(:sum),
                      Dense(4=>1, identity; batch=2); batch=2)
        @test size(model.x) == (8, 2, 2)

        na = NeuralAnsatz(LogPsi(), H1, model, 2)
        x  = QuantumNeuralStates.prepare_input!(na, [vac1, a1], na.x_cpu_buffer)
        QuantumNeuralStates.prepare_chain_input!(na.model, x)
        XE = Array(na.model.xe)

        @test all(iszero, XE[:, :, 1])
        @test XE[m1, :, 2] ≈ T[1, H1.ks[m1][1]]
        @test all(iszero, XE[setdiff(1:8, m1), :, 2])
    end
end



