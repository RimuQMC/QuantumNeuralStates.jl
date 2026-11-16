using Random
using Rimu
using QuantumNeuralStates
using Test

@testset "VMC" begin
    Random.seed!(42)
    T = Float32
    M, N, batch = 3, 2, 3
    addr = BoseFS{missing}{M}()
    H = FroehlichPolaron{T}(addr; D=1, l=3.0, alpha=2, mode_cutoff=N)
    enc = OccupationEncoding((M,), H)
    model = build_model("FCNN", enc, [M, 10, 10, 1], tanh; batch=batch)
    ansatz = NeuralAnsatz(LogPsi(), H, model, batch)

    KA = QuantumNeuralStates.KernelAbstractions
    backend = KA.get_backend(ansatz.model.x)

    addrs_n = [BoseFS{missing}((1, 0, 0)), BoseFS{missing}((0, 1, 0)), BoseFS{missing}((1, 1, 0))]
    ref = [[(a2, v) for (a2, v) in offdiagonals(H * a) if !iszero(v)] for a in addrs_n]
    counts = length.(ref)
    offs = cumsum(counts)
    starts = offs .- counts
    total = offs[end]

    @testset "VMCBuffer" begin
        vb = VMCBuffer(ansatz, addr)
        @test length(vb.addrs_m) == batch
        @test all(==(addr), Array(vb.addrs_m))
        @test size(vb.addrs_offdiag_buf.data, 2) == batch
        @test size(vb.ham_offdiag_buf.data) == size(vb.addrs_offdiag_buf.data)
        @test size(vb.vals_offdiag_buf.data, 1) == 1
        @test length(vb.E_locs) == length(vb.weights) == length(vb.offsets) == batch
        @test !vb.start
        @test size(VMCBuffer(ansatz, addr; K_init=1).addrs_offdiag_buf.data, 1) == 1
    end

    @testset "collect_offdiagonals!" begin
        vb = VMCBuffer(ansatz, addr; K_init=1)                  # forces growth
        collect_offdiagonals!(vb.addrs_offdiag_buf, vb.ham_offdiag_buf, vb.ham_diag,
                              vb.offsets, H, addrs_n, nothing)
        @test Array(vb.offsets) == offs
        @test size(vb.addrs_offdiag_buf.data, 1) >= maximum(counts)
        @test size(vb.ham_offdiag_buf.data, 1) >= maximum(counts)

        A, V = Array(vb.addrs_offdiag_buf.data), Array(vb.ham_offdiag_buf.data)
        @test [(A[k, b], V[k, b]) for b in 1:batch for k in 1:counts[b]] == reduce(vcat, ref)
        @test Array(vb.ham_diag) ≈ [diagonal_element(H * a) for a in addrs_n]

        # with truncation: no collected address violates the mask
        mask = NeuralAnsatz(LogPsi(), H, model, batch; truncation=1).truncation.mask
        collect_offdiagonals!(vb.addrs_offdiag_buf, vb.ham_offdiag_buf, vb.ham_diag,
                              vb.offsets, H, addrs_n, mask)
        o = Array(vb.offsets)          # cumulative: [2, 5, 7]
        c = zeros(Int, batch)          # per-parent counts
        previous = 0
        for b in 1:batch
            c[b] = o[b] - previous     # how many belong to parent b
            previous = o[b]
        end
        # c = [2, 3, 2]
        A = Array(vb.addrs_offdiag_buf.data)
        @test !any(violates_truncation(onr(A[k, b]), mask) for b in 1:batch for k in 1:c[b])
    end

    @testset "proposals" begin
        vb = VMCBuffer(ansatz, addr)
        collect_offdiagonals!(vb.addrs_offdiag_buf, vb.ham_offdiag_buf, vb.ham_diag,
                              vb.offsets, H, addrs_n, nothing)
        addrs_all = vb.addrs_offdiag_buf.data
        rand_vals = KA.zeros(backend, T, batch)

        # Metropolis PROPOSAL: always one of the parent's off-diagonals
        for _ in 1:20
            metropolis_state_proposal!(vb.addrs_m, addrs_all, vb.offsets, rand_vals)
            p = Array(vb.addrs_m)
            @test all(p[b] in first.(ref[b]) for b in 1:batch)
        end

        # CTMC PROPOSAL, uniform log-weights: always one of the parent's off-diagonals
        distro = zeros(T, total)
        for _ in 1:20
            ctmc_state_proposal!(vb.addrs_m, addrs_all, vb.offsets, distro, rand_vals)
            p = Array(vb.addrs_m)
            @test all(p[b] in first.(ref[b]) for b in 1:batch)
        end

        # CTMC, one dominant log-weight per parent (the last one): always picked
        distro[offs] .= 100
        ctmc_state_proposal!(vb.addrs_m, addrs_all, vb.offsets, distro, rand_vals)
        @test Array(vb.addrs_m) == [first(ref[b][end]) for b in 1:batch]

        # parent without off-diagonals keeps its address
        fill!(vb.addrs_m, addr)
        empty_offs = Int32[counts[1], counts[1], counts[1] + counts[3]]     # parent 2 empty
        metropolis_state_proposal!(vb.addrs_m, addrs_all, empty_offs, rand_vals)
        @test Array(vb.addrs_m)[2] == addr
    end

    @testset "metropolis_count_ratio!" begin
        vb = VMCBuffer(ansatz, addr)
        collect_offdiagonals!(vb.addrs_offdiag_buf, vb.ham_offdiag_buf, vb.ham_diag,
                              vb.offsets, H, addrs_n, nothing)
        rand_vals = KA.zeros(backend, T, batch)
        metropolis_state_proposal!(vb.addrs_m, vb.addrs_offdiag_buf.data, vb.offsets, rand_vals)

        ratio = KA.zeros(backend, T, batch)
        metropolis_count_ratio!(ratio, vb.addrs_m, vb.offsets, H, nothing)
        Nm = [count(p -> !iszero(last(p)), offdiagonals(H * a)) for a in Array(vb.addrs_m)]
        @test Array(ratio) ≈ counts ./ Nm
    end

    @testset "get_ctmc_weights!" begin
        log_distro = randn(T, total)
        log_psi = randn(T, batch)
        w = KA.zeros(backend, T, batch)
        get_ctmc_weights!(w, log_distro, Int32.(offs), log_psi)

        lw = [log_psi[b] - log(sum(exp.(log_distro[starts[b]+1:offs[b]]))) for b in 1:batch]
        @test Array(w) ≈ exp.(lw) ./ sum(exp.(lw))
        @test sum(Array(w)) ≈ 1
    end

    @testset "local_energy_stats!" begin
        vb = VMCBuffer(ansatz, addr)
        E = randn(T, batch)
        w = rand(T, batch)
        w ./= sum(w)
        Em, var = QuantumNeuralStates.local_energy_stats!(vb, E, w, batch)
        μ = sum(w .* E)
        @test Array(Em)[1] ≈ μ
        @test Array(var)[1] ≈ sum(w .* (E .- μ) .^ 2)
    end

    @testset "vmc_energy" begin
        vb = VMCBuffer(ansatz, addr)
        buffers = make_buffers(ansatz.model)
        jb = JacobianBuffer(ansatz, buffers)

        for sampler in (:metropolis, :ctmc)
            E, var, new_addrs, acc, w = vmc_energy(H, ansatz, copy(addrs_n), vb, jb;
                                                   vmc_sampler=sampler, burnin=5)
            @test isfinite(Array(E)[1])
            @test Array(var)[1] >= 0
            @test all(0 .<= acc .<= 1)
            @test sum(Array(w)) ≈ 1
            @test length(new_addrs) == batch
        end
        @test_throws ArgumentError vmc_sample!(:foo, vb, jb, H, addrs_n, ansatz)
    end

    @testset "final_elocs_statistics!" begin
        Random.seed!(1)
        vb = VMCBuffer(ansatz, addr)
        buffers = make_buffers(ansatz.model)
        jb = JacobianBuffer(ansatz, buffers)

        n_iter = 200
        B = ansatz.model.batch
        M = num_modes(addr)

        res = Dict{Symbol, Any}()
        for sampler in (:metropolis, :ctmc)
            r, N_all, counts, n_mode = redirect_stdout(devnull) do
                final_elocs_statistics!(sampler, vb, jb, H, copy(addrs_n), ansatz;
                                        n_iter = n_iter, n_skip = 3, burnin = 20,
                                        plot_occ = false)
            end
            res[sampler] = r

            # energy estimate
            @test isfinite(r.mean)
            @test isfinite(r.err) && r.err >= 0

            # occupation statistics
            @test length(N_all) == n_iter * B
            @test sum(values(counts)) == length(N_all)
            @test length(n_mode) == M
            @test all(>=(0), n_mode)
            @test sum(n_mode) ≈ sum(N_all) / length(N_all)
        end

        # both samplers estimate the same ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩
        m, c = res[:metropolis], res[:ctmc]
        @test abs(m.mean - c.mean) <= 5 * hypot(m.err, c.err) + 1e-3
    end
end
