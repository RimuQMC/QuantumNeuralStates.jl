
"""
    run_training_loop(H, ansatz, addr, phases; kwargs...)
        -> E_history, E_err_history, var_history, final_addrs

Train `ansatz` with VMC through a sequence of [`TrainingPhase`](@ref)s. Each phase
runs until its stopping criterion is met or it reaches `max_epochs`, and the next phase
then continues from the current parameters and walker positions. After the last phase,
the parameters are optionally saved, and an unbiased final energy estimate is computed
with [`final_elocs_statistics!`](@ref).

# Initial walkers
The batch of `B = ansatz.model.batch` walkers is initialised in one of two ways:
* **Loaded:** if `load = true` and `loadfile` exists, the walkers stored by
  [`save_master`](@ref) are reused, provided the truncation grid is unchanged and the
  stored batch size equals `B`.
* **Random (otherwise):** the walkers are generated with [`addrs_random`](@ref) from
  `addr`. Only modes that are both coupled by `H` (see [`coupled_modes`](@ref)) and
  kept by the truncation of `ansatz` are populated, with at most `n_max = 3` particles
  per mode.

# Arguments
* `H`: Hamiltonian defined in Rimu.
* `ansatz`: wave-function ansatz, see [`NeuralAnsatz`](@ref).
* `addr`: Rimu Fock address, the same type as used in the definition of `H`. It serves
  as the template for the random initial walkers.
* `phases`: `Vector{TrainingPhase}` run in order,
  e.g. `phases = [TrainingPhase(...), TrainingPhase(...)]`, see [`TrainingPhase`](@ref).

# Keyword Arguments
* `savefile`: file to which the neural network parameters and final walkers are saved,
  see [`save_master`](@ref) (default `"weights.txt"`).
* `loadfile`: file from which parameters and walkers are loaded,
  see [`load_master`](@ref) (default `"weights.txt"`).
* `save`: if `true`, save the parameters at the end of training (default `true`).
* `load`: if `true`, load parameters and walkers from `loadfile` before training
  (default `false`).
* `occplot`: if `true`, plot the occupation statistics of the final estimate
  (default `false`).
* `occfile`: file name for the occupation plots (default `"occupation.pdf"`).
* `markov`: if `true`, log the walker addresses after every epoch (default `false`).
* `markovfile`: file for the Markov-chain log. Meant for diagnostics only, as the file
  can become very large, very quickly (default `"MarkovChain.txt"`).

# Returns
* `E_history`: block-averaged energies of all phases.
* `E_err_history`: per-block error estimates (without autocorrelation analysis).
* `var_history`: block-averaged local-energy variances.
* `final_addrs`: final walker addresses on the device of `ansatz`.

# Notes
* All histories are chained over phases, one entry per block of `block_size` epochs.
* Training uses clipped local energies. The final estimate is unclipped and comes with a
  blocking-analysis error bar, which is the reliable one. It is printed but not returned.
* Neuron and Jacobian statistics, if enabled in `ansatz`, are collected once per block.
"""
function run_training_loop(H, ansatz, addr, phases::Vector{TrainingPhase};
                           savefile::String = "weights.txt", loadfile::String = "weights.txt",
                           save::Bool = true, load::Bool = false,
                           occfile::String = "occupation.png", occplot::Bool=false,
                           markovfile::String = "MarkovChain.txt" , markov::Bool = false)

    @info "Hilbert space dimension: $(@sprintf("%.3e", dimension(H)))"

    # --- initialise addresses ---------------------------------------
    addrs = if isfile(loadfile) && load
        x, grid_change = load_master(ansatz, loadfile)
        if !grid_change && size(x, 2) == ansatz.model.batch
            [typeof(addr)(Tuple(col)) for col in eachcol(x)]
        else
            # fill(addr, ansatz.model.batch) # fill whole batch with same starting address
            coupled = coupled_modes(H, addr)
            truncated = ansatz.truncation === nothing ? (1:num_modes(addr)) : ansatz.truncation.keep
            active_range = intersect(truncated, coupled)
            addrs_random(H, addr, ansatz.model.batch; active = active_range, n_max=3)
        end
    else
        coupled = coupled_modes(H, addr)
        truncated = ansatz.truncation === nothing ? (1:num_modes(addr)) : ansatz.truncation.keep
        active_range = intersect(truncated, coupled)
        addrs_random(H, addr, ansatz.model.batch; active = active_range, n_max=3)
        # fill(addr, ansatz.model.batch) # fill whole batch with same starting address
    end
    backend = KernelAbstractions.get_backend(ansatz.model.x)
    addrs_n = KernelAbstractions.allocate(backend, eltype(addrs), ansatz.model.batch)
    copyto!(addrs_n, addrs)

    if markov
        log_markov_chain(markovfile, addrs_n; start=true)
    end

    # --- shared buffers ---------------------------------------------
    buffers = make_buffers(ansatz.model)
    jac_buf = JacobianBuffer(ansatz, buffers)
    vmc_buf = VMCBuffer(ansatz, addr)
    n_params = length(jac_buf.θ)
    @info "Neural Network parameters: $(n_params)"

    total = memory_estimate((ansatz, buffers, jac_buf, vmc_buf))
    @info "Memory estimate: $(_fmt_bytes(total))"

    # this is for more detailed memory print 
    # parts = ("ansatz"   => ansatz,
    #      "backprop" => buffers,
    #      "jacobian" => jac_buf,
    #      "vmc"      => vmc_buf)
    #
    # println("----- Memory estimate -----")
    # for (name, obj) in parts
    #     println("  ", rpad(name * ":", 10), _fmt_bytes(memory_estimate(obj)))
    # end
    # println("  ", rpad("total:", 10), _fmt_bytes(sum(memory_estimate ∘ last, parts)))
    # println("---------------------------")

    all_E     = Float32[]
    all_E_err = Float32[]
    all_var   = Float32[]

    tmp_neuron_statistics = ansatz.neuron_statistics
    tmp_jacobian_statistics = ansatz.jacobian_statistics
    ansatz.neuron_statistics = false # to plot it only every block
    ansatz.jacobian_statistics = false

    var_track = RunningMean{Float32}() # for tracking of variance spikes
    s_track = RunningMean{Float32}() # for traking of Jacobian scale spikes

    println(repeat("#", 100))
    @printf("  Training with: %d phase(s)", length(phases))
    println()
    println(repeat("#", 100))

    # --- phase loop -------------------------------------------------
    for (pidx, phase) in enumerate(phases)

        println()
        @printf("  Phase %d/%d  |  mode=%s,  optimiser=%s,  vmc_sampler=%s,  max_epochs=%-6d \n",
                pidx, length(phases), phase.mode, phase.optimiser, phase.vmc_sampler, phase.max_epochs)
        println(repeat("─", 100))
        @printf("%-6s %-18s %-12s %-12s %-11s %-11s %-9s %-10s\n",
                "Block", "E", "E_err", "Var", "|ΔE|", "|Δvar|", "Accept", "η_eff (LR)")
        println(repeat("─", 100))

        # truncation checks
        if phase.truncation !== nothing
            if ansatz.truncation === nothing || phase.truncation > ansatz.truncation.k
                change_truncation!(ansatz, H, phase.truncation)
            end
        end
        if ansatz.truncation !== nothing
            println(ansatz.truncation)
        end

        # output weights scaling
        if !isone(phase.output_scale)
            last_layer = last(ansatz.model.layers)
            last_layer.W .*= Float32(phase.output_scale)
            sync_θ!(jac_buf, ansatz)
            set_centering!(ansatz)
        end

        # build phase's optimiser buffer
        opt_symbol, opt_buf, vmc_symbol = _build_opt_buffer(phase, n_params, ansatz, var_track, s_track)

        block = BlockStats()
        E_hist = Float32[]
        E_err_hist = Float32[]
        var_hist = Float32[]
        epoch = 0
        converged = false
        last_accept = NaN
        phase_block_idx = 1

        η = phase.η
        eta_sum = 0f0
        η_dec_idx = isempty(phase.η_decrease) ? 0 : 1 # for decrease η criterion

        while !converged && epoch < phase.max_epochs
            epoch += 1

            burnin = _get_burnin(phase.skip, epoch)

            if epoch % phase.block_size == 0 && tmp_neuron_statistics !== false
                ansatz.neuron_statistics = tmp_neuron_statistics
            end
            if epoch % phase.block_size == 0 && tmp_jacobian_statistics !== false
                ansatz.jacobian_statistics = tmp_jacobian_statistics
            end

            E_mean, variance, last_addrs, acceptance, eta_eff =
                _run_epoch(opt_symbol, vmc_symbol, opt_buf, jac_buf, vmc_buf, H, ansatz, addrs_n;
                           burnin=burnin, mode=phase.mode,
                           λ=phase.λ, η=η)

            eta_sum += eta_eff
            addrs_n = last_addrs
            last_accept = acceptance
            push_epoch!(block, Array(E_mean)[1], Array(variance)[1])

            if markov
                log_markov_chain(markovfile, addrs_n; start=false)
            end

            # --- block analysis -------------------------------------
            if epoch % phase.block_size == 0
                if tmp_neuron_statistics !== false
                    ansatz.neuron_statistics = false
                end
                if tmp_jacobian_statistics !== false
                    ansatz.jacobian_statistics = false
                end

                E_b, E_err_b, var_b = block_summary(block)
                push!(E_hist, E_b)
                push!(E_err_hist, E_err_b)
                push!(var_hist, var_b)
                block = BlockStats()

                # --- η decrease schedule --------------------------------
                if η_dec_idx > 0 && var_b < phase.η_decrease[η_dec_idx][1]
                    η *= Float32(phase.η_decrease[η_dec_idx][2])
                    η_dec_idx = η_dec_idx < length(phase.η_decrease) ? η_dec_idx + 1 : 0
                end

                diff_E = length(E_hist) > 1 ? abs(E_hist[end] - E_hist[end-1]) : NaN
                diff_var = length(var_hist) > 1 ? abs(var_hist[end] - var_hist[end-1]) : NaN

                @printf("%-6d %-18.10f %-12.2e %-12.2e %-11.2e %-11.2e %-9.4f %-10.2e\n",
                        vmc_buf.block_idx, E_b, E_err_b, var_b,
                        isnan(diff_E) ? 0.0 : diff_E,
                        isnan(diff_var) ? 0.0 : diff_var,
                        last_accept, eta_sum/phase.block_size)
                flush(stdout)
                flush(stderr)
                eta_sum = 0f0

                # if vmc_buf.block_idx >= phase.block_min
                if phase_block_idx >= phase.block_min
                    converged = check_stop(phase.stop, E_hist, var_hist,
                                           last_accept, phase.patience)
                end
                vmc_buf.block_idx += 1
                phase_block_idx += 1
            end
        end # epoch loop

        println(repeat("─", 100))
        if converged
            @printf("  Phase %d converged after %d epochs  |  E = %.10f  |  var = %.6f\n",
                    pidx, epoch, last(E_hist), last(var_hist))
        else
            @printf("  Phase %d hit max_epochs (%d)\n", pidx, phase.max_epochs)
        end

        append!(all_E, E_hist)
        append!(all_E_err, E_err_hist)
        append!(all_var, var_hist)
    end # phase loop
    
    if save
        save_master(savefile, jac_buf.θ, addrs_n, ansatz)
    end

    # final VMC energy estimate of trained NN 
    final_elocs_statistics!(last(phases).vmc_sampler, vmc_buf, jac_buf, H, addrs_n, ansatz;
                            occ_file=occfile, plot_occ=occplot)
    
    return all_E, all_E_err, all_var, addrs_n
end
