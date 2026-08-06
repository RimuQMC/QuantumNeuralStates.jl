
"""
    run_training_loop(H, ansatz, addr, phases; kwargs...)
            -> E_history, E_err_history, var_history, final_addrs

Run training through a sequence of TrainingPhase's. Each phase starts when 
the previous one converges (or hits `max_epochs`).

# Variables

* `H`: hamiltonian defined in Rimu.
* `ansatz`: ansatz that evaluates wave-function. See [`NeuralAnsatz`](@ref).
* `addr`: Rimu stype of Fock address also used in `H` definition.
* `phases`: details of each phase are saved in [`TrainingPhase`](@ref) and they 
        can be stuck as: `phases = [TrainingPhase(), TrainingPhase(), ...]`.

# Keyword Arguments

* `savefile`: name of file where parameters of [`NeuralAnsatz`](@ref) are saved.
        See also [`save_master`](@ref). (default `weights.txt`)
* `loadfile`: name of file from which parameters are loaded to [`NeuralAnsatz`](@ref).
        See also [`load_master`](@ref). (default `weights.txt`)
* `save`: boolean value if I want to save parameters at the end of training loop
        (default `true`).
* `load`: boolean value if I want to load saved parameters and weights for the 
        training loop (default `false`).
* `markovfile`: name of file where all Markov Chains of the training loop are
        saved. This is meant for diagnostics only as the number of Markov
        Chains can be enourmous (default `MarkovChain.txt`).
* `markov`: boolean value if I want to save the Markov Chains (default `false`).
"""
function run_training_loop(H, ansatz, addr, phases::Vector{TrainingPhase};
                           savefile::String = "weights.txt", loadfile::String = "weights.txt",
                           save::Bool   = true, load::Bool = false,
                           markovfile::String = "MarkovChain.txt" , markov::Bool = false)

    @info "Hilbert space dimension: $(@sprintf("%.3e", dimension(H)))"

    # --- initialise addresses ---------------------------------------
    addrs_n = if isfile(loadfile) && load
        x = load_master(ansatz, loadfile)
        [typeof(addr)(Tuple(col)) for col in eachcol(x)]
    else
        # addrs_random(H, addr, steps)
        fill(addr, ansatz.model.batch) # fill whole batch with same starting address
        # this is prefered as the neural network randomly sample anyway at the 
        # beginning and also it is necessary for input truncation to work
    end

    if markov
        log_markov_chain(markovfile, addrs_n; start=true)
    end

    # --- shared buffers ---------------------------------------------
    buffers = map(DenseBuffer, ansatz.model.layers)
    jac_buf = JacobianBuffer(ansatz.model, buffers)
    vmc_buf = VMCBuffer(ansatz, addr)
    n_params = length(jac_buf.θ)

    all_E     = Float64[]
    all_E_err = Float64[]
    all_var   = Float64[]

    tmp_neuron_statistics = ansatz.neuron_statistics
    tmp_jacobian_statistics = ansatz.jacobian_statistics
    ansatz.neuron_statistics = false # to plot it only every block
    ansatz.jacobian_statistics = false

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
                "Block", "E_block", "E_err", "Var_block", "|ΔE|", "|Δvar|", "Accept", "η (LR)")
        println(repeat("─", 100))

        if phase.truncation !== nothing
            if ansatz.truncation === nothing || phase.truncation > ansatz.truncation.k
                change_truncation!(ansatz, H, phase.truncation)
            end
        end

        opt_symbol, opt_buf, vmc_symbol = _build_opt_buffer(phase, n_params, ansatz)

        block       = BlockStats()
        E_hist      = Float64[]
        E_err_hist  = Float64[]
        var_hist    = Float64[]
        epoch       = 0
        converged   = false
        last_accept = NaN

        η          = phase.η
        η_dec_idx  = isempty(phase.η_decrease) ? 0 : 1 # for decrease η criterion

        while !converged && epoch < phase.max_epochs
            epoch += 1

            burnin = _get_burnin(phase.skip, epoch)

            if epoch % phase.block_size == 0 && tmp_neuron_statistics !== false
                ansatz.neuron_statistics = tmp_neuron_statistics
            end
            if epoch % phase.block_size == 0 && tmp_jacobian_statistics !== false
                ansatz.jacobian_statistics = tmp_jacobian_statistics
            end

            E_mean, variance, last_addrs, acceptance =
                _run_epoch(opt_symbol, vmc_symbol, opt_buf, jac_buf, vmc_buf, H, ansatz, addrs_n;
                           burnin=burnin, mode=phase.mode,
                           λ=phase.λ, η=η)

            addrs_n     = last_addrs
            last_accept = acceptance
            push_epoch!(block, E_mean, variance)

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
                push!(E_hist,     E_b)
                push!(E_err_hist, E_err_b)
                push!(var_hist,   var_b)
                block = BlockStats()

                # --- η decrease schedule --------------------------------
                if η_dec_idx > 0 && var_b < phase.η_decrease[η_dec_idx][1]
                    η *= Float32(phase.η_decrease[η_dec_idx][2])
                    η_dec_idx = η_dec_idx < length(phase.η_decrease) ? η_dec_idx + 1 : 0
                end

                diff_E   = length(E_hist)   > 1 ? abs(E_hist[end]   - E_hist[end-1])   : NaN
                diff_var = length(var_hist)  > 1 ? abs(var_hist[end] - var_hist[end-1]) : NaN

                @printf("%-6d %-18.10f %-12.2e %-12.2e %-11.2e %-11.2e %-9.4f %-10.2e\n",
                        vmc_buf.block_idx, E_b, E_err_b, var_b,
                        isnan(diff_E)   ? 0.0 : diff_E,
                        isnan(diff_var) ? 0.0 : diff_var,
                        last_accept, η)

                if vmc_buf.block_idx >= phase.block_min
                    converged = check_stop(phase.stop, E_hist, var_hist,
                                           last_accept, phase.patience)
                end
                vmc_buf.block_idx += 1
            end
        end # epoch loop

        println(repeat("─", 100))
        if converged
            @printf("  Phase %d converged after %d epochs  |  E = %.10f  |  var = %.6f\n",
                    pidx, epoch, last(E_hist), last(var_hist))
        else
            @printf("  Phase %d hit max_epochs (%d)\n", pidx, phase.max_epochs)
        end

        append!(all_E,     E_hist)
        append!(all_E_err, E_err_hist)
        append!(all_var,   var_hist)
    end # phase loop
    
    if save
        save_master(savefile, jac_buf.θ, addrs_n, ansatz)
    end

    final_elocs_statistics!(last(phases).vmc_sampler, vmc_buf, jac_buf, H, addrs_n, ansatz)
    
    return all_E, all_E_err, all_var, addrs_n
end
