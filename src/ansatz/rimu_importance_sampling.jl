
# -------------------------------------------------------------------
# --  DISPATCHES FOR RIMU IMPORTANCE SAMPLING -----------------------
# -------------------------------------------------------------------
"""
NeuralAnsatz can be used for Imporance Sampling (IS) in Rimu's FCIQMC.

Because of batched nature of Neural Network it calculates the IS ratio 
(ψ_target/ψ_source) in two stages. Both steps are using `result_dict` 
dictionary based result buffer saved inside [`NeuralAnsatz`](@ref).

First stage:
    It applies `*1/ψ_source` factor after spawning from `source::PDVec`. 
    Because all input configurations are known in `source` already, but
    spawning `targets` are still unknown.
    This is done in [`deposit!`] function. 
Second stage:
    After spawning (and summing all same `target` configurations) we can
    apply `*ψ_target` factor as all `targets` are already known here.
    This is done in [`_ansatz_modify_new!`] function.
"""

# Modified Hamiltonian is not modified here but I need this definition for
# my custom <:AnsatzSampling
function Rimu.Hamiltonians.modify_offdiagonal(
    h::Gutzwiller.AnsatzSampling{A,<:Any,<:Any,<:NeuralAnsatz,<:Any}, src, dst, value
    ) where {A}  
    return dst => value
end

"""
    _ansatz_first_modify!(source, ansatz)

This function computes first Importance Sampling iteration. It evaluates 
input vector of addresses `source` and computes ansatz values.

The result is hold in dictionary `ansatz.result_dict`.

# Variables

* `source`: `Rimu.PDVec` or `Rimu.DVec` type holding addresses.
* `ansatz`: [`NeuralAnsatz`](@ref).

"""
function _ansatz_first_modify!(source, ansatz)
    if ansatz.first_iter == true
        # collect addresses of source - 1st iteration
        empty!(ansatz.addrs_buffer)
        empty!(ansatz.result_buffer)
        empty!(ansatz.result_dict)
        for key in keys(source)
            push!(ansatz.addrs_buffer, key)
        end

        # run new addresses through NeuralAnsatz
        multi_compute_logψ(ansatz, ansatz.addrs_buffer, ansatz.result_buffer)

        sizehint!(ansatz.result_dict, total)
        for (k, v) in zip(ansatz.addrs_buffer, ansatz.result_buffer)
            ansatz.result_dict[k] = exp(v)
        end

        ansatz.first_iter = false
    end
end

"""
    _ansatz_modify_new!(working_memory, ansatz, adj)

This function applies the second stage of the Importance Sampling ratio `*ψ_target`.
It edits walker values in `target` which are saved in first column of `working_memory`
(see Rimu documentation).

# Variables

* `working_memory`: Rimu type of structure.
* `ansatz`: [`NeuralAnsatz`](@ref).
* `adj`: Boolean variable which control if Adjoint version of hamiltonian was applied.
    If so, the Importance Sampling ratio should be inversed.

"""
function _ansatz_modify_new!(working_memory, ansatz, adj)
    # collect spawned addresses in first column of working memory
    empty!(ansatz.addrs_buffer)
    empty!(ansatz.result_buffer)
    empty!(ansatz.result_dict)
    f = Rimu.DictVectors.first_column(working_memory)
    for seg in f.segments
        for key in keys(seg)
            push!(ansatz.addrs_buffer, key)
        end
    end

    # run new addresses through NeuralAnsatz
    multi_compute_logψ(ansatz, ansatz.addrs_buffer, ansatz.result_buffer)

    # New result became source in next iteration
    sizehint!(ansatz.result_dict, total) 
    for (k, v) in zip(ansatz.addrs_buffer, ansatz.result_buffer)
        ansatz.result_dict[k] = exp(v)
    end

    for seg in f.segments
        for key in keys(seg)
            ψ_dst    = ansatz.result_dict[key]  
            seg_val  = seg[key].value 
            if !adj
                seg[key] = Rimu.DictVectors.NonInitiatorValue(seg_val * ψ_dst)
            else
                if ψ_dst < eps(Float64)
                    println("Epsilon triggered *ψ!")
                    seg[key] = Rimu.DictVectors.NonInitiatorValue(seg_val / eps(Float64))
                else
                    seg[key] = Rimu.DictVectors.NonInitiatorValue(seg_val / ψ_dst)
                end
            end
        end
    end
end

"""
    Rimu.Interfaces.apply_operator!(compression, working_memory, target, source, ham, boost)

This is custom dispatch function for calling the [`NeuralAnsatz`](@ref) Importance Sampling (IS)
methods in Rimu. It applies IS ratio in two steps for batch approaches. Firstly, it applies
`*1/ψ_source` factor in spawning step. After spawning step all new `target` addresses are known. 
Secondly, it applies `*ψ_target` factor for `target` and thus compliting the IS ratio.
"""
function Rimu.Interfaces.apply_operator!( 
    compression::Rimu.CompressionStrategy,
    working_memory::Rimu.DictVectors.PDWorkingMemory, target::PDVec, source::PDVec,
    ham::Rimu.FirstOrderTransitionOperator{<:Any,<:Any,
         <:Gutzwiller.AnsatzSampling{Adj,<:Any,<:Any,<:NeuralAnsatz,<:Any}}, boost=1,
) where Adj
    # 1th iteration - first forward run of NN on source
    if ham.hamiltonian.ansatz.first_iter == true
        _ansatz_first_modify!(source, ham.hamiltonian.ansatz)
    end

    # here ansatz modify walker values after spawning (* 1/ψ_source)
    stat_names, stats = Rimu.DictVectors.perform_spawns!(working_memory, source, ham, boost)
    Rimu.DictVectors.collect_local!(working_memory)
    sync_stat_names, sync_stats = Rimu.DictVectors.synchronize_remote!(working_memory)

    # modify new walkers values with (* ψ_target) -> completing guiding ratio
    _ansatz_modify_new!(working_memory, ham.hamiltonian.ansatz, Adj)

    target, comp_stat_names, comp_stats = Rimu.DictVectors.move_and_compress!(compression, target, working_memory)

    stat_names = (stat_names..., comp_stat_names..., sync_stat_names...)
    stats = (stats..., comp_stats..., sync_stats...)

    return stat_names, stats, working_memory, target
end

"""
    Rimu.DictVectors.deposit!(c, k, val, parent) -> deposit!(c, k, val, p_addr => p_value)

This is custom dispatch function for calling the NeuralAnsatz Importance Sampling
methods in Rimu. This affects first stage of applying IS ratio `*1/ψ_source`.
"""
function Rimu.DictVectors.deposit!(
        c, k, val, parent::Pair{<:Rimu.Interfaces.AbstractOperatorColumn{<:Any,<:Any,  
            <:Rimu.FirstOrderTransitionOperator{<:Any,<:Any,
                    <:Gutzwiller.AnsatzSampling{Adj,<:Any,<:Any,<:NeuralAnsatz,<:Any}}}}
) where Adj

    ansatz = first(parent).hamiltonian.hamiltonian.ansatz
    addr = starting_address(first(parent))
    if !Adj 
        if ansatz.result_dict[addr] < eps(Float64)
            val = val / eps(Float64)
        else
            val = val / ansatz.result_dict[addr]
        end
    else
        val = val * ansatz.result_dict[addr]
    end

    return deposit!(c, k, val, addr => last(parent))
end


