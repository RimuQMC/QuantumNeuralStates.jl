# using Gutzwiller
# using Rimu

"""
    NeuralAnsatz(hamiltonian, model, batch_size; kwargs...) <: Gutzwiller.AbstractAnsatz             

A neural network-based variational ansatz for use with Rimu's FCIQMC.
Wraps a [`Chain`](@ref) neural network model and manages the buffers needed for batched evaluation
and importance sampling.

# Arguments
* `hamiltonian`: hamiltonian defined in `Rimu`.
* `model`: a `Chain` neural network model. 
* `batch_size`: number of configurations evaluated in a single forward
                pass. 

# Keyword Arguments
* `input_scale_func`: scaling function for inputs (default: `identity`).
* `max_norm`: maximum particle number for input normalisation. Must be 
                positive if provided.
* `multiforward_buffer`: can be set with `Int` number. This number would determine
                usually bigger batch forward passes. See [`MultiForwardBuffer`](@ref).
* `mean_field`: if set `true` the mean-field would add to neural network wave-funciton
                evaluation. Needs to be manually set, see [`MeanField`](@ref).
* `truncation`: can be used for input space truncation. See [`TruncationBuffer`](@ref).
* `neuron_statistics`: Can be activated with `true` (statistics would be print out to
                terminal), or `filename::String` to be saved in external file. See
                [`neuron_statistics`](@ref).
* `jacobian_statistics`: Can be activated with `true` (statistics would be print out to
                terminal), or `filename::String` to be saved in external file. see
                [`jacobian_statistics`](@ref).

# Example

```
julia> ansatz = NeuralAnsatz(H, model, 1024; input_scale_func=identity, max_norm=1, mean_field=false, 
                       neuron_statistics="./neuron.txt", jacobian_statistics="./jacob.txt", 
                       multiforward_buffer=batch*30)
```

## Fields
* `hamiltonian`: hamiltonian defined in `Rimu`.
* `model`: a neural network type of `Chain` used to evaluate the ansatz. 
* `x_cpu_buffer`: pre-allocated `Float32` buffer for batched network input.
* `addrs_buffer`: dynamically filled buffer of addresses, used during
                Rimu FCIQMC importance sampling.
* `result_buffer`: dynamically filled buffer of network outputs, paired
                with `addrs_buffer` during FCIQMC importance sampling.
* `result_dict`: cache mapping of results of neural network used during
                Rimu FCIQMC importance sampling
* `first_iter`: flag indicating whether the FCIQMC iteration is the first;
                used to control buffer initialisation logic.
* `input_scale_func`: a function applied to raw occupation inputs before they
                are fed to the network. Defaults to `identity` (uniform
                spread). Use e.g. `sqrt` to compress large occupation
                numbers and give smaller values relatively more importance.
* `max_norm`: Maximum total particle number in the system. When set,
                inputs are normalised so the highest possible occupation
                maps to `1`, keeping all inputs in `[0, 1]`. `nothing`
                disables this normalisation - by default.
* `normalisation`: Scale factor derived from `max_norm`. Ensures that
                if neural networks load pre-computed weights, they value
                invariant regardless the normalisation.
* `multi_forward_buffer`: if active it holds additional [`MultiForwardBuffer`](@ref).
                This allows to do forward pass with arbitrary batch size.
* `Mean-Field`: allows using mean-filed estimate with neural network. See
                [`MeanField`](@ref).
* `Truncation`: allows input truncation for graduate training in smaller 
                dimensions. See [`TruncationBuffer`](@ref).
* `neuron_statistics`: allows to print out statistics about neural network. Either,
                to terminal or external file. See [`neuron_statistics`](@ref).
* `jacobian_statistics: allows to print out statistics about jacobian (per-sample
                gradient). Either, to terminal or external file. See 
                [`jacobian_statistics`](@ref). 

"""
mutable struct NeuralAnsatz{A,T<:Real,H,M,X<:AbstractArray,XZ<:AbstractArray,XA<:AbstractArray{A},
                            XR<:AbstractArray,F,D,MFB,MF,TR} <: Gutzwiller.AbstractAnsatz{A,T,0}

    hamiltonian ::H
    model       ::M
    x_cpu_buffer::X
    z_cpu       ::XZ

    # Rimu FCIQMC helper buffers dynamically filled 
    addrs_buffer    ::XA
    result_buffer   ::XR
    result_dict     ::D
    first_iter      ::Bool

    # helper functions for Input preparation
    input_scale_func ::F
    max_norm         ::Union{Nothing, Int}
    normalisation    ::Float32

    # helper buffer structure for NN forward runs where length(input) > batch
    multi_forward_buffer::Union{Nothing,MFB}

    # Mean-Field 
    meanfield::Union{Nothing,MF}

    # Truncation
    truncation::Union{Nothing,TR}

    # Neuron Health Statistics
    neuron_statistics::Union{Bool,String}
    jacobian_statistics::Union{Bool,String}
end
function NeuralAnsatz(hamiltonian, model, batch_size; 
                    input_scale_func=identity, max_norm::Union{Nothing, Int}=nothing, 
                    multiforward_buffer=nothing, mean_field::Bool=false, truncation=nothing,
                    neuron_statistics::Union{Bool,String}=false, jacobian_statistics::Union{Bool,String}=false
    )
    addr = starting_address(hamiltonian)
    dim = size(model.x, 1)
    if batch_size == 1
        x_cpu_buffer = zeros(Float32, dim)
        # x_cpu_buffer = Vector{Float32}(undef, dim)
    else
        x_cpu_buffer = zeros(Float32, dim, batch_size)
        # x_cpu_buffer = Matrix{Float32}(undef, dim, batch_size)
    end
    z_cpu = Matrix{Float64}(undef, size(last(model.layers).z, 1), batch_size)

    A = typeof(addr)
    T = Float64
    H = typeof(hamiltonian)
    M = typeof(model)
    X = typeof(x_cpu_buffer)
    XZ = typeof(z_cpu)

    addrs_buffer  = A[]
    result_buffer = Float64[]
    result_dict   = Dict{A, T}()
    XA = typeof(addrs_buffer)
    XR = typeof(result_buffer)
    D  = typeof(result_dict)
    first_iter = true

    normalisation = if max_norm === nothing
        1.0f0
    elseif max_norm > 0
        Float32(inv(input_scale_func(Float32(max_norm))))
    else
        error("Normalisation must be positive! got $max_norm")
    end
    F = typeof(input_scale_func)

    multi_forward_buffer = if multiforward_buffer === nothing
        nothing
    else
        MultiForwardBuffer(model, addr, multiforward_buffer)
    end
    MFB=typeof(multi_forward_buffer)

    meanfield = if mean_field === false
        nothing
    else
        MeanField(hamiltonian)
    end
    MF=typeof(meanfield)

    trun = build_truncation(hamiltonian, truncation)
    TR=typeof(trun)

    return NeuralAnsatz{A,T,H,M,X,XZ,XA,XR,F,D,MFB,MF,TR}(
                hamiltonian, model, x_cpu_buffer, z_cpu, addrs_buffer, result_buffer, result_dict, 
                first_iter, input_scale_func, max_norm, normalisation, multi_forward_buffer, meanfield, 
                trun, neuron_statistics, jacobian_statistics)
end

"""
    prepare_input!(ansatz, addr, x_cpu_buffer) -> ansatz.x_cpu_buffer

Converts Rimu input notation `addr` into Array{Float32} as input for Neural Network
`ansatz.x_cpu_buffer`. It uses `Rimu.onr()` function for collecting the 
occupation number configurations. It manages all batch sizes.
"""
function prepare_input!(na::NeuralAnsatz, addr, x_cpu_buffer)
    if isa(na.model.layers[1], Dense)
        if na.max_norm === nothing
            x_cpu_buffer .= na.input_scale_func.(Float32.(onr(addr))) #.+ 0.1f0
        else
            x_cpu_buffer .= na.input_scale_func.(Float32.(onr(addr))) .* na.normalisation #.+ 0.1f0
        end
        return x_cpu_buffer
    end
end
function prepare_input!(na::NeuralAnsatz, addrs::AbstractVector, x_cpu_buffer)
    if isa(na.model.layers[1], Dense)
        # Each column is one address from batch 
        if na.max_norm === nothing
            @inbounds for i in eachindex(addrs)
                col = @view x_cpu_buffer[:, i]
                col .= (na.input_scale_func.(Float32.(onr(addrs[i])))) #.+ 0.1f0
            end
        else
            @inbounds for i in eachindex(addrs)
                col = @view x_cpu_buffer[:, i]
                col .= (na.input_scale_func.(Float32.(onr(addrs[i])))) .* na.normalisation #.+ 0.1f0 # norm here is 1/norm corrected
            end
        end
        return x_cpu_buffer
    end
end

"""
    prepare_input_occ!(ansatz, addr, x_cpu_buffer) -> x_cpu_buffer

Similar as [`prepare_input!`](@ref), but always converts to occupation 
representation. This function is used with [`MeanField`](@ref) calculations
as the mean-field require pure occupation number representation.
"""
function prepare_input_occ!(na::NeuralAnsatz, addr, x_cpu_buffer)
    if isa(na.model.layers[1], Dense)
        x_cpu_buffer .= (Float32.(onr(addr))) #.+ 0.1f0
        return x_cpu_buffer
    end
end
function prepare_input_occ!(na::NeuralAnsatz, addrs::AbstractVector, x_cpu_buffer)
    if isa(na.model.layers[1], Dense)
        # Each column is one address from batch 
        @inbounds for i in eachindex(addrs)
            col = @view x_cpu_buffer[:, i]
            col .= Float32.(onr(addrs[i])) #.+ 0.1f0
        end
        return x_cpu_buffer
    end
end
 
"""
    compute_logψ(ansatz, addr) -> logψ
    compute_logψ(ansatz, addr, multi_forward_buffer) -> logψ

Calculates log of the wave-function value prediction of the Neural Network 
from the `addr` input configuration.

If dispatched with `multi_forward_buffer` it allows for computation in 
arbitrary batch size. See [`MultiForwardBuffer`](@ref).
"""
function compute_logψ(na::NeuralAnsatz, addr)
    x = prepare_input!(na, addr, na.x_cpu_buffer)
    logψ = na.model(x)
    return logψ
end
function compute_logψ(na::NeuralAnsatz, addr, multi_forward_buffer)
    x = prepare_input!(na, addr, multi_forward_buffer.x_cpu)
    logψ = na.model(x, multi_forward_buffer)
    return logψ
end

"""
    compute_mflogψ!(ansatz, addr, z) 
    compute_mflogψ!(ansatz, addr, z, multi_forward_buffer)

Similar as [`compute_logψ`](@ref) but with using [`MeanField`](@ref).
The function add in-place mean-field value to result `z`.

If dispatched with `multi_forward_buffer` it allows for computation in 
arbitrary batch size. See [`MultiForwardBuffer`](@ref).
"""
function compute_mflogψ!(na::NeuralAnsatz, addr, z)
        x = prepare_input_occ!(na, addr, na.x_cpu_buffer)
        logψ = na.meanfield(x, z)
    return nothing
end
function compute_mflogψ!(na::NeuralAnsatz, addr, z, multi_forward_buffer)
    x = prepare_input_occ!(na, addr, multi_forward_buffer.x_cpu)
    logψ = na.meanfield(x, z)
    return nothing
end

"""
    multi_compute_logψ!(ansatz, flat_addrs_m, flat_vals_m)

This function allows evaluation of inputs bigger than batch size. It calls
[`compute_logψ`](@ref) (and possibly [`compute_mflogψ!`](@ref)) in loop to accomodate
inputs exceeding batch size and accumulates results into `flat_vals_m` array.

The indexing of input `flat_addrs_m` vector and accumulated result `flat_vals_m` is 
preserved.

# Variables

* `ansatz`: structure of [`NeuralAnsatz`](@ref).
* `flat_addrs_m`: vector of addresses in Rimu format. Can have arbitrary length (
    usually beyond batch size0
* `flat_vals_m`: the result of `logψ` computations are saved in this array which is dynamically
    sized.
"""
function multi_compute_logψ!(ansatz::NeuralAnsatz, flat_addrs_m::AbstractArray, flat_vals_m::AbstractArray)    
    empty!(flat_vals_m) 
    if ansatz.multi_forward_buffer !== nothing
        batch = ansatz.multi_forward_buffer.buffer_size
    else
        batch = size(ansatz.x_cpu_buffer, 2)
    end
    total = length(flat_addrs_m)
    n_chunks = cld(total, batch)
    # chunked NN forward pass over all off-diagonal addresses
    for c in 1:n_chunks
        i_start = (c-1) * batch + 1
        i_end   = min(c * batch, total)
        n_real  = i_end - i_start + 1

        tmp_addrs = view(flat_addrs_m, i_start:i_end)  # slicing of vector for batch size pass

        if ansatz.multi_forward_buffer !== nothing
            raw = compute_logψ(ansatz, tmp_addrs, ansatz.multi_forward_buffer)
            copyto!(ansatz.multi_forward_buffer.z_cpu, raw)    

            if ansatz.meanfield !== nothing 
                compute_mflogψ!(ansatz, tmp_addrs, ansatz.multi_forward_buffer.z_cpu, ansatz.multi_forward_buffer)
            end

            append!(flat_vals_m, view(ansatz.multi_forward_buffer.z_cpu, 1, 1:n_real))
        else
            raw = compute_logψ(ansatz, tmp_addrs)
            copyto!(ansatz.z_cpu, raw)    

            if ansatz.meanfield !== nothing
                compute_mflogψ!(ansatz, tmp_addrs, ansatz.z_cpu)
            end

            append!(flat_vals_m, view(ansatz.z_cpu, 1, 1:n_real))
        end
    end
end

"""
    compute_ψ_64(ansatz, addr) -> Float64.(exp.(logψ))

Similar as [`compute_logψ`](@ref), but returns exp(logψ) values in Float64 format. 
This way the return value is compatible with Rimu format. It is used in Rimu's
Importance Sampling.

## Note
The `Float32 -> Float64` transfer is realised with `copyto!` where we copy 
`model` results to `Float64` buffer and the conversion happens automatically.
"""
function compute_ψ_64(na::NeuralAnsatz, addr)
    x = prepare_input!(na, addr, na.x_cpu_buffer)
    logψ = na.model(x)
    copyto!(na.z_cpu, logψ)
    na.z_cpu .= exp.(na.z_cpu)
    # return exp.(Float64.(logψ .- 40f0))
    return na.z_cpu
end
function compute_ψ_64(na::NeuralAnsatz, addr, multi_forward_buffer)
    x = prepare_input!(na, addr, multi_forward_buffer.x_cpu)
    logψ = na.model(x, multi_forward_buffer)
    copyto!(multi_forward_buffer.z_cpu, logψ)
    multi_forward_buffer.z_cpu .= exp.(multi_forward_buffer.z_cpu)
    # return exp.(Float64.(logψ .- 40f0))
    return multi_forward_buffer.z_cpu
end

"""
    (na)(addr, params)

Calls [`compute_ψ_64`](@ref) function during Rimu Importance Sampling 
calculations. 
"""
function (na::NeuralAnsatz)(addr)
    return compute_ψ_64(na, addr)
end
function (na::NeuralAnsatz)(addr, multi_forward_buffer)
    return compute_ψ_64(na, addr, multi_forward_buffer)
end

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
    This is done in [`deposit!`](@ref) function. 
Second stage:
    After spawning (and summing all same `target` configurations) we can
    apply `*ψ_target` factor as all `targets` are already known here.
    This is done in [`_ansatz_modify_new!`](@ref) function.
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


