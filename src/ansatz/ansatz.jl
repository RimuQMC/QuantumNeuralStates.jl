# using Gutzwiller
# using Rimu

"""
    NeuralAnsatz(ansatz_type, hamiltonian, model, batch_size; kwargs...) <: Gutzwiller.AbstractAnsatz             

A neural network-based variational ansatz for use with Rimu's FCIQMC.
Wraps a [`Chain`](@ref) neural network model and manages the buffers needed for batched evaluation
and importance sampling.

# Arguments
* `ansatz_type`: it is [`AnsatzType`](@ref) which determine how the wave-function ansatz, using neural
        network outputs, should looks like.
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
```julia
julia> M = 10
julia> N = 10
julia> batch = 1024
julia> model = build_model("FCNN", [M, 100, 100, 100, 1], tanh_fast; batch=batch)
julia> addr = near_uniform(BoseFS{N,M})
julia> H = HubbardReal1D(addr; u=0.1)
julia> ansatz = NeuralAnsatz(LogPsi(), H, model, batch)
```

## Fields
* `ansatz_type`: it is [`AnsatzType`](@ref) which determine how the wave-function ansatz, using neural
        network outputs, should looks like.
* `hamiltonian`: hamiltonian defined in `Rimu`.
* `model`: a neural network type of `Chain` used to evaluate the ansatz. 
* `logψ_centering`: this is mean centering over batched `model` output connected with 
                wave-function amplitude - log|ψ| (fighting the gauge invariant for 
                multiplication of wave-function with any number coefficient)
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
mutable struct NeuralAnsatz{AT<:AnsatzType,A,T<:Real,H,M,X<:AbstractArray,XZ<:AbstractArray,XA<:AbstractArray{A},
                            XR<:AbstractArray,D,F,MN<:Union{Nothing,Int},MFB<:Union{Nothing,MultiForwardBuffer},
                            MF<:Union{Nothing,MeanField},TR<:Union{Nothing,TruncationBuffer},
                            NS<:Union{Bool,String}, JS<:Union{Bool,String}} <: Gutzwiller.AbstractAnsatz{A,T,0}
    ansatz_type::AT
    hamiltonian::H
    model::M
    logψ_centering::Float32 # centers model output corresponding to ψ amplitude
    x_cpu_buffer::X
    z_cpu::XZ

    # Rimu FCIQMC helper buffers dynamically filled 
    addrs_buffer::XA
    result_buffer::XR
    result_dict::D
    first_iter::Bool

    # helper functions for Input preparation
    input_scale_func::F
    max_norm::MN
    normalisation::Float32

    # helper buffer structure for NN forward runs where length(input) > batch
    multi_forward_buffer::MFB

    # Mean-Field 
    meanfield::MF

    # Truncation
    truncation::TR

    # Neuron Health Statistics
    neuron_statistics::NS
    jacobian_statistics::JS
end
function NeuralAnsatz(ansatz_type::AnsatzType, hamiltonian, model, batch_size; 
                    input_scale_func=identity, max_norm::Union{Nothing, Int}=nothing, 
                    multiforward_buffer=nothing, mean_field::Bool=false, truncation=nothing,
                    neuron_statistics::Union{Bool,String}=false, jacobian_statistics::Union{Bool,String}=false
    )
    AT=typeof(ansatz_type)
    # safe check of ansatz_type and number of model outputs
    @assert (ansatz_type.num_outputs == size(last(model.layers).z, 1)) "" * 
            "Number of outputs in model (neural network) does not correspond with ansatz_type!"

    addr = starting_address(hamiltonian)
    dim = size(model.x, 1)
    x_cpu_buffer = zeros(Float32, dim, batch_size)
    z_cpu = Matrix{Float64}(undef, size(last(model.layers).z, 1), batch_size)

    nn_output = model(x_cpu_buffer)
    logψ_centering = maximum(nn_output)

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

    F=typeof(input_scale_func)
    normalisation = if max_norm === nothing
        1.0f0
    elseif max_norm > 0
        Float32(inv(input_scale_func(Float32(max_norm))))
    else
        error("Normalisation must be positive! got $max_norm")
    end
    MN=typeof(max_norm)

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

    NS=typeof(neuron_statistics); JS=typeof(jacobian_statistics)

    return NeuralAnsatz{AT,A,T,H,M,X,XZ,XA,XR,D,F,MN,MFB,MF,TR,NS,JS}(
                ansatz_type, hamiltonian, model, logψ_centering, x_cpu_buffer, z_cpu, 
                addrs_buffer, result_buffer, result_dict, first_iter, 
                input_scale_func, max_norm, normalisation, multi_forward_buffer, 
                meanfield, trun, neuron_statistics, jacobian_statistics)
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
        batch = ansatz.model.batch
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

            append!(flat_vals_m, view(ansatz.multi_forward_buffer.z_cpu, :, 1:n_real))
        else
            raw = compute_logψ(ansatz, tmp_addrs)
            copyto!(ansatz.z_cpu, raw)    

            if ansatz.meanfield !== nothing
                compute_mflogψ!(ansatz, tmp_addrs, ansatz.z_cpu)
            end

            append!(flat_vals_m, view(ansatz.z_cpu, :, 1:n_real))
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

