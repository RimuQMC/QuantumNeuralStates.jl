
"""
    BlockStats(energies, variance)

This structure acumulates each iteration's mean energy and variance.
It is used in [`block_summary`](@ref) for statistics over block.
"""
mutable struct BlockStats
    energies::Vector{Float64}
    variances::Vector{Float64}
end
BlockStats() = BlockStats(Float64[], Float64[])

function push_epoch!(bs::BlockStats, E, var)
    push!(bs.energies,  E)
    push!(bs.variances, var)
end

"""
    block_summary(bs::BlockStats)
            -> E_mean, E_err, var_mean

This function calculates statistics of training data over blocks.
""" 
function block_summary(bs::BlockStats)
    E_mean = mean(bs.energies)
    E_std = std(bs.energies)
    E_err = E_std / sqrt(length(bs.energies))
    var_mean = mean(bs.variances)
    return E_mean, E_err, var_mean
end

"""
    StopBuffer

This structure holds all possible options for early stopping in neural 
network training loop. 

It is define either as `Float64`, value for chosen condition, or as 
`Nothing`, which means it is not used. By default all options are
set to `nothing`.

# Note

Special field is `require_all`, boolean. If `true` then all chosen 
conditions needs to be fulfilled before stop condition is triggered.
If `false` than stop condition is triggered when any condition is 
fulfilled.

# Example
```julia
stop = StopBuffer(var_thr=0.01)
stop = StopBuffer(var_thr=0.01, E_thr=0, require_all=true)
```
"""
Base.@kwdef struct StopBuffer
    ΔE_thr::Union{Float64, Nothing} = nothing       # difference of Energy over blocks
    Δvar_thr::Union{Float64, Nothing} = nothing     # difference of variance over blocks
    E_thr::Union{Float64, Nothing} = nothing        # energy of 1 block
    var_thr::Union{Float64, Nothing} = nothing      # variance of 1 block
    accept_thr::Union{Float64, Nothing} = nothing   # acceptence of Markov Chain threshold
    require_all::Bool = false      # if ALL or ANY conditions needs to meet
end

"""
    check_stop(sb::StopBuffer, E_hist, var_hist, accept, patience)

This function is cheching if stop conditions chosen in [`StopBuffer`](@ref) have
been fulfilled or not. If stop condition is defined over blocks, then it will
look only in `patience` number of blocks for this condition.
"""
function check_stop(sb::StopBuffer, E_hist, var_hist, accept, patience)
    results = Bool[]
    n = length(E_hist)
    # chech conditions in last number of "patience" blocks 
    if !isnothing(sb.ΔE_thr) && n >= patience + 1
        push!(results, all(abs(E_hist[i] - E_hist[i-1]) < sb.ΔE_thr
                           for i in (n - patience + 1):n))
    end
    if !isnothing(sb.Δvar_thr) && n >= patience + 1
        push!(results, all(abs(var_hist[i] - var_hist[i-1]) < sb.Δvar_thr
                           for i in (n - patience + 1):n))
    end
    if !isnothing(sb.E_thr) && n >= 1
        push!(results, last(E_hist) < sb.E_thr)
    end
    if !isnothing(sb.var_thr) && n >= 1
        push!(results, last(var_hist) < sb.var_thr)
    end
    if !isnothing(sb.accept_thr) && !isnan(accept)
        push!(results, accept < sb.accept_thr)
    end
    isempty(results) && return false
    return sb.require_all ? Base.all(results) : Base.any(results)
end

"""
    TrainingPhase

One stage of training. This structure allows to define more training phases
with different parameters.

# Variables

* `mode`: define used loss function (and its gradients), see [`apply_loss!`](@ref) 
        (default `:energy`).
* `optimiser`: define what optimiser is used for parameters update 
        (default `:adam`, others `:minSR`, `:minSR_momentum`, `:descent`). See also
        [`adam`](@ref), [`descent`](@ref), [`minSR`](@ref).
* `vmc_sampler`: define type of VMC sampler (default `:metropolis`). See also
        [`vmc_sample!`](@ref).
* `stop`: define a early-stop conditions, see [`StopBuffer`](@ref).
* `η`: learning rate used with parameters update (default `0.001f0`).
* `λ`: Tikhonov regularisation used in [`minSR`](@ref) optimiser (default `0.001f0`).
* `skip`: define how many VMC steps should be skip before parameters update is made.
        Thermalisation of VMC sampler (default `(1, 100)`). Multiple options can be 
        stuck in Tuple of form `(epoch, skip)` - which means from epoch number 
        `epoch` apply skip value `skip`. Example: `[(1, 1000), (100, 500), 
        (1000, 100)]`.
* `η_decrease`: allows to decrease `η` learning rate and can be stuck in similar
        way as `skip` variable. `(var_thr, factor)` when variance threashold is 
        reached than `factor` is applied to `η` as `η_new = η*factor`. Example:
        `[(1, 0.1), (0.1, 0.1)]` (defaultly not use).
* `truncation`: if in [`NeuralAnsatz`](@ref) truncation is defined, this value 
        allows to increase truncation with each phase. Allowing smoother 
        increase in input truncation space. It is recommended to ONLY increase
        truncation value! (defaultly nothing).
* `block_size`: define how many iterations should be stuck to `block` for 
        printout (happens in blocks) and block statistics [`BlockStats`](@ref),
        [`block_summary`](@ref) (default `10`).
* `block_min`: define how many blocks needs to be iterated before early 
        stop-conditions can be applied (default `3`).
* `patience`: used in [`check_stop`](@ref), if block-based stop condition is 
        chosen, the condition is checked in last `patience` number of blocks
        (default `3`).
* `max_epochs`: maximum epochs in learning loop (default `1000`). If none of
        early-stopping conditions are met this is safe-check stop condition. 

# Notes

"""
Base.@kwdef struct TrainingPhase
    mode::Union{Symbol, Tuple} = :energy
    optimiser::Symbol = :adam
    vmc_sampler::Symbol = :metropolis
    stop::StopBuffer = StopBuffer()
    η::Float32 = 0.001f0
    λ::Float32 = 0.001f0
    skip::Vector{Tuple{Int,   Int}} = [(1, 100)]
    η_decrease::Vector{Tuple{Float32,Float32}} = []
    truncation::Union{Nothing, Int} = nothing
    block_size::Int = 10
    block_min::Int = 5
    patience::Int = 3
    max_epochs::Int = 1_000
end


"""Return burnin for the current epoch using the skip schedule."""
function _get_burnin(skip::Vector{Tuple{Int,Int}}, epoch::Int)
    burnin = skip[1][2]            
    for (ep, b) in skip
        if epoch >= ep 
            burnin = b  # last matching burnin stays 
        end
    end
    return burnin
end

"""Build the right optimiser buffer for a phase."""
function _build_opt_buffer(phase, n_params, ansatz)
    opt = phase.optimiser
    vmc = phase.vmc_sampler

    if opt === :minSR return :minSR, minSRBuffer(ansatz.model.batch, n_params, ansatz; 
                                                 velocity=false, β=0.9f0), vmc
    elseif opt === :minSR_momentum return :minSR, minSRBuffer(ansatz.model.batch, n_params, ansatz; 
                                                              velocity=true,  β=0.9f0), vmc
    elseif opt === :descent return :descent, DescentBuffer(ansatz.model.batch, n_params, ansatz), vmc
    elseif opt === :adam return :adam, AdamBuffer(ansatz.model.batch, n_params, ansatz; 
                                                  β1=0.9f0, β2=0.999f0, ε=1f-3), vmc
    else error("Unknown optimiser: $opt")
    end
end

"""Dispatch one epoch to the correct update function."""
function _run_epoch(opt::Symbol, vmc::Symbol, opt_buf, jac_buf, vmc_buf, H, ansatz, addrs_n;
                    burnin, mode, λ, η)
    kw = (; vmc, burnin, mode, λ, η)

    if opt === :minSR return minSR(jac_buf, vmc_buf, opt_buf, H, ansatz, addrs_n; kw...)
    elseif opt === :descent return descent(jac_buf, vmc_buf, opt_buf, H, ansatz, addrs_n; kw...)
    elseif opt === :adam return adam(jac_buf, vmc_buf, opt_buf, H, ansatz, addrs_n; kw...)
    else error("Unknown optimiser: $opt")
    end
end
