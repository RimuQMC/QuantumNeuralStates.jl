# using Rimu: onr

"""
    SCALE_FUNCTIONS

This dictionary holds binding with custom input scaling functions.
Those functions can be defined here and added to the dictionary.
"""
# Define custom input transforms as NAMED functions
sqrtlog1p(x) = sqrt(log1p(x))
log1plog1p(x)  = log1p(log1p(x))

# for load scaling factors
const SCALE_FUNCTIONS = Dict(
    :identity   => identity,
    :sqrt       => sqrt,
    :log1p      => log1p,
    :sqrtlog1p  => sqrtlog1p,
    :log1plog1p => log1plog1p,
)

"""
    update!(chain, jac, θ_new)

Updates all parameters in `chain` given a flat parameter update vector `θ_new` of size (p,).

# Arguments

* `chain`: Neural Network model. See [`Chain`](@ref)
* `jac`: Jacobian buffer which holds `jac.ranges` information about mapping of flatten parameter
    vector to each `chain` layer.
* `θ_new`: flatten parameter vector with new updated weights after optimisation step.

"""
function update!(chain::Chain, jac::JacobianBuffer, θ_new::AbstractVector)
    for (layer, r) in zip(chain.layers, jac.ranges)
        layer.W .= reshape(view(θ_new, r.W), size(layer.W))
        layer.b .= view(θ_new, r.b)
        if !isnothing(r.γ)
            layer.layer_norm.γ .= reshape(view(θ_new, r.γ), size(layer.layer_norm.γ))
            layer.layer_norm.β .= reshape(view(θ_new, r.β), size(layer.layer_norm.β))
        end
    end
    return nothing
end

"""
    addrs_random(ham, addr, batch)

Generates random input vector of addresses (in Rimu notation) for whole batch.
"""
function addrs_random(ham, addr, batch)
  cap_bool = false
  M = num_modes(addr)
  N = try
        ham.mode_cutoff
      catch
        cap_bool = true
        num_particles(addr)
      end
  tmp   = Vector{Int}(undef, M)
  addrs = Vector{typeof(addr)}(undef, batch)

  for i in 1:batch  
    cap = 0
    for m in 1:length(tmp)  # number of modes
        if cap_bool
            remaining_modes = M - m
            if remaining_modes == 0
                tmp[m] = N - cap  # last mode gets whatever is left
            else
                r = rand(0:(N-cap))
                tmp[m] = r 
                cap += r
            end
        else
            r      = rand(0:N)
            tmp[m] = r 
        end
    end
    # a         = typeof(addr)(tmp)
    a = Base.typename(typeof(addr)).wrapper(tmp)
    addrs[i]  = a
  end
  
  return addrs
end

"""
    final_elocs_statistics!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz; batch_iter=100)

This function is run at the end of the [`run_training_loop`](@ref) to see what is variational
energy of learned [`NeuralAnsatz`](@ref). It is using `BlockingAnalysis` from Rimu and
Gutzwiller.

It runs MC sampler from last addresses `addrs` visited in training loop and collects local
energies to calculate the cariational energy. It is using same parameters as the training loop.
The number of samples collected for this analysis is `batch*batch_iter`.
"""
function final_elocs_statistics!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz; batch_iter=100)
    vmc_buf.start = true # sanity check to allow Elocs calculations
    batch = size(last(ansatz.model.layers).z, 2)
    E_block = Vector{Rimu.StatsTools.BlockingResult{Float64}}()

    for i in 1:batch_iter
        new_addrs, E_locs, weights, _, _ = vmc_sample!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz)
        addrs = new_addrs
        if weights === nothing
            result = Rimu.blocking_analysis(E_locs)
            push!(E_block, result) # blocking analysis do mean (1/N uniform weights factor)
        else
            E_locs .= E_locs .* weights .* batch # counterterm for extra blocking analysis norm
            result = Rimu.blocking_analysis(E_locs)
            push!(E_block, result) # blocking analysis do mean (1/N uniform weights factor)
        end
    end
    println()
    println("Final blocking analysis on $(batch_iter*batch) E_locs samples")
    n = batch_iter
    combined_result = Gutzwiller.CombinedBlockingResult(
        mean(r.mean for r in E_block),
        √(mean(r.err^2 for r in E_block) / n),
        √(mean(r.err_err^2 for r in E_block) / n),
        E_block,
    )
    println(combined_result)
    return nothing
end
