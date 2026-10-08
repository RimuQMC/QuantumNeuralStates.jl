# using Rimu: onr

# Define custom input transforms as NAMED functions
sqrtlog1p(x) = sqrt(log1p(x))
log1plog1p(x)  = log1p(log1p(x))

"""
    SCALE_FUNCTIONS

This dictionary holds binding with custom input scaling functions.
Those functions can be defined here and added to the dictionary.
"""
const SCALE_FUNCTIONS = Dict(
    :identity   => identity,
    :sqrt       => sqrt,
    :log1p      => log1p,
    :sqrtlog1p  => sqrtlog1p,
    :log1plog1p => log1plog1p,
)

"""
    update!(ansatz, jac, θ_new)

Updates all parameters in `chain` given a flat parameter update vector `θ_new` of size (p,).

# Arguments
* `ansatz`: Wave-function ansatz, [`NeuralAnsatz`](@ref). See also [`Chain`](@ref)
* `jac`: [`JacobianBuffer`](@ref) which holds `jac.ranges` information about mapping of flatten parameter
    vector to each `chain` layer.
* `θ_new`: flatten parameter vector with new updated weights after optimisation step.
"""
function update!(ansatz, jac::JacobianBuffer, θ_new::AbstractVector)
    chain = ansatz.model
    amplitude_output = view(last(chain.layers).z, 1, :)
    ansatz.logψ_centering = maximum(amplitude_output)

    map((layer, r) -> _set_params!(layer, θ_new, r), chain.layers, jac.ranges)
    return nothing
end

"""
    _set_params!(layer, θ, r::LayerRange)

Copy `layer`'s slice of the flat parameter vector `θ` (given by `r`) into
`layer.W`/`layer.b` (and `layer.layer_norm.γ`/`β` if present). No-op for
`FreeLayer`s (no parameters). Used by [`update!`](@ref).
"""
_set_params!(::FreeLayer, θ, r) = nothing            # no parameters
function _set_params!(layer::ParametricLayer, θ, r::LayerRange)
    layer.W .= reshape(view(θ, r.W), size(layer.W))
    layer.b .= view(θ, r.b)
    if !isnothing(r.γ)
        layer.layer_norm.γ .= reshape(view(θ, r.γ), size(layer.layer_norm.γ))
        layer.layer_norm.β .= reshape(view(θ, r.β), size(layer.layer_norm.β))
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
    final_elocs_statistics!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz;
                            n_iter = 100, n_skip = 5, burnin = 50)

This function is run at the end of the [`run_training_loop`](@ref) to see what is variational
energy of learned [`NeuralAnsatz`](@ref). It is using `BlockingAnalysis` from Rimu and
Gutzwiller.

It runs MC sampler from last addresses `addrs` visited in training loop and collects local
energies to calculate the cariational energy. It is using same parameters as the training loop.
The number of samples collected for this analysis is `batch*batch_iter`.
"""
function final_elocs_statistics!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz;
                                 n_iter = 100, n_skip = 5, burnin = 50)
    ansatz.neuron_statistics = false
    ansatz.jacobian_statistics = false

    # re-equilibrate after the last parameter update
    vmc_buf.start = false
    for _ in 1:burnin
        addrs, = vmc_sample!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz)
    end

    E_t = Vector{Float64}(undef, n_meas)        # time series of weighted batch means

    for t in 1:n_iter
        # decorrelating steps (no E_loc)
        vmc_buf.start = false
        for _ in 1:(n_skip - 1)
            addrs, = vmc_sample!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz)
        end

        # measurement step
        vmc_buf.start = true
        addrs, E_locs, weights, _, _ = vmc_sample!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz)

        e = Float64.(Array(E_locs))
        w = Float64.(Array(weights))
        E_t[t] = sum(w .* e)    # weights already normalized
    end

    res = Rimu.blocking_analysis(E_t)
    println("\nFinal VMC estimate: $n_meas measurements × $(ansatz.model.batch) walkers, every $n_skip steps")
    println(res)
    return res
end

""" 
    safe_denom(x, epsilon)

Function that safe checks near zero values of `x`. The `epsilon` define
how close I want to be careful around zero and return safe value.
"""
@inline function safe_denom(x::T, epsilon::T) where T
    ax = abs(x)
    return ax < epsilon ? copysign(epsilon, x) : x
end

"""
    select_device()

This function check what GPU package was used in script, and return its
GPU() function (`mtl()` - Metal, `cu()` - CUDA). If no GPU backend was 
loaded it returns `identity()` - CPU.
"""
function select_device()
    if isdefined(Main, :CUDA) && Main.CUDA.functional()
        @info "CUDA (cu) was loaded for GPU computations"
        return Main.CUDA.cu
    elseif isdefined(Main, :Metal) && Main.Metal.functional()
        @info "Metal (mtl) was loaded for GPU computations"
        return Main.Metal.mtl
    else
        @info "No functional GPU backend found, using CPU"
        return identity
    end
end
