# using Rimu: onr

# Define custom input transforms as NAMED functions
sqrtlog1p(x) = sqrt(log1p(x))
log1plog1p(x)  = log1p(log1p(x))

"""
    SCALE_FUNCTIONS

Registry of the input scaling functions available for [`NeuralAnsatz`](@ref). Maps a
name (`Symbol`) to the function that is applied to each occupation number before it
enters the network,

```math
x_m = f(n_m) \\cdot \\mathrm{normalisation} ,
```

see [`_prepare_input_kernel!`](@ref) and [`_prepare_input_stream_kernel!`](@ref).

# Available functions
* `:identity`: raw occupation numbers.
* `:sqrt`: `√n`.
* `:log1p`: `log(1 + n)`.
* `:sqrtlog1p`, `:log1plog1p`: composed variants that compress large occupations more
    strongly.

# Adding a new function
Define a scalar function and register it under its own name:

    myscale(x) = ...
    SCALE_FUNCTIONS[:myscale] = myscale      # or add it to the definition below

# Requirements
* The function is called inside GPU kernels on `Float32` values: it must be scalar,
    type-stable (`Float32 → Float32`) and device-compatible (no allocations, no calls
    that only exist on the CPU).
* The key must equal the function's name, `Symbol(nameof(f))`. [`save_master`](@ref)
    stores the function by its name, and [`load_master`](@ref) looks it up here, so a
    mismatch prevents a saved model from being loaded.
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
                            n_iter = 100, n_skip = 5, burnin = 50) -> res

Estimate the variational energy of the trained [`NeuralAnsatz`](@ref) with an error bar.
Run at the end of [`run_training_loop`](@ref).

Starting from the last walker addresses `addrs` of the training loop, the sampler is first
re-equilibrated for `burnin` steps. It then records `n_iter` measurements, separated by
`n_skip` sampler steps each. Every measurement is the weighted batch mean

```math
E_t = \\sum_{b=1}^{B} w_b\\, E_{\\mathrm{loc}}(n_b) ,
```

and the time series ``E_t`` is analysed with `Rimu.blocking_analysis`, which accounts for
the autocorrelation between measurements. In total `n_iter * batch` local energies are used.

# Arguments
* `vmc_sampler`: sampler type, `:metropolis` or `:ctmc`, see [`vmc_sample!`](@ref).
* `vmc_buf`: VMC buffers, see [`VMCBuffer`](@ref).
* `jac_buf`: Jacobian buffer, see [`JacobianBuffer`](@ref).
* `H`: Rimu Hamiltonian.
* `addrs`: walker addresses at the end of training.
* `ansatz`: the trained [`NeuralAnsatz`](@ref).

# Keyword Arguments
* `n_iter`: number of measurements (default `100`).
* `n_skip`: sampler steps per measurement; `n_skip - 1` steps without measurement are made
    in between to reduce correlations (default `5`).
* `burnin`: re-equilibration steps before the first measurement (default `50`).

# Returns
* `res`: result of `Rimu.blocking_analysis`, with the mean energy and its error estimate.

# Notes
* The error bar from the blocking analysis is the reliable one. The `E_err` printed during
    training is computed per block without this correlation analysis and can be too small.
* The mean-field, if present in `ansatz`, is included automatically through the samplers.
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

    E_t = Vector{Float64}(undef, n_iter)        # time series of weighted batch means

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
    println("\nFinal VMC estimate: $n_iter measurements × $(ansatz.model.batch) walkers, every $n_skip steps")
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

"""
    RunningMean{T}

Running mean of `log(x)` (geometric mean of recent epochs) with outlier detection.
Used for spike detection on the energy variance and for drop detection on the Jacobian
scale `s`. Shared across phases, so the baseline survives phase changes.

# Fields
* `rmean`: running mean of `log(x)`.
* `init`: `false` until the first valid value has been seen.
* `n_spike`: number of consecutive outlier epochs.
"""
mutable struct RunningMean{T}
    rmean::T
    init::Bool
    n_spike::Int
end
RunningMean{T}() where {T} = RunningMean{T}(zero(T), false, 0)

"""
    is_outlier!(r, x; up = true, factor = 10, max_spike = 10) -> Bool

`true` if `x` deviates from the running (geometric) mean by more than `factor`:
upward if `up = true` (variance spikes), downward if `up = false` (drops of `s`).
More than `max_spike` outliers in a row are treated as a new regime: the running mean
adopts the current value. Non-finite or non-positive `x` is ignored.
"""
function is_outlier!(r::RunningMean{T}, x::T; up::Bool = true,
                     factor::T = T(10), max_spike::Int = 10) where {T}
    (isfinite(x) && x > zero(T)) || return false
    lx = log(x)
    if !r.init                                         # first valid value of the run
        r.rmean, r.init, r.n_spike = lx, true, 0
        return false
    end
    dev = up ? lx - r.rmean : r.rmean - lx             # > 0: towards the unsafe side
    if dev > log(factor)
        r.n_spike += 1
        if r.n_spike > max_spike                       # persistent → new regime
            r.rmean, r.n_spike = lx, 0
            return false
        end
        return true                                    # genuine outlier
    end
    r.n_spike = 0
    ρ = dev < zero(T) ? T(0.3) : T(0.05)               # fast to the safe side, slow otherwise
    r.rmean = (1 - ρ) * r.rmean + ρ * lx
    return false
end
