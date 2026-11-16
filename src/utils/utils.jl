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

    foreach((layer, r) -> _set_params!(layer, θ_new, r), chain.layers, jac.ranges)
    set_centering!(ansatz)

    return nothing
end

"""
    _set_params!(layer, θ, r::LayerRange)

Copy `layer`'s slice of the flat parameter vector `θ` (given by `r`) into
`layer.W`/`layer.b` (and `layer.layer_norm.γ`/`β` if present). 
Used by [`update!`](@ref).
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
    _get_params!(θ, layer, r::LayerRange)

Copy `layer`'s parameters (`W`, `b`, and `layer_norm.γ`/`β` if present) into
the slice of the flat parameter vector `θ` given by `r`. Inverse of
[`_set_params!`](@ref).
"""
_get_params!(θ, ::FreeLayer, r) = nothing
function _get_params!(θ, layer::ParametricLayer, r::LayerRange)
    view(θ, r.W) .= reshape(layer.W, :)
    view(θ, r.b) .= layer.b
    if !isnothing(r.γ)
        view(θ, r.γ) .= reshape(layer.layer_norm.γ, :)
        view(θ, r.β) .= reshape(layer.layer_norm.β, :)
    end
    return nothing
end

"""
    sync_θ!(jac::JacobianBuffer, ansatz)

Refresh `jac.θ` from the current weights of `ansatz.model`. Call this after
modifying the network's parameters directly.
"""
function sync_θ!(jac::JacobianBuffer, ansatz)
    foreach((layer, r) -> _get_params!(jac.θ, layer, r), ansatz.model.layers, jac.ranges)
    return jac.θ
end

"""
    addrs_random(ham, addr, batch; active = 1:num_modes(addr), n_max = 3)

`batch` random addresses, only on the `active` modes (all others stay 0, e.g.
truncated sites).

* Hamiltonian with `mode_cutoff` (no fixed particle number): each active mode gets an
  independent occupation from `0:min(n_max, mode_cutoff)`.
* Otherwise (fixed particle number N = num_particles(addr)): the N particles are placed
  one at a time on random active modes, each holding at most `cld(N, length(active))`
  particles (the average occupation, rounded up). `n_max` is not used here.
"""
function addrs_random(ham, addr, batch; active = 1:num_modes(addr), n_max::Int = 3)
    M = num_modes(addr)
    fixed_N = !hasproperty(ham, :mode_cutoff)
    tmp = Vector{Int}(undef, M)
    addrs = Vector{typeof(addr)}(undef, batch)

    if fixed_N  # conserving number of particles
        N = num_particles(addr)
        n_cap = cld(N, length(active))
    else        # not bounded number of particles
        n_top = min(n_max, ham.mode_cutoff)
    end

    for i in 1:batch
        fill!(tmp, 0)                                # truncated modes stay 0
        if fixed_N
            for _ in 1:N                             # one particle at a time
                m = rand(active)
                while tmp[m] >= n_cap                # mode full → pick another
                    m = rand(active)
                end
                tmp[m] += 1
            end
        else
            for m in active
                tmp[m] = rand(0:n_top)               # independent, 0 … n_max
            end
        end
        addrs[i] = typeof(addr)(Tuple(tmp))
    end
    return addrs
end

"""
    final_elocs_statistics!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz;
                            n_iter=100, n_skip=5, burnin=50,
                            plot_occ=true, occ_file="occupation.png")
        -> res, N_all, counts, n_mode

Estimate the variational energy of the trained [`NeuralAnsatz`](@ref) with an error bar,
and record the occupation statistics of the sampled walkers.
Run at the end of [`run_training_loop`](@ref).

Starting from the last walker addresses `addrs` of the training loop, the sampler is first
re-equilibrated for `burnin` steps. It then records `n_iter` measurements, separated by
`n_skip` sampler steps each. Every measurement is the weighted batch mean

```math
E_t = \\sum_{b=1}^{B} w_b\\, E_{\\mathrm{loc}}(n_b), \\qquad \\sum_{b=1}^{B} w_b = 1,
```

and the time series ``E_t`` is analysed with `Rimu.blocking_analysis`, which accounts for
the autocorrelation between measurements. In total `n_iter * B` local energies are used.

Local energies are **never clipped** here (`eclamp=false`), so the estimate is unbiased.

# Arguments
* `vmc_sampler`: sampler type, `:metropolis` or `:ctmc`, see [`vmc_sample!`](@ref).
* `vmc_buf`: VMC buffers, see [`VMCBuffer`](@ref).
* `jac_buf`: Jacobian buffer, see [`JacobianBuffer`](@ref).
* `H`: Rimu Hamiltonian.
* `addrs`: walker addresses at the end of training. **Updated in place.**
* `ansatz`: the trained [`NeuralAnsatz`](@ref).

# Keyword Arguments
* `n_iter`: number of measurements (default `100`).
* `n_skip`: sampler steps per measurement. The `n_skip - 1` steps in between are made
  without measuring, to reduce correlations. `n_skip ≤ 1` measures every step
  (default `5`).
* `burnin`: re-equilibration steps before the first measurement (default `50`).
* `plot_occ`: if `true`, plot the occupation statistics (default `true`).
* `occ_file`: file name for the occupation plots (default `"occupation.pdf"`).

# Returns
* `res`: result of `Rimu.blocking_analysis`, with the mean energy and its error estimate.
* `N_all`: total occupation `N` of every measured walker sample (`n_iter * B` entries).
* `counts`: `Dict{Int, Int}` with the number of walker samples for each total occupation `N`.
* `n_mode`: `(M,)` average occupation of each mode over all walker samples.

# Notes
* The error bar from the blocking analysis is the reliable one. The `E_err` printed during
  training is computed per block without this correlation analysis and can be too small.
* If `plot_occ = true`, two bar plots are saved to `occ_file`: the number of walker
  samples per total occupation `N` (log scale) and the average occupation per mode.
* The mean-field, if present in `ansatz`, is included automatically through the samplers.
* `ansatz.neuron_statistics` and `ansatz.jacobian_statistics` are set to `false` and
  are not restored.
* With few measurements (small `n_iter`), the blocking analysis may not converge. Increase
  `n_iter` if `res` reports this.
"""
function final_elocs_statistics!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz;
                                 n_iter = 100, n_skip = 5, burnin = 50,
                                 plot_occ = true, occ_file = "occupation.pdf")
    ansatz.neuron_statistics = false
    ansatz.jacobian_statistics = false

    # re-equilibrate after the last parameter update
    vmc_buf.start = false
    for _ in 1:burnin
        addrs, = vmc_sample!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz; eclamp=false)
    end

    E_t = Vector{Float64}(undef, n_iter)    # time series of weighted batch means
    M = num_modes(first(Array(addrs)))
    N_all = Int[]                           # raw total occupation of every walker sample
    n_mode = zeros(Float64, M)              # summed occupation per mode
    sizehint!(N_all, n_iter * ansatz.model.batch)

    for t in 1:n_iter
        # decorrelating steps (no E_loc)
        vmc_buf.start = false
        for _ in 1:(n_skip - 1)
            addrs, = vmc_sample!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz; eclamp=false)
        end

        # measurement step
        vmc_buf.start = true

        # occupations of every walker (addresses copied to the CPU)
        for ad in Array(addrs)
            o = onr(ad)
            push!(N_all, sum(o))
            for k in 1:M
                n_mode[k] += o[k]
            end
        end

        addrs, E_locs, weights, _, _ = vmc_sample!(vmc_sampler, vmc_buf, jac_buf, H, addrs, ansatz; 
                                                   eclamp=false)

        e = Float64.(Array(E_locs))
        w = Float64.(Array(weights))
        E_t[t] = sum(w .* e)    # weights already normalized
    end
    n_mode ./= length(N_all)                    # average over all walker samples

    res = Rimu.blocking_analysis(E_t)
    println("\nFinal VMC estimate: $n_iter measurements × $(ansatz.model.batch) walkers, every $n_skip steps")
    println(res)

    # raw counts per total occupation N
    counts = Dict{Int, Int}()
    for N in N_all
        counts[N] = get(counts, N, 0) + 1
    end
    N_vals = sort!(collect(keys(counts)))
    println("total occupation: mean = $(sum(N_all) / length(N_all)),  ",
            "min = $(first(N_vals)),  max = $(last(N_vals)),  samples = $(length(N_all))")
    k_max = argmax(n_mode)
    println("per mode: largest average = $(n_mode[k_max]) in mode $k_max,  ",
            "smallest = $(minimum(n_mode))")

    if plot_occ
        p1 = bar(N_vals, [counts[N] for N in N_vals]; yscale = :log10, label = "",
                 xlabel = "total occupation N", ylabel = "walker samples",
                 title = "total occupation ($(length(N_all)) samples)")
        p2 = bar(1:M, n_mode; label = "",
                 xlabel = "mode k", ylabel = "average occupation ⟨n_k⟩",
                 title = "average occupation per mode", xticks = 1:M)
        p = plot(p1, p2; layout = (2, 1), size = (750, 700))
        savefig(p, occ_file)
        println("occupation plots saved to $occ_file")
    end

    return res, N_all, counts, n_mode
end

"""
    coupled_modes(H, addr) -> Vector{Int}

Modes whose occupation `H` can change from the configuration `addr`: every mode that
some off-diagonal move with a non-zero, finite matrix element acts on. Modes outside
this set have a conserved occupation from `addr`, e.g. the k = 0 phonon of the
Froehlich model, and must stay empty in random starts.

# Return
Vector of allowed indices in address modes.
"""
function coupled_modes(H, addr)
    M = num_modes(addr)
    coupled = falses(M)
    col = H * addr
    o0 = onr(addr)
    for (a, v) in offdiagonals(col)
        (iszero(v) || !isfinite(v)) && continue
        o = onr(a)
        for m in 1:M
            o[m] != o0[m] && (coupled[m] = true)     # this move created a particle in m
        end
    end
    return findall(coupled)
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

"""
    set_centering!(ansatz; first=false)

Set `ansatz.logψ_centering` to the maximum of the network output for log ψ,
and return it.

The output of the neural network for log ψ can drift by an arbitrary constant
during training, since the normalization of the wavefunction is not fixed.
Subtracting this centering value keeps the values in a numerically safe range,
for example to avoid overflow in `exp(logψ)`.

# Keyword arguments
- `first::Bool=false`: if `true` (initialization, or after loading weights),
  the centering is computed from `compute_logψ` at the starting address of
  `ansatz.hamiltonian`. If `false` - default (inside the training loop), it is computed by
  re-evaluating `ansatz.model` on the last input cached in `ansatz.model.x`.

# Returns
- The new value of `ansatz.logψ_centering`.
"""
function set_centering!(ansatz; first=false)
    if first # initialization phase / loading weights
        addr = starting_address(ansatz.hamiltonian)
        out = compute_logψ(ansatz, addr)
    else # training loop - centering set from last input saved in NN model::Chain
        out = ansatz.model(ansatz.model.x)
    end
    ansatz.logψ_centering = maximum(out)
    return ansatz.logψ_centering
end
