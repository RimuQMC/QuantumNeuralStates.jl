
"""
    _print_stats(name, arr; io=stdout)

Prints into chosen `io` statistics of array as mean,variance, minimum, and maximum. 
This is meant for investigating health of Neural Network's parameters.
"""
function _print_stats(name, arr; io=stdout)
    v = vec(arr)
    m   = Float32(mean(v))
    s2  = Float32(var(v))
    mn  = Float32(minimum(v))
    mx  = Float32(maximum(v))
    @printf(io, "  %-14s mean=% .4e   var=% .4e   min=% .4e   max=% .4e\n", name, m, s2, mn, mx)
end

"""
    _activation_health(a, z, act; sat_thresh=0.95, io=stdout)

Prints into chosen `io` information about pre-activation and post-activation health in Neural Network (NN).
"""
function _activation_health(a::AbstractMatrix, z::AbstractMatrix, act; sat_thresh=0.95, io=stdout)
    n_total = length(a)

    if act === tanh || act === tanh_fast
        n_sat     = Int(sum(abs.(z) .> Float32(sat_thresh)))     # broadcast + sum, GPU-safe
        n_lin     = Int(sum(abs.(a) .< Float32(0.1)))
        frac_sat  = n_sat / n_total
        frac_lin  = n_lin / n_total
        @printf(io, "   %-14s |z|>%.2f : %6.2f%%  (%d / %d)\n",
                "saturation", sat_thresh, 100*frac_sat, n_sat, n_total)
        @printf(io, "   %-14s |a|<0.10 : %6.2f%%  (%d / %d)   [near-linear regime]\n",
                "under-driven", 100*frac_lin, n_lin, n_total)

    elseif act === relu
        n_dead    = Int(sum(a .<= 0))
        frac_dead = n_dead / n_total
        @printf(io, "   %-14s a<=0        : %6.2f%%  (%d / %d)  [zero-gradient units]\n",
                "dead (ReLU)", 100*frac_dead, n_dead, n_total)

        per_neuron_dead = vec(all(a .<= 0, dims=2))     # reduction along batch dim, GPU-safe
        n_always_dead   = Int(sum(per_neuron_dead))
        n_neurons       = length(per_neuron_dead)
        @printf(io, "   %-14s always-dead : %6.2f%%  (%d / %d neurons dead for ENTIRE batch)\n",
                "dead (ReLU)", 100*n_always_dead/n_neurons, n_always_dead, n_neurons)

    elseif act === gelu
        n_starved    = Int(sum(a .< -3.0))
        frac_starved = n_starved / n_total
        @printf(io, "   %-14s z < -3.0 : %6.2f%%  (%d / %d)  [gradient-starved tail]\n",
                "starved (GELU)", 100*frac_starved, n_starved, n_total)
    else
        @printf(io, "   %-14s (no specific health check for act=%s)\n", "health", act)
    end
end

"""
    neuron_statistics(ansatz; sat_thresh=0.95, idx::Int=0)

This function dispatches different `io` options for Neural Network statistics printing.
Specific variables are saved in [`NeuralAnsatz`](@ref), which determine if statistics is
printed out into terminal, file, or nowhere.

# Keyword Arguments

* `sat_thresh`: represent saturation threshold for sigmoid-like activation functions.
* `idx`: is a dummy variable that accounts for number of epoch/block in training phase.
"""
function neuron_statistics(ansatz; sat_thresh=0.95, idx::Int=0)
    if ansatz.neuron_statistics isa String
        open(ansatz.neuron_statistics, "a") do io
            neuron_statistics(ansatz, io; sat_thresh=sat_thresh, idx=idx)
        end
    elseif ansatz.neuron_statistics === true
        neuron_statistics(ansatz, stdout; sat_thresh=sat_thresh, idx=idx)
    end
end

function neuron_statistics(ansatz, io; sat_thresh=0.95, idx::Int=0)
    println(io)
    println(io, "$(idx) "*"="^90)
    println(io, "  Input  ($(size(ansatz.x_cpu_buffer, 1)))")
    println(io, "  "*"─"^90)
    _print_stats("  input", ansatz.x_cpu_buffer; io=io)

    for (li, layer) in enumerate(ansatz.model.layers)
        if isa(layer, Dense)
            W = layer.W
            b = layer.b
            act = layer.act_func

            println(io, "  "*"─"^90)
            println(io, "  Layer $li  ($(size(W,2)) -> $(size(W,1)),  act=$act)")
            println(io, "  "*"─"^90)

            _print_stats(" W", W; io=io)
            _print_stats(" b", b; io=io)

            a = layer.a
            z = layer.z

            _print_stats(" a (pre-act)", a; io=io)
            _print_stats(" z (post-act)", z; io=io)

            _activation_health(a, z, act; sat_thresh=sat_thresh, io=io)

        else
            error("Unknown layer type: $layer!")
        end
    end
    println(io, "  "*"="^90)
    # return nothing
end

"""
    jacobian_statistics(ansatz, J::AbstractMatrix; lambda::Float32=1f-3, idx::Int=0)

This function dispatches different `io` options for Jacobian statistics printing.
Specific variables are saved in [`NeuralAnsatz`](@ref), which determine if statistics is
printed out into terminal, file, or nowhere.

# Keyword Arguments

* `lambda`: represent threshold for SVD of Jacobian `J` matrix.
* `idx`: is a dummy variable that accounts for number of epoch/block in training phase.
"""
function jacobian_statistics(ansatz, J::AbstractMatrix; lambda::Float32=1f-3, idx::Int=0)
    if ansatz.jacobian_statistics isa String
        open(ansatz.jacobian_statistics, "a") do io
            jacobian_statistics(J, io; lambda=lambda, idx=idx)
        end
    elseif ansatz.jacobian_statistics === true
        jacobian_statistics(J, stdout; lambda=lambda, idx=idx)
    end
end

function jacobian_statistics(J::AbstractMatrix, io; lambda::Float32=1f-3, idx::Int=0)
    println(io)
    n_params, B = size(J)
    layer_name="full_jacobian"

    # --- 1) per-sample gradient norm ---
    norms = vec(sqrt.(sum(abs2, J; dims=1)))
    @printf(io,  "%i [%s] grad-norm   mean=%.4e  var=%.4e  min=%.4e  max=%.4e\n",
            idx, layer_name, Float32(mean(norms)), Float32(var(norms)),
            Float32(minimum(norms)), Float32(maximum(norms)))

    # --- 2) trusted directions vs lambda ---
    O = Array{Float32}(J)
    sv = svdvals(O)                      # descending, length = min(n_params, B)
    sigma_thresh = sqrt(lambda)          # σ² > λ  <=>  σ > sqrt(λ)
    n_trusted = count(>(sigma_thresh), sv)
    n_total   = length(sv)

    col_norms = sqrt.(sum(abs2, O; dims=1)) # ||O_c|| per sample norm
    Ohat = O ./ max.(col_norms, 1f-12) # O_ic / ||O_c||
    G = Ohat' * Ohat # cos(θ) = v * w -> matrix-wise all possible cos(θ_ij) = G[i,j]
    off_diag_mean = (sum(G) - B) / (B*(B-1)) # gives mean cos() of all offdiagoanl elements
    @printf(io,  "  [%s] avg cos(θ_ij) between sample grads: %.4f  (1.0 = fully collinear/degenerate)\n",
            layer_name, Float32(off_diag_mean))

    @printf(io,  "  [%s] trusted directions (σ > sqrt(λ)=%.2e): %d / %d\n",
            layer_name, sigma_thresh, n_trusted, n_total)
end

