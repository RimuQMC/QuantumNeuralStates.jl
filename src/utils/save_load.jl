

"""
    encoding_signature(chain) -> String

Signature of the input encoding of `chain`: type, grid size, channels and any
encoding-specific options, e.g. `OccupationEncoding(grid=(4, 4), C=1)`.
The grid size may differ when loading (see [`load_master`](@ref)).
"""
encoding_signature(chain) = _signature(chain.enc)

"""
    nn_signature(chain) -> String

Signature of the network layers of `chain` (type, shape, kernel, channels, stride,
padding, activation, `LayerNorm`), joined by `" | "`. Must match exactly when loading.
"""
nn_signature(chain) = join(map(_signature, chain.layers), " | ")

"""
    chain_signature(chain) -> String

Full one-line signature (encoding and layers), for display.
"""
chain_signature(chain) = encoding_signature(chain) * " | " * nn_signature(chain)

"""
    _write_architecture(io, chain)

Write the architecture under an `"Architecture:"` header as two lines:
the [`encoding_signature`](@ref) and the [`nn_signature`](@ref).
See [`save_master`](@ref).
"""
function _write_architecture(io, chain)
    println(io, "Architecture:")
    println(io, encoding_signature(chain))
    println(io, nn_signature(chain))
end

"""
    _write_weights(io, θ)

Write weights to an open IO stream.

# Example
    Weights:
        1.0 0.3 11.2 5.9 ...
"""
function _write_weights(io::IO, θ::AbstractVector)
    println(io, "Weights:")
    join(io, Float32.(Array(θ)), ' ')
    println(io)
end

"""
    _write_addrs(io, addrs; num_width=3)

Write addresses to an open IO stream. Each address is converted via `onr()`
to an integer occupation vector and printed with fixed-width formatting.

# Example
    Inputs:
      0   0   0   0   0,   1   0   0   0   0, ...
"""
function _write_addrs(io::IO, addrs::AbstractVector; num_width::Int=3)
    addrs_cpu = Array(addrs)
    println(io, "Inputs:")
    n_addrs = length(addrs_cpu)
    for (j, a) in enumerate(addrs_cpu)
        occ = Int.(onr(a))
        join(io, [@sprintf("%*d", num_width, v) for v in occ], ' ')
        j < n_addrs && print(io, ", ")
    end
    println(io)
end

""" 
    _write_input_scale(io, ansatz)

Write input scaling factors to an open IO stream 
"""
function _write_input_scale(io::IO, ansatz)
    println(io, "Input Scaling:")
    println(io, "    scale function: ", nameof(ansatz.input_scale_func))
    println(io, "    max_norm:       ", ansatz.max_norm === nothing ? "nothing" : ansatz.max_norm)
    println(io, "    normalisation:  ", ansatz.normalisation)
end

""" 
    _write_mf(io, ansatz)

Write into IO stream if mean-field was used for training.
"""
function _write_mf(io::IO, ansatz)
    println(io, "Mean-Field:")
    println(io, "    used: ", ansatz.meanfield !== nothing)
end

"""
    save_master(filename, θ, addrs, ansatz; num_width=3)

Master function that saves everything needed to reconstruct a trained
[`NeuralAnsatz`](@ref): the model's architecture signature, input scaling
configuration, flat parameter vector, and the current VMC walker addresses.

Loaded back with [`load_master`](@ref), which verifies the saved architecture
and input scaling match the freshly-constructed `ansatz` before restoring `θ`.

## Example of txt file
    Architecture:
        OccupationEncoding(grid=(4, 4), C=1)
        Conv((3,),1=>100,relu,stride=1,pad=Periodic,LN=false) | Pool(mean) | Dense(100=>1,identity,LN=false)

    Input Scaling:
        scale function:
        max_norm:
        normalisation:

    Mean-Field:
        used: true

    Weights:
        <Float32 values>

    Inputs:
        0   0   0   0   0,   1   0   0   0   0, ...

# Arguments
* `filename`: path to write to.
* `θ`: flat parameter vector (e.g. `jac.θ`).
* `addrs`: current VMC walker addresses, saved to resume sampling.
* `ansatz`: the `NeuralAnsatz` being saved.

# Keywords
* `num_width`: column width used when formatting `addrs`.

# Notes
* The mean-field itself is not saved: it is rebuilt from the Hamiltonian when the
    `ansatz` is constructed.
"""
function save_master(filename::String, θ::AbstractVector, addrs::AbstractVector, ansatz;
                             num_width::Int=3)
    open(filename, "w") do io
        _write_architecture(io, ansatz.model)
        println(io)
        _write_input_scale(io, ansatz)
        println(io)
        _write_mf(io, ansatz)
        println(io)
        _write_weights(io, θ)
        println(io)
        _write_addrs(io, addrs; num_width)
    end
    @info "Saving $(length(θ)) weights, $(length(addrs)) addresses and " *
          "$(nameof(ansatz.input_scale_func)) input scaling (max_norm = $(ansatz.max_norm)) " *
          "to $filename"
end

# helper: pulls "log1p" from string "scale function: log1p"
parse_kv(line) = strip(split(line, ":", limit=2)[2])

"""
    parse_encoding(line) -> (; type, grid, C, extras)

Parse an encoding signature such as `MomentumEncoding(grid=(16, 16), C=3, occupation=true)`
into its type name, grid size, channel count, and a `Dict` of any further `key=value` options.
"""
function parse_encoding(line::AbstractString)
    m = match(r"^(\w+)\(grid=\(([^)]*)\),\s*C=(\d+)(.*)\)$", strip(line))
    m === nothing && error("Cannot parse encoding signature: $line")
    type   = String(m[1])
    grid   = Tuple(parse.(Int, filter(!isempty, strip.(split(m[2], ',')))))   # handles "(16,)"
    C      = parse(Int, m[3])
    extras = Dict{String,String}()
    for kv in split(m[4], ',')
        kv = strip(kv)
        isempty(kv) && continue
        k, v = split(kv, '='; limit = 2)
        extras[strip(k)] = strip(v)
    end
    return (; type, grid, C, extras)
end

"""
    load_master(ansatz, filename::String) -> x

Load a saved model — written by [`save_master`](@ref) — into `ansatz`,
after verifying it's compatible with the saved one (architecture, 
input encoding, and input scaling).

Returns the saved walker addresses `x` (matrix of `Int`) and `grid_changed`. If the grid
changed, the saved addresses belong to the old grid and must not be used.

# Arguments
* `ansatz`: the `NeuralAnsatz` to load into. See [`NeuralAnsatz`](@ref).
* `filename`: path to the saved file.
"""
function load_master(ansatz, filename::String)
    lines = readlines(filename)
    # find the data lines by looking for the section headers
    arch_idx = findfirst(==("Architecture:"), lines)
    sc_idx = findfirst(==("Input Scaling:"), lines)
    mf_idx = findfirst(==("Mean-Field:"), lines)
    w_idx = findfirst(==("Weights:"), lines)
    i_idx = findfirst(==("Inputs:"),  lines)
    @assert arch_idx !== nothing "Missing 'Architecture:' header"
    @assert sc_idx !== nothing "Missing 'Input Scaling:' header"
    @assert mf_idx !== nothing "Missing 'Mean-Field:' header"
    @assert w_idx !== nothing "Missing 'Weights:' header"
    @assert i_idx !== nothing "Missing 'Inputs:' header"

    # --- encoding check ---
    saved_enc = parse_encoding(lines[arch_idx + 1])
    current_enc = parse_encoding(encoding_signature(ansatz.model))

    saved_enc.type == current_enc.type ||
        error("Encoding type mismatch: saved $(saved_enc.type), current $(current_enc.type)")
    length(saved_enc.grid) == length(current_enc.grid) ||
        error("Grid dimension mismatch: saved $(length(saved_enc.grid))D, " *
              "current $(length(current_enc.grid))D")
    saved_enc.C == current_enc.C ||
        error("Channel mismatch: saved C=$(saved_enc.C), current C=$(current_enc.C)")
    saved_enc.extras == current_enc.extras ||
        error("Encoding options mismatch: saved $(saved_enc.extras), current $(current_enc.extras)")

    grid_changed = saved_enc.grid != current_enc.grid
    grid_changed &&
        @info "Loading weights trained on grid $(saved_enc.grid) into grid $(current_enc.grid) " *
              "— saved walkers are discarded"

    # --- network check (strict) ---
    saved_nn = lines[arch_idx + 2]
    current_nn = nn_signature(ansatz.model)
    saved_nn == current_nn ||
        error("Network mismatch.\n  saved:  $saved_nn\n  current:  $current_nn")

    # --- scaling ---
    saved_scale_name    = Symbol(parse_kv(lines[sc_idx + 1]))         # :log1p
    saved_max_norm      = let s = parse_kv(lines[sc_idx + 2])         # "255" or "nothing"
        s == "nothing" ? nothing : parse(Int, s)
    end
    saved_normalisation = parse(Float32, parse_kv(lines[sc_idx + 3])) # 0.18033688f0
    
    # --- input_scale_func needs to be same ---
    if haskey(SCALE_FUNCTIONS, saved_scale_name)
        if ansatz.input_scale_func === SCALE_FUNCTIONS[saved_scale_name]
            # ansatz.input_scale_func = SCALE_FUNCTIONS[saved_scale_name]
        else
            error("Loaded input_scale_func ($(SCALE_FUNCTIONS[saved_scale_name])) differs from " *
                  "ansatz one ($(ansatz.input_scale_func))")
        end
    else
        error("Unknown scale function '$saved_scale_name'. " *
            "Known options: $(collect(keys(SCALE_FUNCTIONS)))")
    end

    weights_line = lines[w_idx + 1]
    inputs_line  = lines[i_idx + 1]

    # --- mean field ---
    saved_mf = parse(Bool, strip(parse_kv(lines[mf_idx + 1])))   # "true" / "false"

    has_mf = ansatz.meanfield !== nothing
    if saved_mf != has_mf
        error("Weights were trained $(saved_mf ? "with" : "without") a mean-field, " *
              "but the current ansatz has $(has_mf ? "a" : "no") mean-field. " *
              "Construct the ansatz $(saved_mf ? "with" : "without") the mean-field to load these weights.")
    end

    # --- line 2: weights ---
    θ_cpu = parse.(Float32, split(weights_line))

    # --- line 3: inputs (columns separated by ", ") ---
    col_strs = split(inputs_line, ", ")
    cols = [parse.(Int, split(strip(c))) for c in col_strs]
    # check: all columns have same length
    @assert all(length(c) == length(cols[1]) for c in cols) "Input columns have inconsistent lengths"
    x = reduce(hcat, cols)   # Matrix{Int} of size (input_dim, batch)
    
    rs = (); offset = 0
    for layer in ansatz.model.layers
        if !hasparams(layer)        # parameter-free layer check
            rs = (rs..., LayerRange((offset+1):offset, (offset+1):offset))
            continue
        end
        nW = length(layer.W); nb = length(layer.b)
        r_W = (offset+1):(offset+nW); offset += nW
        r_b = (offset+1):(offset+nb); offset += nb
        ln = layer.layer_norm
        if ln !== nothing
            nγ = length(ln.γ); nβ = length(ln.β)
            r_γ = (offset+1):(offset+nγ); offset += nγ
            r_β = (offset+1):(offset+nβ); offset += nβ
            rs = (rs..., LayerRange(r_W, r_b, r_γ, r_β))
        else
            rs = (rs..., LayerRange(r_W, r_b))
        end
    end
    ranges = rs
    p = offset

    @assert length(θ_cpu) == p "Loaded $(length(θ_cpu)) params but model has $p params!"

    # --- load weights into chain (CPU or GPU) ---
    refparam = first(filter(hasparams, ansatz.model.layers))
    θ = similar(refparam.b, length(θ_cpu))
    copyto!(θ, θ_cpu)
    
    scaling_old = saved_normalisation
    scaling_new = ansatz.normalisation
    ratio = Float32(scaling_old/scaling_new) # scaling = 1/N => inverse ratio
    first_param_seen = false
    for (layer, r) in zip(ansatz.model.layers, ranges)
        hasparams(layer) || continue                 # Pool: nothing to load
        layer.W .= reshape(view(θ, r.W), size(layer.W))
        layer.b .= view(θ, r.b)
        if !isnothing(r.γ)
            layer.layer_norm.γ .= reshape(view(θ, r.γ), size(layer.layer_norm.γ))
            layer.layer_norm.β .= reshape(view(θ, r.β), size(layer.layer_norm.β))
        end
        if !first_param_seen
            layer.W .*= ratio
            first_param_seen = true
        end
    end

    @info "Weights loaded from $filename ($p parameters)"
    return x, grid_changed
end


"""
    log_markov_chain(filename, addrs; start=true, num_width)

Save one vector of addresses as one line in `filename`.

# Keyword Arguments

* `start=true`: create/overwrite file, write first line
* `start=false`: append new line (no override)
* `num_width`: character width reserved by integer (default 3)

Each address is written as its Int occupation-number vector,
addresses separated by " | ".

# Example 
    0, 1, 2, 3, 4, 5 | 1, 0, 0, 2, 1, 3 | ...
"""
function log_markov_chain(filename::String, addrs::AbstractVector; start::Bool=false, num_width::Int=3)
    mode = start ? "w" : "a"
    addrs_cpu = Array(addrs)

    open(filename, mode) do io
        line = join([_format_addr(Int.(onr(a)); w=num_width) for a in addrs_cpu], " | ")
        println(io, line)
    end

    if mode === "w"
        @info "Markov chain will be saved in: $filename"
    end
    return nothing
end

"""Format one address vector with fixed-width integers."""
function _format_addr(occ::AbstractVector{Int}; w::Int=3)
    inner = join([@sprintf("%*d", w, n) for n in occ], ", ")
    return "$inner"
end
