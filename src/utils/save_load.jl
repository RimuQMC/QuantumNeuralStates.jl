
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
    _write_layernorm_flag(io, chain)

Write LayerNorm flag if used: true/false for every layer in chain.
"""
function _write_layernorm_flag(io, chain)
    println(io, "LayerNorm:")
    join(io, [layer.layer_norm !== nothing ? "true" : "false" for layer in chain.layers], " ")
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
function _write_addrs(io::IO, addrs::Vector; num_width::Int=3)
    println(io, "Inputs:")
    n_addrs = length(addrs)
    for (j, a) in enumerate(addrs)
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
    save_weights(filename, θ)

Save flatten weights vector "θ" to a plain-text file. See [`_write_weights`].
"""
function save_weights(filename::String, θ::AbstractVector)
    open(filename, "w") do io
        _write_weights(io, θ)
    end
    @info "Weights saved to $filename ($(length(θ)) parameters)"
end

"""
    save_addrs(filename, addrs; num_width=3)

Save addresses to a plain-text file. Addresses are separated by ", " on 
a single line. See [`_write_addrs`].
"""
function save_addrs(filename::String, addrs::Vector; num_width::Int=3)
    open(filename, "w") do io
        _write_addrs(io, addrs; num_width)
    end
    @info "Addresses saved to $filename ($(length(addrs)) addresses)"
end

"""
    save_input_scale(filename, ansatz)

Save information about input scaling function and about input normalisations.
See [`_write_input_scale`].
"""
function save_input_scale(filename::String, ansatz)
    open(filename, "w") do io
        _write_input_scale(io, ansatz)
    end
    @info "Input scaling saved to $filename.)"
end

"""
    save_master(filename, θ, addrs; num_width=3)

Master function, that saves all important Neural Network configurations. 

## Example of txt file
    Input Scaling:
        scale function: 
        max_norm:
        normalisation:

    Weights:
        <Float32 values>

    LayerNorm:
        true, false, ....

    Inputs:
        0   0   0   0   0,   1   0   0   0   0, ...

"""
function save_master(filename::String, θ::AbstractVector, addrs::Vector, ansatz;
                             num_width::Int=3)
    open(filename, "w") do io
        _write_input_scale(io, ansatz)
        println(io)
        _write_weights(io, θ)
        println(io)
        _write_layernorm_flag(io, ansatz.model)
        println(io)
        _write_addrs(io, addrs; num_width)
    end
    @info "Saving ($(length(θ)) weights, $(length(addrs)) addresses, and $(nameof(ansatz.input_scale_func)) 
    input scaling function with norm of $(ansatz.max_norm)) to $filename"
end

# helper: pulls "log1p" from string "scale function: log1p"
parse_kv(line) = strip(split(line, ":", limit=2)[2])

"""
    load_master(chain, filename::String) -> x

Loads all configurations from plain text file into the Neural Network model `chain` 
and returns the saved inputs `x` as a matrix of `Int` (For reconstraction in Rimu notation). 
If only one column was saved, `vec(x)` is called to get a flat vector.

# Arguments

* `chain`: Neural Network model. See [`Chain`](@ref).
* `filename`: "name_of_file.txt".

"""
function load_master(ansatz, filename::String)
    lines = readlines(filename)
    # find the data lines by looking for the section headers
    sc_idx= findfirst(==("Input Scaling:"), lines)
    w_idx = findfirst(==("Weights:"), lines)
    ln_idx = findfirst(==("LayerNorm:"), lines)
    i_idx = findfirst(==("Inputs:"),  lines)
    @assert sc_idx!== nothing "Missing 'Input Scaling:' header"
    @assert w_idx !== nothing "Missing 'Weights:' header"
    @assert i_idx !== nothing "Missing 'Inputs:' header"

    # --- line 1: scaling ----------
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

    # --- line 2: weights ---
    θ_cpu = parse.(Float32, split(weights_line))

    # --- line 2.5: layer normalisation flags ---
    saved_ln_flags = parse.(Bool, split(lines[ln_idx + 1]))
    current_ln_flags = [layer.layer_norm !== nothing for layer in ansatz.model.layers]
    @assert saved_ln_flags == current_ln_flags "LayerNorm configuration mismatch: saved=$saved_ln_flags, " * 
        "current=$current_ln_flags"

    # --- line 3: inputs (columns separated by ", ") ---
    col_strs = split(inputs_line, ", ")
    cols = [parse.(Int, split(strip(c))) for c in col_strs]
    # check: all columns have same length
    @assert all(length(c) == length(cols[1]) for c in cols) "Input columns have inconsistent lengths"
    x = reduce(hcat, cols)   # Matrix{Int} of size (input_dim, batch)

    rs     = ()
    offset = 0
    for layer in ansatz.model.layers
        nW = length(layer.W)
        nb = length(layer.b)
        if layer.layer_norm !== nothing
            nγ = length(layer.layer_norm.γ)
            nβ = length(layer.layer_norm.β)
            r = LayerRange(
                offset+1          : offset+nW,
                offset+nW+1       : offset+nW+nb,
                offset+nW+nb+1    : offset+nW+nb+nγ,
                offset+nW+nb+nγ+1 : offset+nW+nb+nγ+nβ
            )
            offset += nW + nb + nγ + nβ
        else
            r = LayerRange(offset+1 : offset+nW, offset+nW+1 : offset+nW+nb)
            offset += nW + nb
        end
        rs = (rs..., r)
    end
    ranges = rs
    p      = offset

    @assert length(θ_cpu) == p "Loaded $(length(θ_cpu)) params but model has $p params!"

    # --- load weights into chain (CPU or GPU) ---
    l = first(ansatz.model.layers)
    θ = similar(l.b, length(θ_cpu))
    copyto!(θ, θ_cpu)
    
    scaling_old = saved_normalisation
    scaling_new = ansatz.normalisation
    ratio = Float32(scaling_old/scaling_new) # scaling = 1/N => inverse ratio
    for (i, (layer, range)) in enumerate(zip(ansatz.model.layers, ranges))
        layer.W .= reshape(view(θ, range.W), size(layer.W))
        layer.b .= view(θ, range.b)
        if range.γ !== nothing
            layer.layer_norm.γ .= reshape(view(θ, range.γ), size(layer.layer_norm.γ))
            layer.layer_norm.β .= reshape(view(θ, range.β), size(layer.layer_norm.β))
        end
        if i == 1
            # row_sum_W = vec(sum(layer.W, dims=2))
            layer.W .*= ratio
            # layer.b .-= 0.1f0 .* (ratio - 1f0) .* row_sum_W
        end
    end

    @info "Weights loaded from $filename ($p parameters, input size $(size(x)))"
    # if size(x, 2) == 1
    #     x = vec(x)      # if batch = 1, x is vector (not matrix)
    # end
    return x
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
function log_markov_chain(filename::String, addrs; start::Bool=false, num_width::Int=3)
    mode = start ? "w" : "a"

    open(filename, mode) do io
        line = join([_format_addr(Int.(onr(a)); w=num_width) for a in addrs], " | ")
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
