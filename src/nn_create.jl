
"""
    build_model(type, enc, sizes, activation; kwargs...) 

Kinda helping macro to build full Neural Network (is not needed).
"""

function build_model(type::String, enc::InputEncoding, sizes::Vector{Int}, activation;
                     batch::Int = 1, device::Function = identity, layer_norm = false)
    length(sizes) >= 1 || error("`sizes` needs at least the number of outputs")

    if type == "FCNN"
        layers = []
        n_in   = nsites(enc) * nchannels(enc)   

        for n_out in sizes[1:end-1]     # one hidden layer per entry
            push!(layers, Dense(n_in=>n_out, activation;
                                batch=batch, device=device, layer_norm=layer_norm))
            n_in = n_out
        end
        push!(layers, Dense(n_in=>sizes[end], identity; batch=batch, device=device))

        return Chain(enc, layers...; device=device, batch=batch)
    end

    error("unknown model type \"$type\" (available: \"FCNN\")")
end
