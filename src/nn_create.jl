
"""
    build_model(type, sizes, activation; kwargs...) 

Kinda helping macro to build full Neural Network (is not needed).
"""

function build_model(type::String, sizes::Vector{Int}, activation::Function; 
        batch::Int=1, device::Function = identity, Layer_Norm=false
)
    layers = []
    if type == "FCNN"
        for i in 1:length(sizes)-2
            push!(layers, Dense(sizes[i], sizes[i+1], activation; batch=batch, device=device, Layer_Norm=Layer_Norm))
        end
        push!(layers, Dense(sizes[end-1], sizes[end], identity; batch=batch, device=device))
        return Chain(layers...; device=device, batch=batch)
    end
end
