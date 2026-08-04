using QuantumNeuralStates
using Documenter
using DocumenterInterLinks

DocMeta.setdocmeta!(QuantumNeuralStates, :DocTestSetup, :(using QuantumNeuralStates); recursive=true)

links = InterLinks(
    "Rimu" => "https://rimuqmc.github.io/Rimu.jl/stable/objects.inv",
)

makedocs(;
    modules=[QuantumNeuralStates],
    authors="Krystof Krsek <krsek.k@gmail.com> and contributors",
    sitename="QuantumNeuralStates.jl",
    # checkdocs = :none, # ignore all leftover code functions not defined in .md
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Guide" => "index.md",
        "User Documentation" => [
            "Neural Networks" => "neuralnetworks.md",
            "Variational Monte Carlo" => "vmc.md",
            "Neural Ansatz" => "ansatz.md",
            "Optimisers" => "optimisers.md",
            "Input / Output" => "io.md",
            "Utils" => "utils.md"
        ],
        "API"   => "API.md",
    ],
    plugins = [links]
)

deploydocs(
    repo = "github.com/RimuQMC/QuantumNeuralStates.jl.git",
    push_preview = true,
)
