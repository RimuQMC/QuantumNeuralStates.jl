
"""
    TruncationBuffer(H, k; type=:center, metric=:l1)

This buffer is part of [`NeuralAnsatz`](@ref) and it introduce input
truncation. 

# Arguments

* `H`: Hamiltonian defiend in `Rimu`. It is needed for extracting system
        geometry.
* `k`: this value define how small truncation I want to use

# Keyword Arguments

* `type`: define a switch between different truncation schemes. For now
        only `:center` (truncate input space from center futher) is defined.
* `metric`: define another parameter involving geometry of truncation.
        (how I determine nearest closest neighbours)

# Note

* `type options`: :center
* `metric options`: :l1, :l2, :linf 
"""
struct TruncationBuffer
    k::Int
    type::Symbol
    metric::Symbol

    dims::Tuple
    keep::Vector{Int} # indices of flatten vector to keep
    mask::Vector{Bool}
end
function TruncationBuffer(H, k::Int; type::Symbol=:center, metric::Symbol=:l1)
    dims  = grid_dims(H)
    
    if type === :center
        order = center_order(dims; metric=metric)
    else
        error("Invalid type of truncation mask: $type")
    end

    keep  = sort(order[1:k])
    mask  = falses(prod(dims))
    mask[keep] .= true  # indices that are in keep have true value -> rest false
    println("----- Truncation: k = $k, type = $type, metric = $metric -----")
    return TruncationBuffer(k, type, metric, dims, keep, mask)
end

"""
    grid_dims(H)

This function should extract geometry information from hamiltonian. So far,
if hamiltonian contains `:geometry` type it is used. Otherwise, problem is 
set to be one dimensional.
"""
function grid_dims(H)
    if hasproperty(H, :geometry)
        g = H.geometry
        hasmethod(size, Tuple{typeof(g)}) && return size(g)         # e.g. size(g) == (L,L)
        hasproperty(g, :dims) && return Tuple(g.dims)
    end
    # Fallback: derive L from D and M if only those are exposed
    # D = typeof(H).parameters[3] # dimension is 3rd type
    D = 1
    M = length(onr(starting_address(H))) 
    # L = round(Int, M^(1/D))
    L = M 
    return ntuple(_ -> L, D)
end

"""
    build_truncation(hamiltonian, truncation)

This function is called inside [`NeuralAnsatz`](@ref) if truncation should be
created. 
"""
function build_truncation(hamiltonian, truncation)
    truncation === nothing && return nothing

    if truncation isa Integer
        return TruncationBuffer(hamiltonian, truncation)
    elseif truncation isa Tuple
        length(truncation) in 1:3 ||
            error("Truncation tuple must have 1–3 elements (k, type, metric), got $(length(truncation))")
        k      = truncation[1]
        type   = length(truncation) >= 2 ? truncation[2] : :center
        metric = length(truncation) >= 3 ? truncation[3] : :l1
        return TruncationBuffer(hamiltonian, k; type=type, metric=metric)
    else
        error("Invalid truncation input! Choose ::Int(k) or ::Tuple(k, type, metric), got $(typeof(truncation))")
    end
end

"""
    change_truncation!(ansatz, H, k)

This function allows to change truncations size `k`. The `truncation` buffer inside
`NeuralAnsatz` is then rewrite with new truncation size.

It is important that truncation can only gradually increase! Otherwise some instability
can be observed!
"""
function change_truncation!(ansatz, H, k::Int)
    old = ansatz.truncation
    @assert old === nothing || k > old.k "New k ($k) must be larger than current k ($(old.k))"

    if old === nothing
        ansatz.truncation = TruncationBuffer(H, k)
    else
        ansatz.truncation = TruncationBuffer(H, k; type=old.type, metric=old.metric)
    end

    return nothing
end

"""
    violates_truncation(occ, mask) -> Bool

This function guard `vmc` samplers before sampling outside of truncation region.
[`TruncationBuffer`](@ref) holds variable `mask` which contains information about
allowed occupation numbers in sampled addresses. If proposed address is allowed 
returns `true`. If proposed address is not allowed return `false`.
"""
@inline function violates_truncation(occ, mask)
    @inbounds for i in eachindex(occ)
        !mask[i] && occ[i] != 0 && return true
    end
    return false
end

