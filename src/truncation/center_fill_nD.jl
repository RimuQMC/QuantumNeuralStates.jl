"""
    center_order(dims::NTuple{D,Int}; metric=:l1)

Permutation of the linear indices 1:prod(dims) (column-major, matching
reshape(input, dims...)) sorted by distance from the geometric centre,
closest first. Works for any D: 

* `D=1`: expands outward from the middle of a line. 
* `D=2`: gives the expanding diamond shape from the center. 
* `D=3`: gives an octahedron (:l1), a cube (:linf), or a ball (:l2).
"""
function center_order(dims::NTuple{D,Int}; metric::Symbol=:l1) where {D}
    centre = ntuple(d -> (dims[d] + 1) / 2, D)
    cart   = CartesianIndices(dims)
    M      = length(cart)

    dist(ci) = begin
        diffs = ntuple(d -> abs(ci[d] - centre[d]), D)
        metric === :l1   ? sum(diffs) :
        metric === :linf ? maximum(diffs) :
        metric === :l2   ? sqrt(sum(abs2, diffs)) :
        error("unknown metric $metric")
    end

    idx = collect(1:M)
    sort!(idx; by = i -> (dist(cart[i]), Tuple(cart[i])))   # tuple tiebreak = deterministic
    return idx
end

