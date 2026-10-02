

"""
    _actname(f) -> String

Short display name for an activation: a function's name, or `"(name1, name2)"`
for a tuple of activations (per-range activations).
"""
_actname(f::Function) = string(nameof(f))
_actname(fs::Tuple)   = "(" * join(_actname.(fs), ", ") * ")"

"""
    _typename(x) -> String

The name of `x`'s type, as a string.
"""
_typename(x) = string(nameof(typeof(x)))

"""
    _funcname(f) -> String

`f`'s name if it's a `Function`, else its type's name (for callable structs).
"""
_funcname(f) = f isa Function ? string(nameof(f)) : _typename(f)   # also handles callable structs

"""
    _devname(f) -> String

Short display name for a `device` function: `"CPU"` for `identity`, else the
function's name (`mtl`, `cu`, ...).
"""
_devname(f) = f === identity ? "CPU" : string(nameof(f))   # mtl → "mtl", cu → "cu"

"""
    _flag_mem(x) -> String

`"true  (size)"` / `"false  (0 B)"` — whether `x` is present, and its
[`memory_estimate`](@ref), formatted for display.
"""
_flag_mem(x) = string(x !== nothing, "  (", _fmt_bytes(memory_estimate(x)), ")")

"""
    _group(n::Integer) -> String

Nicer number `n` print with thousands separators, e.g. `19905 -> "19,905"`.
"""
_group(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")   # 19905 → "19,905"

"""
    _fmt_bytes(b::Integer) -> String

Format a byte count with the largest sensible unit (`B`, `KiB`, `MiB`, ...).
"""
function _fmt_bytes(b::Integer)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    x, i = float(b), 1
    while x ≥ 1024 && i < length(units)
        x /= 1024; i += 1
    end
    return i == 1 ? "$b B" : string(round(x; digits = 2), " ", units[i])
end

"""
    memory_estimate(x) -> Int

Estimate the memory (in bytes) held by all arrays (only) reachable from `x`, searching
struct fields, tuples, and named tuples recursively. Works for CPU and GPU arrays. 
Assumes every array is referenced only once.
"""
function memory_estimate(x)
    x isa AbstractArray && return _mem_array(x)
    x isa Union{Tuple, NamedTuple} && return sum(memory_estimate, x; init = 0)
    x isa Union{Nothing, Number, Symbol, AbstractString, Function, Type, Module} && return 0

    T = typeof(x)
    isstructtype(T) || return 0
    s = 0
    for i in 1:fieldcount(T)
        isdefined(x, i) && (s += memory_estimate(getfield(x, i)))
    end
    return s
end

"""
    _mem_array(A::AbstractArray) -> Int

The array case of [`memory_estimate`](@ref): counts the parent for views/
reshapes/adjoints, `sizeof(A)` for a bitstype array, or recurses per element
otherwise.
"""
function _mem_array(A::AbstractArray)
    P = parent(A)
    P !== A && return memory_estimate(P)       # views, reshapes, adjoints: count the parent
    isbitstype(eltype(A)) && return sizeof(A)  
    s = 0                                      
    for i in eachindex(A)
        isassigned(A, i) && (s += memory_estimate(A[i]))
    end
    return s
end
