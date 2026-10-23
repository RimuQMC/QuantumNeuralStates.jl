
# using KernelAbstractions
# using SpecialFunctions: loggamma          # used only at setup, on the CPU

"""
    MeanField(H; backend = KernelAbstractions.CPU())

If mean-field is enabled in [`NeuralAnsatz`](@ref), then this 
structure adds mean-field log-amplitude on top of the neural-network output,

```math
\\log\\psi(n) = \\log\\psi_{\\mathrm{NN}}(n) + \\log\\psi_{\\mathrm{MF}}(n).
```

The mean field is defined by a function `mf_func(o, params, T)` that returns the
log-amplitude (of type `T`) of a configuration with occupation numbers `o = onr(addr)`,
together with its parameters `params`. Both are returned by a Hamiltonian-specific
method of [`_choose_meanfield_function`](@ref). Parameters are built once on the CPU
and moved to `backend`; the evaluation runs as a kernel on that backend.

# Arguments
* `H`: Rimu Hamiltonian for which the mean field is constructed.

# Keyword Arguments
* `backend`: KernelAbstractions backend on which the parameters live and the kernels run,
    e.g. `KernelAbstractions.CPU()` (default), `MetalBackend()`, or `CUDABackend()`. Must 
    match the backend of the arrays passed to `(mf)`. Resolved in 
    [`NeuralAnsatz`](@ref) structure.

# Notes
* To add a mean field for a new Hamiltonian, define a function `mf_func(o, params, T)`
    and a method `_choose_meanfield_function(H::Rimu.Hamiltonian; backend)` returning
    `(mf_func, params)`.
* `params` should be a `Tuple` or `NamedTuple` of arrays and scalars, so it is converted
    to device arrays automatically when passed to the kernel.
* `mf_func` runs inside a kernel: it must not allocate and may only call device-compatible
    functions (e.g. no `loggamma`; use a look-up table instead).
* The mean field does not depend on the network parameters `θ`, so it does not enter the
    Jacobian.
"""
struct MeanField{F,P}
    func::F          # mf_func(o, params, T) -> T
    params::P
end

function MeanField(H; backend = KernelAbstractions.CPU(), eltype = Float32)
    func, params = _choose_meanfield_function(H; backend, eltype)
    return MeanField{typeof(func), typeof(params)}(func, params)
end

"""
    (mf)(addrs, result)            # addrs::AbstractVector, size (B,)
    (mf)(addrs, offsets, result)   # addrs::AbstractMatrix, size (K, B), ragged columns

Add the mean-field log-amplitude of each configuration to `result` in place.

# Arguments
* `addrs`: walker addresses (B,), or off-diagonal addresses (K, B) where column `b` holds
    `offsets[b] - offsets[b-1]` valid entries.
* `offsets`: (B,) cumulative number of valid off-diagonals per column, see
    [`_column_start`](@ref).
* `result`: log-amplitudes to add to, (B,) for walkers or (total,) in stream order for
    off-diagonals, `total = offsets[end]`.

# Notes
* For the off-diagonals, entry `k` of column `b` is added to
    `result[_column_start(offsets, b) + k]`, the same stream order as in
    [`_prepare_input_stream_kernel!`](@ref).
* Apply it to `log ψ` after [`log_psi!`](@ref), for both the walkers and their
    off-diagonals, so that all ratios `ψ(n')/ψ(n)` contain the mean field.
"""
function (mf::MeanField)(addrs::AbstractVector, result::AbstractArray)
    backend = KernelAbstractions.get_backend(result)
    _mf_addrs_kernel!(backend)(result, addrs, mf.func, mf.params;
                               ndrange = length(addrs))     # (B,)
    return result
end
function (mf::MeanField)(addrs::AbstractMatrix, offsets, result::AbstractArray)
    backend = KernelAbstractions.get_backend(result)
    _mf_offdiag_kernel!(backend)(result, addrs, offsets, mf.func, mf.params;
                                 ndrange = size(addrs))     # (K, B)
    return result
end

"""
    _mf_addrs_kernel!(result, addrs, mf_func, params)

Kernel that adds the mean-field log-amplitude of each walker to `result`. Launched with
`ndrange = length(addrs)`, one thread per walker `b`, which evaluates
`mf_func(onr(addrs[b]), params, T)` and adds it to `result[b]`.
"""
@kernel function _mf_addrs_kernel!(result, @Const(addrs), mf_func, params)
    b = @index(Global, Linear)                     # walker b
    T = eltype(result)
    o = onr(@inbounds addrs[b])
    @inbounds result[b] += mf_func(o, params, T)
end

"""
    _mf_offdiag_kernel!(result, addrs, offsets, mf_func, params)

Kernel that adds the mean-field log-amplitude of each valid off-diagonal to `result`.
Launched with a 2D `ndrange = (K, B)`, one thread per slot `(k, b)` of `addrs`. Threads
with `k > offsets[b] - offsets[b-1]` (unfilled slots) do nothing; the others add
`mf_func(onr(addrs[k, b]), params, T)` to `result[_column_start(offsets, b) + k]`.

Unlike [`_prepare_input_stream_kernel!`](@ref), no binary search
([`_find_column`](@ref)) is needed, since the kernel is launched over `(k, b)` directly
and writes into the final stream position.
"""
@kernel function _mf_offdiag_kernel!(result, @Const(addrs), @Const(offsets), mf_func, params)
    k, b  = @index(Global, NTuple)                 # slot k of walker b
    start = _column_start(offsets, b)
    nb    = Int(@inbounds offsets[b]) - start      # valid off-diagonals of walker b
    if k <= nb
        T = eltype(result)
        o = onr(@inbounds addrs[k, b])
        @inbounds result[start + k] += mf_func(o, params, T)
    end
end

"""
    _to_backend(backend, x) -> y

Allocate an array of the same element type and size as `x` on `backend` and copy `x`
into it. Used to move the mean-field parameters built on the CPU to the device.
"""
function _to_backend(backend, x::AbstractArray)
    y = KernelAbstractions.allocate(backend, eltype(x), size(x))
    copyto!(y, x)
    return y
end

"""
    _choose_meanfield_function(H; backend) -> (mf_func, params)

Return the mean-field log-amplitude function `mf_func(o, params, T)` and its parameters
`params` for Hamiltonian `H`. Define a method for each Hamiltonian that supports a mean
field; the generic fallback throws an error.

See [`MeanField`](@ref).
"""
_choose_meanfield_function(H; backend = KernelAbstractions.CPU()) =
    error("no mean field defined for $(typeof(H))")

function _choose_meanfield_function(H::Rimu.FroehlichPolaron{T};
                                    backend = KernelAbstractions.CPU(),
                                    eltype = Float32, n_table::Int = 256) where T
    S = eltype
    eta, f = solve_self_consistent_eta(H)

    log_f = S[iszero(fi) ? S(-35) : S(log(abs(fi))) for fi in f]    # (M,) fixed
    lgf = S[loggamma(T(n) + 1) for n in 0:n_table-1]                # look-up table of log(n!)   
    const_term = S(-0.5 * sum(f .^ 2))                              # scalar, fixed

    params = (_to_backend(backend, log_f), _to_backend(backend, lgf), const_term)
    return _mf_func_froehlich, params
end
