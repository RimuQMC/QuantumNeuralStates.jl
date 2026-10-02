# using Gutzwiller
# using Rimu

"""
    NeuralAnsatz(ansatz_type, hamiltonian, model, batch_size; kwargs...) <: Gutzwiller.AbstractAnsatz             

A neural network-based variational ansatz for use with Rimu's FCIQMC.
Wraps a [`Chain`](@ref) neural network model and manages the buffers needed for batched evaluation
and importance sampling.

# Arguments
* `ansatz_type`: it is [`AnsatzType`](@ref) which determine how the wave-function ansatz, using neural
        network outputs, should looks like.
* `hamiltonian`: hamiltonian defined in `Rimu`.
* `model`: a `Chain` neural network model. 
* `batch_size`: number of configurations evaluated in a single forward
                pass. 

# Keyword Arguments
* `input_scale_func`: scaling function for inputs (default: `identity`).
* `max_norm`: maximum particle number for input normalisation. Must be 
                positive if provided.
* `multiforward_buffer`: can be set with `Int` number. This number would determine
                usually bigger batch forward passes. See [`MultiForwardBuffer`](@ref).
* `mean_field`: if set `true` the mean-field would add to neural network wave-funciton
                evaluation. Needs to be manually set, see [`MeanField`](@ref).
* `truncation`: can be used for input space truncation. See [`TruncationBuffer`](@ref).
* `neuron_statistics`: Can be activated with `true` (statistics would be print out to
                terminal), or (NOT WORKING - `filename::String` to be saved in external file). See
                [`neuron_statistics`](@ref).
* `jacobian_statistics`: Can be activated with `true` (statistics would be print out to
                terminal), or (NOT WORKING - `filename::String` to be saved in external file). See
                [`jacobian_statistics`](@ref).

# Example
```julia
julia> M = 10
julia> N = 10
julia> batch = 1024
julia> model = build_model("FCNN", [M, 100, 100, 100, 1], tanh_fast; batch=batch)
julia> addr = near_uniform(BoseFS{N,M})
julia> H = HubbardReal1D(addr; u=0.1)
julia> ansatz = NeuralAnsatz(LogPsi(), H, model, batch)
```

## Fields
* `ansatz_type`: it is [`AnsatzType`](@ref) which determine how the wave-function ansatz, using neural
        network outputs, should looks like.
* `hamiltonian`: hamiltonian defined in `Rimu`.
* `model`: a neural network type of `Chain` used to evaluate the ansatz. 
* `logψ_centering`: this is max value centering over batched `model` output connected with 
                wave-function amplitude - log|ψ| (fighting the gauge invariant for 
                multiplication of wave-function with arbitrary number coefficient)
* `x_cpu_buffer`: pre-allocated `Float32` buffer for batched network input.
* `addrs_buffer`: dynamically filled buffer of addresses, used during
                Rimu FCIQMC importance sampling.
* `result_buffer`: dynamically filled buffer of network outputs, paired
                with `addrs_buffer` during FCIQMC importance sampling.
* `result_dict`: cache mapping of results of neural network used during
                Rimu FCIQMC importance sampling
* `first_iter`: flag indicating whether the FCIQMC iteration is the first;
                used to control buffer initialisation logic.
* `input_scale_func`: a function applied to raw occupation inputs before they
                are fed to the network. Defaults to `identity` (uniform
                spread). Use e.g. `sqrt` to compress large occupation
                numbers and give smaller values relatively more importance.
* `max_norm`: Maximum total particle number in the system. When set,
                inputs are normalised so the highest possible occupation
                maps to `1`, keeping all inputs in `[0, 1]`. `nothing`
                disables this normalisation - by default.
* `normalisation`: Scale factor derived from `max_norm`. Ensures that
                if neural networks load pre-computed weights, they value
                invariant regardless the normalisation.
* `multi_forward_buffer`: if active it holds additional [`MultiForwardBuffer`](@ref).
                This allows to do forward pass with arbitrary batch size.
* `Mean-Field`: allows using mean-filed estimate with neural network. See
                [`MeanField`](@ref).
* `Truncation`: allows input truncation for graduate training in smaller 
                dimensions. See [`TruncationBuffer`](@ref).
* `neuron_statistics`: allows to print out statistics about neural network. Either,
                to terminal or external file. See [`neuron_statistics`](@ref).
* `jacobian_statistics: allows to print out statistics about jacobian (per-sample
                gradient). Either, to terminal or external file. See 
                [`jacobian_statistics`](@ref). 

"""
mutable struct NeuralAnsatz{AT<:AnsatzType,A,T<:Real,H,M,X<:AbstractArray,XZ<:AbstractArray,XA<:AbstractArray{A},
                            XR<:AbstractArray,D,F,MN<:Union{Nothing,Int},MFB<:Union{Nothing,MultiForwardBuffer},
                            MF<:Union{Nothing,MeanField},TR<:Union{Nothing,TruncationBuffer},
                            NS<:Union{Bool,String}, JS<:Union{Bool,String}} <: Gutzwiller.AbstractAnsatz{A,T,0}
    ansatz_type::AT
    hamiltonian::H
    model::M
    logψ_centering::Float32 # centers model output corresponding to ψ amplitude
    x_cpu_buffer::X
    z_cpu::XZ

    # Rimu FCIQMC helper buffers dynamically filled 
    addrs_buffer::XA
    result_buffer::XR
    result_dict::D
    first_iter::Bool

    # helper functions for Input preparation
    input_scale_func::F
    max_norm::MN
    normalisation::Float32

    # helper buffer structure for NN forward runs where length(input) > batch
    multi_forward_buffer::MFB

    # Mean-Field 
    meanfield::MF

    # Truncation
    truncation::TR

    # Neuron Health Statistics
    neuron_statistics::NS
    jacobian_statistics::JS
end
function NeuralAnsatz(ansatz_type::AnsatzType, hamiltonian, model, batch_size; 
                    input_scale_func=identity, max_norm::Union{Nothing, Int}=nothing, 
                    multiforward_buffer=nothing, mean_field::Bool=false, truncation=nothing,
                    neuron_statistics::Union{Bool,String}=false, jacobian_statistics::Union{Bool,String}=false
    )
    AT=typeof(ansatz_type)
    # safe check of ansatz_type and number of model outputs
    @assert (ansatz_type.num_outputs == size(last(model.layers).z, 1)) "" * 
            "Number of outputs in model (neural network) does not correspond with ansatz_type!"
    model.enc isa NoEncoding &&
    error("NeuralAnsatz needs an encoding that maps Fock states to network inputs " *
          "(OccupationEncoding or MomentumEncoding); NoEncoding is for plain NN use")

    addr = starting_address(hamiltonian)
    dim = size(model.x, 1)
    x_cpu_buffer = zeros(Float32, size(model.x)[1:end-1]..., batch_size)
    z_cpu = Matrix{Float64}(undef, size(last(model.layers).z, 1), batch_size)

    nn_output = model(x_cpu_buffer)
    logψ_centering = maximum(nn_output)

    A = typeof(addr)
    T = Float64
    H = typeof(hamiltonian)
    M = typeof(model)
    X = typeof(x_cpu_buffer)
    XZ = typeof(z_cpu)

    addrs_buffer  = A[]
    result_buffer = Float64[]
    result_dict   = Dict{A, T}()
    XA = typeof(addrs_buffer)
    XR = typeof(result_buffer)
    D  = typeof(result_dict)
    first_iter = true

    F=typeof(input_scale_func)
    normalisation = if max_norm === nothing
        1.0f0
    elseif max_norm > 0
        Float32(inv(input_scale_func(Float32(max_norm))))
    else
        error("Normalisation must be positive! got $max_norm")
    end
    MN=typeof(max_norm)

    multi_forward_buffer = if multiforward_buffer === nothing
        nothing
    else
        MultiForwardBuffer(model, addr, multiforward_buffer)
    end
    MFB=typeof(multi_forward_buffer)

    meanfield = if mean_field === false
        nothing
    else
        MeanField(hamiltonian)
    end
    MF=typeof(meanfield)

    trun = build_truncation(hamiltonian, truncation)
    TR=typeof(trun)

    NS=typeof(neuron_statistics); JS=typeof(jacobian_statistics)

    return NeuralAnsatz{AT,A,T,H,M,X,XZ,XA,XR,D,F,MN,MFB,MF,TR,NS,JS}(
                ansatz_type, hamiltonian, model, logψ_centering, x_cpu_buffer, z_cpu, 
                addrs_buffer, result_buffer, result_dict, first_iter, 
                input_scale_func, max_norm, normalisation, multi_forward_buffer, 
                meanfield, trun, neuron_statistics, jacobian_statistics)
end

function Base.show(io::IO, ::MIME"text/plain", a::NeuralAnsatz)
    rows = (
        "hamiltonian" => _typename(a.hamiltonian),
        "ansatz type" => _typename(a.ansatz_type) * "()",
        "input scale" => string(_funcname(a.input_scale_func),
                                         ", max_norm = ", repr(a.max_norm)),
        "multi-forward buffer" => _flag_mem(a.multi_forward_buffer),
        "mean field" => _flag_mem(a.meanfield),
        "truncation" => sprint(print, a.truncation),
        "total memory estimate" => _fmt_bytes(memory_estimate(a)),
    )
    w = maximum(length ∘ first, rows)

    print(io, "NeuralAnsatz")
    for (k, v) in rows
        print(io, "\n  ", rpad(k * ":", w + 1), " ", v)
    end
end


"""
    prepare_input!(na::NeuralAnsatz, addr, x) -> x
    prepare_input!(na::NeuralAnsatz, addrs::AbstractVector, x) -> x

Write the scaled occupation numbers `input_scale_func(onr(addr)) * normalisation`
into the occupation channel of the input buffer `x` (shape of the `Chain` input).
A single `addr` fills every column; a vector `addrs` fills one column per address.

Runs as a KernelAbstractions kernel on the device of `x`:
* CPU: `x` an `Array`, `addrs` a `Vector`
* GPU: `x` a device array (e.g. `model.x`), `addrs` a device array of addresses

Addresses must be isbits for GPU use, and `input_scale_func` must compile on the
device. Columns beyond `length(addrs)` are left untouched.
"""
function prepare_input!(na::NeuralAnsatz, addr, x)
    enc = na.model.enc
    M, C, B = nsites(enc), nchannels(enc), size(x, ndims(x))
    num_modes(addr) == M ||
        error("address has $(num_modes(addr)) modes, encoding expects $M sites")

    buf = reshape(x, M, C, B)
    backend = KernelAbstractions.get_backend(buf)
    _prepare_input_single_kernel!(backend)(buf, addr, occupation_channel(enc),
                                           na.input_scale_func, na.normalisation;
                                           ndrange = B)
    KernelAbstractions.synchronize(backend)
    return x
end
function prepare_input!(na::NeuralAnsatz, addrs::AbstractVector, x)
    enc = na.model.enc
    M, C, B = nsites(enc), nchannels(enc), size(x, ndims(x))
    length(addrs) <= B ||
        error("got $(length(addrs)) addresses but the buffer has only $B columns")
    num_modes(eltype(addrs)) == M ||
        error("addresses have $(num_modes(eltype(addrs))) modes, encoding expects $M sites")
    isempty(addrs) && return x

    buf = reshape(x, M, C, B)
    backend = KernelAbstractions.get_backend(buf)
    _prepare_input_kernel!(backend)(buf, addrs, occupation_channel(enc),
                                    na.input_scale_func, na.normalisation;
                                    ndrange = length(addrs))
    KernelAbstractions.synchronize(backend)
    return x
end

# vector of addresses: column b gets addrs[b]
@kernel function _prepare_input_kernel!(buf, @Const(addrs), ch::Int, f, s)
    b = @index(Global, Linear)                 # this thread's sample
    o = onr(addrs[b])                          # occupations of sample b (SVector)
    for m in 1:length(o)
        buf[m, ch, b] = f(Float32(o[m])) * s
    end
end

# single address: every column gets the same occupations
@kernel function _prepare_input_single_kernel!(buf, addr, ch::Int, f, s)
    b = @index(Global, Linear)                 # this thread's column
    o = onr(addr)
    for m in 1:length(o)
        buf[m, ch, b] = f(Float32(o[m])) * s
    end
end

"""
    prepare_input_occ!(ansatz, addr, x_cpu_buffer) -> x_cpu_buffer

Similar as [`prepare_input!`](@ref), but always converts to occupation 
representation. This function is used with [`MeanField`](@ref) calculations
as the mean-field require pure occupation number representation.
"""
function prepare_input_occ!(na::NeuralAnsatz, addr, x_cpu_buffer)
    if isa(na.model.layers[1], Dense)
        x_cpu_buffer .= (Float32.(onr(addr))) #.+ 0.1f0
        return x_cpu_buffer
    end
end
function prepare_input_occ!(na::NeuralAnsatz, addrs::AbstractVector, x_cpu_buffer)
    if isa(na.model.layers[1], Dense)
        # Each column is one address from batch 
        @inbounds for i in eachindex(addrs)
            col = @view x_cpu_buffer[:, i]
            col .= Float32.(onr(addrs[i])) #.+ 0.1f0
        end
        return x_cpu_buffer
    end
end
 
"""
    compute_logψ(ansatz, addr) -> logψ
    compute_logψ(ansatz, addr, multi_forward_buffer) -> logψ

Calculates log of the wave-function value prediction of the Neural Network 
from the `addr` input configuration.

If dispatched with `multi_forward_buffer` it allows for computation in 
arbitrary batch size. See [`MultiForwardBuffer`](@ref).
"""
function compute_logψ(na::NeuralAnsatz, addr)
    # x = prepare_input!(na, addr, na.x_cpu_buffer)
    x = prepare_input!(na, addr, na.model.x)
    logψ = na.model(x)
    return logψ
end
function compute_logψ(na::NeuralAnsatz, addr, multi_forward_buffer)
    # x = prepare_input!(na, addr, multi_forward_buffer.x_cpu)
    x = prepare_input!(na, addr, multi_forward_buffer.x)
    logψ = na.model(x, multi_forward_buffer)
    return logψ
end

"""
    compute_mflogψ!(ansatz, addr, z) 
    compute_mflogψ!(ansatz, addr, z, multi_forward_buffer)

Similar as [`compute_logψ`](@ref) but with using [`MeanField`](@ref).
The function add in-place mean-field value to result `z`.

If dispatched with `multi_forward_buffer` it allows for computation in 
arbitrary batch size. See [`MultiForwardBuffer`](@ref).
"""
function compute_mflogψ!(na::NeuralAnsatz, addr, z)
    x = prepare_input_occ!(na, addr, na.x_cpu_buffer)
    logψ = na.meanfield(x, z)
    return nothing
end
function compute_mflogψ!(na::NeuralAnsatz, addr, z, multi_forward_buffer)
    x = prepare_input_occ!(na, addr, multi_forward_buffer.x_cpu)
    logψ = na.meanfield(x, z)
    return nothing
end

# """
#     multi_compute_logψ!(ansatz, flat_addrs_m, flat_vals_m)
#
# This function allows evaluation of inputs bigger than batch size. It calls
# [`compute_logψ`](@ref) (and possibly [`compute_mflogψ!`](@ref)) in loop to accomodate
# inputs exceeding batch size and accumulates results into `flat_vals_m` array.
#
# The indexing of input `flat_addrs_m` vector and accumulated result `flat_vals_m` is 
# preserved.
#
# # Variables
#
# * `ansatz`: structure of [`NeuralAnsatz`](@ref).
# * `flat_addrs_m`: vector of addresses in Rimu format. Can have arbitrary length (
#     usually beyond batch size0
# * `flat_vals_m`: the result of `logψ` computations are saved in this array which is dynamically
#     sized.
# """
"""
    multi_compute_logψ!(ansatz, addrs_buf, vals_buf, offsets) -> vals

This function evaluates the Neural Network on all valid off-diagonal addresses, with
an arbitrary number of them. All valid off-diagonals are treated as one continuous stream, 
which is cut into forward passes.

Each forward pass consists of:
1. [`_prepare_input_stream_kernel!`](@ref) — fills the network input `xe` with the
   occupations of the next `n` off-diagonals of the stream,
2. the forward pass of the model,
3. a contiguous copy of the `n` outputs into `vals`, in stream order.

The outputs are stored packed, in stream order: column `g` of `vals` holds the outputs
of stream position `g`. The valid part is `view(vals, :, 1:total)` with
`total = last(offsets)`; everything beyond is unused capacity (dummies).

# Variables
* `ansatz`: structure of [`NeuralAnsatz`](@ref).
* `addrs_buf`: [`GPUGrowRowBuffer`](@ref) holding off-diagonal addresses as `(K, B)`:
    column `b` holds the off-diagonals spawned by parent address.
* `vals_buf`: [`GPUGrowColumnBuffer`](@ref) for the network outputs `(n_out, capacity)`;
    grown to at least `total` columns.
* `offsets`: device vector of length `B`, the cumulative number of valid off-diagonals
    per column (`cumsum` of the counts). Column `b` holds `offsets[b] - offsets[b-1]`
    valid off-diagonals (with `offsets[0] = 0`).

# Indices
* `j`: column of the network input `xe` (and output `raw`) in the current forward
    pass, `1 … n`. Every thread of a kernel handles one `j`.
* `offset_current`: number of stream positions already processed by earlier passes
    (`0, fwd, 2fwd, …`).
* `g = offset_current + j`: global position in the stream of valid off-diagonals,
    `1 … total`, continuous over all passes.
* `b = _find_column(offsets, g)`: column of `addrs` that stream position `g` belongs
    to (see [`_find_column`](@ref)).
* `k = g - _column_start(offsets, b)`: row of that off-diagonal within column `b`
    (see [`_column_start`](@ref)).

`b` and `k` are only needed to read the address `addrs[k, b]`; the output position is
`g` itself.

## Note
The forward size `fwd` is the number of columns of the input buffer: the
[`MultiForwardBuffer`](@ref) `buffer_size` if present, otherwise the model `batch` size.
"""
function multi_compute_logψ!(ansatz::NeuralAnsatz, addrs_buf::GPUGrowRowBuffer,
                             vals_buf::GPUGrowColumnBuffer, offsets)
    addrs = addrs_buf.data                                  # (K, B)
    total = isempty(offsets) ? 0 : Int(maximum(offsets))   # = last(offsets)
    vals  = ensure_capacity!(vals_buf, total)               # (n_out, capacity)
    n_out = size(vals, 1)

    model = ansatz.model
    mfb   = ansatz.multi_forward_buffer
    xe, x = mfb !== nothing ? (mfb.xe, mfb.x) : (model.xe, model.x)
    fwd   = size(x, ndims(x))                               # columns of one forward pass

    ch, f, s = occupation_channel(model.enc), ansatz.input_scale_func, ansatz.normalisation
    backend  = KernelAbstractions.get_backend(vals)

    for offset_current in 0:fwd:(total - 1)
        n = min(fwd, total - offset_current)                # valid off-diagonals in this pass

        _prepare_input_stream_kernel!(backend)(xe, addrs, offsets, offset_current,
                                               ch, f, s; ndrange = n)

        raw = mfb !== nothing ? model(x, mfb) : model(x)    # forward pass

        # raw columns 1:n → vals columns offset_current+1 : offset_current+n
        # copyto!(view(vals, :, offset_current+1 : offset_current+n), view(raw, :, 1:n))
        copyto!(vals, n_out * offset_current + 1, raw, 1, n_out * n)
    end
    KernelAbstractions.synchronize(backend)
    return vals
end

"""
    _prepare_input_stream_kernel!(xe, addrs, offsets, offset_current, ch, f, s)

Kernel that fills the network input for one forward pass. Launched with `ndrange = n`,
one thread per input column `j`. Thread `j` takes stream position
`g = offset_current + j`, finds its slot `(k, b)` with [`_find_column`](@ref) and
[`_column_start`](@ref), and writes the scaled occupations
`f(onr(addrs[k, b])) * s` into `xe[:, ch, j]`.

# Variables
* `xe`: network input viewed as `xe[m, c, j]` (site, channel, batch position).
* `addrs`: off-diagonal addresses `(K, B)`.
* `offsets`: cumulative valid counts per column, length `B`.
* `offset_current`: stream positions processed by earlier passes.
* `ch`: occupation channel of the input encoding.
* `f`, `s`: `input_scale_func` and normalisation of the [`NeuralAnsatz`](@ref).

See [`multi_compute_logψ!`](@ref) for the meaning of the indices.
"""
@kernel function _prepare_input_stream_kernel!(xe, @Const(addrs), @Const(offsets),
                                               offset_current::Int, ch::Int, f, s)
    j = @index(Global, Linear)                  # column of xe in this forward pass
    g = offset_current + j                      # position in the stream
    b = _find_column(offsets, g)                # column of addrs
    k = g - _column_start(offsets, b)           # off-diagonal within column b
    o = onr(addrs[k, b])
    for m in 1:length(o)
        xe[m, ch, j] = f(Float32(o[m])) * s
    end
end

"""
    _find_column(offsets, g) -> b

Return the column `b` of `addrs` that stream position `g` belongs to: the first `b`
with `offsets[b] ≥ g`, i.e. the first column that has not ended before position `g`.
Uses a binary search. Columns without valid off-diagonals have
the same `offsets` value as their predecessor and are skipped automatically.
"""
@inline function _find_column(offsets, g)
    lo, hi = 1, length(offsets)
    while lo < hi
        mid = (lo + hi) ÷ 2
        if offsets[mid] < g
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

"""
    _column_start(offsets, b) -> Int

Number of stream positions before column `b`: the end of the previous column,
`offsets[b - 1]`, or `0` for the first column. Column `b` occupies stream positions 
`_column_start(offsets, b) + 1 : offsets[b]`, so the row of position `g` within its 
column is `k = g - _column_start(offsets, b)`.
"""
@inline _column_start(offsets, b) = b == 1 ? 0 : Int(offsets[b - 1])
# function multi_compute_logψ!(ansatz::NeuralAnsatz, addrs_buf::GPUGrowRowBuffer,
#                              vals_buf::GPUGrowColumnBuffer, offsets)
#     addrs = addrs_buf.data                          # (K, B)
#     K, B  = size(addrs)
#     vals  = ensure_capacity!(vals_buf, K * B)       # (n_out, capacity)
#
#     mfb   = ansatz.multi_forward_buffer
#     model = ansatz.model
#     xe, x = mfb !== nothing ? (mfb.xe, mfb.x) : (model.xe, model.x)
#     batch = size(x, ndims(x))
#
#     n_cols = batch ÷ K                              # addrs columns per forward pass
#     n_cols >= 1 || error("forward batch ($batch) smaller than K ($K) - num_offdiagonals.")
#     n_chunks = cld(B, n_cols)
#
#     backend = KernelAbstractions.get_backend(vals)
#     ch, f, s = occupation_channel(model.enc), ansatz.input_scale_func, ansatz.normalisation
#
#     for c in 1:n_chunks
#         i_start = (c - 1) * n_cols + 1
#         i_end   = min(c * n_cols, B)
#         n_real  = i_end - i_start + 1               # columns of addrs in this chunk
#
#         _prepare_input_chunk_kernel!(backend)(xe, addrs, offsets, i_start, ch, f, s;
#                                               ndrange = n_real)
#         KernelAbstractions.synchronize(backend)
#
#         raw = mfb !== nothing ? model(x, mfb) : model(x)          # forward pass
#
#         _store_outputs_chunk_kernel!(backend)(vals, raw, offsets, i_start, K;
#                                               ndrange = n_real)
#         KernelAbstractions.synchronize(backend)
#     end
#     return vals
# end
# # writes the valid off-diagonals of addrs[:, b] into xe
# @kernel function _prepare_input_chunk_kernel!(xe, @Const(addrs), @Const(offsets),
#                                               i_start::Int, ch::Int, f, s)
#     n = @index(Global, Linear)                      # position in the chunk
#     b = i_start + n - 1                             # column of addrs
#     K = size(addrs, 1)
#     for k in 1:offsets[b]                           # valid off-diagonals only
#         o = onr(addrs[k, b])
#         for m in 1:length(o)
#             xe[m, ch, (n - 1) * K + k] = f(Float32(o[m])) * s
#         end
#     end
# end
# # copies the outputs of the valid off-diagonals back to vals
# @kernel function _store_outputs_chunk_kernel!(vals, @Const(raw), @Const(offsets),
#                                               i_start::Int, K::Int)
#     n = @index(Global, Linear)                      # position in the chunk
#     b = i_start + n - 1                             # column of addrs
#     for k in 1:offsets[b]
#         for o in 1:size(raw, 1)
#             vals[o, (b - 1) * K + k] = raw[o, (n - 1) * K + k]
#         end
#     end
# end
# function multi_compute_logψ!(ansatz::NeuralAnsatz, flat_addrs_m::AbstractArray, 
#                              vals_buf::GPUGrowBuffer)
#     total = length(flat_addrs_m)
#     flat_vals_m = ensure_capacity!(vals_buf, total) # view into vals_buf.data, sized exactly to `total`
#
#     buf   = ansatz.multi_forward_buffer
#     batch = buf !== nothing ? buf.buffer_size : ansatz.model.batch
#     n_chunks = cld(total, batch)
#
#     for c in 1:n_chunks
#         i_start = (c-1) * batch + 1
#         i_end   = min(c * batch, total)
#         n_real  = i_end - i_start + 1
#         tmp_addrs = view(flat_addrs_m, i_start:i_end)
#
#         raw = buf !== nothing ? compute_logψ(ansatz, tmp_addrs, buf) : compute_logψ(ansatz, tmp_addrs)
#
#         if ansatz.meanfield !== nothing
#             mfresult = buf !== nothing ? compute_mflogψ(ansatz, tmp_addrs, buf) : compute_mflogψ(ansatz, tmp_addrs)
#             raw .= raw .+ mfresult
#         end
#
#         copyto!(view(flat_vals_m, :, i_start:i_end), view(raw, :, 1:n_real))
#     end
#     return flat_vals_m
# end
# function multi_compute_logψ!(ansatz::NeuralAnsatz, flat_addrs_m::AbstractArray, flat_vals_m::AbstractArray)    
#     empty!(flat_vals_m) 
#     if ansatz.multi_forward_buffer !== nothing
#         batch = ansatz.multi_forward_buffer.buffer_size
#     else
#         batch = ansatz.model.batch
#     end
#     total = length(flat_addrs_m)
#     n_chunks = cld(total, batch)
#     # chunked NN forward pass over all off-diagonal addresses
#     for c in 1:n_chunks
#         i_start = (c-1) * batch + 1
#         i_end   = min(c * batch, total)
#         n_real  = i_end - i_start + 1
#
#         tmp_addrs = view(flat_addrs_m, i_start:i_end)  # slicing of vector for batch size pass
#
#         if ansatz.multi_forward_buffer !== nothing
#             raw = compute_logψ(ansatz, tmp_addrs, ansatz.multi_forward_buffer)
#             copyto!(ansatz.multi_forward_buffer.z_cpu, raw)    
#
#             if ansatz.meanfield !== nothing 
#                 compute_mflogψ!(ansatz, tmp_addrs, ansatz.multi_forward_buffer.z_cpu, ansatz.multi_forward_buffer)
#             end
#
#             append!(flat_vals_m, view(ansatz.multi_forward_buffer.z_cpu, :, 1:n_real))
#         else
#             raw = compute_logψ(ansatz, tmp_addrs)
#             copyto!(ansatz.z_cpu, raw)    
#
#             if ansatz.meanfield !== nothing
#                 compute_mflogψ!(ansatz, tmp_addrs, ansatz.z_cpu)
#             end
#
#             append!(flat_vals_m, view(ansatz.z_cpu, :, 1:n_real))
#         end
#     end
# end

"""
    compute_ψ_64(ansatz, addr) -> Float64.(exp.(logψ))

Similar as [`compute_logψ`](@ref), but returns exp(logψ) values in Float64 format. 
This way the return value is compatible with Rimu format. It is used in Rimu's
Importance Sampling.

## Note
The `Float32 -> Float64` transfer is realised with `copyto!` where we copy 
`model` results to `Float64` buffer and the conversion happens automatically.
"""
function compute_ψ_64(na::NeuralAnsatz, addr)
    x = prepare_input!(na, addr, na.x_cpu_buffer)
    logψ = na.model(x)
    copyto!(na.z_cpu, logψ)
    na.z_cpu .= exp.(na.z_cpu)
    # return exp.(Float64.(logψ .- 40f0))
    return na.z_cpu
end
function compute_ψ_64(na::NeuralAnsatz, addr, multi_forward_buffer)
    x = prepare_input!(na, addr, multi_forward_buffer.x_cpu)
    logψ = na.model(x, multi_forward_buffer)
    copyto!(multi_forward_buffer.z_cpu, logψ)
    multi_forward_buffer.z_cpu .= exp.(multi_forward_buffer.z_cpu)
    # return exp.(Float64.(logψ .- 40f0))
    return multi_forward_buffer.z_cpu
end

"""
    (na)(addr, params)

Calls [`compute_ψ_64`](@ref) function during Rimu Importance Sampling 
calculations. 
"""
function (na::NeuralAnsatz)(addr)
    return compute_ψ_64(na, addr)
end
function (na::NeuralAnsatz)(addr, multi_forward_buffer)
    return compute_ψ_64(na, addr, multi_forward_buffer)
end

