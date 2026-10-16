

"""
    GPUGrowColumnBuffer(backend, T, fixed, init_capacity)
    GPUGrowColumnBuffer(backend, T, init_capacity)          # 1D

Device buffer of shape `(fixed..., capacity)`. The LAST dimension `capacity` grows,
all leading dimensions are fixed. Typical use: network outputs `(n_out, n)`.

`ensure_capacity!(buf, n)` guarantees `capacity ≥ n` and returns the full `buf.data`. 
The valid part is chosen by the caller, e.g. `view(buf.data, :, 1:n)`. 
Growing does not preserve old contents.
"""
mutable struct GPUGrowColumnBuffer{T, N, AT<:AbstractArray{T,N}, F<:Tuple}
    data::AT
    fixed::F
    capacity::Int
end
function GPUGrowColumnBuffer(backend, ::Type{T}, fixed::Tuple, init_capacity::Int) where {T}
    data = KernelAbstractions.allocate(backend, T, fixed..., init_capacity)
    return GPUGrowColumnBuffer(data, fixed, init_capacity)
end
GPUGrowColumnBuffer(backend, ::Type{T}, init_capacity::Int) where {T} =
    GPUGrowColumnBuffer(backend, T, (), init_capacity)

function ensure_capacity!(buf::GPUGrowColumnBuffer{T}, total::Int) where {T}
    if total > buf.capacity
        new_capacity = ceil(Int, total * 1.2)
        backend = KernelAbstractions.get_backend(buf.data)
        olddata = buf.data
        buf.data = KernelAbstractions.allocate(backend, T, buf.fixed..., new_capacity)
        buf.capacity = new_capacity
        KernelAbstractions.unsafe_free!(olddata)
    end
    # return view(buf.data, ntuple(_ -> Colon(), length(buf.fixed))..., 1:total)
    return buf.data
end

"""
    GPUGrowRowBuffer(backend, T, init_capacity, fixed)
    GPUGrowRowBuffer(backend, T, init_capacity)       # 1D

Device buffer of shape `(capacity, fixed...)`. The FIRST dimension `capacity` grows,
all trailing dimensions are fixed. Typical use: off-diagonal slots `(K, B)`,
where each parent's `K` slots are contiguous (column `b`).

`ensure_capacity!(buf, n)` guarantees `capacity ≥ n` and returns the full `buf.data`. 
The valid part is chosen by the caller, e.g. `view(buf.data, 1:counts[b], b)`. 
Growing does not preserve old contents.
"""
mutable struct GPUGrowRowBuffer{T, N, AT<:AbstractArray{T,N}, F<:Tuple}
    data::AT
    capacity::Int
    fixed::F
end
function GPUGrowRowBuffer(backend, ::Type{T}, init_capacity::Int, fixed::Tuple) where {T}
    data = KernelAbstractions.allocate(backend, T, init_capacity, fixed...)
    return GPUGrowRowBuffer(data, init_capacity, fixed)
end
GPUGrowRowBuffer(backend, ::Type{T}, init_capacity::Int) where {T} =
    GPUGrowRowBuffer(backend, T, init_capacity, ())

function ensure_capacity!(buf::GPUGrowRowBuffer{T}, total::Int) where {T}
    if total > buf.capacity
        new_capacity = ceil(Int, total * 1.2)
        backend = KernelAbstractions.get_backend(buf.data)
        olddata = buf.data
        buf.data = KernelAbstractions.allocate(backend, T, new_capacity, buf.fixed...)
        buf.capacity = new_capacity
        KernelAbstractions.unsafe_free!(olddata)
    end
    return buf.data
end

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
