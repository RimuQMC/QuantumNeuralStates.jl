mutable struct GPUGrowBuffer{T, AT<:AbstractArray{T,2}}
    data::AT
    fixed_dim::Int
    capacity::Int
end

function GPUGrowBuffer(backend, ::Type{T}, fixed_dim::Int, init_capacity::Int) where {T}
    data = KernelAbstractions.allocate(backend, T, fixed_dim, init_capacity)
    return GPUGrowBuffer(data, fixed_dim, init_capacity)
end

function ensure_capacity!(buf::GPUGrowBuffer{T}, total::Int) where {T}
    if total > buf.capacity
        new_capacity = ceil(Int, total * 1.2)
        backend = KernelAbstractions.get_backend(buf.data)   # infer backend from current array
        newdata = KernelAbstractions.allocate(backend, T, buf.fixed_dim, new_capacity)
        buf.data = newdata
        buf.capacity = new_capacity
    end
    return view(buf.data, :, 1:total)
end

function multi_compute_logψ!(ansatz::NeuralAnsatz, flat_addrs_m::AbstractArray, vals_buf::GPUGrowBuffer)
    total = length(flat_addrs_m)
    flat_vals_m = ensure_capacity!(vals_buf, total) # view into vals_buf.data, sized exactly to `total`

    buf   = ansatz.multi_forward_buffer
    batch = buf !== nothing ? buf.buffer_size : ansatz.model.batch
    n_chunks = cld(total, batch)

    for c in 1:n_chunks
        i_start = (c-1) * batch + 1
        i_end   = min(c * batch, total)
        n_real  = i_end - i_start + 1
        tmp_addrs = view(flat_addrs_m, i_start:i_end)

        raw = buf !== nothing ? compute_logψ(ansatz, tmp_addrs, buf) : compute_logψ(ansatz, tmp_addrs)

        if ansatz.meanfield !== nothing
            mfresult = buf !== nothing ? compute_mflogψ(ansatz, tmp_addrs, buf) : compute_mflogψ(ansatz, tmp_addrs)
            raw .= raw .+ mfresult
        end

        copyto!(view(flat_vals_m, :, i_start:i_end), view(raw, :, 1:n_real))
    end
    return flat_vals_m   # a view — valid until the next `ensure_capacity!` call that grows the buffer
end
