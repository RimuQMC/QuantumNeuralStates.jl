using LinearAlgebra

mutable struct Conv{T, K <: AbstractArray{T}, V <: AbstractVector{T},
                     F <: Function, G <: Function}
    W        ::K        # kernel (dimension of window, C_in, C_out = number of kernels)
    b        ::V
    act_func ::F
    act_deriv::G
    a        ::Union{Nothing, AbstractArray{T}}    # will be initialised in first forward pass
    z        ::Union{Nothing, AbstractArray{T}}    # will be initialised in first forward pass

    stride   ::Int

    # applying NormLayer or not
    layer_norm::Union{LayerNorm, Nothing}
end

function Conv(kernel_size::NTuple{N, Int}, channels::Pair{Int,Int}, act::Function;
              stride::Int=1, device::Function=identity, Layer_Norm=false) where N
    # L_in is dimension of input 
    T = Float32
    C_in, C_out = channels.first, channels.second
    fan_in  = prod(kernel_size) * C_in
    fan_out = prod(kernel_size) * C_out
    if act === relu || act === gelu
        std = T(sqrt(2.0 / fan_in))               # He
    else
        std = T(sqrt(2.0 / (fan_in + fan_out)))   # Glorot / Xavier
    end

    # kernel shape: (spatial_dims..., C_in, C_out)
    W = device(randn(T, (kernel_size..., C_in, C_out)) .* std)
    b = device(zeros(T, C_out))

    act_d = get(ACT_DERIV, act, nothing)
    if act_d === nothing 
        error("No derivative registered for $act. Use only function define in activations.jl or define yours there.")
    end

    layer_norm = if Layer_Norm===false
        nothing
    else
        LayerNorm(C_out, batch, device)
    end

    K, V, F, G = typeof(W), typeof(b), typeof(act), typeof(act_d)
    return Conv{T,K,V,F,G}(W, b, act, act_d, nothing, nothing, stride, layer_norm)
end

function forward(layer::Conv, x::AbstractArray)
    # First pass = initialize `a` and `z`
    if layer.a === nothing
        N_spatial = ndims(x) - 2        # x is (L=spatial..., C_in, batch)
        C_in      = size(x)[end-1]
        C_out     = size(layer.W)[end]
        batch     = size(x)[end]

        K         = size(layer.W)[1:N_spatial]  # W -> (K,C_in, C_out)
        L_in      = size(x)[1:N_spatial]        # x -> (L_in, C_in, batch)

        L_out_vec = Vector{Int}(undef, N_spatial)
        for i in 1:N_spatial
            L_out_vec[i] = div((L_in[i] - K[i]), layer.stride) + 1
        end
        L_out     = Tuple(L_out_vec)

        layer.z = fill!(similar(layer.W, L_out..., C_out, batch), zero(eltype(layer.W)))
        layer.a = fill!(similar(layer.W, L_out..., C_out, batch), zero(eltype(layer.W)))
    end

    # FORWARD PASS


end



