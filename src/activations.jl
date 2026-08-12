#using NNlib

relu_deriv(x::T)     where T<:Real = x > zero(T) ? one(T) : zero(T)
sigmoid_deriv(x::T)  where T<:Real = (s = sigmoid(x); s * (one(T) - s))
tanh_deriv(x::T)     where T<:Real = one(T) - tanh(x)^2
identity_deriv(x::T) where T<:Real = one(T)

sigmoid_fast_deriv(x::T) where T<:Real = (s = sigmoid_fast(x); s * (one(T) - s))
tanh_fast_deriv(x::T)    where T<:Real = one(T) - tanh_fast(x)^2

gelu_deriv(x::T) where T<:Real = (c = T(sqrt(2/π));
                                  u = c * (x + T(0.044715) * x*x*x);
                                  th = tanh(u);
                                  T(0.5) * (one(T) + th) +
                                  T(0.5) * x * (one(T) - th*th) * c * (one(T) + T(0.134145) * x*x))

# Dictionary mapping of activation functions and their derivatives
"""
    ACT_DERIV

This directory holds mapping of activation functions to its derivatives.
Both activations and derivatives can be defined here if needed.
"""
const ACT_DERIV = Dict(
    tanh => tanh_deriv,
    tanh_fast => tanh_fast_deriv,
    relu => relu_deriv,
    sigmoid => sigmoid_deriv,
    sigmoid_fast => sigmoid_fast_deriv,
    identity => identity_deriv,
    gelu => gelu_deriv,
)
