
"""
    MeanField(H)

If mean-field is enabled in [`NeuralAnsatz`](@ref), then this 
structure contains mean-field function and necessary parameters.

# Arguments

* `H`: Hamiltonian defined in Rimu for which I want to add mean-field.

This structure is meant to be general and user should define its own
dispatch on [`_choose_meanfield_function`](@ref) that returns mean-field
`func` and `params`.

Mean-field is then called with function [`(mf)`](@ref). So far this 
happens on CPU only!
"""
struct MeanField{F,P}
    func::F
    params::P
end
function MeanField(H)
    func, params = _choose_meanfield_function(H)
    
    F=typeof(func); P=typeof(params)
    return MeanField{F,P}(func, params)
end

"""
    (mf)(input, result) -> result

This function calls mean-field function with `input` (same input array as in
Neural Network model). 

The result is added into the `result` array (so it can be stuck on top of 
Neural Network result). This is important details as the result CANNOT be overwritten
by mean-field! Only add on top of existing result!
"""
function (mf::MeanField)(input::AbstractArray, result::AbstractArray)
    mf.func(result, input, mf.params)
    return result
end

"""
    _choose_meanfield_function(H::typeof(H)) -> func, params

This function allows dispatch on different type of Hamiltonians with different
mean-field needs. User should create custom function which returns mean-field
function `func` and needed parameters `params` for its evalation. 

# Note
* `params` can be arbitrary type best fitted for passing necessary values.
* When defining mean-field function read also [`(mf)`](@ref) for closer details.
"""
# function _choose_meanfield_function(H::Rimu.FroehlichPolaronND{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any})
function _choose_meanfield_function(H::Rimu.FroehlichPolaron)
    eta, f = solve_self_consistent_eta(H)

    small_val = -35
    log_f = [fi == 0.0 ? small_val : log(abs(fi)) for fi in f]   # (M,) fixed
    log_f_row = reshape(log_f, 1, :)
    const_term = -0.5 * sum(f .^ 2)                         # scalar, fixed

    params = (log_f_row, const_term)
    return forward_logMF_FroehlichND!, params
end
