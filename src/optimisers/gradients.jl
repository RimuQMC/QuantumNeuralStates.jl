"""
    apply_loss_mode!(grad, E_locs, w, E_mean, variance, α, mode)

Here are defined loss functions that someone can use in optimisation problems related
with neural network training with VMC approach. It calculates the gradient in place of
`grad` buffer.

Anyone can define its own loss function with symbol specifier used in [`TrainingPhase`](@ref).
It is important to emphatize that those gradients of loss functions are always calculated 
w.r.t. log|ψ|!

```math
\\frac{\\partial{\\mathcal{L}}}{log\\psi}  
```
"""
function apply_loss_mode!(grad, E_locs, w, E_mean, variance, α, mode)
    if mode === :energy
        @. grad += α * 2 * w * (E_locs - E_mean)
    elseif mode === :variance
        @. grad += α * 2 * w * ((E_locs - E_mean)^2 - variance)
    elseif mode === :std
        σ = sqrt(variance)
        @. grad += α * w * ((E_locs - E_mean)^2 - variance) / σ
    elseif mode === :logvar
        @. grad += α * 2 * w * ((E_locs - E_mean)^2 - variance) / (variance + 1)
    # elseif mode === :norm
    #     @. grad += α * 4 * w * raw_norm * (raw_norm - 1)
    else
        error("Invalid loss mode: $mode")
    end
end

"""
    apply_loss_composite!(grad, E_locs, w, E_mean, variance, modes)

If `modes` is Tuple it allows to use multiple loss functions. 

# Notes

when defining modes, someone can use either `::Symbol` defined in [`apply_loss_mode!`](@ref)
for single loss function or use `::Tuple` as ((::Symbol, ::Number), ...) where the extra 
number value determins with what weight that loss function is applied.

## Example
((L₁, 0.8), (L₂, 0.2), ...) -> L = 0.8*L₁ + 0.2*L₂ + ...
"""
function apply_loss_composite!(grad, E_locs, w, E_mean, variance, modes)
    for (mode, α) in modes
        apply_loss_mode!(grad, E_locs, w, E_mean, variance, α, mode)
    end
end

"""
    apply_loss!(grad, E_locs, w, E_mean, variance, mode)

This function calculates, in-place `grad`, gradients of loss function w.r.t. logψ. It 
handle single loss function but also composite loss functions defined in `mode`.

# Variables

* `grad`: array buffer (vector) in which gradients are saved.
* `E_locs`: vector of local energies calculated from VMC.
* `w`: weights from VMC samplers (uniform - MCMC / custom - CTMC). See
    [`metropolis_sample!`](@ref) and [`ctmc_sample!`](@ref).
* `E_mean`: mean energy calculated from `E_locs` weighted by `w`.
* `variance`: variance of `E_locs`.
* `mode`: holds information about what loss function is applied. Each loss 
    function needs to be defined in [`apply_loss_mode!`](@ref) with its own symbol.
    See also [`apply_loss_composite!`](@ref).
"""
function apply_loss!(grad, E_locs, w, E_mean, variance, mode)
    fill!(grad, 0.0)

    if mode isa Tuple
        apply_loss_composite!(grad, E_locs, w, E_mean, variance, mode)
    else
        apply_loss_mode!(grad, E_locs, w, E_mean, variance, 1.0, mode)
    end
end
