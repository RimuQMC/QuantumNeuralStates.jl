# Neural Networks

```@meta
CurrentModule = QuantumNeuralStates
```
## Activation functions
The neural network forward pass computes activation of it current output `z = f(a)`, where
`a` is pre-activation output, `z` is post-activation output, and `f()` is the activation
function (non-linear).

This package contains few most known activation functions as:
* `identity`: this is only exception, as it is used in the final network output as linear activation
    for continuous variable estimate.
* `relu`: unbounded, non-negative function.
* `tanh`: bounded function in range [-1,1].
* `tanh_fast`: bounded function in range [-1,1]. Currently not working on `Metal` GPUs backend.
* `sigmoid`: bounded function in range [0,1].
* `sigmoid_fast`: bounded function in range [0,1].
* `gelu`: unbounded function in positive ranges and suppressed in negative ranges. Currently not 
    working on `Metal` GPUs backend.
* `gelu_fast`: unbounded function in positive ranges and suppressed in negative ranges. Currently 
    not working on `Metal` GPUs backend.

Above functions are exported from `NNlib` package (exception `gelu_fast`), but everyone can define its 
own function inside `./src/activations.jl`. As the code also needs derivatives of activation functions
(for backpropagation purposes), those derivatives needs to be defined again inside `./src/activations.jl`
file with `ACT_DERIV` Dictionary based lookup `act => act_deriv`.

```@docs
ACT_DERIV
apply_act!
apply_act_deriv!
```

## Loss functions and gradients
```@docs
apply_loss!
apply_loss_mode!
apply_loss_composite!
```
## Layers
```@docs
AbstractLayer
ParametricLayer
FreeLayer
```
### Dense
```@docs
Dense
DenseBuffer
```
### Conv
```@docs
Conv
ConvBuffer
PadMode
NoPad
Periodic
Zeros
```
### Pool
```@docs
Pool
PoolBuffer
```
## Chain
```@docs
Chain
prepare_chain_input!
n_params
forward
LayerNorm
ln_forward!
MultiForwardBuffer
```
## Backpropagation
```@docs
JacobianBuffer
back_jacobian!
flatten_jacobian!
back!
ln_backward!
update!
```

## Index

```@index
Pages = ["neuralnetworks.md"]
```
