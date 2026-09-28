# QuantumNeuralStates

Machine Learning package for representing quantum wave-function using Neural Networks.
This package is designed to work in batched approach supporting GPU acceleration.

The package is designed around [Rimu.jl](https://github.com/RimuQMC/Rimu.jl).

## Installation

QuantumNeuralStates.jl is not yet registered. To install it, run

```julia
import Pkg; Pkg.add(url="https://github.com/RimuQMC/QuantumNeuralStates.jl")
```

## Usage Guide

```julia
using Rimu
using QuantumNeuralStates
```

First we need to decide if we want to use GPU for Neural Networks calculations or not.
If so, we need to also include GPU julia packages: `CUDA`, `Metal`, (`AMDGPU` - not tested).
We also need to define `device` which would hold GPU function (default `device = identity` -
CPU)

```julia
using Metal
device = select_device() # or manually Metal.mtl / CUDA.cu
```

Next we set up the quantum system using `Rimu` interface. In this case 1D real Hubbard model.

```
M=10 # number of sites
N=50 # number of particles
addr = near_uniform(BoseFS{N,M})
H = HubbardReal1D(addr; u=0.1)
```

Now we define the Neural Network itself. We define our network `Chain` as combination of three
convolutional layers `Conv` with kernel size `(3,)`, number of features in each layer `32`, 
activation function `relu`, and `Periodic()` pedding. We will connect the `Conv` layers via
`Pool` (output dimension reduction) to fully-connected layers `Dense`. Last layer's activation
layer is `identity` with one output as we want to predict pure real wave-function (see more below). 
This whole package is designed for different networks architecture and layer combinations.

```julia
batch = 1024
act = relu
pad = Periodic()
model = Chain(Conv((3,), 1=>32, act; batch=batch, device=device, pad=pad),
              Conv((3,), 32=>32, act; batch=batch, device=device, pad=pad),
              Conv((3,), 32=>32, act; batch=batch, device=device, pad=pad),
              Pool(:mean; device=device),
              Dense(32=>32, tanh; batch=batch, device=device, layer_norm=true), 
              Dense(32=>1, identity; batch=batch, device=device); 
              batch=batch, device=device, input_size=(10,))
```

So far we only defined pure neural network. We need to define physical `ansatz` that would 
represent the wave-function. The neural network output is one number which would represent 
the logarithm of the wave-function `log|ψ|` (`LogPsi()` type of ansatz). The wave-function 
ansatz can be customly defined as a subtype of abstract `AnsatzType` (there you can also 
define how many outputs ansatz needs).

```julia
ansatz = NeuralAnsatz(LogPsi(), H, model, batch)
```

Lastly, we need to define training parameters. This is done inside `TrainingPhase` struct,, which
can be stacked. We can also define keyword arguments about details if we want to save/load
result of training.

```julia
phases = [
    TrainingPhase(
        mode       = :energy,
        optimiser  = :adam,
        vmc_sampler= :metropolis,
        stop       = StopBuffer(var_thr=1000),
        η          = 0.001f0,
        skip       = [(1, 50), (300, 200)],
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 500,
    ),
    TrainingPhase(
        mode       = :energy,
        optimiser  = :minSR,
        vmc_sampler= :ctmc,
        stop       = StopBuffer(ΔE_thr=0.00005, var_thr=1),
        η          = 0.001f0,
        λ          = 0.001f0,
        skip       = [(1, 20)],
        η_decrease = [(1, 0.1)], 
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 1000,
    ),
]

SAVEFILE     = "./weights/example.txt"
SAVE_WEIGHTS = true
LOADFILE     = ""
LOAD_WEIGHTS = false
```

Finally we can run the training loop.

```julia
E, E_err, var, new_addrs = run_training_loop(H, ansatz, addr, phases; 
                                            savefile=SAVEFILE, loadfile=LOADFILE, 
                                            save=SAVE_WEIGHTS, load=LOAD_WEIGHTS)

```

The return values are holding the full training history in blocks. 

Example of training output is showed below. Both example code `example.jl` and output
`example.log` can be find inside the package.

```
❯ julia --project=. example.jl
[ Info: Metal (mtl) was loaded for GPU computations
[ Info: Hilbert space dimension: 1.257e+10
[ Info: Neural Network parameters: 7489
[ Info: Total memory estimate: 124.68 MiB
####################################################################################################
  Training with: 2 phase(s)
####################################################################################################

  Phase 1/2  |  mode=energy,  optimiser=adam,  vmc_sampler=metropolis,  max_epochs=500
────────────────────────────────────────────────────────────────────────────────────────────────────
Block  E                  E_err        Var          |ΔE|        |Δvar|      Accept    η (LR)
────────────────────────────────────────────────────────────────────────────────────────────────────
1      -71.6861211235     1.04e+00     1.67e+02     0.00e+00    0.00e+00    0.9512    1.00e-03
2      -75.5132949892     8.24e-01     1.35e+02     3.83e+00    3.17e+01    0.8320    1.00e-03
3      -82.7109810417     3.67e-01     6.26e+01     7.20e+00    7.27e+01    0.7637    1.00e-03
4      -85.0026737031     1.99e-01     3.44e+01     2.29e+00    2.82e+01    0.7686    1.00e-03
5      -86.2369671306     6.12e-02     2.06e+01     1.23e+00    1.38e+01    0.7285    1.00e-03
6      -86.8131895655     6.22e-02     1.90e+01     5.76e-01    1.65e+00    0.7617    1.00e-03
────────────────────────────────────────────────────────────────────────────────────────────────────
  Phase 1 converged after 60 epochs  |  E = -86.8131895655  |  var = 18.969346

  Phase 2/2  |  mode=energy,  optimiser=minSR,  vmc_sampler=ctmc,  max_epochs=1000
────────────────────────────────────────────────────────────────────────────────────────────────────
Block  E                  E_err        Var          |ΔE|        |Δvar|      Accept    η (LR)
────────────────────────────────────────────────────────────────────────────────────────────────────
7      -87.2503069227     3.19e-02     9.28e+00     0.00e+00    0.00e+00    1.0000    1.00e-03
8      -87.4415385223     3.13e-02     5.38e+00     1.91e-01    3.90e+00    1.0000    1.00e-03
9      -87.6000121929     1.69e-02     3.01e+00     1.58e-01    2.37e+00    1.0000    1.00e-03
10     -87.6418938369     1.54e-02     2.42e+00     4.19e-02    5.86e-01    1.0000    1.00e-03
11     -87.7180320774     1.05e-02     2.06e+00     7.61e-02    3.66e-01    1.0000    1.00e-03
12     -87.7366279076     9.79e-03     1.72e+00     1.86e-02    3.36e-01    1.0000    1.00e-03
13     -87.7623779841     1.04e-02     1.46e+00     2.58e-02    2.58e-01    1.0000    1.00e-03
14     -87.8221424469     9.56e-03     1.28e+00     5.98e-02    1.84e-01    1.0000    1.00e-03
15     -87.8341310239     1.01e-02     1.18e+00     1.20e-02    9.55e-02    1.0000    1.00e-03
16     -87.8363731661     5.15e-03     1.07e+00     2.24e-03    1.11e-01    1.0000    1.00e-03
17     -87.8637776697     1.08e-02     9.95e-01     2.74e-02    7.68e-02    1.0000    1.00e-04
────────────────────────────────────────────────────────────────────────────────────────────────────
  Phase 2 converged after 110 epochs  |  E = -87.8637776697  |  var = 0.994559
┌ Info: Saving (7489 weights, 1024 addresses, and identity input scaling function with norm of nothing) to
└ ./weights/example.txt

Final blocking analysis on 102400 E_locs samples
CombinedBlockingResult{Float64}
  mean = -87.8581 ± 0.0079
  with uncertainty of ± 0.00023585025405172975
  Combined from 100 blocking results. (k ∈ 1 … 4)
```

Another training example is showned in `example2.jl`. In this example we train two output
neural network with first output activation function `identity` and second `tanh`, using
`LogPsiSignTanh()` ansatz. This ansatz allows to predict real wave-functions with signs.


