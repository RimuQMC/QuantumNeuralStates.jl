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

Now we define the Neural Network itself. 

Firstly, we need to define what will be our input. The
most default is `OccupationEncoding((dims..., H))` as pure occupation number input grid (the
dimensions are defined by `dims...` Tuple). 

Secondly, we define our network `Chain` as combination of 
three convolutional layers `Conv` with kernel size `(3,)`, number of features in each layer `32`, 
activation function `relu`, and `Periodic()` pedding. We will connect the `Conv` layers via
`Pool` (output dimension reduction) to fully-connected layers `Dense`. Last layer's activation
layer is `identity` with one output as we want to predict pure real wave-function (see more below). 
This whole package is designed for different networks architecture and layer combinations.

```julia
batch = 1024
act = relu
pad = Periodic()
input = OccupationEncoding((10,), H; device=device)
model = Chain(input,
              Conv((3,), nchannels(input)=>32, act; batch=batch, device=device, pad=pad),
              Conv((3,), 32=>32, act; batch=batch, device=device, pad=pad),
              Conv((3,), 32=>32, act; batch=batch, device=device, pad=pad),
              Pool(:mean; device=device),
              Dense(32=>32, tanh; batch=batch, device=device, layer_norm=true), 
              Dense(32=>1, identity; batch=batch, device=device); 
              batch=batch, device=device)
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
[ Info: Total memory estimate: 124.72 MiB
####################################################################################################
  Training with: 2 phase(s)
####################################################################################################

  Phase 1/2  |  mode=energy,  optimiser=adam,  vmc_sampler=metropolis,  max_epochs=500
────────────────────────────────────────────────────────────────────────────────────────────────────
Block  E                  E_err        Var          |ΔE|        |Δvar|      Accept    η (LR)
────────────────────────────────────────────────────────────────────────────────────────────────────
1      -74.4890867922     8.48e-01     1.36e+02     0.00e+00    0.00e+00    0.9062    1.00e-03
2      -79.7448248401     7.35e-01     8.66e+01     5.26e+00    4.95e+01    0.7861    1.00e-03
3      -83.9752008669     1.73e-01     5.84e+01     4.23e+00    2.82e+01    0.7627    1.00e-03
4      -85.8995638030     1.04e-01     2.58e+01     1.92e+00    3.25e+01    0.7549    1.00e-03
5      -86.8934642452     4.70e-02     1.40e+01     9.94e-01    1.19e+01    0.7402    1.00e-03
6      -87.2132227072     6.73e-02     1.46e+01     3.20e-01    6.59e-01    0.7500    1.00e-03
────────────────────────────────────────────────────────────────────────────────────────────────────
  Phase 1 converged after 60 epochs  |  E = -87.2132227072  |  var = 14.610711

  Phase 2/2  |  mode=energy,  optimiser=minSR,  vmc_sampler=ctmc,  max_epochs=1000
────────────────────────────────────────────────────────────────────────────────────────────────────
Block  E                  E_err        Var          |ΔE|        |Δvar|      Accept    η (LR)
────────────────────────────────────────────────────────────────────────────────────────────────────
7      -87.3900796444     3.78e-02     1.24e+01     0.00e+00    0.00e+00    1.0000    1.00e-03
8      -87.6015161571     2.55e-02     5.09e+00     2.11e-01    7.27e+00    1.0000    1.00e-03
9      -87.6873272386     1.42e-02     3.50e+00     8.58e-02    1.60e+00    1.0000    1.00e-03
10     -87.7525259735     2.07e-02     3.06e+00     6.52e-02    4.32e-01    1.0000    1.00e-03
11     -87.8307316718     1.34e-02     1.64e+00     7.82e-02    1.42e+00    1.0000    1.00e-03
12     -87.8542562406     9.03e-03     1.36e+00     2.35e-02    2.88e-01    1.0000    1.00e-03
13     -87.8614857501     1.08e-02     1.22e+00     7.23e-03    1.42e-01    1.0000    1.00e-03
14     -87.8705928733     7.71e-03     1.10e+00     9.11e-03    1.19e-01    1.0000    1.00e-03
15     -87.9021396178     1.10e-02     9.23e-01     3.15e-02    1.73e-01    1.0000    1.00e-04
────────────────────────────────────────────────────────────────────────────────────────────────────
  Phase 2 converged after 90 epochs  |  E = -87.9021396178  |  var = 0.923180
┌ Info: Saving (7489 weights, 1024 addresses, and identity input scaling function with norm of nothing) to
└ ./weights/example.txt

Final blocking analysis on 102400 E_locs samples
CombinedBlockingResult{Float64}
  mean = -87.9188 ± 0.008
  with uncertainty of ± 0.000338377680949751
  Combined from 100 blocking results. (k ∈ 1 … 7)
```

Another training example is showned in `example2.jl`. In this example we train two output
neural network with first output activation function `identity` and second `tanh`, using
`LogPsiSignTanh()` ansatz. This ansatz allows to predict real wave-functions with signs.


