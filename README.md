# QuantumNeuralStates

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://rimuqmc.github.io/QuantumNeuralStates.jl/dev/)
[![Build Status](https://github.com/RimuQMC/QuantumNeuralStates.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/RimuQMC/QuantumNeuralStates.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/RimuQMC/QuantumNeuralStates.jl/badge.svg?branch=main)](https://coveralls.io/github/RimuQMC/QuantumNeuralStates.jl?branch=main)


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

Next we set up the quantum system using `Rimu` interface. In this case 1D Froehlich Polaron.

> [!WARNING]
> For GPU calculations use only Hamiltonians that support `Float32` precision! 
> Otherwise use CPU only.

```
N = 10 # phonon cutoff per site
M = 11 # number of sites
addr = BoseFS{missing, M}()
H = FroehlichPolaron{Float32}(addr; D = 1, alpha = 2, l = 5, mode_cutoff=N)
```

Now we define the Neural Network itself. 

Firstly, we need to define what will be our input. The
most default is `OccupationEncoding((dims...), H)` as pure occupation number input grid (the
dimensions are defined by `dims...` Tuple). 

Secondly, we define our neural network `Chain` as combination of 
three dense layers `Dense` and activation function `relu`. Last layer's activation is 
`identity` with one output as we want to predict pure real wave-function (see more below). 
This whole package is designed for different networks architecture and layer combinations.

> [!IMPORTANT]
> Last layer needs to be `Dense` every time! If you use e.g. `Conv` layers, then set up
> dummy `Dense(1=>1,...)` as final output (as in commented example).
> This is because `Dense` hold native structure for wave-function ansatz type - see below.


```julia
batch = 1024
act = relu
# pad = Zeros()
input = OccupationEncoding((11,), H; device=device)
model = Chain(input,
              # FCNN example
              Dense(nsites(input)*nchannels(input)=>32, act; batch=batch, device=device), 
              Dense(32=>32, act; batch=batch, device=device), 
              Dense(32=>32, act; batch=batch, device=device), 
              Dense(32=>1, identity; batch=batch, device=device); 

              # CNN example
              # Conv((3,3), nchannels(input)=>32, act; batch=batch, device=device, pad=pad),
              # Conv((3,3), 32=>32, act; batch=batch, device=device, pad=pad),
              # Conv((3,3), 32=>32, act; batch=batch, device=device, pad=pad),
              # Conv((3,3), 32=>1, identity; batch=batch, device=device, pad=pad),
              # Pool(:sum; device=device),
              # Dense(1=>1, identity; batch=batch, device=device, init_scale=0.1);

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

> [!TIP]
> Start first phases with `metropolis` sampler. At the beginning of training it is more
> stable using uniform sampler. Later switch to `ctmc` sampler as it is more efficient 
> (although more unpredictible in first stages).

```julia
phases = [
    TrainingPhase(
        mode       = :energy,
        optimiser  = :adam,
        vmc_sampler= :metropolis,
        stop       = StopBuffer(var_thr=1000),
        η          = 0.001f0,
        skip       = [(1, 50)], 
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 500,
    ),
    TrainingPhase(
        mode       = :energy,
        optimiser  = :minSR,
        vmc_sampler= :ctmc,
        stop       = StopBuffer(var_thr=0.5),
        η          = 0.01f0,
        skip       = [(1, 20)],
        η_decrease = [(1, 0.1)],
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 300,
    ),
]

SAVEFILE = "./weights/example.txt"
SAVE_WEIGHTS = true
LOADFILE = ""
LOAD_WEIGHTS = false
OCCFILE = "./fig/example_walker_distro.pdf"
OCC_PLOT = true

```

Finally we can run the training loop.

```julia
E, E_err, var, new_addrs = run_training_loop(H, ansatz, addr, phases; 
                                savefile=SAVEFILE, save=SAVE_WEIGHTS,
                                loadfile=LOADFILE, load=LOAD_WEIGHTS, 
                                occfile=OCCFILE, occplot=OCC_PLOT)

```

The return values are holding the full training history in blocks. 

Example of training output is showed below. Both example code `example.jl` and output
`example.log` can be find inside the package. You can find saved parameters in 
`./weights/example.jl` and final walker distribution (occupation plot) in
`./fig/example_walker_distro.pdf`.

```
❯ julia --project=. example.jl
[ Info: Metal (mtl) was loaded for GPU computations
[ Info: Hilbert space dimension: 2.853e+11
[ Info: Neural Network parameters: 2529
[ Info: Memory estimate: 33.73 MiB
####################################################################################################
  Training with: 2 phase(s)
####################################################################################################

  Phase 1/2  |  mode=energy,  optimiser=adam,  vmc_sampler=metropolis,  max_epochs=500
────────────────────────────────────────────────────────────────────────────────────────────────────
Block  E                  E_err        Var          |ΔE|        |Δvar|      Accept    η_eff (LR)
────────────────────────────────────────────────────────────────────────────────────────────────────
1      907.7288208008     5.04e+01     1.71e+06     0.00e+00    0.00e+00    0.7803    1.00e-03
2      782.4915771484     3.19e+01     1.16e+06     1.25e+02    5.45e+05    0.7578    1.00e-03
3      557.8919677734     1.80e+01     6.15e+05     2.25e+02    5.45e+05    0.7295    1.00e-03
4      320.6023254395     2.64e+01     2.21e+05     2.37e+02    3.94e+05    0.5967    1.00e-03
5      160.5218353271     4.65e+00     4.08e+04     1.60e+02    1.80e+05    0.5381    1.00e-03
6      121.7953338623     4.08e+00     1.83e+04     3.87e+01    2.25e+04    0.5039    1.00e-03
7      99.9105834961      1.56e+00     1.12e+04     2.19e+01    7.11e+03    0.4600    1.00e-03
8      88.8359375000      1.07e+00     8.08e+03     1.11e+01    3.10e+03    0.4316    1.00e-03
9      80.6103057861      1.11e+00     6.14e+03     8.23e+00    1.94e+03    0.3730    1.00e-03
10     75.5901031494      6.90e-01     6.04e+03     5.02e+00    9.52e+01    0.3770    1.00e-03
11     71.3423004150      6.30e-01     4.65e+03     4.25e+00    1.39e+03    0.2930    1.00e-03
12     65.0676727295      4.49e-01     2.40e+03     6.27e+00    2.25e+03    0.2568    1.00e-03
13     61.7502212524      5.44e-01     1.24e+03     3.32e+00    1.16e+03    0.2383    1.00e-03
14     59.7186470032      3.40e-01     9.90e+02     2.03e+00    2.46e+02    0.2363    1.00e-03
────────────────────────────────────────────────────────────────────────────────────────────────────
  Phase 1 converged after 140 epochs  |  E = 59.7186470032  |  var = 990.136902

  Phase 2/2  |  mode=energy,  optimiser=minSR,  vmc_sampler=ctmc,  max_epochs=300
────────────────────────────────────────────────────────────────────────────────────────────────────
Block  E                  E_err        Var          |ΔE|        |Δvar|      Accept    η_eff (LR)
────────────────────────────────────────────────────────────────────────────────────────────────────
15     49.1356239319      1.75e+00     2.05e+02     0.00e+00    0.00e+00    0.7412    6.56e-03
16     36.9942512512      1.02e+00     3.85e+01     1.21e+01    1.66e+02    0.7617    9.08e-03
17     26.6224727631      9.92e-01     7.66e+01     1.04e+01    3.81e+01    0.7686    6.49e-03
18     15.3240280151      1.19e+00     6.46e+01     1.13e+01    1.20e+01    0.7422    6.66e-03
19     4.2937507629       8.15e-01     2.83e+01     1.10e+01    3.63e+01    0.7471    9.24e-03
20     -0.4476626515      1.58e-01     4.51e+00     4.74e+00    2.38e+01    0.7383    1.00e-02
21     -1.3234083652      4.45e-02     1.60e+00     8.76e-01    2.91e+00    0.7510    1.00e-02
22     -1.6271784306      3.20e-02     1.06e+00     3.04e-01    5.43e-01    0.7510    1.00e-02
23     -1.8302958012      1.89e-02     6.74e-01     2.03e-01    3.83e-01    0.7246    1.00e-02
24     -1.9219939709      5.39e-03     4.79e-01     9.17e-02    1.96e-01    0.7451    1.00e-03
────────────────────────────────────────────────────────────────────────────────────────────────────
  Phase 2 converged after 100 epochs  |  E = -1.9219939709  |  var = 0.478544
[ Info: Saving 2529 weights, 1024 addresses and identity input scaling (max_norm = nothing) to ./weights/example.txt

Final VMC estimate: 100 measurements × 1024 walkers, every 5 steps
BlockingResult{Float64}
  mean = -1.9202 ± 0.003
  with uncertainty of ± 0.000432004831521941
  from 25 blocks after 2 transformations (k = 3).

total occupation: mean = 3.610322265625,  min = 0,  max = 17,  samples = 102400
per mode: largest average = 0.863125 in mode 6,  smallest = 0.014609375
occupation plots saved to ./fig/example_walker_distro.pdf
```

Another training example is showned in `example2.jl`. In this example we train two output
neural network with first output activation function `identity` and second `tanh`, using
`LogPsiSignTanh()` ansatz. This ansatz allows to predict real wave-functions with signs.

