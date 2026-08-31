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
device = Metal.mtl
```

Next we set up the quantum system using `Rimu` interface. In this case 1D real Hubbard model.

```
M=10 # number of sites
N=50 # number of particles
addr = near_uniform(BoseFS{N,M})
H = HubbardReal1D(addr; u=0.1)
```

Now we define the Neural Network itself. We define fully-connected network with 3 hidden layers 
with 100 neurons each, using `tanh` as activation function in all layers (except output). 

```julia
batch = 1024
model = build_model("FCNN", [M, 100, 100, 100, 1], tanh; batch=batch, device=device)
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
[ Info: Metal (mtl) was loaded for GPU computations
[ Info: Hilbert space dimension: 1.257e+10
####################################################################################################
  Training with: 2 phase(s)
####################################################################################################

  Phase 1/2  |  mode=energy,  optimiser=adam,  vmc_sampler=metropolis,  max_epochs=500
────────────────────────────────────────────────────────────────────────────────────────────────────
Block  E_block            E_err        Var_block    |ΔE|        |Δvar|      Accept    η (LR)
────────────────────────────────────────────────────────────────────────────────────────────────────
1      -75.4216519638     6.61e-01     1.33e+02     0.00e+00    0.00e+00    0.8955    1.00e-03
2      -81.3519736909     5.46e-01     6.47e+01     5.93e+00    6.83e+01    0.7715    1.00e-03
3      -84.7818395856     2.18e-01     2.59e+01     3.43e+00    3.88e+01    0.7393    1.00e-03
4      -86.4517515764     1.21e-01     1.77e+01     1.67e+00    8.23e+00    0.7148    1.00e-03
5      -86.9696813192     3.33e-02     1.06e+01     5.18e-01    7.08e+00    0.7480    1.00e-03
6      -87.3760715511     4.56e-02     7.66e+00     4.06e-01    2.93e+00    0.7295    1.00e-03
────────────────────────────────────────────────────────────────────────────────────────────────────
  Phase 1 converged after 60 epochs  |  E = -87.3760715511  |  var = 7.657163

  Phase 2/2  |  mode=energy,  optimiser=minSR,  vmc_sampler=ctmc,  max_epochs=1000
────────────────────────────────────────────────────────────────────────────────────────────────────
Block  E_block            E_err        Var_block    |ΔE|        |Δvar|      Accept    η (LR)
────────────────────────────────────────────────────────────────────────────────────────────────────
7      -87.5751969399     1.45e-02     4.95e+00     0.00e+00    0.00e+00    1.0000    1.00e-03
8      -87.6655968089     1.94e-02     3.81e+00     9.04e-02    1.13e+00    1.0000    1.00e-03
9      -87.7501034566     1.27e-02     2.92e+00     8.45e-02    8.98e-01    1.0000    1.00e-03
10     -87.7844468009     1.65e-02     2.24e+00     3.43e-02    6.75e-01    1.0000    1.00e-03
11     -87.8482979563     1.24e-02     1.73e+00     6.39e-02    5.10e-01    1.0000    1.00e-03
12     -87.8799704622     1.22e-02     1.56e+00     3.17e-02    1.75e-01    1.0000    1.00e-03
13     -87.8954311235     1.25e-02     1.28e+00     1.55e-02    2.78e-01    1.0000    1.00e-03
14     -87.9222552485     6.18e-03     1.05e+00     2.68e-02    2.28e-01    1.0000    1.00e-03
15     -87.9450072279     6.82e-03     1.03e+00     2.28e-02    1.77e-02    1.0000    1.00e-03
16     -87.9676880299     1.06e-02     7.93e-01     2.27e-02    2.38e-01    1.0000    1.00e-04
────────────────────────────────────────────────────────────────────────────────────────────────────
  Phase 2 converged after 100 epochs  |  E = -87.9676880299  |  var = 0.793347
┌ Info: Saving (22001 weights, 1024 addresses, and identity input scaling function with norm of
└ nothing) to ./weights/example.txt

Final blocking analysis on 102400 E_locs samples
CombinedBlockingResult{Float64}
  mean = -87.9509 ± 0.0081
  with uncertainty of ± 0.0002743148796123271
  Combined from 100 blocking results. (k ∈ 1 … 5)

```

Another training example is showned in `example2.jl`. In this example we train two output
neural network with first output activation function `identity` and second `tanh`, using
`LogPsiSignTanh()` ansatz. This ansatz allows to predict real wave-functions with signs.


