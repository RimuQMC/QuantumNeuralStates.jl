module QuantumNeuralStates

using LinearAlgebra
using NNlib: tanh, relu, sigmoid, identity, tanh_fast, sigmoid_fast, gelu
using NNlib: scatter
using Rimu
using Gutzwiller
using Printf
using Statistics
using Random
using KernelAbstractions
using GPUArraysCore: @allowscalar
using Atomix
using SpecialFunctions: loggamma

const PACKAGE_NAME = "QuantumNeuralStates"

@doc """

    Quantum Neural States

Machine Learning package designed to work with Rimu ([online](https://RimuQMC.github.io/Rimu.jl/)).
"""
# QuantumNeuralStates

function __init__()
    default_logger()
end

include("./activations.jl")
include("./layers/layers_helpers.jl")
include("./layers/layernorm.jl")
include("./layers/dense.jl")
include("./layers/conv.jl")
include("./layers/pool.jl")
include("./input_encoding.jl")
include("./chain.jl")
include("./backpropagation/back_dense.jl")
include("./backpropagation/back_conv.jl")
include("./backpropagation/back_pool.jl")
include("./backpropagation/backpropagation.jl") # needs to be included as last backpropagation file
include("./utils/save_load.jl")
include("./utils/network_health_statistics.jl")
include("./utils/utils.jl")
include("./utils/gpu_helpers.jl")
include("./utils/io.jl")

# Dispatch functions for Importance Sampling in Rimu
import Rimu.DictVectors: deposit!   
import Rimu.Interfaces:  apply_operator!

# meanfields 
include("./meanfield/meanfield.jl")
export MeanField
include("./meanfield/FroehlichND_MF.jl")

# truncations
include("./truncation/truncation.jl")
export TruncationBuffer, violates_truncation
include("./truncation/center_fill_nD.jl")

# ansatz
include("./ansatz/ansatz_type.jl")
export AnsatzType, LogPsi, LogPsiSignTanh
export psi, log_psi!
include("./ansatz/ansatz.jl")
export NeuralAnsatz, prepare_input!, prepare_input_occ!, compute_logψ, multi_compute_logψ!
include("./ansatz/rimu_importance_sampling.jl")

# VMC
include("./vmc/vmc_helpers.jl")
export VMCBuffer
include("./vmc/local_energy.jl")
export calculate_local_energy!
include("./vmc/metropolis.jl")
export metropolis_sample!, metropolis_heatbath_sample!
include("./vmc/ctmc.jl")
export ctmc_sample!, ctmc_heatbath_sample!
include("./vmc/vmc.jl")
export vmc_sample!, vmc_energy

# optimisers
include("./optimisers/gradients.jl")
export apply_loss!
include("./optimisers/cg_solver.jl")
export cg_solve!
include("./optimisers/minSR.jl")
export compute_minSR!, minSR, minSRBuffer
include("./optimisers/descent.jl")
export descent, DescentBuffer
include("./optimisers/adam.jl")
export adam, AdamBuffer

# training 
include("./training/training_helpers.jl")
export StopBuffer, TrainingPhase
include("./training/training.jl")
export run_training_loop
include("./nn_create.jl")
export build_model

export nchannels, nsites, NoEncoding, OccupationEncoding, MomentumEncoding
export Chain, Dense, Conv, Pool
export NoPad, Periodic, Zeros # padding structs
export JacobianBuffer, make_buffers, back_jacobian!, back!
export forward, back_jacobian!, update!, addrs_random, final_elocs_statistics!
export neuron_statistics, jacobian_statistics
export save_master, load_master, log_markov_chain
export memory_estimate, select_device

export tanh, relu, sigmoid, gelu, tanh_fast, sigmoid_fast, identity, gelu_fast
export tanh_deriv, relu_deriv, sigmoid_deriv, identity_deriv, tanh_fast_deriv, sigmoid_fast_deriv, gelu_deriv, gelu_fast_deriv


end
