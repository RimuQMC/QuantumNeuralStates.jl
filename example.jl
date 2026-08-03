# using LinearAlgebra
using Rimu
using QuantumNeuralStates
# using Printf

# Choose what GPU you are using
using Metal     # device = mtl
# using CUDA      # device = cu

# include("./QNS.jl")
# using .QNS

# -------------------------------------------------------------------
# Choosing what GPU wull be running (if none -> CPU run is chosen)
# -------------------------------------------------------------------
# you can manually choose on what device (CPU/GPU) the Neural Network would run
# but this generally picks the GPU way if kept here
device = identity
try
    CUDA.functional()
    global device = CUDA.cu
    @info "CUDA (cu) was loaded for GPU computations"
catch
end
try
    Metal.functional()
    global device = Metal.mtl
    @info "Metal (mtl) was loaded for GPU computations"
catch
end

# -------------------------------------------------------------------
# Quantum System
# -------------------------------------------------------------------
N = 50 # number of particles
M = 10 # number of sites

# -------------------------------------------------------------------
# NN Model
# -------------------------------------------------------------------
batch  = 1024
# Fully connected Neural Network with 3 hidden layers and in each layer 100 neurons
model  = build_model("FCNN", [M, 100, 100, 100, 1], tanh; batch=batch, device=device, Layer_Norm=false);

# --------------------------------------------------------------------------------------------------------------------------------------
# ALL VARIABLES
# --------------------------------------------------------------------------------------------------------------------------------------
phases = [
    TrainingPhase(
        mode       = :energy,
        optimiser  = :adam,
        vmc_sampler= :ctmc,
        stop       = StopBuffer(var_thr=1000),
        η          = 0.001f0,
        skip       = [(1, 100), (300, 200)],  # (epoch, B) → burnin B
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 500,
    ),
    TrainingPhase(
        mode       = :energy,
        optimiser  = :minSR,
        vmc_sampler= :metropolis,
        stop       = StopBuffer(ΔE_thr=0.00005, var_thr=1),
        η          = 0.001f0,
        λ          = 0.001f0,
        skip       = [(1, 300)],
        η_decrease = [(1, 0.1)], #  (var_thr, factor), if var < thr → η *= factor
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 1000,
    ),
]

# RIMU VARIABLES
addr = near_uniform(BoseFS{N,M});
H = HubbardReal1D(addr; u=0.1);
ansatz  = NeuralAnsatz(H, model, batch); # NN ansatz for wave-function


# filename where learned weights (and inputs) will be stored AND if I want to load saved weights (and inputs)
SAVEFILE     = "./weights/example.txt"
SAVE_WEIGHTS = true
LOADFILE     = ""
LOAD_WEIGHTS = false
MARKOVFILE   = "MarkovChain.txt" # saving Markov Chain
SAVE_MARKOV  = false

# --------------------------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------------------------------------------------------------------------------------------

block_E_history, block_E_err_history, block_var_history, new_addrs = run_training_loop(H, ansatz, addr, phases; 
                                                                            savefile=SAVEFILE, loadfile=LOADFILE, 
                                                                            save=SAVE_WEIGHTS, load=LOAD_WEIGHTS, 
                                                                            markovfile=MARKOVFILE, markov=SAVE_MARKOV);

