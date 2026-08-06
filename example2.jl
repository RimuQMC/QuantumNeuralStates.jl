using Rimu
using QuantumNeuralStates

# Choose what GPU you are using
# using Metal     # device = mtl
using CUDA      # device = cu

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
N = 10 # number of particles
M = 10 # number of sites

# -------------------------------------------------------------------
# NN Model
# -------------------------------------------------------------------
batch  = 1024
# Fully connected Neural Network with 3 hidden layers and in each layer 100 neurons
# model  = build_model("FCNN", [M, 100, 100, 100, 2], tanh_fast; batch=batch, device=device, Layer_Norm=false);
act = tanh_fast
model = Chain(Dense(M, 200, act; batch=batch, device=device, Layer_Norm=true),
              Dense(200, 200, act; batch=batch, device=device, Layer_Norm=true),
              Dense(200, 200, act; batch=batch, device=device, Layer_Norm=true),
              Dense(200, 2, (identity, act); batch=batch, device=device); 
              device=device, batch=batch)

# --------------------------------------------------------------------------------------------------------------------------------------
# ALL VARIABLES
# --------------------------------------------------------------------------------------------------------------------------------------
phases = [
    TrainingPhase(
        mode       = :energy,
        optimiser  = :minSR,
        vmc_sampler= :ctmc,
        stop       = StopBuffer(ΔE_thr=0.00005, var_thr=1),
        η          = 0.001f0,
        λ          = 0.001f0,
        skip       = [(1, 20)],
        η_decrease = [(1, 0.1)], #  (var_thr, factor), if var < thr → η *= factor
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 1000,
    ),
]

# RIMU VARIABLES
addr = BoseFS{N,M}(5=>10);
H = HubbardMom1D(addr);
ansatz  = NeuralAnsatz(LogPsiSignTanh(), H, model, batch); # NN ansatz for wave-function


# filename where learned weights (and inputs) will be stored AND if I want to load saved weights (and inputs)
SAVEFILE     = "./weights/example_sign.txt"
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

