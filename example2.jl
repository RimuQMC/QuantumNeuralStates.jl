using Rimu
using QuantumNeuralStates

# Choose what GPU you are using
using Metal     # device = mtl
# using CUDA      # device = cu

# -------------------------------------------------------------------
# Choosing what GPU wull be running (if none -> CPU run is chosen)
# -------------------------------------------------------------------
# you can manually choose on what device (CPU/GPU) the Neural Network would run
# but this generally picks the GPU based on what package is loaded
device = select_device()

# -------------------------------------------------------------------
# Quantum System
# -------------------------------------------------------------------
N = 10 # number of particles
M = 10 # number of sites
addr = BoseFS{N,M}(5=>10); # Rimu
H = HubbardMom1D(addr); # Rimu

# -------------------------------------------------------------------
# NN Model
# -------------------------------------------------------------------
batch  = 1024
act = tanh_fast
input = MomentumEncoding((M,), H; device=device)
model = Chain(input,
              Dense(M*nchannels(input)=>200, act; batch=batch, device=device, layer_norm=true),
              Dense(200=>200, act; batch=batch, device=device, layer_norm=true),
              Dense(200=>200, act; batch=batch, device=device, layer_norm=true),
              Dense(200=>2, (identity, act); batch=batch, device=device); 
              device=device, batch=batch)

ansatz  = NeuralAnsatz(LogPsiSignTanh(), H, model, batch); # NN ansatz for wave-function

# --------------------------------------------------------------------------------------------------------------------------------------
# ALL TRAINING VARIABLES
# --------------------------------------------------------------------------------------------------------------------------------------
phases = [
    TrainingPhase(
        mode       = :energy,
        optimiser  = :minSR,
        vmc_sampler= :metropolis,
        stop       = StopBuffer(var_thr=40),
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

