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
addr = BoseFS{N,M}(5=>10) # Rimu
H = HubbardMom1D{Float32}(addr) # Rimu

# -------------------------------------------------------------------
# NN Model
# -------------------------------------------------------------------
batch = 1024
act = tanh
input = MomentumEncoding((M,), H; device=device)
model = Chain(input,
              Dense(nsites(input)*nchannels(input)=>64, act; batch=batch, device=device, layer_norm=true),
              Dense(64=>64, act; batch=batch, device=device, layer_norm=true),
              Dense(64=>64, act; batch=batch, device=device, layer_norm=true),
              Dense(64=>2, (identity, act); batch=batch, device=device); 
              device=device, batch=batch)

ansatz  = NeuralAnsatz(LogPsiSignTanh(), H, model, batch); # NN ansatz for wave-function

# --------------------------------------------------------------------------------------------------------------------------------------
# ALL TRAINING VARIABLES
# --------------------------------------------------------------------------------------------------------------------------------------
phases = [
    TrainingPhase(
        mode       = :energy,
        optimiser  = :minSR,
        vmc_sampler= :ctmc,
        stop       = StopBuffer(ΔE_thr=0.00005, var_thr=1),
        η          = 0.01f0,
        λ          = 0.001f0,
        skip       = [(1, 20)],
        η_decrease = [(10, 0.1)], #  (var_thr, factor), if var < thr → η *= factor
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 100,
    ),
]

# filename where learned weights (and inputs) will be stored AND if I want to load saved weights (and inputs)
SAVEFILE = "./weights/example_sign.txt"
SAVE_WEIGHTS = true
LOADFILE = ""
LOAD_WEIGHTS = false

# --------------------------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------------------------------------------------------------------------------------------

E, E_err, var, new_addrs = run_training_loop(H, ansatz, addr, phases; 
                                savefile=SAVEFILE, loadfile=LOADFILE, 
                                save=SAVE_WEIGHTS, load=LOAD_WEIGHTS)
