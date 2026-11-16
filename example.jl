using Rimu
using QuantumNeuralStates

# Choose what GPU backend you are using
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
N = 10 # phonon cutoff per site
M = 11 # number of sites
addr = BoseFS{missing, M}() # Rimu
H = FroehlichPolaron{Float32}(addr; D = 1, alpha = 2, l = 5, mode_cutoff=N) # Rimu

# -------------------------------------------------------------------
# NN Model
# -------------------------------------------------------------------
batch = 1024
act = relu
pad = Zeros()
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

ansatz = NeuralAnsatz(LogPsi(), H, model, batch) # NN ansatz for wave-function

# -------------------------------------------------------------------
# ALL TRAINING VARIABLES
# -------------------------------------------------------------------
phases = [
    TrainingPhase(
        mode       = :energy,
        optimiser  = :adam,
        vmc_sampler= :metropolis,
        stop       = StopBuffer(var_thr=1000),
        η          = 0.001f0,
        skip       = [(1, 50)],  # (epoch, B) → burnin B
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
        skip       = [(1, 20)],  # (epoch, B) → burnin B
        η_decrease = [(1, 0.1)], # (var_thr, factor), if var < thr → η *= factor
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 300,
    ),
]

# filename where learned weights (and inputs) will be stored AND if I want to load saved weights (and inputs)
SAVEFILE = "./weights/example.txt"
SAVE_WEIGHTS = true
LOADFILE = ""
LOAD_WEIGHTS = false
MARKOVFILE = "MarkovChain.txt" # saving Markov Chain
SAVE_MARKOV = false
OCCFILE = "./fig/example_walker_distro.pdf" # saving plot of final walker distribution
OCC_PLOT = true

# --------------------------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------------------------------------------------------------------------------------------

E, E_err, var, new_addrs = run_training_loop(H, ansatz, addr, phases; 
                                savefile=SAVEFILE, loadfile=LOADFILE, 
                                save=SAVE_WEIGHTS, load=LOAD_WEIGHTS, 
                                occfile=OCCFILE, occplot=OCC_PLOT,
                                markovfile=MARKOVFILE, markov=SAVE_MARKOV)
