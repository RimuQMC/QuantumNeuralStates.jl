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
N = 50 # number of particles
# M = 10 # number of sites
# addr = near_uniform(BoseFS{N,M}) # Rimu
# H = HubbardReal1D{Float32}(addr; u=0.1) # Rimu
M = 16 # number of sites
addr = BoseFS{missing, M}()                                  # phonon vacuum, variable phonon number
H    = FroehlichPolaron{Float32}(addr; D = 2, alpha = 1, l = 6)

# -------------------------------------------------------------------
# NN Model
# -------------------------------------------------------------------
batch  = 1024
act = relu
pad = Periodic()
input = OccupationEncoding((4,4), H; device=device)
model = Chain(input,
              # Dense(nsites(input)*nchannels(input)=>32, tanh; batch=batch, device=device, layer_norm=true), 
              # Dense(32=>32, tanh; batch=batch, device=device, layer_norm=true), 
              # Dense(32=>32, tanh; batch=batch, device=device, layer_norm=true), 
              Conv((3,3), nchannels(input)=>32, act; batch=batch, device=device, pad=pad),
              Conv((3,3), 32=>32, act; batch=batch, device=device, pad=pad),
              Conv((3,3), 32=>32, act; batch=batch, device=device, pad=pad),
              Pool(:mean; device=device),
              Dense(32=>32, tanh; batch=batch, device=device, layer_norm=true), 
              Dense(32=>1, identity; batch=batch, device=device); 
              batch=batch, device=device)

ansatz = NeuralAnsatz(LogPsi(), H, model, batch)#; multiforward_buffer=batch*500) # NN ansatz for wave-functiwn

# -------------------------------------------------------------------
# ALL TRAINING VARIABLES
# -------------------------------------------------------------------
phases = [
    TrainingPhase(
        mode       = :energy,
        optimiser  = :adam,
        vmc_sampler= :ctmc,
        stop       = StopBuffer(var_thr=1000),
        η          = 0.001f0,
        skip       = [(1, 20), (300, 200)],  # (epoch, B) → burnin B
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 30,
    ),
    # TrainingPhase(
    #     mode       = :energy,
    #     optimiser  = :minSR,
    #     vmc_sampler= :ctmc,
    #     stop       = StopBuffer(ΔE_thr=0.00005, var_thr=1),
    #     η          = 0.001f0,
    #     λ          = 0.001f0,
    #     skip       = [(1, 20)],
    #     η_decrease = [(1, 0.1)], #  (var_thr, factor), if var < thr → η *= factor
    #     block_size = 10, 
    #     block_min  = 6, 
    #     patience   = 3,
    #     max_epochs = 30,
    # ),
]

# filename where learned weights (and inputs) will be stored AND if I want to load saved weights (and inputs)
SAVEFILE = "./weights/example.txt"
SAVE_WEIGHTS = false
LOADFILE = ""
LOAD_WEIGHTS = false
MARKOVFILE = "MarkovChain.txt" # saving Markov Chain
SAVE_MARKOV = false

# --------------------------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------------------------------------------------------------------------------------------

t = @timed run_training_loop(H, ansatz, addr, phases; 
                                                                            savefile=SAVEFILE, loadfile=LOADFILE, 
                                                                            save=SAVE_WEIGHTS, load=LOAD_WEIGHTS, 
                                                                            markovfile=MARKOVFILE, markov=SAVE_MARKOV);

println(t.time)
