using Rimu
using QuantumNeuralStates

# Choose what GPU you are using
using Metal     # device = mtl
# using CUDA      # device = cu
using Random
Random.seed!(1234)

# -------------------------------------------------------------------
# Choosing what GPU wull be running (if none -> CPU run is chosen)
# -------------------------------------------------------------------
# you can manually choose on what device (CPU/GPU) the Neural Network would run
# but this generally picks the GPU based on what package is loaded
device = select_device()

# -------------------------------------------------------------------
# Quantum System
# -------------------------------------------------------------------
N = 255 # number of particles
M = 7*7 # number of sites
addr = BoseFS{missing, M}()                                  # phonon vacuum, variable phonon number
H    = FroehlichPolaron{Float32}(addr; D = 2, alpha = 2, l = 5, mode_cutoff=N)

# -------------------------------------------------------------------
# NN Model
# -------------------------------------------------------------------
batch  = 1024
act = relu
pad = Zeros()
input = OccupationEncoding((7,7), H; device=device)
model = Chain(input,
              # Dense(nsites(input)*nchannels(input)=>64, act; batch=batch, device=device), 
              # Dense(64=>64, act; batch=batch, device=device), 
              # Dense(64=>64, act; batch=batch, device=device), 
              # Dense(64=>64, act; batch=batch, device=device), 
              # Dense(64=>64, act; batch=batch, device=device), 
              # Dense(64=>1, identity; batch=batch, device=device); 
              Conv((3,3), nchannels(input)=>32, act; batch=batch, device=device, pad=pad),
              Conv((3,3), 32=>32, act; batch=batch, device=device, pad=pad),
              Conv((3,3), 32=>32, act; batch=batch, device=device, pad=pad),
              Conv((3,3), 32=>1, identity; batch=batch, device=device, pad=pad),
              Pool(:sum; device=device),
              Dense(1=>1, identity; batch=batch, device=device, init_scale=0.1);
              batch=batch, device=device)

ansatz = NeuralAnsatz(LogPsi(), H, model, batch; 
                      input_scale_func=identity, mean_field=false)#, jacobian_statistics=true, neuron_statistics=true)#; multiforward_buffer=batch*500) # NN ansatz for wave-functiwn

# -------------------------------------------------------------------
# ALL TRAINING VARIABLES
# -------------------------------------------------------------------
phases = [
    # TrainingPhase(
    #     mode       = :energy,
    #     optimiser  = :minSR,
    #     vmc_sampler= :ctmc,
    #     stop       = StopBuffer(var_thr=0.001),
    #     η          = 0.01f0,
    #     skip       = [(1, 20)],  # (epoch, B) → burnin B
    #     # η_decrease = [(1, 0.1)], #  (var_thr, factor), if var < thr → η *= factor
    #     # skip       = [(1, 20)],  # (epoch, B) → burnin B
    #     # truncation = 3*3,
    #     block_size = 10, 
    #     block_min  = 6, 
    #     patience   = 3,
    #     max_epochs = 200,
    # ),
    TrainingPhase(
        mode       = :energy,
        optimiser  = :minSR,
        vmc_sampler= :ctmc,
        stop       = StopBuffer(var_thr=1000),
        η          = 0.01f0,
        skip       = [(1, 20)],  # (epoch, B) → burnin B
        # skip       = [(1, 20)],  # (epoch, B) → burnin B
        # truncation = 3*3,
        d_loose = 0.3f0,
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 1000,
    ),
    TrainingPhase(
        mode       = :energy,
        optimiser  = :minSR,
        vmc_sampler= :ctmc,
        stop       = StopBuffer(var_thr=0.001),
        η          = 0.01f0,
        skip       = [(1, 20)],  # (epoch, B) → burnin B
        # η_decrease = [(1, 0.1)], #  (var_thr, factor), if var < thr → η *= factor
        # skip       = [(1, 20)],  # (epoch, B) → burnin B
        # truncation = 3*3,
        block_size = 10, 
        block_min  = 6, 
        patience   = 3,
        max_epochs = 200,
    ),
]

# filename where learned weights (and inputs) will be stored AND if I want to load saved weights (and inputs)
SAVEFILE = "./weights_poster/CNN_3Conv-32_relu_M49_N255_alpha2.0_3.txt"
SAVE_WEIGHTS = true
LOADFILE = "./weights_poster/CNN_3Conv-32_relu_M49_N255_alpha2.0_2.txt"
LOAD_WEIGHTS = true
MARKOVFILE = "MarkovChain.txt" # saving Markov Chain
SAVE_MARKOV = false
OCCFILE = "./fig/CNN_3Conv-32_relu_M49_N255_alpha2.0_3.png"
OCC_PLOT = true

# --------------------------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------------------------------------------------------------------------------------------

t = @timed run_training_loop(H, ansatz, addr, phases; 
                            savefile=SAVEFILE, loadfile=LOADFILE, 
                            save=SAVE_WEIGHTS, load=LOAD_WEIGHTS, 
                            occfile=OCCFILE, occplot=OCC_PLOT,
                            markovfile=MARKOVFILE, markov=SAVE_MARKOV);

println(t.time)
