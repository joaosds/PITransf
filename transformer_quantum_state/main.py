from model import TransformerModel
from Hamiltonian import FermionicModel
from optimizer import Optimizer
import numpy as np
import os
import torch 
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker
import seaborn
import sys
import time  # for tracking cpu time
from matplotlib.ticker import LogLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot import create_plots

torch.manual_seed(0)
torch.nn.TransformerEncoder.enable_nested_tensor = False
plt.style.use("myrcparams.mplstyle")

# Parse command line arguments
t0 = float(sys.argv[1])
N0 = int(sys.argv[2])
basislabel = str(sys.argv[3])
n_iter0 = int(sys.argv[4])
n_batch0 = int(sys.argv[5])
n_unique00 = int(sys.argv[6])
detuned = str(sys.argv[7])
sec_batch = int(sys.argv[8])
sampler = str(sys.argv[9])
cudan = str(sys.argv[10])
embedding_size = int(sys.argv[11])
n_head = int(sys.argv[12])
n_layers = int(sys.argv[13])

# Get results directory from environment variable or use default
results_dir = os.environ.get('RESULTS_DIR', 'results')
if len(sys.argv) > 14:
    results_dir = sys.argv[14]

torch.set_printoptions(precision=5)
rc("font", family="serif", serif="cm10")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# Device configuration
if torch.cuda.is_available():
    device_id = int(cudan) if cudan.isdigit() else 0
    device = torch.device(f"cuda:{device_id}")
    torch.set_default_device(f"cuda:{device_id}")
    torch.set_default_dtype(torch.float32)
    print(f"Using CUDA device: {device_id}")
else:
    print("Using CPU")
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    device = "cpu"

# Create necessary directories
os.makedirs(results_dir, exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Set up output directories
base_path = results_dir  # Use the results directory passed from main script
system_output_dir = os.path.join(base_path, f"n_{N0}")
plots_output_dir = os.path.join("plots", f"n_{N0}")

os.makedirs(system_output_dir, exist_ok=True)
os.makedirs(plots_output_dir, exist_ok=True)

print(f"Results will be saved to: {system_output_dir}")
print(f"Plots will be saved to: {plots_output_dir}")

# System configuration
system_size = [[N0]]
t = t0
U = "HF"
id = "HF-basis"
identifier = f"review_{t}_{basislabel}_{detuned}"

# Plotting flags
plot_it = True
plot_n = True
pcaplot = True
plot_it_hist = False
plot_it_hist_weigh = True
plot_sample_dynamics = False

# Initialize Hamiltonian
Hamiltonians = [FermionicModel(system_size, t)]
param_dim = Hamiltonians[0].param_dim

start = time.time()

# Model parameters
n_hid = embedding_size
dropout = 0
minibatch = 1000

# Create transformer model
model = TransformerModel(
    system_size,
    param_dim,
    embedding_size,
    n_head,
    n_hid,
    n_layers,
    dropout=dropout,
    minibatch=minibatch,
)
num_params = sum([param.numel() for param in model.parameters()])
model.to(device)

# Save configuration
folder = "results/"
name = type(Hamiltonians[0]).__name__
save_str = f"{name}_{embedding_size}_{n_head}_{n_layers}"

param_range = None  # use default param range
use_SR = False

# Define basis transformations
Uk_band_base = torch.tensor(
    [
        [1, -1j],
        [1, 1j],
    ],
    dtype=torch.complex64,
) / torch.sqrt(torch.tensor(2.0, dtype=torch.complex64))

Uk_chiral_base = torch.tensor(
    [
        [1, 0],
        [0, 1],
    ],
    dtype=torch.complex64,
)

# Create basis arrays for system size
Uk_chiral_list = []
Uk_band_list = []

for _ in range(system_size[0][0]):
    Uk_chiral_list.append(Uk_chiral_base)
    Uk_band_list.append(Uk_chiral_base)

Uk_band = torch.stack(Uk_band_list)
Uk_chiral = torch.stack(Uk_chiral_list)

# Load HF data
try:
    # Read HF energy
    ehflist = []
    with open("enhf.txt", "r") as file:
        for line in file:
            array = eval(line.strip())
            ehflist.append(array)

    # Read HF occupation
    hfocclist = []
    with open("ukocc.txt", "r") as file:
        for line in file:
            array = eval(line.strip())
            hfocclist.append(array)

    # Load numpy arrays
    Ehf0 = ehflist[0].real
    Ed = np.load("ed.npy").real[0]
    Uk_hf = torch.tensor(np.load("uk.npy"), dtype=torch.complex64)
    Nk_hf = torch.tensor(np.load("nk.npy"), dtype=torch.complex64).T

    print(f"✓ Loaded HF data: Ehf={Ehf0:.6f}, Ed={Ed:.6f}")

except FileNotFoundError as e:
    print(f"⚠ Warning: Could not load HF data: {e}")
    print("Using default values...")
    Ehf0 = 0.0
    Ed = 0.0
    Uk_hf = Uk_chiral
    Nk_hf = torch.zeros((N0, 3), dtype=torch.complex64)

# Select basis based on input
if basislabel == "chiral":
    basis = Uk_chiral
    Ehf = Ehf0
elif basislabel == "band":
    basis = Uk_band
    Ehf = Ehf0
elif basislabel == "hf":
    basis = Uk_hf
    Ehf = Ehf0
else:
    raise ValueError(f"Invalid basis: {basislabel}. Valid options: hf, chiral, band")

print(f"Using {basislabel} basis")
print(f"System parameters: Ehf={Ehf:.6f}, Ed={Ed:.6f}, N={N0}, n_unique={n_unique00}")

# Initialize optimizer
optim = Optimizer(model, Hamiltonians)

n_iter = n_iter0
n_batch = n_batch0
n_unique = n_unique00

# Initial guess for alpha parameter
alfa0 = torch.tensor(1.50)
alfa11 = (1.0 + torch.tanh(alfa0)) / 2
alfa21 = torch.abs(1.0 - alfa11**2)
alfa31 = alfa11 * torch.sqrt(alfa21)

print("Starting optimization...")

# Run optimization
(
    loss,
    Er,
    Ei,
    Ev,
    Er2,
    Ei2,
    Ev2,
    Er3,
    Ei3,
    Ev3,
    psi,
    sample_weight,
    samples,
    norm,
    grad,
    alfa1,
    alfa2,
    alfa3,
    vscore,
    vscore2,
    occupation,
    embed,
) = optim.train(
    N0,
    n_iter,
    t,
    basis,
    Ehf,
    alfa0,
    alfa11,
    alfa21,
    alfa31,
    sec_batch,
    device,
    detuned,
    sampler=sampler,
    batch=n_batch,
    max_unique=n_unique,
    param_range=param_range,
    fine_tuning=False,
    use_SR=use_SR,
    ensemble_id=int(use_SR),
)

end = time.time()

# Calculate runtime
loss = loss.detach()
elapsed_time_seconds = end - start
hours = int(elapsed_time_seconds / 3600)
minutes = int((elapsed_time_seconds % 3600) / 60)
seconds = round(float(elapsed_time_seconds % 60), 3)

print(f"✓ Optimization completed!")
print(f"Run time: {hours:03}:{minutes:02}:{seconds:06.3f}")

# Process results
momentum = Hamiltonians[0].k.clone()
n_batch2 = len(psi)
n = int(Hamiltonians[0].n)
t = Hamiltonians[0].t

# Calculate band occupations
band = torch.zeros(n, 3, dtype=torch.cfloat, device=device)

for j in range(3):
    for i in range(n):
        band[i, j] = occupation[i, :, j].sum()

for j in range(3):
    band[:, j] += alfa1[-1] * Nk_hf[:, j]

n, batch = samples.shape
print(f"Final alpha: {alfa1[-1]:.6f}")
print(f"Number of parameters: {num_params}")
print(f"Unique samples: {batch}")

# Utility functions for file output
def append_to_file(filename, *variables, precision=16):
    """Append variables to file with specified precision"""
    if not os.path.exists(filename):
        open(filename, "w").close()

    with open(filename, "a") as file:
        line = " ".join(
            format_value(get_last_element_or_value(var), precision) for var in variables
        )
        file.write(line + "\n")

def get_last_element_or_value(var):
    """Extract last element from list/tuple or return value"""
    if isinstance(var, (list, tuple)):
        return var[-1] if var else None
    return var

def format_value(value, precision):
    """Format value with appropriate precision"""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)

# Save results to file
output_filename = os.path.join(system_output_dir, f"{basislabel}_{N0}.txt")
try:
    append_to_file(
        output_filename,
        t,
        Er[-1],
        Ev[-1],
        Ehf,
        alfa1[-1],
        vscore[-1],
        vscore2[-1],
        norm[-1],
        Ed,
    )
    print(f"✓ Results saved to: {output_filename}")
except Exception as e:
    print(f"✗ Error saving results: {e}")

# Prepare data for plotting
band = band.real
Nk_hf = Nk_hf.real

print(f"Final energies: Ed={Ed:.6f}, Ehf={Ehf:.6f}, Er_final={Er[-1]:.6f}")

# Import and run plotting functions
if any([plot_it, plot_n, plot_it_hist_weigh, pcaplot]):
    try:
        # Prepare data dictionary for plotting
        data = {
            'system_output_dir': f"{base_path}/n_{N0}",  # Use your existing path
            'identifier': identifier,
            't': t,
            'U': U,
            'n': n,
            'N0': N0,
            'embedding_size': embedding_size,
            'n_head': n_head,
            'n_layers': n_layers,
            'hours': hours,
            'minutes': minutes,
            'seconds': seconds,
            'n_unique00': n_unique00,
            'batch': batch,
            'sampler': sampler,
            'num_params': num_params,
            'n_iter': n_iter,
            'Er': Er,
            'Er2': Er2,
            'Er3': Er3,
            'Ehf': Ehf,
            'Ed': Ed,
            'norm': norm,
            'vscore': vscore,
            'alfa1': alfa1,
            'alfa2': alfa2,
            'alfa3': alfa3,
            'grad': grad,
            'loss': loss,
            'momentum': momentum,
            'band': band,
            'Nk_hf': Nk_hf,
            'samples': samples,
            'sample_weight': sample_weight,
            'device': device,
            'n_batch2': n_batch2,
            'model': model,  # This is the key addition!
        }

        # Set plotting flags
        flags = {
            'plot_n': plot_n,
            'plot_it': plot_it,
            'plot_it_hist_weigh': plot_it_hist_weigh,
            'pcaplot': pcaplot,
        }

        # Generate all plots
        create_plots(data, flags)        
    except ImportError as e:
        print(f"⚠ Warning: Could not import plotting module: {e}")
        print("Skipping plot generation...")
    except Exception as e:
        print(f"✗ Error generating plots: {e}")

print("✓ Script completed successfully!")

