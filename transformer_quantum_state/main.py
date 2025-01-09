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

from model_utils import treesamplerumap
import umap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import KernelPCA


torch.manual_seed(0)
# import scienceplots
torch.nn.TransformerEncoder.enable_nested_tensor = False
plt.style.use("myrcparams.mplstyle")

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
torch.set_printoptions(precision=5)
rc("font", family="serif", serif="cm10")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

if torch.cuda.is_available():
    device = torch.device("cuda:3")  ## specify the GPU id's, GPU id's start from 0.
    torch.set_default_device("cuda:3")
    torch.set_default_dtype(torch.float32)
else:
    print("CPU")
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    device = "cpu"


try:
    os.mkdir("results/")
except FileExistsError:
    pass

base_path = "/media/local-scratch/SCRATCH_NOBACKUP/sobral"
folder_path = f"{base_path}/n_{N0}"
os.makedirs(folder_path, exist_ok=True)
# ----------------------------------------------------------------------------------------
# System sizes from 10 to 41.
# system_sizes = np.arange(6, 7, 2).reshape(-1, 1)  # Same as length in perle

# Read from script in bash

n_unique0 = n_unique00


folder_path = f"plots/n_{N0}"
os.makedirs(folder_path, exist_ok=True)
# n_unique0 = math.comb(N0,2) + math.comb(N0,1) # Truncation of Hilbert Space


system_size = [[N0]]

t = t0
U = "HF"
id = "HF-basis"
identifier = f"t2_{t}_{basislabel}_{detuned}"
plot_it = True
plot_n = True
plot_it_hist = False
plot_it_hist_weigh = True
plot_sample_dynamics = False
Hamiltonians = [FermionicModel(system_size, t)]

# ----------------------------------------------------------------------------------------
# (ii). Define the transformer model and parameters
# If I use everything equal to 2 we converge to the wrong energy! Investigate that
param_dim = Hamiltonians[0].param_dim  # print(param_dim)


start = time.time()


# embedding_size = 100
# embedding_size = 14
# n_head = 2
n_hid = embedding_size
# n_layers = 1
dropout = 0
minibatch = 1000


# embedding_size = 128
# n_head = 16
# n_hid = embedding_size
# n_layers = 16
# minibatch = 6400
# dropout = 0
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
# ----------------------------------------------------------------------------------------
# print("Number of parameters: ", num_params)
folder = "results/"
name = type(Hamiltonians[0]).__name__
save_str = f"{name}_{embedding_size}_{n_head}_{n_layers}"

param_range = None  # use default param range
use_SR = False

# Defining chiral and band basis according to system size
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

Uk_chiral_list = []
Uk_band_list = []

# Loop to create N identical tensors
for _ in range(system_size[0][0]):
    Uk_chiral_list.append(Uk_chiral_base)
    Uk_band_list.append(Uk_chiral_base)

# Stack the tensors
Uk_band = torch.stack(Uk_band_list)
Uk_chiral = torch.stack(Uk_chiral_list)

# print(Uk_band.shape)
# print(Uk_chiral.shape)


# Defining the HF-basis
filebasisen = "enhf.txt"
filebasis = "uk.txt"
filebasisocc = "ukocc.txt"

# Read the file
ehflist = []
edlist = []
hfbasislist = []
hfocclist = []
with open("enhf.txt", "r") as file:
    for line in file:
        # Strip whitespace and convert the line to a list of integers
        array = eval(line.strip())
        ehflist.append(array)


with open("ukocc.txt", "r") as file:
    for line in file:
        # Strip whitespace and convert the line to a list of integers
        array = eval(line.strip())
        hfocclist.append(array)

# Defining the HF energy
Ehf0 = ehflist[0].real
# Ehf = 0
Ed = np.load("ed.npy").real[0]
Uk_hf = torch.tensor(np.load("uk.npy"), dtype=torch.complex64)
Nk_hf = torch.tensor(np.load("nk.npy"), dtype=torch.complex64).T


# Pytorch version
# Ed = torch.load("ed.pt", map_location=device).real[0]
# Uk_hf = torch.load("uk.pt", map_location=device).to(dtype=torch.complex128)
# Nk_hf = torch.load("nk.pt", map_location=device).to(dtype=torch.complex128).T

if basislabel == "chiral":
    basis = Uk_chiral
    Ehf = Ehf0
elif basislabel == "band":
    basis = Uk_band
    Ehf = Ehf0
elif basislabel == "hf":
    basis = Uk_hf
    Ehf = Ehf0

print(f"Ehf={Ehf}, Ed={Ed}, N={N0}, nuni={n_unique0}")
# ----------------------------------------------------------------------------------------
# (iii) Optimization of the W parameters in the transformer.
optim = Optimizer(model, Hamiltonians)

n_iter = n_iter0
n_batch = n_batch0
# n_unique = 2 ** int(Hamiltonians[0].n)
n_unique = n_unique0


# Come only from V(q) term that's why it's a constant


# First guess for alpha

alfa0 = torch.tensor(1.50)
alfa11 = (1.0 + torch.tanh(alfa0)) / 2
alfa21 = torch.abs(1.0 - alfa11**2)
alfa31 = alfa11 * torch.sqrt(alfa21)

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
    sampler=sampler,
    batch=n_batch,
    max_unique=n_unique,
    param_range=param_range,
    fine_tuning=False,
    use_SR=use_SR,
    ensemble_id=int(use_SR),
)

end = time.time()

loss = loss.detach()

elapsed_time_seconds = end - start
# Convert seconds to hours, minutes, and seconds
hours = int(elapsed_time_seconds / 3600)
minutes = int((elapsed_time_seconds % 3600) / 60)
seconds = round(float(elapsed_time_seconds % 60), 3)

# Print the runtime in the format of hours:minutes:seconds
print(f"Run time: {hours:03}:{minutes:02}:{seconds:03}")
# # ----------------------------------------------------------------------------------------

momentum = Hamiltonians[0].k.clone()
n_batch2 = len(psi)
n = int(Hamiltonians[0].n)
t = Hamiltonians[0].t

band = torch.zeros(n, 3, dtype=torch.cfloat, device=device)

for j in range(3):
    for i in range(n):
        band[i, j] = occupation[i, :, j].sum()

for j in range(3):
    band[:, j] += alfa1[-1] * Nk_hf[:, j]


n, batch = samples.shape
print(alfa1[-1])
print("Number of parameters: ", num_params)
print(f"nuni={batch}")


def append_to_file(filename, *variables, precision=16):
    # Check if the file exists, if not create it
    if not os.path.exists(filename):
        open(filename, "w").close()

    # Open the file in append mode
    with open(filename, "a") as file:
        # Convert all variables to strings, handling different types
        line = " ".join(
            format_value(get_last_element_or_value(var), precision) for var in variables
        )
        # Append the line to the file, adding a newline character
        file.write(line + "\n")


def get_last_element_or_value(var):
    if isinstance(var, (list, tuple)):
        return var[-1] if var else None
    return var


def format_value(value, precision):
    if isinstance(value, torch.Tensor):
        # Extract the float value from the tensor
        value = value.item()
    if isinstance(value, float):
        # Format float with specified precision
        return f"{value:.{precision}f}"
    return str(value)


# Example usage

filename = f"{basislabel}_{N0}.txt"
try:
    append_to_file(
        filename,
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
    print("Data appended successfully.")
except Exception as e:
    print(f"An error occurred: {e}")

# Example usage

band = band.real
Nk_hf = Nk_hf.real
print(Nk_hf.shape)

print(Ed, Ehf, Ehf0, Er[-1])
if plot_n:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.axvline(x=torch.pi / 2)
    ax1.axvline(x=-torch.pi / 2)

    momentum = momentum.cpu()
    band = band.cpu()
    Nk_hf = Nk_hf.cpu()

    ax1.scatter(
        momentum[:, 1],
        band[:, 0].real,
        marker="d",
        color="orange",
        label=r"N$_{x} (TQS)$",
    )

    ax1.plot(
        momentum[:, 1],
        Nk_hf[:, 0],
        color="orange",
        linestyle="--",
        label=r"N$_{x} (HF)$",
    )

    ax1.scatter(
        momentum[:, 1], band[:, 1].real, marker="d", color="black", label=r"N$_{y}$"
    )
    ax1.plot(momentum[:, 1], Nk_hf[:, 1], color="black", linestyle="--")
    ax1.scatter(
        momentum[:, 1], band[:, 2].real, marker="d", color="red", label=r"N$_{z}$"
    )
    ax1.plot(momentum[:, 1], Nk_hf[:, 2], linestyle="--", color="red")

    ax1.legend()
    ax1.set_xlabel(r"Momentum $|\mathbf{k}|$")
    ax1.set_ylabel(r"$|\mathbf{N}|$")

    difference = torch.abs(torch.abs(band[:, :].real) - torch.abs(Nk_hf[:, :].real))

    ax2.minorticks_on()

    ax2.plot(
        momentum[:, 1],
        difference[:, 0],
        marker="d",
        color="orange",
        label=r"$\Delta$N$_{x}$",
    )
    ax2.plot(
        momentum[:, 1],
        difference[:, 1],
        marker="d",
        color="black",
        label=r"$\Delta$N$_{y}$",
    )
    ax2.plot(
        momentum[:, 1],
        difference[:, 2],
        marker="d",
        color="red",
        label=r"$\Delta$N$_{z}$",
    )

    ax2.legend()
    ax2.set_xlabel(r"Momentum $|\mathbf{k}|$")
    ax2.set_ylabel(r"$|\Delta \mathbf{N}|$")

    plt.tight_layout()
    plt.savefig(f"{base_path}/n_{n}/{identifier}_occupation_t_{t}.pdf")

if plot_it:
    iter = torch.zeros((n_iter + 1))
    for i in range(n_iter + 1):
        iter[i] = i
    plt.clf()
    plt.cla()

    Er = Er.cpu()
    fig, ax = plt.subplots()

    # plt.gca().set_ylim(bottom=0)
    if N0 < 11:
        param = 1 - abs(Ehf / Ed)
        print(param)
        plt.axhline(y=Ed, color="blue", label="ED")
        plt.axhline(y=Ehf, color="red", label=r"$E_{\text{HF}}$")
        y_range = abs(Er[-1]) * 0.005
        # Ed_adjusted = Ed + 0.005 * Ed  # Ed + 5% of Ed
        # plt.ylim(Ed_adjusted, Ehf + y_range)
        # plt.ylim(Ehf + y_range / 2, Ehf + y_range)
    # plt.ylim(Er[-1] - y_range, Er[-1] + y_range)
    elif N0 >= 11:
        plt.axhline(y=Ehf, color="red", label=r"$E_{\text{HF}}$")
        y_range = abs(Ehf) * 0.01
    # plt.ylim(Er[-1] - y_range, Er[-1] + y_range)
    # plt.ylim(Ehf - y_range, Ehf + y_range / 2)

    text_content = (
        f"N = {n}\n"
        f"t = {t}\n"
        f"U = {U}\n"
        f"$d_e$ = {embedding_size}, \n"
        f"$n_{{head}}$ = {n_head}, $n_{{layers}}$ = {n_layers} \n"
        f"RTime: {hours:02}:{minutes:02}:{seconds:02} \n"
        f"$N_{{ps}}$ = {n_unique00}, $N_s$ = {batch}  ({sampler})\n"
        f"$N_{{TQS}}$ = {num_params}"
    )
    # Add a text box with variables
    ax.text(
        0.5,
        0.6,
        text_content,
        bbox=dict(facecolor="white", alpha=0.5),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )

    Er = Er.cpu()
    Ev = Ev.cpu()
    Er2 = Er2.cpu()
    Ev2 = Ev2.cpu()
    Er3 = Er3.cpu()
    Ev3 = Ev3.cpu()
    iter = iter.cpu()
    ax.plot(
        iter[:],
        Er[:],
        color="black",
        label=r"$\text{E}_{GS}$",
    )
    ax.set_ylim(-65, -64)

    # First inset (original)
    axins1 = inset_axes(
        ax,
        width="40%",
        height="30%",
        loc="lower right",
        bbox_to_anchor=(-0.1, 0.2, 1, 1),
        bbox_transform=ax.transAxes,
    )

    axins1.plot(
        iter[:], Er3[:], color="green", label=r"$\text{E}_{\Lambda}^{(1)}$"
    )
    axins1.plot(
        iter[:], Er2[:], color="darkgray", label=r"$\text{E}_{s,s^{\prime}}$"
    )
    axins1.set_xlabel(r"Iterations $i$")
    axins1.set_ylabel(r"$E/N$")

    # Second inset (new)
    axins2 = inset_axes(
        ax,
        width="40%",
        height="30%",
        loc="upper right",
        bbox_to_anchor=(-0.1, 0.6, 1, 1),
        bbox_transform=ax.transAxes,
    )

    axins2.plot(
        iter[:], Er3[:], color="green", label=r"$\text{E}_{\Lambda}^{(1)}$"
    )
    axins2.plot(
        iter[:], Er2[:], color="darkgray", label=r"$\text{E}_{s,s^{\prime}}$"
    )
    axins2.set_ylim(-6.5, -5)  # Set the specified y-range
    axins2.set_xlabel(r"Iterations $i$")
    axins2.set_ylabel(r"$E/N$")
    # axins.set_ylim(-0.005, 0.005)

    # Finalize the main plot
    ax.legend()
    ax.set_xlabel(r"Iterations $i$")
    ax.set_ylabel(r"E/N")
    plt.savefig(f"{base_path}/n_{n}/{identifier}_energyconv{t}.pdf")

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    norm = norm.cpu()

    plt.plot(
        iter[100:-1],
        norm[100:-1],
        linestyle="--",
        color="black",
    )

    plt.xlabel(r"Iterations $i$")
    plt.ylabel(r"Normalization")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{base_path}/n_{n}/{identifier}_normalization_{t}.pdf")

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    # plt.scatter(true[:], predict[:, 0])
    # plt.title(f"N = {n}, t = {t}, U = {U}, g = {g}, d_emb={embedding_size}")
    plt.yscale("log")
    vscore = vscore.cpu()

    # plt.ylim(vscore[-1] + vscore[-1]*10**(-2), vscore[-1] - vscore[-1]*10**2)
    plt.plot(
        iter[:],
        vscore[:],
        linestyle="--",
        color="black",
    )

    plt.xlabel(r"Iterations $i$")
    plt.ylabel(r"V-score")
    ax = plt.gca()
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$")
    )

    plt.tight_layout()
    plt.savefig(f"{base_path}/n_{n}/{identifier}_vscore_{t}.pdf")

    # alfa plots
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    alfa1 = alfa1.cpu()
    alfa2 = alfa2.cpu()
    alfa3 = alfa3.cpu()

    ax.plot(
        iter[100:-1], alfa1[100:-1], linestyle="-", color="black", label=r"$\alpha_{1}$"
    )
    ax.set_xlabel(r"Iterations $i$")
    ax.set_ylabel(r"Reweighting factors $\alpha$")

    axins1 = inset_axes(
        ax,
        width="35%",
        height="25%",
        loc="lower right",
        bbox_to_anchor=(-0.1, 0.2, 1, 1),
        bbox_transform=ax.transAxes,
    )

    axins1.plot(
        iter[100:-1],
        alfa2[100:-1],
        linestyle="-",
        color="darkgray",
        label=r"$\alpha_{s,s^{\prime}}$",
    )

    axins1.plot(
        iter[100:-1],
        alfa3[100:-1],
        linestyle="-",
        color="green",
        label=r"$\alpha_{\Lambda}^{(1)}$",
    )

    # Adjust inset plots
    for axins in [axins1]:
        axins.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        axins.tick_params(axis="both", which="major")

    # Add a single legend for all plots
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axins1.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.savefig(f"{base_path}/n_{n}/{identifier}_alpha_{t}.pdf")

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    grad = grad.cpu()
    plt.plot(
        iter[1:-1],
        -grad[1:-1],
        color="red",
        label=r"-$\nabla E_{\alpha}$",
    )
    ax.set_ylim(grad[-1] + grad[-1] * 10, grad[-1] - grad[-1] * 10)

    plt.legend()
    plt.xlabel(r"Iterations $i$")
    plt.ylabel(r"$-\nabla E_{\alpha}$")
    plt.tight_layout()
    plt.savefig(f"{base_path}/n_{n}/{identifier}_nablae_{t}.pdf")

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    loss = loss.cpu()
    plt.plot(
        iter[1:-1],
        loss[1:-1],
        color="red",
    )
    ax.set_ylim(loss[-1] - loss[-1] * 5, loss[-1] + loss[-1] * 5)

    plt.xlabel(r"Iterations $i$")
    plt.ylabel(r"Loss")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{base_path}/n_{n}/{identifier}_loss_{t}.pdf")

if plot_it_hist_weigh:
    print(samples.shape, "aqiu carai")

    def create_dual_histogram(
        samples, sample_weight, n, t, U, embedding_size, device, n_batch2, identifier
    ):
        plt.clf()
        plt.cla()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Define colormap for both subplots
        cmap = plt.colormaps["coolwarm"].resampled(n)

        # First subplot (original histogram)
        labels = torch.arange(len(sample_weight))
        colord = torch.zeros(samples.shape[1], device=device)
        for i in range(n_batch2 - 1):
            colord[i] = torch.sum(samples[:, i])
        sample_weight = sample_weight.cpu()
        colord = colord.cpu()
        labels = labels.cpu()
        bars = ax1.bar(
            labels[0:-1],
            sample_weight[0:-1] * batch,
            color=[cmap(i / n) for i in range(len(colord))],
        )
        norm = plt.Normalize(vmin=0, vmax=n - 1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label(r" Electrons on band 1 ($N_{e}^{1}$)")
        cbar.ax.tick_params(labelsize=12)

        def assign_colors(bars, colord, cmap, N):
            colord = np.clip(colord.numpy(), 0, N - 1)
            color_map = {i: cmap(i / (N - 1)) for i in range(N)}
            for bar, color_index in zip(bars, colord):
                bar.set_facecolor(color_map[int(color_index)])

        assign_colors(bars, colord, cmap, n)
        ax1.set_xlabel(r"Sample index $\mathbf{s}$")
        ax1.set_ylabel(r"Relative Occurrence $r_\mathbf{s}$")
        ax1.tick_params(axis="both", which="major", labelsize=12)

        # Second subplot (new occupancy histogram)
        ne = torch.sum(samples, dim=0)

        unique_ne, inverse_indices = torch.unique(ne, return_inverse=True)
        inverse_indices = inverse_indices.cpu()
        unique_ne = unique_ne.cpu()
        nefinal = torch.zeros_like(unique_ne, dtype=torch.float)

        if sample_weight.dim() == 1:
            weights_sum = sample_weight
        elif sample_weight.dim() == 2:
            weights_sum = sample_weight.sum(dim=0)
        else:
            raise ValueError(
                f"Unexpected sample_weight dimension: {sample_weight.dim()}"
            )

        nefinal.index_add_(0, inverse_indices, weights_sum)

        x = unique_ne.cpu().numpy()
        y = nefinal.cpu().numpy()

        # threshold = 0.1 * y.max()
        # y_plot = np.where(y < threshold, y * 10, y)
        # small_bar_mask = y < threshold
        y_plot = y
        small_bar_mask = y

        norm2 = plt.Normalize(vmin=0, vmax=n - 1)
        bars2 = ax2.bar(
            x,
            y,
            color=[cmap(norm2(i)) for i in x],
            edgecolor="black",
        )

        for i, (height, small) in enumerate(zip(y_plot, small_bar_mask)):
            label = rf"{y[i]/10:.2e}" if small else f"{y[i]:.2f}"
            ax2.text(
                x[i],
                height + 0.01,
                label,
                ha="center",
                va="bottom",
                rotation=0,
            )

        ax2.set_xlabel(r"Number of electrons on band 1 ($N_e^{1}$)")
        ax2.set_ylabel("Total Weight")
        ax2.set_xticks(range(int(x.min()), int(x.max()) + 1))

        sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax2)
        cbar2.set_label(r"Number of electrons on band 1 ($N_e^{1}$)")

        plt.tight_layout()
        plt.savefig(
            f"{base_path}/n_{n}/{identifier}_dual_hist_{t}.pdf",
            dpi=300,
            bbox_inches="tight",
        )

    # Example usage:
    # if sampler == "tree":
    create_dual_histogram(
        samples,
        sample_weight,
        n,
        t,
        U,
        embedding_size,
        device,
        n_batch2,
        identifier,
    )
    # elif sampler == "normal":
    #     a = 1
    # write second function
# If you want to use the trained model just call it with something like
# model.load_state_dict(torch.load(f'{folder}ckpt_100000_{save_str}_0.ckpt'))

plot_attention = False
if plot_attention:
    # tgt_sent = trans.split()

    # h = torch.tensor([])

    def draw(data, x, y, ax):
        seaborn.heatmap(
            data,
            xticklabels=x,
            square=True,
            yticklabels=y,
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            ax=ax,
        )

    for layer in range(1, 6, 2):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Encoder Layer", layer + 1)
        for h in range(4):
            draw(
                model.encoder.layers[layer].self_attn.attn[0, h].data,
                sent,
                sent if h == 0 else [],
                ax=axs[h],
            )
        plt.savefig(f"{base_path}/n_{n}/{identifier}_attentionmatrix_{t}.pdf")


pcaplot = False
if pcaplot == True:
    ns = torch.sum(samples, dim=0, dtype=torch.int8)
    idxns = torch.where(ns == n)[0]

    print(model)
    samples, sample_weight = treesamplerumap(model)
    n, batch = samples.shape
    _, embed = model.forward(samples)
    print(samples.shape, embed.shape)
    print(np.shape(embed), "dimensions")

    ns = torch.sum(samples, dim=0, dtype=torch.int8)
    idxns = torch.where(ns == n)[0]

    if len(idxns) > 0:  # If there are columns to remove
        embed = embed.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        # Convert idxns to numpy array if it's not already
        idxns_np = idxns.detach().cpu().numpy() if isinstance(idxns, torch.Tensor) else idxns

        # Create a boolean mask for columns to keep
        mask = np.ones(samples.shape[1], dtype=bool)
        mask[idxns_np] = False

        # Use the mask to keep only the columns we want
        newsamples = torch.tensor(samples[:, mask])
        newembed = torch.tensor(embed[:, mask.T])
        print(newembed.shape, "aaaaa")
        newembed0 = newembed.permute(1, 0, 2)  # Shape: (batch, n, d_e)
        print(newembed0.shape, "aaaaa")
        newembed0 = newembed0.reshape(batch - 1, -1)
        newembed1 = newembed.sum(dim=0)
    else:
        newsamples = samples
        newembed = embed
        newembed0 = newembed.permute(1, 0, 2)  # Shape: (batch, n, d_e)
        newembed0 = newembed0.reshape(batch, -1)
        newembed1 = newembed.sum(dim=0)
    print(
        newsamples.shape, newembed.shape, newembed1.shape, newembed0.shape, "aaaaaaaaa"
    )
    with open("samples.txt", "w") as file:
        for i in range(batch-1):
            values = [
                f"{v}" for v in newsamples[:, i].tolist()
            ]  # Adjust .4f to control decimal places
            file.write(f"{i}: {values}\n")

    def plot_embeddings_with_color(
        samples,
        embed,
        do_pca=True,
        spectrum_pca=True,
        do_umap=True,
        do_tsne=True,
        do_kpca=True,
        label_points=False,
        name="aaa",
    ):
        plt.close("all")
        plt.clf()
        plt.cla()

        # Determine active plots
        top_plots = sum([do_pca, do_tsne])

        # Create figure with subplot layout and space for colorbar
        fig = plt.figure(figsize=(12, 8))
        # Create grid with space for colorbar (width_ratios adds space on the right)
        gs = plt.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.1])

        # Create axes based on which plots are active
        if top_plots == 2:
            ax1 = fig.add_subplot(gs[0, 0])  # PCA
            ax2 = fig.add_subplot(gs[0, 1])  # t-SNE
            ax3 = fig.add_subplot(gs[1, 0])  # UMAP on bottom left
            ax4 = fig.add_subplot(gs[1, 1])  # Kernel PCA on bottom right
            # Create a special axes for colorbar
            cax = fig.add_subplot(gs[:, 2])  # Colorbar axes spans both rows
            axs = [ax1, ax2, ax3, ax4]
        elif top_plots == 1:
            ax1 = fig.add_subplot(gs[0, 0:2])  # Single top plot takes full width
            ax3 = fig.add_subplot(gs[1, 0])  # UMAP on bottom left
            ax4 = fig.add_subplot(gs[1, 1])  # Kernel PCA on bottom right
            cax = fig.add_subplot(gs[:, 2])  # Colorbar axes spans both rows
            axs = [ax1, ax3, ax4]
        else:
            ax3 = fig.add_subplot(gs[:, 0])  # UMAP on left
            ax4 = fig.add_subplot(gs[:, 1])  # Kernel PCA on right
            cax = fig.add_subplot(gs[:, 2])  # Colorbar axes spans both rows
            axs = [ax3, ax4]

        # Define colormap
        n = len(samples)
        cmap = plt.colormaps["coolwarm"].resampled(n)

        # Calculate colord
        colord = torch.zeros(samples.shape[1])
        for i in range(samples.shape[1]):
            colord[i] = torch.sum(samples[:, i])
        ns = torch.sum(samples, dim=0, dtype=torch.int8)
        colord = colord.cpu().numpy()

        # Reshape embed tensor
        # _, states, _ = embed.shape
        # X = embed.detach().cpu().numpy().reshape(states, -1)
        X = embed.detach().cpu().numpy()
        print("Here is the dimension of the embedding", X.shape)

        # Define normalization
        norm = plt.Normalize(vmin=colord.min(), vmax=colord.max())

        # Function to create scatter plot
        def create_scatter(ax, X_2d, title):
            # Add index labels
            scatter = ax.scatter(
                X_2d[:, 0], X_2d[:, 1], c=colord, cmap=cmap, norm=norm, s=50
            )

            if label_points == True:
                sample_indices = range(len(X_2d))

                for idx, (x, y) in enumerate(X_2d):
                    ax.annotate(
                        str(sample_indices[idx]),
                        (x, y),
                        xytext=(3, 3),  # 3 points offset
                        textcoords="offset points",
                        fontsize=10,
                        alpha=0.7,
                    )

            ax.set_xlabel("First PC")
            ax.set_ylabel("Second PC")
            ax.set_title(title)
            plt.setp(ax, xticks=[], yticks=[])
            return scatter

        plot_index = 0
        last_scatter = None

        # PCA (if enabled, will be on top left)
        if do_pca:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            last_scatter = create_scatter(axs[plot_index], X_pca, "PCA")
            plot_index += 1

        if spectrum_pca == True:
            X_pcafull = PCA(n_components=min(batch-1, embedding_size))
            X_pcafull = X_pcafull.fit(X)
            print(n, "number of features")
            # X_pcafull = X_pcafull.fit(X)
            # print(X_pcafull.shape, "aqui")
            embed_ind = np.arange(min(batch-1,embedding_size))
            axs[plot_index].scatter(
                embed_ind, X_pcafull.explained_variance_ratio_, color="red"
            )
            # axs[plot_index].scatter(
            #     embed_ind, X_pcafull.explained_variance_, color="blue", marker="d"
            # )
            axs[plot_index].set_xlabel("Principal component index $i$")
            axs[plot_index].set_ylabel("Normalized Eigenvalues of PCs")
            plot_index += 1

        # t-SNE (if enabled, will be on top right)
        if do_tsne:
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)
            last_scatter = create_scatter(axs[plot_index], X_tsne, "t-SNE")
            plot_index += 1

        # UMAP (if enabled, will be on bottom left)
        if do_umap:
            reducer = umap.UMAP(
                random_state=42,
            )
            X_umap = reducer.fit_transform(X)
            last_scatter = create_scatter(axs[plot_index], X_umap, "UMAP")
            plot_index += 1

        # # Kernel PCA (if enabled, will be on bottom right)
        # if do_kpca:
        #     kpca = KernelPCA(
        #         n_components=2,
        #         kernel="rbf",  # You can try different kernels: 'rbf', 'polynomial', 'cosine', etc.
        #         random_state=42,
        #     )
        #     X_kpca = kpca.fit_transform(X)
        #     last_scatter = create_scatter(axs[plot_index], X_kpca, "Kernel PCA")

        # Add colorbar using the dedicated colorbar axes
        if last_scatter is not None:
            plt.colorbar(
                last_scatter, cax=cax, label=r"Electrons on band 1 ($N_{e}^{1}$)"
            )

        plt.tight_layout()
        string = f"{base_path}/n_{n}/{name}_{identifier}embed2d_{t}.pdf"
        plt.savefig(string, dpi=180, bbox_inches="tight")

    # Usage:
    plot_embeddings_with_color(
        newsamples,
        newembed0,
        do_pca=True,
        do_umap=True,
        do_tsne=True,
        name="firstemb",
    )
    plot_embeddings_with_color(
        newsamples,
        newembed1,
        do_pca=True,
        do_umap=True,
        do_tsne=True,
        name="summedemb",
    )
