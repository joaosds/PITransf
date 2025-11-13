import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.style.use("myrcparams.mplstyle")

import glob


def read_data_file(filename):
    data = np.loadtxt(filename, unpack=True)
    return {
        "t": data[0],
        "Er": data[1],
        "Ev": data[2],
        "Ehf": data[3],
        "alfa1": data[4],
        "vscore": data[5],
        "vscore2": data[6],
        "norm": data[7],
        "Ed": data[8],
    }


def read_all_data(pattern):
    all_data = {}
    for filename in glob.glob(pattern):
        basis, n0 = filename.split("_")[0], filename.split("_")[1].split(".")[0]
        if basis not in all_data:
            all_data[basis] = {}
        all_data[basis][n0] = read_data_file(filename)
    return all_data


def vscore(n, var, average):
    vscore = n * var / (average**2)
    return vscore


# Read all data files
hf_8soap = read_all_data("datanew3/hf_8soap.txt")
hf_12soap = read_all_data("datanew3/hf_12soap.txt")
hf_14soap = read_all_data("datanew3/hf_14.txt")
hf_14soap2 = read_all_data("datanew3/hf_14nu2.txt")
hf_16soap = read_all_data("datanew3/hf_16soap.txt")
hf_20soap = read_all_data("datanew3/hf_20soap.txt")

hf_6soap = read_all_data("datanew3/hf_6soap.txt")
hf_10chiral = read_all_data("datanew3/hf_10.txt")
chiral_10 = read_all_data("datanew3/chiral_10_new.txt")
hf_10 = read_all_data("datanew3/hf_102.txt")
hf_10soap = read_all_data("datanew3/hf_10soap.txt")
band_10 = read_all_data("datanew3/band_10_new.txt")


name = "datanew3/hf"
name2 = "6soap"
hft6soap = hf_6soap[name][name2]["t"]
hfEr6soap = hf_6soap[name][name2]["Er"]
hfEv6soap = hf_6soap[name][name2]["Ev"]
hf6soap = hf_6soap[name][name2]["Ehf"]
ed6soap = hf_6soap[name][name2]["Ed"]
hfalpha6soap = hf_6soap[name][name2]["alfa1"]
normhf6soap = hf_6soap[name][name2]["norm"]
vscorehf6soap = hf_6soap[name][name2]["vscore"]

hfEr6soap = hfEr6soap / (normhf6soap**2)


name = "datanew3/hf"
name2 = "10soap"
hft10soap = hf_10soap[name][name2]["t"]
hfEr10soap = hf_10soap[name][name2]["Er"]
hfEv10soap = hf_10soap[name][name2]["Ev"]
hf10soap = hf_10soap[name][name2]["Ehf"]
ed10soap = hf_10soap[name][name2]["Ed"]
hfalpha10soap = hf_10soap[name][name2]["alfa1"]
normhf10soap = hf_10soap[name][name2]["norm"]
vscorehf10soap = hf_10soap[name][name2]["vscore"]

hfEr10soap = hfEr10soap / normhf10soap


name = "datanew3/hf"
name2 = "12soap"
hft12soap = hf_12soap[name][name2]["t"]
hfEr12soap = hf_12soap[name][name2]["Er"]
hfEv12soap = hf_12soap[name][name2]["Ev"]
hf12soap = hf_12soap[name][name2]["Ehf"]
ed12soap = hf_12soap[name][name2]["Ed"]
hfalpha12soap = hf_12soap[name][name2]["alfa1"]
normhf12soap = hf_12soap[name][name2]["norm"]
vscorehf12soap = hf_12soap[name][name2]["vscore"]

hfEr12soap = hfEr12soap / normhf12soap


name = "datanew3/hf"
name2 = "14nu2"
hft14soap2 = hf_14soap2[name][name2]["t"]
hfEr14soap2 = hf_14soap2[name][name2]["Er"]
hfEv14soap2 = hf_14soap2[name][name2]["Ev"]
hf14soap2 = hf_14soap2[name][name2]["Ehf"]
ed14soap2 = hf_14soap2[name][name2]["Ed"]
hfalpha14soap2 = hf_14soap2[name][name2]["alfa1"]
normhf14soap2 = hf_14soap2[name][name2]["norm"]
vscorehf14soap2 = hf_14soap2[name][name2]["vscore"]

hfEr14soap2 = hfEr14soap2 / normhf14soap2



name = "datanew3/hf"
name2 = "14"
hft14soap = hf_14soap[name][name2]["t"]
hfEr14soap = hf_14soap[name][name2]["Er"]
hfEv14soap = hf_14soap[name][name2]["Ev"]
hf14soap = hf_14soap[name][name2]["Ehf"]
ed14soap = hf_14soap[name][name2]["Ed"]
hfalpha14soap = hf_14soap[name][name2]["alfa1"]
normhf14soap = hf_14soap[name][name2]["norm"]
vscorehf14soap = hf_14soap[name][name2]["vscore"]

hfEr14soap = hfEr14soap / normhf14soap


name = "datanew3/hf"
name2 = "16soap"
hft16soap = hf_16soap[name][name2]["t"]
hfEr16soap = hf_16soap[name][name2]["Er"]
hfEv16soap = hf_16soap[name][name2]["Ev"]
hf16soap = hf_16soap[name][name2]["Ehf"]
ed16soap = hf_16soap[name][name2]["Ed"]
hfalpha16soap = hf_16soap[name][name2]["alfa1"]
normhf16soap = hf_16soap[name][name2]["norm"]
vscorehf16soap = hf_16soap[name][name2]["vscore"]

hfEr16soap = hfEr16soap / normhf16soap


name = "datanew3/hf"
name2 = "20soap"
hft20soap = hf_20soap[name][name2]["t"]
hfEr20soap = hf_20soap[name][name2]["Er"]
hfEv20soap = hf_20soap[name][name2]["Ev"]
hf20soap = hf_20soap[name][name2]["Ehf"]
ed20soap = hf_20soap[name][name2]["Ed"]
hfalpha20soap = hf_20soap[name][name2]["alfa1"]
normhf20soap = hf_20soap[name][name2]["norm"]
vscorehf20soap = hf_20soap[name][name2]["vscore"]

hfEr20soap = hfEr20soap / normhf20soap


name = "datanew3/hf"
name2 = "8soap"
hft8soap = hf_8soap[name][name2]["t"]
hfEr8soap = hf_8soap[name][name2]["Er"]
hfEv8soap = hf_8soap[name][name2]["Ev"]
hf8soap = hf_8soap[name][name2]["Ehf"]
ed8soap = hf_8soap[name][name2]["Ed"]
hfalpha8soap = hf_8soap[name][name2]["alfa1"]
normhf8soap = hf_8soap[name][name2]["norm"]
vscorehf8soap = hf_8soap[name][name2]["vscore"]

hfEr8soap = hfEr8soap / normhf8soap

name = "datanew3/chiral"
name2 = "10"
chiral10t = chiral_10[name][name2]["t"]
chiralEr10 = chiral_10[name][name2]["Er"]
chiralEv10 = chiral_10[name][name2]["Ev"]
chiralhf10 = chiral_10[name][name2]["Ehf"]
norm10chiral = chiral_10[name][name2]["norm"]
chiralalpha10 = chiral_10[name][name2]["alfa1"]
vscorechiral10 = chiral_10[name][name2]["vscore"]
hf10chiral = chiral_10[name][name2]["Ehf"]
chiraled10 = chiral_10[name][name2]["Ed"]

chiralEr10 = chiralEr10 / norm10chiral

name = "datanew3/hf"
name2 = "102"
hft10 = hf_10[name][name2]["t"]
hfEr10 = hf_10[name][name2]["Er"]
hfEv10 = hf_10[name][name2]["Ev"]
hf10 = hf_10[name][name2]["Ehf"]
ed10 = hf_10[name][name2]["Ed"]
normhf10 = hf_10[name][name2]["norm"]
hfalpha10 = hf_10[name][name2]["alfa1"]
vscorehf10 = hf_10[name][name2]["vscore"]

hfEr10 = hfEr10 / normhf10


name = "datanew3/band"
name2 = "10"
band10t = band_10[name][name2]["t"]
bandEr10 = band_10[name][name2]["Er"]
bandEv10 = band_10[name][name2]["Ev"]
bandhf10 = band_10[name][name2]["Ehf"]
banded10 = band_10[name][name2]["Ed"]
norm10band = band_10[name][name2]["norm"]
bandalpha10 = band_10[name][name2]["alfa1"]
vscoreband10 = band_10[name][name2]["vscore"]

bandEr10 = bandEr10 / norm10band

vscorehf10 = vscore(10, hfEv10soap / normhf10soap, hfEr10soap * 10 / normhf10soap)
vscorechiral10 = vscore(10, chiralEv10 / norm10chiral, chiralEr10 * 10 / norm10chiral)
vscoreband10 = vscore(10, bandEv10 / norm10band, bandEr10 * 10 / norm10band)


# Function to set y-axis to scientific notation
def set_axis_scientific(ax, axis="y"):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # Set the scientific notation threshold
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


# Alternative function using FuncFormatter
def scientific_notation(x, pos):
    return f"${x:.1e}$"


# Function to add gray background stripe
def add_background_stripe(ax):
    ax.axvspan(0.04, 0.11, facecolor="gray", alpha=0.3)


# First three plots remain the same
delta_e_hf6soap = abs(hfEr6soap - hf6soap)
delta_e_hf8soap = abs(hfEr8soap - hf8soap)
delta_e_hf10soap = abs(hfEr10soap - hf10soap)
delta_e_hf12soap = abs(hfEr12soap - hf12soap)
delta_e_hf14soap = abs(hfEr14soap - hf14soap)
delta_e_hf14soap2 = abs(hfEr14soap2 - hf14soap2)
delta_e_hf16soap = abs(hfEr16soap - hf16soap)
delta_e_hf20soap = abs(hfEr20soap - hf20soap)
delta_ed_hf6soap = abs(ed6soap) - abs(hf6soap)
delta_ed_hf8soap = abs(ed8soap) - abs(hf8soap)
delta_ed_hf10soap = abs(ed10soap) - abs(hf10soap)
delta_ed_hf12soap = abs(ed12soap) - abs(hf12soap)

cmap = plt.get_cmap("viridis")
norm = mcolors.Normalize(vmin=0, vmax=10)
color1 = cmap(norm(0))
color2 = cmap(norm(1))
color3 = cmap(norm(2))
color4 = cmap(norm(3))
color5 = cmap(norm(4))
color6 = cmap(norm(5))
color7 = cmap(norm(6))

colors = sns.color_palette("tab20", n_colors=7)
color1 = colors[0]
color2 = colors[1]
color3 = colors[2]
color4 = colors[3]
color5 = colors[4]
color6 = colors[5]
color7 = colors[6]


blue = "#00007dff"
yellow = "#ff8800ff"
red = "#da0010ff"

# Plot 2: vscore vs t_Uarray
# ax2 = fig.add_subplot(gs[0, 1])
# ax2 = fig.add_subplot(gs[0:2, 1])  # Adjust size of first plot
# # ax2.plot(hft6, vscorehf6, marker="o", color=color1, label="N=6")
# # ax2.plot(hft8, vscorehf8, marker="d", color=color2, label="N=8")
#
markers = ["o", "s", "d", "p", "H", "*", "h"]

fig1, ax1 = plt.subplots()

ax1.set_xlabel(r"$t/U$")
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
# ax1.set_yscale("log")
# ax1.yaxis.get_offset_text().set_x(-0.075) ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
ax1.set_ylim(0, 3.5e-3)  # Set y-axis limit
ax1.set_xlim(0.02, 0.16)  # Set y-axis limit
ax1.set_ylabel(r"$\delta E_{\text{HF}}$")


# Create discrete colormap
N_values = [6, 8, 10, 12]
N_values2 = [6, 8, 10, 12, 14, 16, 20, 25]
# Using seaborn colormap

# cmap = plt.cm.colors.ListedColormap(colors)
# norm = plt.cm.colors.BoundaryNorm(boundaries=N_values + [14], ncolors=len(N_values))
#
# # Add colorbar with discrete colors
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# cbar = plt.colorbar(
#     sm,
#     ax=ax1,
#     label="$N_{e}$",
#     boundaries=N_values2 + [14],
#     ticks=N_values2,
# )


# Choose which N_values to use
selected_values = N_values2  # or N_values

# Create boundaries by including the last value
boundaries = selected_values.copy()
# if selected_values[-1] != 20:  # If the last value isn't 20, add the next step
#     boundaries.append(
#         selected_values[-1] + (selected_values[1] - selected_values[0])
#     )  # Add one more step

# Create colors and colormap
colors = sns.color_palette("Spectral", n_colors=8)
cmap = plt.cm.colors.ListedColormap(colors)
norm = plt.cm.colors.BoundaryNorm(boundaries=boundaries, ncolors=len(selected_values))

# Example plot to demonstrate the colormap
# fig, ax = plt.subplots(figsize=(10, 2))
cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax1,
    # orientation="horizontal",
    label="$N_{e}$",
    boundaries=boundaries,
    ticks=selected_values,
)

# Create proxy artists for legend with intended styles
tqs_proxy = plt.Line2D(
    [0],
    [0],
    color="gray",
    marker=markers[2],
    linestyle="none",
    label="TQS",
    markersize=8,
)
ed_proxy = plt.Line2D(
    [0], [0], color="gray", linestyle="-", label=r"$|E_{\text{ED}}- E_{\text{HF}}|$"
)

# Add legend with only the proxy artists
ax1.legend(handles=[tqs_proxy, ed_proxy], loc="upper left")

# Original plotting code remains the same, just remove the label parameters
for i, N in enumerate(N_values):
    ax1.plot(eval(f"hft{N}soap"), eval(f"delta_ed_hf{N}soap"), color="gray")
    ax1.scatter(
        eval(f"hft{N}soap"),
        eval(f"delta_e_hf{N}soap"),
        marker=markers[2],
        s=60,
        color=colors[i],
        edgecolor="black",  # Add black edge
        linewidth=1.5,  # Edge width
    )


ax1.scatter(
    hft14soap,
    delta_e_hf14soap,
    marker=markers[2],
    color=colors[4],
    s=60,
    edgecolor="black",  # Add black edge
    linewidth=1.5,  # Edge width
)
ax1.scatter(
    hft16soap,
    delta_e_hf16soap,
    marker=markers[2],
    color=colors[6],
    s=60,
    edgecolor="black",  # Add black edge
    linewidth=1.5,  # Edge width
)
print(delta_e_hf20soap)
ax1.scatter(
    hft20soap,
    delta_e_hf20soap,
    marker=markers[2],
    color=colors[7],
    s=60,
    edgecolor="black",  # Add black edge
    linewidth=1.5,  # Edge width
)
# ax1.plot(
#     hft12,
#     delta_ed_hf12,
#     linestyle="--",
#     color=color4,
#     label="N = 12",
# )
ax1.scatter(
    hft14soap2,
    delta_e_hf14soap2,
    marker="o",
    color="white",
        s=80,
    edgecolor="Black",  # Add black edge
    linewidth=1.5,  # Edge width
)

# Add legend

# Create inset axes in ax1
axins = ax1.inset_axes([0.6, 0.70, 0.35, 0.35])


# ed_tqs_diff6 = abs((ed6 - hfEr6) / ed6)
ed_tqs_diff6soap = abs((ed6soap - hfEr6soap))
# ed_tqs_diff8 = abs((ed8 - hfEr8) / ed8)
ed_tqs_diff8soap = abs((ed8soap - hfEr8soap))
ed_tqs_diff10soap = abs((ed10soap - hfEr10soap))
ed_tqs_diff12soap = abs((ed12soap - hfEr12soap))
# ed_tqs_diff20soap = abs((ed20soap - hfEr12soap)/ed12soap)
# ed_tqs_diff10 = abs((ed10 - hfEr10) / ed10)
# ed_tqs_diff12 = abs((ed12 - hfEr12) / ed12)
# Copy ax4 plots to axins
# Loop for inset plots
for i, N in enumerate(N_values):
    axins.plot(
        eval(f"hft{N}soap"),
        eval(f"ed_tqs_diff{N}soap"),
        marker=markers[2],
        color=colors[i],
        linestyle="--",
        markersize=5,
    )

# Configure inset formatting
axins.set_xlabel(r"$t/U$")
axins.set_ylabel(r"$\delta E_{\text{ED}}$")
axins.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axins.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

plt.tight_layout()
plt.savefig(
    "first_plot.svg",
    dpi=300,
)

fig2, ax3 = plt.subplots()


for i, N in enumerate(N_values):
    ax3.plot(
        eval(f"hft{N}soap"),
        np.sqrt(eval(f"hfalpha{N}soap")),
        linestyle="--",
        color=colors[i],
    )
    ax3.scatter(
        eval(f"hft{N}soap"),
        np.sqrt(eval(f"hfalpha{N}soap")),
        marker=markers[2],
        s=60,
        edgecolor="black",  # Add black edge
        linewidth=1.5,  # Edge width
        color=colors[i],
    )


ax3.scatter(
    hft20soap,
    np.sqrt(hfalpha20soap),
    # linestyle="--",
    marker=markers[2],
    s=60,
    edgecolor="black",  # Add black edge
    linewidth=1.5,  # Edge width
    color=colors[4],
)

ax3.scatter(
    hft16soap,
    np.sqrt(hfalpha16soap),
    # linestyle="--",
    marker=markers[2],
    s=60,
    edgecolor="black",  # Add black edge
    linewidth=1.5,  # Edge width
    color=colors[6],
)


ax3.scatter(
    hft20soap,
    np.sqrt(hfalpha20soap),
    marker=markers[2],
    s=60,
    edgecolor="black",  # Add black edge
    linewidth=1.5,  # Edge width
    color=colors[7],
)

ax3.scatter(
    hft14soap2,
    np.sqrt(hfalpha14soap2),
    marker="o",
    color="white",
        s=80,
    edgecolor="Black",  # Add black edge
    linewidth=1.5,  # Edge width
)

add_background_stripe(ax3)
ax3.set_xlabel(r"$t/U$")
ax3.set_ylabel(r"$\alpha$")


plt.tight_layout()
plt.savefig(
    "pannelscalingalpha.svg",
    # format="eps",
    dpi=300,
)
# bbox_inches="tight",
# transparent=False,
