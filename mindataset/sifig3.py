import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

plt.style.use("myrcparams.mplstyle")

# Your data arrays (t_Uarray, ehf, tqshf, vscorehf, alfahf, tqsband, vscoreband, alfaband, tqschiral, vscorechiral, alfachiral) should be defined here
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
hf_6 = read_all_data("datanew3/hf_6.txt")
hf_8 = read_all_data("datanew3/hf_8.txt")
hf_8soap = read_all_data("datanew3/hf_8soap.txt")
hf_12 = read_all_data("datanew3/hf_12.txt")
hf_12soap = read_all_data("datanew3/hf_12soap.txt")
hf_62 = read_all_data("datanew3/hf_62.txt")
hf_6soap = read_all_data("datanew3/hf_6soap.txt")
hf_10chiral = read_all_data("datanew3/hf_10.txt")
chiral_10 = read_all_data("datanew3/chiral_10_new.txt")
hf_10 = read_all_data("datanew3/hf_102.txt")
hf_10soap = read_all_data("datanew3/hf_10soap.txt")
band_10 = read_all_data("datanew3/band_10_new.txt")
hf_15 = read_all_data("datanew3/hf_15.txt")
hf_14 = read_all_data("datanew3/hf_14.txt")
hf_30 = read_all_data("datanew3/hf_30.txt")

name = "datanew3/hf"
name2 = "6"
hft6 = hf_6[name][name2]["t"]
hfEr6 = hf_6[name][name2]["Er"]
hfEv6 = hf_6[name][name2]["Ev"]
hf6 = hf_6[name][name2]["Ehf"]
ed6 = hf_6[name][name2]["Ed"]
hfalpha = hf_6[name][name2]["alfa1"]
normhf6 = hf_6[name][name2]["norm"]
vscorehf6 = hf_6[name][name2]["vscore"]

hfEr6 = hfEr6 / (normhf6**2)


name = "datanew3/hf"
name2 = "6soap"
hft6soap = hf_6soap[name][name2]["t"]
hfEr6soap = hf_6soap[name][name2]["Er"]
hfEv6soap = hf_6soap[name][name2]["Ev"]
hf6soap = hf_6soap[name][name2]["Ehf"]
ed6soap = hf_6soap[name][name2]["Ed"]
hfalphasoap = hf_6soap[name][name2]["alfa1"]
normhf6soap = hf_6soap[name][name2]["norm"]
vscorehf6soap = hf_6soap[name][name2]["vscore"]

hfEr6soap = hfEr6soap / (normhf6soap**2)

name = "datanew3/hf"
name2 = "8"
hft8 = hf_8[name][name2]["t"]
hfEr8 = hf_8[name][name2]["Er"]
hfEv8 = hf_8[name][name2]["Ev"]
hf8 = hf_8[name][name2]["Ehf"]
ed8 = hf_8[name][name2]["Ed"]
hfalpha8 = hf_8[name][name2]["alfa1"]
normhf8 = hf_8[name][name2]["norm"]
vscorehf8 = hf_8[name][name2]["vscore"]

hfEr8 = hfEr8 / normhf8


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

name = "datanew3/hf"
name2 = "12"
hft12 = hf_12[name][name2]["t"]
hfEr12 = hf_12[name][name2]["Er"]
hfEv12 = hf_12[name][name2]["Ev"]
hf12 = hf_12[name][name2]["Ehf"]
ed12 = hf_12[name][name2]["Ed"]
hfalpha12 = hf_12[name][name2]["alfa1"]
normhf12 = hf_12[name][name2]["norm"]
vscorehf12 = hf_12[name][name2]["vscore"]

hfEr12 = hfEr12 / normhf12


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
name2 = "15"
hft15 = hf_15[name][name2]["t"]
hfEr15 = hf_15[name][name2]["Er"]
hfEv15 = hf_15[name][name2]["Ev"]
hf15 = hf_15[name][name2]["Ehf"]
hfalpha15 = hf_15[name][name2]["alfa1"]
hf15norm = hf_15[name][name2]["norm"]
vscorehf15 = hf_15[name][name2]["vscore"]

hfEr15 = hfEr15 / hf15norm


name = "datanew3/hf"
name2 = "14"
hft14 = hf_14[name][name2]["t"]
hfEr14 = hf_14[name][name2]["Er"]
hfEv14 = hf_14[name][name2]["Ev"]
hf14 = hf_14[name][name2]["Ehf"]
hfalpha14 = hf_14[name][name2]["alfa1"]
hf14norm = hf_14[name][name2]["norm"]
vscorehf14 = hf_14[name][name2]["vscore"]

hfEr14 = hfEr14 / hf14norm

name = "datanew3/hf"
name2 = "62"
hft62 = hf_62[name][name2]["t"]
hfEr62 = hf_62[name][name2]["Er"]
hfEv62 = hf_62[name][name2]["Ev"]
hf62 = hf_62[name][name2]["Ehf"]
ed62 = hf_62[name][name2]["Ed"]
hfalpha62 = hf_62[name][name2]["alfa1"]
hfalpha62 = hf_62[name][name2]["norm"]
vscorehf62 = hf_62[name][name2]["vscore"]


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


name = "datanew3/hf"
name2 = "30"
hft30 = hf_30[name][name2]["t"]
hfEr30 = hf_30[name][name2]["Er"]
hfEv30 = hf_30[name][name2]["Ev"]
hf30 = hf_30[name][name2]["Ehf"]
ed30 = hf_30[name][name2]["Ed"]
hfalpha30 = hf_30[name][name2]["alfa1"]
vscorehf30 = hf_30[name][name2]["vscore"]


vscorehf6 = vscore(6, hfEv6 * 6, (hfEr6) * 6)
vscorehf8 = vscore(8, hfEv8 / normhf8, hfEr8 * 8 / normhf8)
vscorehf10 = vscore(10, hfEv10soap / normhf10soap, hfEr10soap * 10 / normhf10soap)
vscorechiral10 = vscore(10, chiralEv10 / norm10chiral, chiralEr10 * 10 / norm10chiral)
vscoreband10 = vscore(10, bandEv10 / norm10band, bandEr10 * 10 / norm10band)

# vscorehf62 = vscore(6, hfEv62, hfEr62 * 6)
# vscorehf10 = vscore(10, hfEv10, hfEr10 * 10)
# vscorehf20 = vscore(20, hfEv20 * 20, hfEr20 * 20)
# vscorehf30 = vscore(30, hfEv30 * 30, hfEr30 * 30)


# Create a 2x2 grid
fig = plt.figure(figsize=(12, 12))  # Adjust figure size to accommodate new subplot
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])  # Changed to 4 rows


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
delta_e_hf6 = abs((hfEr6 - hf6) / hf6)
delta_e_hf6soap = abs((hfEr6soap - hf6soap) / hf6soap)
delta_e_hf8soap = abs((hfEr8soap - hf8soap) / hf8soap)
delta_e_hf10soap = abs((hfEr10soap - hf10soap) / hf10soap)
delta_e_hf12soap = abs((hfEr12soap - hf12soap) / hf12soap)
delta_ed_hf6 = abs((ed6 - hf6) / hf6)
delta_ed_hf8 = abs((ed8 - hf8) / hf8)
delta_ed_hf10 = abs((ed10soap - hf10soap) / hf10soap)
delta_ed_hf12 = abs((ed12 - hf12) / hf12)
delta_e_hf66 = (abs(hfEr6) - abs(ed6)) * 6
delta_e_hf8 = abs((hfEr8 - hf8) / hf8)
delta_e_hf88 = abs(hfEr8) - abs(ed8)
delta_e_hf10 = abs((hfEr10 - hf10) / hf10)
delta_e_hf12 = abs((hfEr12 - hf12) / hf12)
delta_e_hf14 = abs((hfEr14 - hf14) / hf14)
delta_e_chiral10 = abs(chiralEr10) - abs(hf10chiral)
delta_e_band10 = abs(bandEr10) - abs(bandhf10)
delta_e_hf100 = abs(hfEr10soap) - abs(ed10soap)
delta_e_hf15 = abs(hfEr15) - abs(hf15)
delta_e_hf30 = abs(hfEr30) - abs(hf30)


# delta_e_hf6 = (abs(hfEr6) - abs(hf6))/abs(hf6)
# delta_e_hf6soap = (abs(hfEr6soap) - abs(hf6soap))/abs(hf6soap)
# delta_e_hf8soap = (abs(hfEr8soap) - abs(hf8soap))/abs(hf8soap)
# delta_e_hf10soap = (abs(hfEr10soap) - abs(hf10soap))/abs(hf10soap)
# delta_e_hf12soap = (abs(hfEr12soap) - abs(hf12soap))/abs(hf12soap)
# delta_ed_hf6 = (abs(ed6) - abs(hf6))/abs(hf6)
# delta_ed_hf8 = (abs(ed8) - abs(hf8))/abs(hf8)
# delta_ed_hf10 = (abs(ed10) - abs(hf10))/abs(hf10)
# delta_ed_hf12 = (abs(ed12) - abs(hf12))/abs(hf12)
# delta_e_hf8 = (abs(hfEr8) - abs(hf8))/abs(hf8)
# delta_e_hf10 = (abs(hfEr10) - abs(hf10))/abs(hf10)
# delta_e_hf12 = (abs(hfEr12) - abs(hf12))/abs(hf12)
# delta_e_hf14 = (abs(hfEr14) - abs(hf14))/abs(hf14)
# delta_e_chiral10 = abs(chiralEr10) - abs(hf10chiral)
#

cmap = plt.get_cmap("viridis")
norm = mcolors.Normalize(vmin=0, vmax=10)
color1 = cmap(norm(0))
color2 = cmap(norm(1))
color3 = cmap(norm(2))
color4 = cmap(norm(3))
color5 = cmap(norm(4))
color6 = cmap(norm(5))
color7 = cmap(norm(6))

cmap = plt.get_cmap("tab20")
# For 7 colors, space them evenly across the full range
colors = [cmap(i) for i in np.linspace(0, 1, 7)]

# Or assign individually:
color1 = cmap(0)  # First color
color2 = cmap(0.20)  # ~1/6 of the way
color3 = cmap(0.33)  # ~2/6 of the way
color4 = cmap(0.5)  # Middle
color5 = cmap(0.67)  # ~4/6 of the way
color6 = cmap(0.83)  # ~5/6 of the way
color7 = cmap(1)  # Last color

# ax1.plot(
#     hft6,
#     delta_e_hf6,
#     marker="o",
#     color=color1,
# )
# ax1.plot(
#     hft8,
#     delta_e_hf8,
#     marker="d",
#     color=color2,
# )

blue = "#00007dff"
yellow = "#ff8800ff"
red = "#da0010ff"
# ax1 = fig.add_subplot(gs[0:2, 0:2])  # Adjust size of first plot
ax1 = fig.subplots()
ax1.plot(hft10soap, delta_e_hf10soap, marker="o", color=red, label="HF-basis")
ax1.plot(band10t, delta_e_band10, marker="p", color=yellow, label="Band-basis")
ax1.plot(chiral10t, delta_e_chiral10, marker="d", color=blue, label="chiral-basis")
ax1.plot(hft10soap, delta_ed_hf10, linestyle="--", color="Black", label="ED")
# ax1.plot(
#     hft10,
#     delta_e_hf10,
#     marker="p",
#     color=color3,
# )
# ax1.plot(
#     hft15,
#     delta_e_hf15,
#     marker="p",
#     color="red",
# )
ax1.set_xlabel(r"$t/U$")
ax1.legend()
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
# ax1.set_yscale("log")
# ax1.yaxis.get_offset_text().set_x(-0.075) ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
ax1.set_ylabel(r"$\delta E_{\text{HF}} = |E_{\text{TQS}}-E_{\text{HF}}|$")

# Plot 2: vscore vs t_Uarray
# ax2 = fig.add_subplot(gs[0, 1])
# ax2 = fig.add_subplot(gs[0:2, 1])  # Adjust size of first plot
# # ax2.plot(hft6, vscorehf6, marker="o", color=color1, label="N=6")
# # ax2.plot(hft8, vscorehf8, marker="d", color=color2, label="N=8")
#

markers = ["o", "s", "d", "p"]
# ax2.plot(hft10, vscorehf10, marker=markers[0], color=red)
# ax2.plot(chiral10t, vscorechiral10, marker=markers[1], color=blue)
# ax2.plot(band10t, vscoreband10, marker=markers[1], color=yellow)
# # ax2.plot(hft15, vscorehf15, marker="+", color="red", label="N=15")
# # ax2.plot(hft15, vscorehf15, marker="+", color="red", label="N=15")
# ax2.set_ylabel("V-score")
# ax2.set_yscale("log")

# New Plot 4: ED to TQS differences
# ax4 = fig.add_subplot(gs[2:4, 0])  # Place below ax1
ax4 = fig.subplots()
ed_tqs_diff6 = -abs(ed6) + abs(hfEr6)
ed_tqs_diff8 = -abs(ed8) + abs(hfEr8)
ed_tqs_diff10 = abs((ed10soap - hfEr10soap) / ed10soap)
ed_tqs_diffchiral10 = abs((chiraled10 - chiralEr10) / chiraled10)
ed_tqs_diffband10 = abs((banded10 - bandEr10) / banded10)

ax4.plot(
    hft10soap,
    ed_tqs_diff10,
    marker=markers[0],
    color=red,
)

ax4.plot(
    band10t,
    ed_tqs_diffband10,
    marker=markers[1],
    color=yellow,
)

ax4.plot(
    chiral10t,
    ed_tqs_diffchiral10,
    marker=markers[2],
    color=blue,
)
ax4.set_xlabel(r"$t/U$")
ax4.set_ylabel(r"$\Delta E_{\text{ED}} = |E_{\text{TQS}}| - |E_{\text{ED}}|$")
# ax4.set_yscale("log")
# ax4.set_ylim(1e-5,1e-20)
ax4.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax4.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax4.yaxis.get_offset_text().set_x(-0.075)
ax4.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

# plt.tight_layout()
# plt.savefig("b.pdf")
# plt.show()
# plt.show()
#

plt.close()
# fig = plt.figure(figsize=(12, 12))  # Adjust figure size to accommodate new subplot
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])  # Changed to 4 rows
# ax1 = fig.add_subplot(gs[0:2, 0:2])  # Adjust size of first plot
fig1, ax1 = plt.subplots()

ax1.scatter(
    hft6,
    delta_e_hf6,
    marker=markers[0],
    color=color1,
    s=20,
    label="N = 6 (Adam)",
)
ax1.scatter(
    hft8,
    delta_e_hf8,
    marker=markers[1],
    s=20,
    color=color2,
    label="N = 8",
)
ax1.scatter(
    hft10,
    delta_e_hf10,
    marker=markers[2],
    s=20,
    color=color3,
    label="N = 10",
)
ax1.scatter(
    hft12,
    delta_e_hf12,
    marker=markers[3],
    color=color4,
    s=20,
    label="N = 12",
)

ax1.scatter(
    hft6soap,
    delta_e_hf6soap,
    marker=markers[0],
    s=60,
    color=color1,
    facecolor="None",
    label="N = 6 (Soap)",
)

ax1.scatter(
    hft8soap,
    delta_e_hf8soap,
    marker=markers[1],
    s=60,
    color=color2,
    facecolor="None",
    label="N = 8",
)

ax1.scatter(
    hft10soap,
    delta_e_hf10soap,
    marker=markers[2],
    s=60,
    color=color3,
    facecolor="None",
    label="N = 10",
)
ax1.scatter(
    hft12soap,
    delta_e_hf12soap,
    marker=markers[3],
    color=color4,
    s=60,
    facecolor="None",
    label="N = 12",
)


ax1.plot(
    hft6,
    delta_ed_hf6,
    linestyle="--",
    color=color1,
    label="N = 6 (ED)",
)
ax1.plot(
    hft8,
    delta_ed_hf8,
    linestyle="--",
    color=color2,
    label="N = 8",
)
ax1.plot(
    hft10soap,
    delta_ed_hf10,
    linestyle="--",
    color=color3,
    label="N = 10",
)
ax1.plot(
    hft12,
    delta_ed_hf12,
    linestyle="--",
    color=color4,
    label="N = 12",
)

# ax1.plot(
#     hft14,
#     delta_e_hf14,
#     marker="p",
#     color=color5,
#     label="N = 14",
# )
# ax1.plot(
#     hft15,
#     delta_e_hf15,
#     marker="p",
#     color="brown",
#     label="N = 15",
# )
# ax1.plot(
#     hft20,
#     delta_e_hf20,
#     marker="p",
#     color=color6,
#     label="N = 20",
# )
#
# ax1.plot(
#     hft30,
#     delta_e_hf30,
#     marker="p",
#     color=color7,
#     label="N = 30",
# )
ax1.set_xlabel(r"$t/U$")
ax1.legend()
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
# ax1.set_yscale("log")
# ax1.yaxis.get_offset_text().set_x(-0.075) ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
ax1.set_ylabel(r"$\delta E_{\text{HF}} = |\left(E_{\text{TQS}}-E_{\text{HF}}\right)/E_{\text{HF}}|$")

# Plot 2: vscore vs t_Uarray
# ax2 = fig.add_subplot(gs[0, 1])
# ax2 = fig.add_subplot(gs[0:2, 1])  # Adjust size of first plot
# ax2.plot(hft6, vscorehf6, marker="o", color=color1, label="N=6")
# ax2.plot(hft6soap, vscorehf6soap, marker="o", color=red, label="N=6soap")
# ax2.plot(hft8, vscorehf8, marker="d", color=color2, label="N=8")
# ax2.plot(hft10, vscorehf10, marker="p", color=color3)
# ax2.plot(hft12, vscorehf12, marker="p", color=color4)
# ax2.plot(hft14, vscorehf14, marker="p", color=color5)
# # ax2.plot(hf10chiralt, vscorechiral10, marker="p", color=color3, label="N=10")
# ax2.plot(hft15, vscorehf15, marker="+", color="brown", label="N=15")
# # ax2.plot(hft15, vscorehf15, marker="+", color="red", label="N=15")
# ax2.set_ylabel("V-score")
# ax2.set_yscale("log")
#
# Plot 3: alfa vs t_Uarray

plt.tight_layout()
plt.savefig("a.pdf")
# ax3 = fig.add_subplot(gs[2:4, 1])  # Place below ax1
fig3, ax3 = plt.subplots()
add_background_stripe(ax3)
ax3.plot(
    hft6,
    np.sqrt(hfalpha),
    marker=markers[0],
    linestyle="--",
    markersize=4,
    color=color1,
    label="N = 6 (Adam)",
)
ax3.plot(
    hft8,
    np.sqrt(hfalpha8),
    linestyle="--",
    dashes=(5, 5),
    marker=markers[1],
    markersize=4,
    color=color2,
    label="N = 8",
)
ax3.plot(
    hft10,
    np.sqrt(hfalpha10),
    linestyle="--",
    dashes=(5, 5),
    marker=markers[2],
    markersize=4,
    color=color3,
    label="N = 10",
)
ax3.plot(
    hft12,
    np.sqrt(hfalpha12),
    linestyle="--",
    dashes=(5, 5),
    marker=markers[3],
    markersize=4,
    color=color4,
    label="N = 12",
)
ax3.scatter(
    hft6soap,
    np.sqrt(hfalphasoap),
    marker=markers[0],
    s=60,
    facecolor="None",
    color=color1,
    label="N = 6 (Soap)",
)
ax3.scatter(
    hft8soap,
    np.sqrt(hfalpha8soap),
    marker=markers[1],
    s=60,
    facecolor="None",
    color=color2,
    label="N = 8",
)
ax3.scatter(
    hft10soap,
    np.sqrt(hfalpha10soap),
    marker=markers[2],
    s=60,
    facecolor="None",
    color=color3,
    label="N = 10 ",
)
ax3.scatter(
    hft12soap,
    np.sqrt(hfalpha12soap),
    marker=markers[3],
    s=60,
    facecolor="None",
    color=color4,
    label="N = 12 (Soap)",
)
# ax3.plot(hft14, np.sqrt(hfalpha14), marker="p", color=color5, label="N = 14")
# ax3.plot(hft15, np.sqrt(hfalpha15), marker="+", color="brown", label="N = 15")
# ax3.plot(hft20, np.sqrt(hfalpha20), marker="+", color=color6, label="N = 20")
# ax3.plot(hft30, np.sqrt(hfalpha30), marker="+", color=color7, label="N = 30")
ax3.set_xlabel(r"$t/U$")
ax3.set_ylabel(r"$\alpha$")

# Original plot code remains the same until the end, then add:

# Create an inset axes
axins = ax3.inset_axes(
    [0.6, 0.3, 0.34, 0.34]
)  # [x, y, width, height] in relative coordinates

# Plot the same data in the inset
axins.plot(
    hft6,
    np.sqrt(hfalpha),
    marker=markers[0],
    linestyle="--",
    markersize=4,
    color=color1,
)
axins.plot(
    hft8,
    np.sqrt(hfalpha8),
    linestyle="--",
    dashes=(5, 5),
    marker=markers[1],
    markersize=4,
    color=color2,
)
axins.plot(
    hft10,
    np.sqrt(hfalpha10),
    linestyle="--",
    dashes=(5, 5),
    marker=markers[2],
    markersize=4,
    color=color3,
)
axins.plot(
    hft12,
    np.sqrt(hfalpha12),
    linestyle="--",
    dashes=(5, 5),
    marker=markers[3],
    markersize=4,
    color=color4,
)

axins.scatter(
    hft6soap,
    np.sqrt(hfalphasoap),
    marker=markers[0],
    s=60,
    facecolor="None",
    color=color1,
)
axins.scatter(
    hft8soap,
    np.sqrt(hfalpha8soap),
    marker=markers[1],
    s=60,
    facecolor="None",
    color=color2,
)
axins.scatter(
    hft10soap,
    np.sqrt(hfalpha10soap),
    marker=markers[2],
    s=60,
    facecolor="None",
    color=color3,
)
axins.scatter(
    hft12soap,
    np.sqrt(hfalpha12soap),
    marker=markers[3],
    s=60,
    facecolor="None",
    color=color4,
)

# Get the x-range for the last 10 points
x_min = hft6[-20]  # Assuming hft6 is your x-axis data
x_max = hft6[-5]
y_min = min(np.sqrt(hfalpha[-22:]))  # Get minimum y-value in the range
y_max = max(np.sqrt(hfalpha[-10:]))  # Get maximum y-value in the range

# Add some padding to the y-range
y_padding = (y_max - y_min) * 0.1
y_min -= y_padding
y_max += y_padding

# Set the limits for the inset plot
axins.set_xlim(x_min, x_max)
axins.set_ylim(y_min, y_max)

# Draw box showing the zoomed region
ax3.indicate_inset_zoom(axins)
plt.tight_layout()
plt.savefig("b.pdf")

# Optional: Adjust tick parameters for the inset
# axins.tick_params(labelsize=8)


# New Plot 4: ED to TQS differences
# ax4 = fig.add_subplot(gs[2:4, 0])  # Place below ax1
fig4, ax4 = plt.subplots()
ed_tqs_diff6 = abs((ed6 - hfEr6) / ed6)
ed_tqs_diff6soap = abs((ed6soap - hfEr6soap) / ed6soap)
ed_tqs_diff8 = abs((ed8 - hfEr8) / ed8)
ed_tqs_diff8soap = abs((ed8soap - hfEr8soap) / ed8soap)
ed_tqs_diff10soap = abs((ed10soap - hfEr10soap) / ed10soap)
ed_tqs_diff12soap = abs((ed12soap - hfEr12soap) / ed12soap)
ed_tqs_diff10 = abs((ed10 - hfEr10) / ed10)
ed_tqs_diff12 = abs((ed12 - hfEr12) / ed12)
# ed_tqs_diffchiral10 = -abs(ed10chiral) + abs(chiralEr10)

ax4.plot(
    hft6,
    ed_tqs_diff6,
    marker=markers[0],
    color=color1,
    linestyle="--",
    markersize=4,
    label="N=6 (Adam)",
)


ax4.plot(
    hft8,
    ed_tqs_diff8,
    marker=markers[1],
    color=color2,
    linestyle="--",
    markersize=4,
    label="N=8",
)

ax4.plot(
    hft10,
    ed_tqs_diff10,
    marker=markers[2],
    linestyle="--",
    color=color3,
    markersize=4,
    label="N=10",
)

ax4.plot(
    hft12,
    ed_tqs_diff12,
    marker=markers[3],
    linestyle="--",
    color=color4,
    label="N=12",
)

ax4.scatter(
    hft6soap,
    ed_tqs_diff6soap,
    marker=markers[0],
    color=color1,
    s=60,
    # linestyle="--",
    facecolor="None",
    label="N=6 (Soap)",
)
ax4.scatter(
    hft8soap,
    ed_tqs_diff8soap,
    marker=markers[1],
    color=color2,
    s=60,
    facecolor="None",
    label="N=8 (Soap)",
)
ax4.scatter(
    hft10soap,
    ed_tqs_diff10soap,
    marker=markers[2],
    s=60,
    color=color3,
    facecolor="None",
    label="N=10 (Soap)",
)
ax4.scatter(
    hft12soap,
    ed_tqs_diff12soap,
    marker=markers[3],
    s=60,
    color=color4,
    facecolor="None",
    label="N=12 (Soap)",
)


# ax4.plot(
#     chiral10t,
#     ed_tqs_diffchiral10,
#     marker="p",
#     color="black",
#     label="N=10",
# )
ax4.set_xlabel(r"$t/U$")
ax4.set_ylabel(
    r"$\delta E_{\text{ED}} = |(E_{\text{TQS}} - E_{\text{ED}})/ E_{\text{ED}}|$"
)
# ax4.set_yscale("log")
# ax4.set_ylim(1e-5,1e-20)
ax4.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax4.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax4.yaxis.get_offset_text().set_x(-0.075)
ax4.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

# plt.tight_layout()
# plt.savefig("pannel_scaling.png", dpi=150)
# plt.show()
plt.tight_layout()
plt.savefig("c.pdf")
