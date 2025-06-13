import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from model_utils import treesamplerumap

# Set matplotlib parameters
plt.style.use("myrcparams.mplstyle")


def create_plots(data, flags=None):
    """
    Main plotting function that coordinates all plot generation
    
    Args:
        data: Dictionary containing all plotting data
        flags: Dictionary containing plotting flags (optional - will use defaults if None)
    """
    
    # Set default flags if none provided
    if flags is None:
        flags = {
            'plot_n': True,
            'plot_it': True,
            'plot_it_hist_weigh': True,
            'pcaplot': True,
        }
    
    print("Generating plots...")
    
    # Ensure output directory is outside transformer_quantum_state folder
    if 'system_output_dir' not in data or not data['system_output_dir']:
        data['system_output_dir'] = os.path.join("..", "plots")
    
    # Create output directory if it doesn't exist
    os.makedirs(data['system_output_dir'], exist_ok=True)
    
    try:
        if flags.get('plot_n', False):
            plot_occupation_comparison(data)
            print("✓ Occupation comparison plots created")
        
        if flags.get('plot_it', False):
            plot_energy_convergence(data)
            plot_normalization(data)
            plot_vscore(data)
            plot_alpha_evolution(data)
            plot_gradient(data)
            plot_loss(data)
            print("✓ Training iteration plots created")
        
        if flags.get('plot_it_hist_weigh', False):
            plot_histogram_weighted(data)
            print("✓ Weighted histogram plots created")
        
        if flags.get('pcaplot', False):
            plot_pca_analysis(data)
            print("✓ PCA analysis plots created")
            
    except Exception as e:
        print(f"Error generating plots: {e}")
        raise


# ============================= OCCUPATION ANALYSIS =============================

def plot_occupation_comparison(data):
    """
    Plot occupation number comparison between TQS and Hartree-Fock
    
    Creates two subplots:
    1. Occupation numbers vs momentum for different bands
    2. Differences between TQS and HF predictions
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    momentum = data['momentum'].cpu()
    band = data['band'].cpu()
    Nk_hf = data['Nk_hf'].cpu()
    
    # Reference lines at ±π/2
    ax1.axvline(x=torch.pi / 2, color='gray', alpha=0.5, linestyle='-')
    ax1.axvline(x=-torch.pi / 2, color='gray', alpha=0.5, linestyle='-')

    # Plot occupation numbers
    ax1.scatter(
        momentum[:, 1], band[:, 0].real, marker="d", color="orange", 
        label=r"N$_{x}$ (TQS)", alpha=0.7
    )
    ax1.plot(
        momentum[:, 1], Nk_hf[:, 0], color="orange", linestyle="--", 
        label=r"N$_{x}$ (HF)", linewidth=2
    )
    ax1.scatter(
        momentum[:, 1], band[:, 1].real, marker="d", color="black", 
        label=r"N$_{y}$", alpha=0.7
    )
    ax1.plot(momentum[:, 1], Nk_hf[:, 1], color="black", linestyle="--", linewidth=2)
    ax1.scatter(
        momentum[:, 1], band[:, 2].real, marker="d", color="red", 
        label=r"N$_{z}$", alpha=0.7
    )
    ax1.plot(momentum[:, 1], Nk_hf[:, 2], linestyle="--", color="red", linewidth=2)

    ax1.legend(fontsize=12)
    ax1.set_xlabel(r"Momentum $|\mathbf{k}|$", fontsize=14)
    ax1.set_ylabel(r"$|\mathbf{N}|$", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot differences
    difference = torch.abs(torch.abs(band[:, :].real) - torch.abs(Nk_hf[:, :].real))
    ax2.minorticks_on()

    ax2.plot(
        momentum[:, 1], difference[:, 0], marker="d", color="orange", 
        label=r"$\Delta$N$_{x}$", markersize=4
    )
    ax2.plot(
        momentum[:, 1], difference[:, 1], marker="d", color="black", 
        label=r"$\Delta$N$_{y}$", markersize=4
    )
    ax2.plot(
        momentum[:, 1], difference[:, 2], marker="d", color="red", 
        label=r"$\Delta$N$_{z}$", markersize=4
    )

    ax2.legend(fontsize=12)
    ax2.set_xlabel(r"Momentum $|\mathbf{k}|$", fontsize=14)
    ax2.set_ylabel(r"$|\Delta \mathbf{N}|$", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    filename = os.path.join(data['system_output_dir'], f"{data['identifier']}_occupation_t_{data['t']}.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# ============================= TRAINING ANALYSIS =============================

def plot_energy_convergence(data):
    """Plot energy convergence during training"""
    
    n_iter = data['n_iter']
    iter_range = torch.arange(n_iter + 1).cpu()
    
    plt.clf()
    plt.cla()

    Er = data['Er'].cpu()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Add reference lines for small systems
    if data['N0'] < 11:
        plt.axhline(y=data['Ed'], color="blue", label="ED", linewidth=2)
        plt.axhline(y=data['Ehf'], color="red", label=r"$E_{\text{HF}}$", linewidth=2)
    else:
        plt.axhline(y=data['Ehf'], color="red", label=r"$E_{\text{HF}}$", linewidth=2)

    # Add information text box
    text_content = (
        f"N = {data['n']}\n"
        f"t = {data['t']}\n"
        f"U = {data['U']}\n"
        f"$d_e$ = {data['embedding_size']}, \n"
        f"$n_{{head}}$ = {data['n_head']}, $n_{{layers}}$ = {data['n_layers']} \n"
        f"RTime: {data['hours']:02}:{data['minutes']:02}:{data['seconds']:02} \n"
        f"$N_{{ps}}$ = {data['n_unique00']}, $N_s$ = {data['batch']}  ({data['sampler']})\n"
        f"$N_{{TQS}}$ = {data['num_params']}"
    )
    
    ax.text(
        0.5, 0.6, text_content, bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="bottom", horizontalalignment="left",
        transform=ax.transAxes, fontsize=10
    )

    # Plot main energy
    ax.plot(iter_range, Er, color="black", label=r"$\text{E}_{GS}$", linewidth=2)
    
    # Set y-axis range around the final value
    final_energy = Er[-1].item()
    energy_range = abs(final_energy) * 0.01  # 1% range around final value
    ax.set_ylim(final_energy - energy_range, final_energy + energy_range)

    # Add inset plots
    axins1 = inset_axes(
        ax, width="40%", height="30%", loc="lower right",
        bbox_to_anchor=(-0.1, 0.2, 1, 1), bbox_transform=ax.transAxes,
    )

    Er3 = data['Er3'].cpu()
    Er2 = data['Er2'].cpu()
    
    axins1.plot(iter_range, Er3, color="green", label=r"$\text{E}_{\Lambda}^{(1)}$", linewidth=1.5)
    axins1.plot(iter_range, Er2, color="darkgray", label=r"$\text{E}_{s,s^{\prime}}$", linewidth=1.5)
    axins1.set_xlabel(r"Iterations $i$", fontsize=10)
    axins1.set_ylabel(r"$E/N$", fontsize=10)

    ax.legend(fontsize=12)
    ax.set_xlabel(r"Iterations $i$", fontsize=14)
    ax.set_ylabel(r"E/N", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    filename = os.path.join(data['system_output_dir'], f"{data['identifier']}_energyconv{data['t']}.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_normalization(data):
    """Plot normalization evolution"""
    
    n_iter = data['n_iter']
    iter_range = torch.arange(n_iter + 1).cpu()
    norm = data['norm'].cpu()

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.plot(
        iter_range[100:-1], norm[100:-1], linestyle="--", color="black", linewidth=2
    )

    plt.xlabel(r"Iterations $i$", fontsize=14)
    plt.ylabel(r"Normalization", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(data['system_output_dir'], f"{data['identifier']}_normalization_{data['t']}.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_vscore(data):
    """Plot V-score evolution"""
    
    n_iter = data['n_iter']
    iter_range = torch.arange(n_iter + 1).cpu()
    vscore = data['vscore'].cpu()

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plt.yscale("log")
    plt.plot(iter_range, vscore, linestyle="--", color="black", linewidth=2)

    plt.xlabel(r"Iterations $i$", fontsize=14)
    plt.ylabel(r"V-score", fontsize=14)
    ax = plt.gca()
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$")
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(data['system_output_dir'], f"{data['identifier']}_vscore_{data['t']}.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_alpha_evolution(data):
    """Plot alpha parameter evolution"""
    
    n_iter = data['n_iter']
    iter_range = torch.arange(n_iter + 1).cpu()
    
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    alfa1 = data['alfa1'].cpu()
    alfa2 = data['alfa2'].cpu()
    alfa3 = data['alfa3'].cpu()

    ax.plot(
        iter_range[100:-1], alfa1[100:-1], linestyle="-", color="black", 
        label=r"$\alpha_{1}$", linewidth=2
    )
    ax.set_xlabel(r"Iterations $i$", fontsize=14)
    ax.set_ylabel(r"Reweighting factors $\alpha$", fontsize=14)
    ax.grid(True, alpha=0.3)

    axins1 = inset_axes(
        ax, width="35%", height="25%", loc="lower right",
        bbox_to_anchor=(-0.1, 0.2, 1, 1), bbox_transform=ax.transAxes,
    )

    axins1.plot(
        iter_range[100:-1], alfa2[100:-1], linestyle="-", color="darkgray",
        label=r"$\alpha_{s,s^{\prime}}$", linewidth=1.5
    )
    axins1.plot(
        iter_range[100:-1], alfa3[100:-1], linestyle="-", color="green",
        label=r"$\alpha_{\Lambda}^{(1)}$", linewidth=1.5
    )

    # Format inset
    axins1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axins1.tick_params(axis="both", which="major")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axins1.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=12)

    filename = os.path.join(data['system_output_dir'], f"{data['identifier']}_alpha_{data['t']}.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_gradient(data):
    """Plot gradient evolution"""
    
    n_iter = data['n_iter']
    iter_range = torch.arange(n_iter + 1).cpu()
    grad = data['grad'].cpu()
    
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plt.plot(
        iter_range[1:-1], -grad[1:-1], color="red", 
        label=r"-$\nabla E_{\alpha}$", linewidth=2
    )
    ax.set_ylim(grad[-1] + grad[-1] * 10, grad[-1] - grad[-1] * 10)

    plt.legend(fontsize=12)
    plt.xlabel(r"Iterations $i$", fontsize=14)
    plt.ylabel(r"$-\nabla E_{\alpha}$", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(data['system_output_dir'], f"{data['identifier']}_nablae_{data['t']}.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss(data):
    """Plot loss evolution"""
    
    n_iter = data['n_iter']
    iter_range = torch.arange(n_iter + 1).cpu()
    loss = data['loss'].cpu()
    
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plt.plot(iter_range[1:-1], loss[1:-1], color="red", linewidth=2)
    ax.set_ylim(loss[-1] - loss[-1] * 5, loss[-1] + loss[-1] * 5)

    plt.xlabel(r"Iterations $i$", fontsize=14)
    plt.ylabel(r"Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(data['system_output_dir'], f"{data['identifier']}_loss_{data['t']}.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# ============================= HISTOGRAM ANALYSIS =============================

def plot_histogram_weighted(data):
    """Create dual histogram plots"""
    
    plt.clf()
    plt.cla()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Extract data
    samples = data['samples']
    sample_weight = data['sample_weight']
    n = data['n']
    t = data['t']
    U = data['U']
    embedding_size = data['embedding_size']
    device = data['device']
    n_batch2 = data['n_batch2']
    batch = data['batch']

    # Define colormap
    cmap = plt.colormaps["coolwarm"].resampled(n)

    # First subplot - sample weights
    labels = torch.arange(len(sample_weight))
    colord = torch.zeros(samples.shape[1], device=device)
    
    for i in range(n_batch2 - 1):
        colord[i] = torch.sum(samples[:, i])
        
    sample_weight = sample_weight.cpu()
    colord = colord.cpu()
    labels = labels.cpu()
    
    bars = ax1.bar(
        labels[0:-1], sample_weight[0:-1] * batch,
        color=[cmap(i / n) for i in range(len(colord))],
    )
    
    norm = plt.Normalize(vmin=0, vmax=n - 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label(r" Electrons on band 1 ($N_{e}^{1}$)", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    def assign_colors(bars, colord, cmap, N):
        colord = np.clip(colord.numpy(), 0, N - 1)
        color_map = {i: cmap(i / (N - 1)) for i in range(N)}
        for bar, color_index in zip(bars, colord):
            bar.set_facecolor(color_map[int(color_index)])

    assign_colors(bars, colord, cmap, n)
    ax1.set_xlabel(r"Sample index $\mathbf{s}$", fontsize=14)
    ax1.set_ylabel(r"Relative Occurrence $r_\mathbf{s}$", fontsize=14)
    ax1.tick_params(axis="both", which="major", labelsize=12)

    # Second subplot - occupancy histogram
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
        raise ValueError(f"Unexpected sample_weight dimension: {sample_weight.dim()}")

    nefinal.index_add_(0, inverse_indices, weights_sum)

    x = unique_ne.cpu().numpy()
    y = nefinal.cpu().numpy()
    y_plot = y

    norm2 = plt.Normalize(vmin=0, vmax=n - 1)
    bars2 = ax2.bar(
        x, y, color=[cmap(norm2(i)) for i in x], edgecolor="black",
    )

    for i, height in enumerate(y_plot):
        label = f"{y[i]:.2f}"
        ax2.text(x[i], height + 0.01, label, ha="center", va="bottom", rotation=0)

    ax2.set_xlabel(r"Number of electrons on band 1 ($N_e^{1}$)", fontsize=14)
    ax2.set_ylabel("Total Weight", fontsize=14)
    ax2.set_xticks(range(int(x.min()), int(x.max()) + 1))

    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax2)
    cbar2.set_label(r"Number of electrons on band 1 ($N_e^{1}$)", fontsize=12)

    plt.tight_layout()
    
    filename = os.path.join(data['system_output_dir'], f"{data['identifier']}_dual_hist_{data['t']}.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# ============================= PCA ANALYSIS =============================

def plot_pca_analysis(data):
    """Plot PCA analysis of embeddings"""
    
    try:
        model = data.get('model')
        if model is None:
            print("Model not available for PCA analysis")
            return
            
        print("Starting PCA analysis...")
        
        # Get samples from the model using treesamplerumap
        try:
            samples, sample_weight = treesamplerumap(model)
        except Exception as e:
            print(f"Error with treesamplerumap: {e}")
            print("treesamplerumap function not available or failed")
            return
            
        n, batch = samples.shape
        _, embed = model.forward(samples)
        
        print(f"Samples shape: {samples.shape}, Embed shape: {embed.shape}")
        
        # Filter out full occupation states
        ns = torch.sum(samples, dim=0, dtype=torch.int8)
        idxns = torch.where(ns == n)[0]

        if len(idxns) > 0:
            embed = embed.detach().cpu().numpy()
            samples_np = samples.detach().cpu().numpy()
            idxns_np = idxns.detach().cpu().numpy() if isinstance(idxns, torch.Tensor) else idxns

            mask = np.ones(samples_np.shape[1], dtype=bool)
            mask[idxns_np] = False

            newsamples = torch.tensor(samples_np[:, mask])
            newembed = torch.tensor(embed[:, mask.T])
            newembed0 = newembed.permute(1, 0, 2)
            newembed0 = newembed0.reshape(batch - 1, -1)
            newembed1 = newembed.sum(dim=0)
            effective_batch = batch - 1
        else:
            newsamples = samples
            newembed = embed
            newembed0 = newembed.permute(1, 0, 2)
            newembed0 = newembed0.reshape(batch, -1)
            newembed1 = newembed.sum(dim=0)
            effective_batch = batch

        print(f"Filtered samples shape: {newsamples.shape}, Filtered embed shape: {newembed0.shape}")

        # Save samples info
        output_dir = data['system_output_dir']
        with open(os.path.join(output_dir, "samples.txt"), "w") as file:
            for i in range(effective_batch):
                values = [f"{v}" for v in newsamples[:, i].tolist()]
                file.write(f"{i}: {values}\n")

        # Create embedding plots
        plot_embeddings_with_color(
            newsamples, newembed0, data, name="firstemb"
        )
        plot_embeddings_with_color(
            newsamples, newembed1, data, name="summedemb"
        )
        
        print("✓ PCA analysis completed successfully")
        
    except Exception as e:
        print(f"Error in PCA analysis: {e}")
        import traceback
        traceback.print_exc()


def plot_embeddings_with_color(samples, embed, data, name="embedding"):
    """Plot embeddings with PCA, t-SNE, UMAP"""
    
    plt.close("all")
    plt.clf()
    plt.cla()

    # Create figure layout
    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.1])

    ax1 = fig.add_subplot(gs[0, 0])  # PCA
    ax2 = fig.add_subplot(gs[0, 1])  # t-SNE
    ax3 = fig.add_subplot(gs[1, 0])  # UMAP
    ax4 = fig.add_subplot(gs[1, 1])  # Spectrum
    cax = fig.add_subplot(gs[:, 2])  # Colorbar
    axs = [ax1, ax2, ax3, ax4]

    n = len(samples)
    cmap = plt.colormaps["coolwarm"].resampled(n)

    colord = torch.zeros(samples.shape[1])
    for i in range(samples.shape[1]):
        colord[i] = torch.sum(samples[:, i])
    colord = colord.cpu().numpy()

    X = embed.detach().cpu().numpy()
    norm = plt.Normalize(vmin=colord.min(), vmax=colord.max())

    print(f"X shape for PCA: {X.shape}, colord shape: {colord.shape}")

    def create_scatter(ax, X_2d, title):
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1], c=colord, cmap=cmap, norm=norm, s=50
        )
        ax.set_xlabel("First Component")
        ax.set_ylabel("Second Component")
        ax.set_title(title)
        plt.setp(ax, xticks=[], yticks=[])
        return scatter

    last_scatter = None

    try:
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        last_scatter = create_scatter(axs[0], X_pca, "PCA")
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    except Exception as e:
        print(f"PCA failed: {e}")
        axs[0].text(0.5, 0.5, f"PCA failed:\n{str(e)}", ha='center', va='center', transform=axs[0].transAxes)

    try:
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
        X_tsne = tsne.fit_transform(X)
        create_scatter(axs[1], X_tsne, "t-SNE")
        print("t-SNE completed successfully")
    except Exception as e:
        print(f"t-SNE failed: {e}")
        axs[1].text(0.5, 0.5, f"t-SNE failed:\n{str(e)}", ha='center', va='center', transform=axs[1].transAxes)

    try:
        # UMAP
        reducer = umap.UMAP(random_state=42, n_neighbors=min(15, X.shape[0]-1))
        X_umap = reducer.fit_transform(X)
        create_scatter(axs[2], X_umap, "UMAP")
        print("UMAP completed successfully")
    except Exception as e:
        print(f"UMAP failed: {e}")
        axs[2].text(0.5, 0.5, f"UMAP failed:\n{str(e)}", ha='center', va='center', transform=axs[2].transAxes)

    try:
        # PCA spectrum
        batch = data.get('batch', samples.shape[1])
        embedding_size = data.get('embedding_size', X.shape[1])
        
        n_components = min(X.shape[0], X.shape[1], 50)  # Limit to reasonable number
        if n_components < 2:
            n_components = min(X.shape[0], X.shape[1])
            
        X_pcafull = PCA(n_components=n_components)
        X_pcafull = X_pcafull.fit(X)
        
        embed_ind = np.arange(n_components)
        axs[3].scatter(embed_ind, X_pcafull.explained_variance_ratio_, color="red", s=20)
        axs[3].set_xlabel("Principal component index $i$")
        axs[3].set_ylabel("Normalized Eigenvalues of PCs")
        axs[3].set_title("PCA Spectrum")
        print(f"PCA spectrum completed with {n_components} components")
    except Exception as e:
        print(f"PCA spectrum failed: {e}")
        axs[3].text(0.5, 0.5, f"PCA spectrum failed:\n{str(e)}", ha='center', va='center', transform=axs[3].transAxes)

    # Add colorbar
    if last_scatter is not None:
        plt.colorbar(last_scatter, cax=cax, label=r"Electrons on band 1 ($N_{e}^{1}$)")
    else:
        # Create a dummy colorbar if no scatter plots worked
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, cax=cax, label=r"Electrons on band 1 ($N_{e}^{1}$)")

    plt.tight_layout()
    
    filename = os.path.join(data['system_output_dir'], f"{name}_{data['identifier']}_embed2d_{data['t']}.pdf")
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved embedding plot: {filename}")


# ============================= UTILITY FUNCTIONS =============================

def ensure_directory_exists(directory):
    """Ensure a directory exists, create it if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def safe_tensor_to_cpu(tensor):
    """Safely convert tensor to CPU, handling both tensor and non-tensor inputs"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu()
    return tensor


def format_scientific_notation(value, precision=2):
    """Format a value in scientific notation for plot labels"""
    if abs(value) < 1e-3 or abs(value) > 1e3:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def create_colormap_for_states(n_states):
    """Create a colormap suitable for visualizing quantum states"""
    if n_states <= 10:
        return plt.cm.tab10
    elif n_states <= 20:
        return plt.cm.tab20
    else:
        return plt.cm.viridis


# ============================= ERROR HANDLING =============================

class PlottingError(Exception):
    """Custom exception for plotting errors"""
    pass


def handle_plotting_error(func_name, error, data_keys=None):
    """Standard error handling for plotting functions"""
    error_msg = f"Error in {func_name}: {str(error)}"
    
    if data_keys:
        missing_keys = [key for key in data_keys if key not in globals().get('current_data', {})]
        if missing_keys:
            error_msg += f"\nMissing data keys: {missing_keys}"
    
    print(error_msg)
    
    # Optionally create an error plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, f"Plot generation failed:\n{func_name}\n\n{str(error)}", 
            ha='center', va='center', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    return fig


# ============================= MAIN INTERFACE =============================

def generate_all_plots(data_dict, output_dir=None, flags=None):
    """
    Convenience function to generate all available plots
    
    Args:
        data_dict: Dictionary containing all necessary data
        output_dir: Output directory (optional, will use data_dict['system_output_dir'] if not provided)
        flags: Dictionary of plot flags (optional, will enable all plots if not provided)
    """
    
    if output_dir:
        data_dict['system_output_dir'] = output_dir
    
    if flags is None:
        flags = {
            'plot_n': True,
            'plot_it': True,
            'plot_it_hist_weigh': True,
            'pcaplot': True,
        }
    
    try:
        create_plots(data_dict, flags)
        print("\n" + "=" * 50)
        print(f"Output directory: {data_dict['system_output_dir']}")
        print("=" * 50)
    except Exception as e:
        print(f"\nERROR: Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
