import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})


def plt_error_evolution(cost_history, save_path=None):
    """Plot the evolution of the training MSE over epochs."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_history, label="Training MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return 

def plt_nn_optimizer_comparison(results, layers, plots_dir):
    """Save the optimizer MSE comparison for any NN architecture as a PDF."""

    # Determine architecture from the weight matrices
    architecture = [layers[0][0].shape[0]]
    architecture += [W.shape[1] for W, b in layers]

    # Architecture for plot title and filename
    architecture_title = str(architecture)
    architecture_name = "-".join(map(str, architecture))

    # Create the figure
    fig, ax = plt.subplots(figsize=(9, 5))
    for optimizer, result in results.items():
        if optimizer == "sgd_inv_t" or optimizer == "rmsprop_inv_t" or optimizer == "adam_inv_t":
            if optimizer == "sgd_inv_t":
                ax.plot(result["cost_history"], label=optimizer.upper(), linestyle='--', color="blue")
            elif optimizer == "rmsprop_inv_t":
                ax.plot(result["cost_history"], label=optimizer.upper(), linestyle='--', color="red")
            elif optimizer == "adam_inv_t":
                ax.plot(result["cost_history"], label=optimizer.upper(), linestyle='--', color="green")
        else:
            if optimizer == "sgd":
                ax.plot(result["cost_history"], label=optimizer.upper(), color="blue")
            elif optimizer == "rmsprop":
                ax.plot(result["cost_history"], label=optimizer.upper(), color="red")
            elif optimizer == "adam":
                ax.plot(result["cost_history"], label=optimizer.upper(), color="green")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title(f"Optimizer comparison - NN {architecture_title}")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    # Create the output directory if necessary
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Automatically construct the PDF filename
    save_path = plots_dir / f"NN_optimizers_{architecture_name}.pdf"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

