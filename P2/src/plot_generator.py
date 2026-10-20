import numpy as np
import matplotlib.pyplot as plt

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

    return fig, ax
