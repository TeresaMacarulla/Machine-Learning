import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

def plt_e_learning_curves(results, plots_dir, start_epoch=1):
    """Compare training and validation histories for MNIST classifiers."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, (model_name, result) in enumerate(results.items()):

        color = f"C{i}"
        epochs = np.arange(len(result["cost_history"]))
        selected_epochs = epochs[start_epoch:]

        # Cross-entropy
        axes[0].plot(selected_epochs, result["cost_history"][start_epoch:], color=color, linestyle="--", label=f"{model_name} - train")
        axes[0].plot(selected_epochs, result["val_cost_history"][start_epoch:], color=color, linestyle="-", label=f"{model_name} - validation")

        # Classification accuracy
        axes[1].plot(selected_epochs, result["accuracy_history"][start_epoch:], color=color, linestyle="--", label=f"{model_name} - train")
        axes[1].plot(selected_epochs, result["val_accuracy_history"][start_epoch:], color=color, linestyle="-", label=f"{model_name} - validation")

    axes[0].set_title("Cross-entropy evolution")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].set_yscale("log")
    axes[1].set_title("Classification accuracy evolution")
    axes[1].set_ylabel("Accuracy")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(plots_dir / "MNIST_E_learning_curves.pdf",  bbox_inches="tight")
    plt.close(fig)

