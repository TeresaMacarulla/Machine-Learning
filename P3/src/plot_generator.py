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

def plt_learning_curves(results, plots_dir, start_epoch=1):
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

    fig.savefig(plots_dir / "Learning_curves.pdf",  bbox_inches="tight")
    plt.close(fig)

def plt_activation_architecture_heatmaps(results, plots_dir):
    """Compare activation functions and architectures."""

    architectures = list(results.keys())
    activations = ["sigmoid", "relu", "leaky_relu"]
    val_accuracy = np.array([
        [
            100 * results[arch][act]["val_accuracy_history"][-1]
            for act in activations
        ]
        for arch in architectures
    ])

    val_ce = np.array([
        [
            results[arch][act]["val_cost_history"][-1]
            for act in activations
        ]
        for arch in architectures
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, values, title, fmt in [
        (axes[0], val_accuracy, "Validation accuracy (%)", ".2f"),
        (axes[1], val_ce, "Validation cross-entropy", ".3f")
    ]:

        im = ax.imshow(values, cmap="Blues", aspect="auto")

        ax.set_xticks(range(len(activations)))
        ax.set_xticklabels(
            ["Sigmoid", "ReLU", "Leaky ReLU"]
        )

        ax.set_yticks(range(len(architectures)))
        ax.set_yticklabels(architectures)

        ax.set_xlabel("Hidden activation")
        ax.set_ylabel("Architecture")
        ax.set_title(title)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):

                value = values[i, j]

                ax.text(
                    j, i,
                    f"{value:{fmt}}",
                    ha="center",
                    va="center",
                    color="white" if im.norm(value) > 0.55 else "black",
                    fontsize=11
                )

        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.tight_layout()

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        plots_dir / "Activation_architecture_heatmaps.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

