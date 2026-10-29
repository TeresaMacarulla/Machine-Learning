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

def plt_nn_optimizer_comparison(results, layers, plots_dir):
    """Compare validation-MSE histories using the best learning rate per method."""

    # Determine the architecture
    architecture = [layers[0][0].shape[0]]
    architecture += [W.shape[1] for W, b in layers]
    architecture_name = "-".join(map(str, architecture))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "plain_gd": "purple",
        "plain_sgd": "blue",
        "rmsprop_sgd": "red",
        "adam_sgd": "green"
    }

    for method, rate_results in results.items():
        # Exclude non-finite validation results
        valid = [
            (eta, result)
            for eta, result in rate_results.items()
            if np.isfinite(result["val_cost_history"][-1])
        ]
        if not valid:
            continue

        # Select the learning rate with the lowest final validation MSE
        best_eta, best_result = min(valid, key=lambda item: item[1]["val_cost_history"][-1])
        # Remove the schedule suffix to identify the base optimizer
        base_method = method.removesuffix("_invt")
        # Solid = constant; dashed = inverse time
        linestyle = "--" if method.endswith("_invt") else "-"

        ax.plot(
            best_result["val_cost_history"],
            label=f"{method} (eta={best_eta:g})",
            color=colors[base_method],
            linestyle=linestyle
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_yscale("log")
    ax.set_title(f"Optimizer comparison - NN {architecture}")
    ax.grid(True)
    ax.legend(fontsize=10, ncol=2)

    fig.tight_layout()

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_path = (plots_dir / f"NN_optimizer_validation_{architecture_name}.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plt_nn_learning_rate_comparison(results, layers, plots_dir):
    """Save validation MSE versus learning rate for each optimizer."""

    architecture = [layers[0][0].shape[0]]
    architecture += [W.shape[1] for W, b in layers]
    architecture_name = "-".join(map(str, architecture))

    fig, ax = plt.subplots(figsize=(9, 5))

    for optimizer, rate_results in results.items():
        rates = sorted(rate_results.keys())
        val_mses = [
            rate_results[eta]["val_cost_history"][-1]
            for eta in rates
        ]
        ax.plot(
            rates, val_mses,
            marker="o",
            label=optimizer.upper()
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Validation MSE")
    ax.set_title(f"Learning-rate comparison - NN {architecture}")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=10)
    fig.tight_layout()

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_path = (
        plots_dir / f"NN_learning_rates_{architecture_name}.pdf"
    )

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plt_ols_vs_nn(x_train, y_train, x_test, y_test, x_grid, y_exact, y_ols, nn_predictions, degree, plots_dir):
    """Save a comparison of OLS and NN approximations of Runge's function."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(x_train, y_train, s=12, alpha=0.35, label="Training observations")
    ax.scatter(x_test, y_test, s=25, marker="x", label="Test observations")
    ax.plot(x_grid, y_exact, "k-", linewidth=2, label="Exact Runge function")
    ax.plot(x_grid, y_ols, "--", linewidth=2, label=f"OLS (degree {degree})")

    for name, predictions in nn_predictions.items():
        ax.plot(x_grid, predictions, linewidth=2, label=name)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("OLS and neural-network approximation of Runge's function")
    ax.grid(True)
    ax.legend(fontsize=10)
    fig.tight_layout()

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_path = plots_dir / f"Runge_OLS_d{degree}_vs_NN.pdf"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plt_activ_arch_comparison(best_configs_c, architectures_c, plots_dir):
    """Compare final validation MSE across activations and architectures."""

    architecture_names = list(architectures_c.keys())
    x_positions = np.arange(len(architecture_names))
    fig, ax = plt.subplots(figsize=(9, 5))

    for activation_name in ("sigmoid", "relu", "leaky_relu"):
        val_mses = [
            best_configs_c[arch_name][activation_name]["result"]["val_cost_history"][-1]
            for arch_name in architecture_names
        ]
        ax.plot(x_positions, val_mses, marker="o", linewidth=2, label=activation_name)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(architecture_names)
    ax.set_xlabel("Network architecture")
    ax.set_ylabel("Final validation MSE")
    ax.set_title("Activation functions and network architecture")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1))

    fig.tight_layout()

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_path = plots_dir / "NN_activation_architecture_comparison.pdf"

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plt_learning_curves(best_results, architecture, plots_dir, start_epoch=1):
    """Compare training and validation histories for the three activations."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (activation_name, configuration) in enumerate(best_results.items()):

        result = configuration["result"]
        eta = configuration["eta"]
        train_history = result["cost_history"]
        val_history = result["val_cost_history"]
        epochs = np.arange(len(train_history))
        color = f"C{i}"

        # Dashed line: training MSE
        ax.plot(epochs[start_epoch:], train_history[start_epoch:], color=color, linestyle="--", alpha=0.8, label=f"{activation_name} - train")
        # Solid line: validation MSE
        ax.plot(epochs[start_epoch:], val_history[start_epoch:], color=color, linestyle="-", linewidth=1.5, label=f"{activation_name} - validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"Training and validation MSE - NN {architecture}")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), fontsize=10)

    fig.tight_layout()

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    architecture_name = "-".join(map(str, architecture))
    save_path = plots_dir / f"NN_C_learning_curves_{architecture_name}.pdf"

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

