# good data set from https://github.com/Pa-Sto/kilterboardie/tree/main

from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split

from utilities import (
    load_kilter_dataset_sparse,
    create_layers_batch,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    leaky_relu,
    leaky_relu_derivative,
    softmax,
    train_neural_network
    )

from plot_generator import (
    plt_learning_curves,
    plt_activation_architecture_heatmaps
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR_1 = PLOTS_DIR / "part_1"
PLOTS_DIR_2 = PLOTS_DIR / "part_2"
PLOTS_DIR_3 = PLOTS_DIR / "part_3"

# ============================================================
# Load and prepare the dataset
# ============================================================

# Load and prepare Kilter Board dataset
DATA_DIR = PROJECT_ROOT / "data"

print("-------------------------------------------------------------------------------")
print(" LOADING KILTER-BOARD DATA ")
print("-------------------------------------------------------------------------------")

X_kilter, y_kilter = load_kilter_dataset_sparse(DATA_DIR)

print("\nDataset summary")
print("X shape:", X_kilter.shape)
print("y shape:", y_kilter.shape)

# Inspect the number of samples per grade
grades, counts = np.unique(y_kilter + 3, return_counts=True)

"""
for grade, count in zip(grades, counts):
    print(f"V{grade}: {count} climbing problems")
"""
    
# First split: 70% training, 30% validation + test
X_kilter_train, X_temp, y_kilter_train, y_temp = train_test_split(X_kilter, y_kilter, test_size=0.30, random_state=2026, stratify=y_kilter)

# Second split: divide the remaining 30% equally into 15% validation and 15% test
X_kilter_val, X_kilter_test, y_kilter_val, y_kilter_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=2026, stratify=y_temp)

print("\nDataset split")
print("Training:", X_kilter_train.shape, y_kilter_train.shape)
print("Validation:", X_kilter_val.shape, y_kilter_val.shape)
print("Test:", X_kilter_test.shape, y_kilter_test.shape)
print()

Y_kilter_train = np.eye(11, dtype=np.float32)[y_kilter_train]
Y_kilter_val = np.eye(11, dtype=np.float32)[y_kilter_val]

print("-------------------------------------------------------------------------------")
print(" OPTION MENU ")
print("-------------------------------------------------------------------------------")
print("1: Compare Multinominal Logistic Regression vs. two NN architectures.")
print("2: Explore different Learning-rates for two NN architectures.")
print("3: Explore different activation functions for two NN architectures.")
print("-------------------------------------------------------------------------------")
number = float(input("Choose a number: "))
print()

if number == 1:

    # ============================================================
    # Apply Multinomial logistic-regression baseline and NNs
    # ============================================================

    architectures = {
        "Logistic reg." : [4761, 11],
        "NN 1x100": [4761, 100, 11],
        "NN 2x100": [4761, 100, 100, 11]
    }

    results = {}
    for network_name, architecture in architectures.items():

        n_hidden = len(architecture) - 2
        # He initialization for ReLU hidden layers
        layers = create_layers_batch(network_input_size=architecture[0], layer_output_sizes=architecture[1:], seed=2026, hidden_activation="relu")
        # ReLU hidden layers and softmax output
        activations = [relu] * n_hidden + [softmax]
        # No separate softmax derivative is needed
        activation_ders = [relu_derivative] * n_hidden + [None]

        # Train the network
        result = train_neural_network(X_kilter_train, Y_kilter_train, layers, activations, activation_ders, eta0=0.001,
                                    optimizer="adam", batch_size=128, n_epochs=15, schedule="constant", seed=2026,
                                    X_val=X_kilter_val, y_val=Y_kilter_val)
        results[network_name] = result

    for network_name, architecture in architectures.items():
        result = results[network_name]
        print(f"\n{network_name}")
        print(f"Architecture: {architecture}")
        print(f"Training cross-entropy: "
            f"{result['cost_history'][-1]:.6f}")
        print(f"Validation cross-entropy: "
            f"{result['val_cost_history'][-1]:.6f}")
        print(f"Training accuracy: "
            f"{100 * result['accuracy_history'][-1]:.2f}%")
        print(f"Validation accuracy: "
            f"{100 * result['val_accuracy_history'][-1]:.2f}%")

    plt_learning_curves(results, PLOTS_DIR_1, start_epoch=1)

    print("\n VALIDATION DIAGNOSTICS")

    for model_name, result in results.items():
        best_ce_epoch = np.argmin(result["val_cost_history"])
        best_acc_epoch = np.argmax(result["val_accuracy_history"])
        print(
            f"{model_name}: "
            f"Best CE epoch={best_ce_epoch}, "
            f"Best accuracy epoch={best_acc_epoch}"
        )

    print("-------------------------------------------------------------------------------")
    print(" Find the resulting plots in P3/plots/part_1 ")
    print("-------------------------------------------------------------------------------")

if number == 2:

    # ============================================================
    # Learning-rate investigation for NNs
    # ============================================================

    architectures = {
        "NN 1x100": [4761, 100, 11],
        "NN 2x100": [4761, 100, 100, 11]
    }
        
    learning_rates = [3e-4, 1e-3, 3e-3]
    learning_rate_results = {}

    for network_name, architecture in architectures.items():

        print(f"CALCULATING {network_name}")
        learning_rate_results[network_name] = {}
        n_hidden = len(architecture) - 2
        activations = [relu] * n_hidden + [softmax]
        activation_ders = [relu_derivative] * n_hidden + [None]

        for eta in learning_rates:
            print(f"Learning rate: {eta}")
            # Same initialization for a controlled comparison
            layers = create_layers_batch(architecture[0], architecture[1:], seed=2026, hidden_activation="relu")
            result = train_neural_network(X_kilter_train, Y_kilter_train, layers, activations, activation_ders, eta0=eta,
                                        optimizer="adam", batch_size=128, n_epochs=15, schedule="constant",
                                        seed=2026, X_val=X_kilter_val, y_val=Y_kilter_val)
            learning_rate_results[network_name][eta] = result

    # Print all six configurations
    print("\nLEARNING-RATE COMPARISON")
    print(
        f"{'Network':<8} {'Eta':>10} "
        f"{'Train CE':>12} {'Val CE':>12} "
        f"{'Train Acc':>12} {'Val Acc':>12}"
    )
    print("-" * 74)

    for network_name, rate_results in learning_rate_results.items():
        for eta, result in sorted(rate_results.items()):
            print(
                f"{network_name:<8} "
                f"{eta:>10.4g} "
                f"{result['cost_history'][-1]:>12.5f} "
                f"{result['val_cost_history'][-1]:>12.5f} "
                f"{100 * result['accuracy_history'][-1]:>11.2f}% "
                f"{100 * result['val_accuracy_history'][-1]:>11.2f}%"
                )

    best_rates_e = {}
    for network_name, rate_results in learning_rate_results.items():
        best_eta, best_result = min(rate_results.items(), key=lambda item: (-item[1]["val_accuracy_history"][-1], item[1]["val_cost_history"][-1]))
        best_rates_e[network_name] = {
            "eta": best_eta,
            "result": best_result
        }
        print(
            f"\n{network_name} - Best learning rate: {best_eta:g}"
            f"\nValidation accuracy: "
            f"{100 * best_result['val_accuracy_history'][-1]:.2f}%"
            f"\nValidation cross-entropy: "
            f"{best_result['val_cost_history'][-1]:.6f}"
        )

if number == 3:
            
# ============================================================
# Activation functions
# ============================================================

    architectures = {
        "NN 1x100": [4761, 100, 11],
        "NN 2x100": [4761, 100, 100, 11]
    }

    activation_options = {
        "sigmoid": (sigmoid, sigmoid_derivative),
        "relu": (relu, relu_derivative),
        "leaky_relu": (leaky_relu, leaky_relu_derivative)
    }

    activation_arch_results = {}

    for arch_name, architecture in architectures.items():
        print(f"CALCULATING {arch_name}")
        activation_arch_results[arch_name] = {}
        n_hidden = len(architecture) - 2

        for activation_name, (activation, derivative) in activation_options.items():

            # Initialize for the chosen hidden activation
            print(f"Activation function: {activation_name}")
            activations = [activation] * n_hidden + [softmax]
            activation_ders = [derivative] * n_hidden + [None]
            layers = create_layers_batch(network_input_size=architecture[0], layer_output_sizes=architecture[1:], seed=2026, hidden_activation=activation_name)

            # Select the best learning rate from the previous part: 0.001 for 1x100 and 0.0003 for 2x100
            if arch_name == "NN 1x100":
                result = train_neural_network(X_kilter_train, Y_kilter_train, layers, activations, activation_ders, eta0=0.001, optimizer="adam",
                                            batch_size=128, n_epochs=15, schedule="constant", seed=2026, X_val=X_kilter_val, y_val=Y_kilter_val)
            if arch_name == "NN 2x100":
                result = train_neural_network(X_kilter_train, Y_kilter_train, layers, activations, activation_ders, eta0=0.0003, optimizer="adam",
                                            batch_size=128, n_epochs=15, schedule="constant", seed=2026, X_val=X_kilter_val, y_val=Y_kilter_val)
            activation_arch_results[arch_name][activation_name] = result

    # Print results
    print("\n ACTIVATION FUNCTION / ARCHITECTURE COMPARISON")
    print(
        f"{'Architecture':<13} {'Activation':<12} "
        f"{'Train CE':>10} {'Val CE':>10} "
        f"{'Train Acc':>11} {'Val Acc':>11} "
        f"{'Best CE epoch':>14}"
    )
    print("-" * 89)

    for arch_name, activation_results in activation_arch_results.items():

        for activation_name, result in activation_results.items():
            train_ce = result["cost_history"][-1]
            val_ce = result["val_cost_history"][-1]
            train_acc = result["accuracy_history"][-1]
            val_acc = result["val_accuracy_history"][-1]
            best_ce_epoch = int(np.argmin(result["val_cost_history"]))
            print(
                f"{arch_name:<13} "
                f"{activation_name:<12} "
                f"{train_ce:>10.5f} "
                f"{val_ce:>10.5f} "
                f"{100 * train_acc:>10.2f}% "
                f"{100 * val_acc:>10.2f}% "
                f"{best_ce_epoch:>14d}"
            )

    # Select best configurations
    best_configs_e_b = {}
    print("\nBEST ACTIVATION FOR EACH ARCHITECTURE")

    for arch_name, activation_results in activation_arch_results.items():

        best_activation, best_result = min(
            activation_results.items(),
            key=lambda item: (
                -item[1]["val_accuracy_history"][-1],
                item[1]["val_cost_history"][-1]
            )
        )
        best_configs_e_b[arch_name] = {
            "architecture": architectures[arch_name],
            "activation": best_activation,
            "result": best_result
        }
        print(
            f"{arch_name}: "
            f"Activation={best_activation}, "
            f"Val accuracy={100 * best_result['val_accuracy_history'][-1]:.2f}%, "
            f"Val CE={best_result['val_cost_history'][-1]:.5f}"
        )

    plt_activation_architecture_heatmaps(activation_arch_results, PLOTS_DIR_3)

    print("-------------------------------------------------------------------------------")
    print(" Find the resulting plots in P3/plots/part_3 ")
    print("-------------------------------------------------------------------------------")







    






