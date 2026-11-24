# data set from: https://huggingface.co/datasets/Ken1m/KilterBoard
# good data set from https://github.com/Pa-Sto/kilterboardie/tree/main

from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split

from utilities import (
    load_kilter_dataset,
    create_layers_batch,
    relu,
    relu_derivative,
    softmax,
    train_neural_network
    )

from plot_generator import (
    plt_e_learning_curves
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"

# ============================================================
# Load and prepare the dataset
# ============================================================

# Load and prepare Kilter Board dataset
DATA_DIR = PROJECT_ROOT / "data"

X_kilter, y_kilter = load_kilter_dataset(DATA_DIR)

print("\nDataset summary")
print("X shape:", X_kilter.shape)
print("y shape:", y_kilter.shape)

# Inspect the number of samples per grade
grades, counts = np.unique(y_kilter + 3, return_counts=True)

for grade, count in zip(grades, counts):
    print(f"V{grade}: {count} climbing problems")

# First split: 70% training, 30% validation + test
X_kilter_train, X_temp, y_kilter_train, y_temp = train_test_split(X_kilter, y_kilter, test_size=0.30, random_state=2026, stratify=y_kilter)

# Second split: divide the remaining 30% equally into 15% validation and 15% test
X_kilter_val, X_kilter_test, y_kilter_val, y_kilter_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=2026, stratify=y_temp)

print("\nDataset split")
print("Training:", X_kilter_train.shape, y_kilter_train.shape)
print("Validation:", X_kilter_val.shape, y_kilter_val.shape)
print("Test:", X_kilter_test.shape, y_kilter_test.shape)


# ============================================================
# Apply Multinomial logistic-regression baseline and NNs
# ============================================================

architectures = {
    "Logistic reg." : [4761, 11],
    "NN 1": [4761, 100, 11],
    "NN 2": [4761, 100, 100, 11]
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
    result = train_neural_network(X_kilter_train, y_kilter_train, layers, activations, activation_ders, eta0=0.001,
                                  optimizer="adam", batch_size=128, n_epochs=15, schedule="constant", seed=2026,
                                  X_val=X_kilter_val, y_val=y_kilter_val, task="classification")
    results[network_name] = result

for network_name, architecture in architectures.items():
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

plt_e_learning_curves(results, PLOTS_DIR, start_epoch=1)

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
print(" Find the resulting plots in P3/plots ")
print("-------------------------------------------------------------------------------")








    






