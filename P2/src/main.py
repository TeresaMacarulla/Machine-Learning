import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from pathlib import Path
import jax
import jax.numpy as jnp

from utilities import (
    artificial_data,
    create_layers_batch,
    feed_forward_saver_batch,
    identity,
    sigmoid,
    identity_derivative,
    sigmoid_derivative,
    mean_squared_error,
    mse_output_derivative,
    backpropagation_batch,
    gradient_check,
    optimize_regression_stochastic)

from plot_generator import (
    plt_error_evolution
)

# Path to the root P2 folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the plots folder
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Part b): Writing my own neural network code
# ============================================================

# Data generation 
baseline_n=100
seed=2026
baseline_sigma=0.1
x, y = artificial_data( n=baseline_n, sigma=baseline_sigma, seed=seed)

# Prepare the data for the neural network
X_nn = x.reshape(-1, 1)
y_nn = y.reshape(-1, 1)

architecture_1 = [1, 50, 1]
architecture_2 = [1, 100, 100, 1]

layers_1 = create_layers_batch(architecture_1[0], architecture_1[1:])
layers_2 = create_layers_batch(architecture_2[0], architecture_2[1:])

# Define the activation functions
activations_1 = [sigmoid, identity]
activations_2 = [sigmoid, sigmoid, identity]

# Calculate predictions
layer_in_1, zs_1, y_pred_1 = feed_forward_saver_batch(X_nn, layers_1, activations_1)
layer_in_2, zs_2, y_pred_2 = feed_forward_saver_batch(X_nn, layers_2, activations_2)

# Activation derivatives corresponding to each layer
activation_ders_1 = [sigmoid_derivative, identity_derivative]
activation_ders_2 = [sigmoid_derivative, sigmoid_derivative, identity_derivative]

# Network 1: [1, 50, 1]
print("Network 1:")
check_1 = gradient_check(X_nn, y_nn, layers_1, activations_1, activation_ders_1)
print()

# Network 2: [1, 100, 100, 1]
print("Network 2:")
check_2 = gradient_check(X_nn, y_nn, layers_2, activations_2, activation_ders_2)

assert check_1 and check_2, "Gradient verification failed."

# Train Network 1 using mini-batch SGD
result_1 = optimize_regression_stochastic(X_nn, y_nn, layers_1, activations_1, activation_ders_1,
                                          eta0=0.01, batch_size=32, n_epochs=300, schedule="constant",
                                          seed=2026)

# Retrieve the results
trained_layers_1 = result_1["layers"]
cost_history_1 = result_1["cost_history"]

print("\nSGD training results - Network 1")
print(f"Initial MSE: {cost_history_1[0]:.6f}")
print(f"Final MSE:   {cost_history_1[-1]:.6f}")
print(f"Updates:     {result_1['updates']}")

plt_error_evolution(cost_history=cost_history_1, save_path=PLOTS_DIR / "MSE_evolution.pdf") 