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
    optimize_regression_stochastic,
    fit_input_scaler, 
    scale_inputs)

from plot_generator import (
    plt_error_evolution,
    plt_nn_optimizer_comparison
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

# Use the 80/20% split for the final comparison
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2026)

# Raw neural-network inputs
X_train_raw = x_train.reshape(-1, 1)
X_test_raw = x_test.reshape(-1, 1)

# Fit the scaler ONLY on the training inputs
input_means, input_stds = fit_input_scaler(X_train_raw)

# Scale training and test inputs with the same parameters
X_train_nn = scale_inputs(X_train_raw, input_means, input_stds)
X_test_nn = scale_inputs(X_test_raw, input_means, input_stds)

# Targets remain in their original units
y_train_nn = y_train.reshape(-1, 1)
y_test_nn = y_test.reshape(-1, 1)

architecture_1 = [1, 50, 1]
architecture_2 = [1, 100, 100, 1]

layers_1 = create_layers_batch(architecture_1[0], architecture_1[1:])
layers_2 = create_layers_batch(architecture_2[0], architecture_2[1:])

# Define the activation functions
activations_1 = [sigmoid, identity]
activations_2 = [sigmoid, sigmoid, identity]

# Calculate predictions
layer_in_1, zs_1, y_pred_1 = feed_forward_saver_batch(X_train_nn, layers_1, activations_1)
layer_in_2, zs_2, y_pred_2 = feed_forward_saver_batch(X_train_nn, layers_2, activations_2)

# Activation derivatives corresponding to each layer
activation_ders_1 = [sigmoid_derivative, identity_derivative]
activation_ders_2 = [sigmoid_derivative, sigmoid_derivative, identity_derivative]

# Network 1: [1, 50, 1]
print("Network 1:")
check_1 = gradient_check(X_train_nn, y_train_nn, layers_1, activations_1, activation_ders_1)
print()

# Network 2: [1, 100, 100, 1]
print("Network 2:")
check_2 = gradient_check(X_train_nn, y_train_nn, layers_2, activations_2, activation_ders_2)
print()

assert check_1 and check_2, "Gradient verification failed."

# Train the Networks using SGD, RMSprop or adam
optimizers = ["sgd", "sgd_inv_t", "rmsprop", "rmsprop_inv_t", "adam", "adam_inv_t"]
results_1 = {}
results_2 = {}

# Constant learning-rate schedule
for optimizer in optimizers:
    if optimizer == "sgd" or optimizer == "rmsprop" or optimizer == "adam":
        results_1[optimizer] = optimize_regression_stochastic(X_train_nn, y_train_nn, layers_1, activations_1, activation_ders_1,
                                                        eta0=0.01, optimizer=optimizer, batch_size=32, n_epochs=300,
                                                        schedule="constant", seed=2026)
        results_2[optimizer] = optimize_regression_stochastic(X_train_nn, y_train_nn, layers_2, activations_2, activation_ders_2,
                                                        eta0=0.01, optimizer=optimizer, batch_size=32, n_epochs=300,
                                                        schedule="constant", seed=2026)
    if optimizer == "sgd_inv_t" or optimizer == "rmsprop_inv_t" or optimizer == "adam_inv_t":
        results_1[optimizer] = optimize_regression_stochastic(X_train_nn, y_train_nn, layers_1, activations_1, activation_ders_1,
                                                        eta0=0.01, optimizer=optimizer, batch_size=32, n_epochs=300,
                                                        schedule="inverse_time", seed=2026)
        results_2[optimizer] = optimize_regression_stochastic(X_train_nn, y_train_nn, layers_2, activations_2, activation_ders_2,
                                                        eta0=0.01, optimizer=optimizer, batch_size=32, n_epochs=300,
                                                        schedule="inverse_time", seed=2026)

    # Retrieve the results
    history_1 = results_1[optimizer]["cost_history"]
    history_2 = results_2[optimizer]["cost_history"]
    print(f"Network 1:")
    print(f"Optimizer: {optimizer}")
    print(f"Initial MSE: {history_1[0]:.6f}")
    print(f"Final MSE:   {history_1[-1]:.6f}")
    print(f"Updates:     {results_1[optimizer]['updates']}")
    print()
    print(f"Network 2:")
    print(f"Optimizer: {optimizer}")
    print(f"Initial MSE: {history_2[0]:.6f}")
    print(f"Final MSE:   {history_2[-1]:.6f}")
    print(f"Updates:     {results_2[optimizer]['updates']}")
    print()

plt_nn_optimizer_comparison(results_1, layers_1, PLOTS_DIR)
plt_nn_optimizer_comparison(results_2, layers_2, PLOTS_DIR)
