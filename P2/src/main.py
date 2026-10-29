import numpy as np
import argparse
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
    relu,
    relu_derivative,
    leaky_relu,
    leaky_relu_derivative,
    prepare_data,
    mean_squared_error,
    gradient_check,
    optimize_regression_stochastic,
    fit_input_scaler, 
    scale_inputs,
    fit_and_evaluate,
    r2_score,
    runge, 
    design_matrix, 
    scale_design_matrix, 
    predict)

from plot_generator import (
    plt_nn_optimizer_comparison,
    plt_nn_learning_rate_comparison,
    plt_ols_vs_nn,
    plt_activ_arch_comparison,
    plt_learning_curves
)

# Path to the root P2 folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the plots folder
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR_1 = PLOTS_DIR/ "part_1"
PLOTS_DIR_2 = PLOTS_DIR/ "part_2"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR_1.mkdir(parents=True, exist_ok=True)
PLOTS_DIR_2.mkdir(parents=True, exist_ok=True)

# ================================================================================
# Part a): Prepare the data
# ================================================================================

# Data generation 
baseline_n=200
seed=2026
baseline_sigma=0.1
x, y = artificial_data( n=baseline_n, sigma=baseline_sigma, seed=seed)

# Use the 80/20% split for the final comparison: 80% will be training+validation and 20% will be test
# Training updates the weights; validation selects hyperparameters; testing gives an independent final evaluation.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2026)

# Split the existing 80% set into fitting and validation subsets
x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=2027)

# Prepare input matrices
X_fit_raw = x_fit.reshape(-1, 1)
X_val_raw = x_val.reshape(-1, 1)

# Fit scaler on the fitting subset ONLY
fit_means, fit_stds = fit_input_scaler(X_fit_raw)

X_fit_nn = scale_inputs(X_fit_raw, fit_means, fit_stds)
X_val_nn = scale_inputs(X_val_raw, fit_means, fit_stds)

# Prepare targets
y_fit_nn = y_fit.reshape(-1, 1)
y_val_nn = y_val.reshape(-1, 1)

if __name__ == "__main__":

    print(" OPTION MENU ")
    print("-------------------------------------------------------------------------------")
    print("1: Compare two NN architectures and implement hyperparameter-selection.")
    print("2: Explore different activation functions and NN architectures.")
    number = float(input("Choose a number: "))
    print()

    if number == 1:

    # ================================================================================
    # Part b): Write my own neural network code and implement hyperparameter-selection
    # ================================================================================

        architecture_1 = [1, 50, 1]
        architecture_2 = [1, 100, 100, 1]

        layers_1 = create_layers_batch(architecture_1[0], architecture_1[1:])
        layers_2 = create_layers_batch(architecture_2[0], architecture_2[1:])

        # Define the activation functions
        activations_1 = [sigmoid, identity]
        activations_2 = [sigmoid, sigmoid, identity]

        # Calculate predictions
        layer_in_1, zs_1, y_pred_1 = feed_forward_saver_batch(X_fit_nn, layers_1, activations_1)
        layer_in_2, zs_2, y_pred_2 = feed_forward_saver_batch(X_fit_nn, layers_2, activations_2)

        # Activation derivatives corresponding to each layer
        activation_ders_1 = [sigmoid_derivative, identity_derivative]
        activation_ders_2 = [sigmoid_derivative, sigmoid_derivative, identity_derivative]

        # Network 1: [1, 50, 1]
        print()
        print("Network 1:")
        check_1 = gradient_check(X_fit_nn, y_fit_nn, layers_1, activations_1, activation_ders_1)
        print()

        # Network 2: [1, 100, 100, 1]
        print("Network 2:")
        check_2 = gradient_check(X_fit_nn, y_fit_nn, layers_2, activations_2, activation_ders_2)
        print()

        assert check_1 and check_2, "Gradient verification failed."

        # Train the Networks using GD, SGD, RMSprop, adam,constant learning rate, inverse time learning rate...
        learning_rates = {
            "plain_gd": [0.01, 0.05, 0.1],
            "plain_gd_invt": [0.01, 0.05, 0.1],
            "plain_sgd": [0.01, 0.05, 0.1],
            "plain_sgd_invt": [0.01, 0.05, 0.1],
            "rmsprop_sgd": [1e-4, 1e-3, 1e-2],
            "rmsprop_sgd_invt": [1e-4, 1e-3, 1e-2],
            "adam_sgd": [1e-4, 1e-3, 1e-2],
            "adam_sgd_invt": [1e-4, 1e-3, 1e-2]
        }
        experiment_settings = {
            # method name: (optimizer, schedule, full_batch)
            "plain_gd":          ("sgd",     "constant",     True),
            "plain_gd_invt":     ("sgd",     "inverse_time", True),
            "plain_sgd":         ("sgd",     "constant",     False),
            "plain_sgd_invt":    ("sgd",     "inverse_time", False),
            "rmsprop_sgd":       ("rmsprop", "constant",     False),
            "rmsprop_sgd_invt":  ("rmsprop", "inverse_time", False),
            "adam_sgd":          ("adam",    "constant",     False),
            "adam_sgd_invt":     ("adam",    "inverse_time", False)
        }
        networks = {
            "Network 1": (layers_1, activations_1, activation_ders_1),
            "Network 2": (layers_2, activations_2, activation_ders_2)
        }
        tuning_results = {}

        for network_name, (layers, acts, ders) in networks.items():
            tuning_results[network_name] = {}

            for method, rates in learning_rates.items():
                tuning_results[network_name][method] = {}
                # Identify the optimizer, schedule and batch regime
                optimizer, schedule, full_batch = experiment_settings[method]
                batch_size = len(X_fit_nn) if full_batch else 32

                for eta in rates:
                    result = optimize_regression_stochastic(X_fit_nn, y_fit_nn, layers, acts, ders, eta0=eta, optimizer=optimizer,
                                                            batch_size=batch_size, n_epochs=300, schedule=schedule, seed=2026,
                                                            X_val=X_val_nn, y_val=y_val_nn)
                    tuning_results[network_name][method][eta] = result
                    train_mse = result["cost_history"][-1]
                    val_mse = result["val_cost_history"][-1]
                    print(
                        f"{network_name} | {method:18s} "
                        f"| eta={eta:.4g} "
                        f"| Train MSE={train_mse:.6f} "
                        f"| Val MSE={val_mse:.6f}"
                    )

        best_configs = {}
        for network_name, results in tuning_results.items():
            candidates = []

            for method, rate_results in results.items():
                for eta, result in rate_results.items():
                    val_mse = result["val_cost_history"][-1]
                    if np.isfinite(val_mse):
                        candidates.append((val_mse, method, eta))

            best_val_mse, best_method, best_eta = min(candidates)
            best_configs[network_name] = {"method": best_method, "eta": best_eta}

            print(f"\nBest configuration for {network_name}:")
            print(f"  Optimizer: {best_method}")
            print(f"  Learning rate: {best_eta}")
            print(f"  Validation MSE: {best_val_mse:.6f}")

        plt_nn_learning_rate_comparison(tuning_results["Network 1"], layers_1, PLOTS_DIR_1)
        plt_nn_learning_rate_comparison(tuning_results["Network 2"], layers_2, PLOTS_DIR_1)

        plt_nn_optimizer_comparison(tuning_results["Network 1"], layers_1, PLOTS_DIR_1)
        plt_nn_optimizer_comparison(tuning_results["Network 2"], layers_2, PLOTS_DIR_1)

    # ============================================================
    # Part b): Final NN training and evaluation
    # ============================================================

        # Recombine fitting and validation data through the original
        # 80% training set
        (X_train_nn, X_test_nn, y_train_nn, y_test_nn, train_means, train_stds) = prepare_data(x_train, x_test, y_train, y_test)

        # Selected best configurations
        final_nn_results = {}
        for network_name, (layers, acts, ders) in networks.items():

            # Selected hyperparameters from the validation experiment
            selected = best_configs[network_name]
            optimizer, schedule, full_batch = experiment_settings[selected["method"]]
            batch_size = len(X_train_nn) if full_batch else 32

            # Retrain using all 160 training observations
            result = optimize_regression_stochastic(X_train_nn, y_train_nn, layers, acts, ders, eta0=selected["eta"], optimizer=optimizer, 
                                                    batch_size=batch_size, n_epochs=300, schedule=schedule, seed=2026)
            trained_layers = result["layers"]
            _, _, y_train_pred = feed_forward_saver_batch(X_train_nn, trained_layers, acts)
            _, _, y_test_pred = feed_forward_saver_batch(X_test_nn, trained_layers, acts)

            final_nn_results[network_name] = {
                "layers": trained_layers,
                "method": selected["method"],
                "eta": selected["eta"],
                "y_train_pred": y_train_pred,
                "y_test_pred": y_test_pred,
                "mse_train": mean_squared_error(y_train_nn, y_train_pred),
                "mse_test": mean_squared_error(y_test_nn, y_test_pred)
            }

            print(f"\n{network_name}")
            print(f"Method: {selected['method']}")
            print(f"Learning rate: {selected['eta']}")
            print(f"Training MSE: {final_nn_results[network_name]['mse_train']:.6f}")
            print(f"Test MSE: {final_nn_results[network_name]['mse_test']:.6f}")

    # ============================================================
    # Part b): OLS benchmark from Project 1
    # ============================================================

        best_degree = 14
        ols_result = fit_and_evaluate(x_train, x_test, y_train, y_test, degree=best_degree, method="ols")

        print("\nOLS benchmark (Project 1)")
        print(f"Polynomial degree: {best_degree}")
        print(f"Training MSE: {ols_result['mse_train']:.6f}")
        print(f"Test MSE: {ols_result['mse_test']:.6f}")

        print("\nFINAL MODEL COMPARISON")
        print(f"{'Model':<26} {'Train MSE':>12} {'Test MSE':>12} {'Test R²':>12}")
        print("-" * 66)

        print(
            f"{'OLS (degree 14)':<26} "
            f"{ols_result['mse_train']:>12.6f} "
            f"{ols_result['mse_test']:>12.6f} "
            f"{r2_score(y_test, ols_result['y_test_pred']):>12.6f}"
        )

        for network_name, result in final_nn_results.items():
            # Flatten predictions to match the 1D target arrays
            y_test_pred = result["y_test_pred"].ravel()
            print(
                f"{network_name:<26} "
                f"{result['mse_train']:>12.6f} "
                f"{result['mse_test']:>12.6f} "
                f"{r2_score(y_test, y_test_pred):>12.6f}"
            )

        # Dense grid for plotting smooth fitted functions
        x_grid = np.linspace(-1.0, 1.0, 500)
        y_exact = runge(x_grid)

        # OLS predictions using degree-14 polynomial features
        X_grid_ols = design_matrix(x_grid, best_degree)
        X_grid_ols = scale_design_matrix(X_grid_ols, ols_result["feature_means"], ols_result["feature_stds"])
        y_ols_grid = predict(X_grid_ols, ols_result["theta"])

        # NN predictions using the training-set input scaler
        X_grid_nn = scale_inputs(x_grid.reshape(-1, 1), train_means, train_stds)

        nn_predictions = {}
        for network_name, result in final_nn_results.items():

            acts = networks[network_name][1]
            _, _, prediction = feed_forward_saver_batch(X_grid_nn, result["layers"], acts)
            nn_predictions[network_name] = prediction.ravel()

        # Save the final comparison figure
        plt_ols_vs_nn(x_train, y_train, x_test, y_test, x_grid, y_exact, y_ols_grid, nn_predictions, best_degree, PLOTS_DIR_1)

        print()
        print("-------------------------------------------------------------------------------")
        print(" Find the resulting plots in P2/plots/part_1 ")
        print("-------------------------------------------------------------------------------")

    if number == 2:
    # ============================================================
    # Part c): Compare activation functions, depth and width
    # ============================================================

        # Activation functions and their derivatives
        activation_options = {
            "sigmoid": (sigmoid, sigmoid_derivative),
            "relu": (relu, relu_derivative),
            "leaky_relu": (leaky_relu, leaky_relu_derivative)
        }
        architectures_c = {
            "1x50":       [1, 50, 1],
            "1x100":      [1, 100, 1],
            "2x50":       [1, 50, 50, 1],
            "2x100":      [1, 100, 100, 1]
        }

        # Starting learning-rate grid for Adam
        learning_rates_c = [1e-4, 1e-3, 1e-2]
        results_c = {}
        best_configs_c = {}

        for arch_name, architecture in architectures_c.items():

            results_c[arch_name] = {}
            best_configs_c[arch_name] = {}
            # Number of hidden layers
            n_hidden = len(architecture) - 2

            for activation_name, (activation, derivative) in activation_options.items():

                # Initialize parameters for the selected activation
                layers = create_layers_batch(
                    network_input_size=architecture[0],
                    layer_output_sizes=architecture[1:],
                    seed=2026,
                    hidden_activation=activation_name
                )

                # Use the selected activation for all hidden layers and identity for the output layer
                activations = [activation] * n_hidden + [identity]
                activation_ders = [derivative] * n_hidden + [identity_derivative]
                results_c[arch_name][activation_name] = {}

                for eta in learning_rates_c:
                    result = optimize_regression_stochastic(X_fit_nn, y_fit_nn, layers, activations, activation_ders,
                                                            eta0=eta, optimizer="adam", batch_size=32, n_epochs=300,
                                                            schedule="constant", seed=2026, X_val=X_val_nn, y_val=y_val_nn)
                    results_c[arch_name][activation_name][eta] = result

                # Select learning rate using final validation MSE
                best_eta = min(learning_rates_c,
                    key=lambda eta: results_c[arch_name][activation_name][eta]["val_cost_history"][-1])
                best_result = results_c[arch_name][activation_name][best_eta]
                best_configs_c[arch_name][activation_name] = {
                    "eta": best_eta,
                    "result": best_result
                }
                print(
                    f"{arch_name:7s} | {activation_name:10s} "
                    f"| eta={best_eta:.4g} "
                    f"| Train MSE={best_result['cost_history'][-1]:.6f} "
                    f"| Val MSE={best_result['val_cost_history'][-1]:.6f}"
                )

    # ============================================================
    # Part c): Visualization and overfitting analysis
    # ============================================================

        # Compare final validation MSE across architectures and activations
        plt_activ_arch_comparison(best_configs_c, architectures_c, PLOTS_DIR_2)
        # Compare training/validation learning curves for each architecture
        for arch_name, architecture in architectures_c.items():
            plt_learning_curves(best_configs_c[arch_name], architecture, PLOTS_DIR_2, start_epoch=30)

        print("\nPART C - GENERALIZATION DIAGNOSTICS")
        print(f"{'Architecture':<10} {'Activation':<12} "
            f"{'Final gap':>12} {'Best epoch':>12} {'Min val MSE':>12}")
        print("-" * 65)

        for arch_name, activation_results in best_configs_c.items():

            for activation_name, configuration in activation_results.items():
                result = configuration["result"]
                train_history = result["cost_history"]
                val_history = result["val_cost_history"]
                gap = val_history[-1] - train_history[-1]
                best_epoch = int(np.argmin(val_history))
                min_val_mse = val_history[best_epoch]
                print(
                    f"{arch_name:<10} {activation_name:<12} "
                    f"{gap:>12.6f} {best_epoch:>12d} "
                    f"{min_val_mse:>12.6f}"
                )
   
    # ============================================================
    # Step C.4: Final held-out evaluation
    # ============================================================

        # Select the best configuration using FINAL validation MSE
        best_val_mse, best_arch_name, best_activation_name = min(
            (
                configuration["result"]["val_cost_history"][-1],
                arch_name,
                activation_name
            )
            for arch_name, activation_results in best_configs_c.items()
            for activation_name, configuration in activation_results.items()
        )
        best_eta = best_configs_c[best_arch_name][best_activation_name]["eta"]

        print("\nBEST PART C CONFIGURATION")
        print(f"Architecture: {architectures_c[best_arch_name]}")
        print(f"Activation: {best_activation_name}")
        print(f"Learning rate: {best_eta}")
        print(f"Validation MSE: {best_val_mse:.6f}")

        # Prepare all 160 training observations and 40 test observations
        (X_train_nn, X_test_nn, y_train_nn, y_test_nn, train_means, train_stds) = prepare_data(x_train, x_test, y_train, y_test)

        # Sigmoid baselines from Part b and selected Part c configuration
        models_to_compare = {
            "Sigmoid [1,50,1]": ("1x50", "sigmoid", 0.01),
            "Sigmoid [1,100,100,1]": ("2x100", "sigmoid", 0.01),
            f"Part C: {best_activation_name} {best_arch_name}": (
                best_arch_name, best_activation_name, best_eta
            )
        }
        final_results_c = {}

        for model_name, (arch_name, activation_name, eta) in models_to_compare.items():

            architecture = architectures_c[arch_name]
            activation, derivative = activation_options[activation_name]
            n_hidden = len(architecture) - 2
            # Reinitialize from the same seed used during validation
            layers = create_layers_batch(
                architecture[0],
                architecture[1:],
                seed=2026,
                hidden_activation=activation_name
            )

            activations = [activation] * n_hidden + [identity]
            activation_ders = [derivative] * n_hidden + [identity_derivative]

            # Train using the complete training dataset
            result = optimize_regression_stochastic(X_train_nn, y_train_nn, layers, activations, activation_ders, eta0=eta, 
                                                    optimizer="adam", batch_size=32, n_epochs=300, schedule="constant", seed=2026)
            trained_layers = result["layers"]

            _, _, y_train_pred = feed_forward_saver_batch(X_train_nn, trained_layers, activations)
            _, _, y_test_pred = feed_forward_saver_batch(X_test_nn, trained_layers, activations)

            final_results_c[model_name] = {
                "layers": trained_layers,
                "activations": activations,
                "mse_train": mean_squared_error(y_train_nn, y_train_pred),
                "mse_test": mean_squared_error(y_test_nn, y_test_pred),
                "r2_test": r2_score(y_test, y_test_pred.ravel())
            }

        # OLS benchmark on the identical train/test split
        ols_result_c = fit_and_evaluate(x_train, x_test, y_train, y_test, degree=14, method="ols")

        print("\nFINAL PART C COMPARISON")
        print(f"{'Model':<35} {'Train MSE':>12} {'Test MSE':>12} {'Test R2':>12}")
        print("-" * 75)
        print(
            f"{'OLS (degree 14)':<35} "
            f"{ols_result_c['mse_train']:>12.6f} "
            f"{ols_result_c['mse_test']:>12.6f} "
            f"{r2_score(y_test, ols_result_c['y_test_pred']):>12.6f}"
        )

        for model_name, result in final_results_c.items():
            print(
                f"{model_name:<35} "
                f"{result['mse_train']:>12.6f} "
                f"{result['mse_test']:>12.6f} "
                f"{result['r2_test']:>12.6f}"
            )

    # ============================================================
    # Part c): Final fitted-function comparison
    # ============================================================

        x_grid = np.linspace(-1.0, 1.0, 500)
        y_exact = runge(x_grid)
        # Polynomial OLS prediction
        X_grid_ols = design_matrix(x_grid, 14)
        X_grid_ols = scale_design_matrix(X_grid_ols, ols_result_c["feature_means"], ols_result_c["feature_stds"])
        y_ols_grid = predict(X_grid_ols, ols_result_c["theta"])

        # Neural-network predictions
        X_grid_nn = scale_inputs(x_grid.reshape(-1, 1), train_means, train_stds)
        nn_predictions_c = {}

        for model_name, result in final_results_c.items():
            _, _, y_grid_pred = feed_forward_saver_batch(X_grid_nn, result["layers"], result["activations"])
            nn_predictions_c[model_name] = y_grid_pred.ravel()

        # Save figure in the Part c folder
        plt_ols_vs_nn(x_train, y_train, x_test, y_test, x_grid, y_exact, y_ols_grid, nn_predictions_c, degree=14, plots_dir=PLOTS_DIR_2)

        print("-------------------------------------------------------------------------------")
        print(" Find the resulting plots in P2/plots/part_2 ")
        print("-------------------------------------------------------------------------------")

