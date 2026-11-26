# good data set from https://github.com/Pa-Sto/kilterboardie/tree/main
# scikit-learn logistic regression https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from time import perf_counter

from utilities import (
    load_kilter_dataset_sparse,
    feed_forward_saver_batch,
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
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR_0 = RESULTS_DIR / "part_0"
RESULTS_DIR_1 = RESULTS_DIR / "part_1"
RESULTS_DIR_2 = RESULTS_DIR / "part_2"
RESULTS_DIR_3 = RESULTS_DIR / "part_3"

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
print("0: Compare custom vs. Scikit-learn multinomial logistic regression.")
print("1: Compare Multinominal Logistic Regression vs. two NN architectures.")
print("2: Explore different Learning-rates for two NN architectures.")
print("3: Explore different activation functions for two NN architectures.")
print("-------------------------------------------------------------------------------")
number = float(input("Choose a number: "))
print()

if number == 0:

    # ============================================================
    # Custom vs. Scikit-learn multinomial logistic regression
    # ============================================================

    # Custom logistic regression
    print("\nTraining custom logistic regression...")

    architecture = [4761, 11]
    layers = create_layers_batch(network_input_size=architecture[0], layer_output_sizes=architecture[1:], seed=2026)

    start_time = perf_counter()

    custom_result = train_neural_network(X_kilter_train, Y_kilter_train, layers, [softmax], [None], eta0=0.001, optimizer="adam", batch_size=128,
                                         n_epochs=40, schedule="constant", seed=2026, X_val=X_kilter_val, y_val=Y_kilter_val, regularization="l2",
                                         lmbda=1e-4, early_stopping_patience=5)

    custom_time = perf_counter() - start_time

    # Calculate predictions using the trained weights
    trained_layers = custom_result["best_layers"]

    _, _, custom_train_prob = feed_forward_saver_batch(X_kilter_train, trained_layers, [softmax])
    _, _, custom_val_prob = feed_forward_saver_batch(X_kilter_val, trained_layers, [softmax])

    print("Epochs completed:", custom_result["epochs"])
    print("Best epoch:", custom_result["best_epoch"])
    print("Best validation CE:", custom_result["best_val_ce"])

    # Scikit-learn logistic regression
    print("\nTraining Scikit-learn logistic regression...")

    sklearn_lr = LogisticRegression(solver="saga", C=1.0, max_iter=1000, tol=1e-3, fit_intercept=True, class_weight=None, random_state=2026)
    start_time = perf_counter()

    sklearn_lr.fit(X_kilter_train, y_kilter_train)

    sklearn_time = perf_counter() - start_time

    assert np.array_equal(sklearn_lr.classes_, np.arange(11))

    sklearn_train_prob = sklearn_lr.predict_proba(X_kilter_train)
    sklearn_val_prob = sklearn_lr.predict_proba(X_kilter_val)
    print("Iterations:", sklearn_lr.n_iter_)

    # Evaluate both implementations
    probabilities = {
        "Custom": (custom_train_prob, custom_val_prob),
        "Scikit-learn": (sklearn_train_prob, sklearn_val_prob)
    }
    training_times = {
        "Custom": custom_time,
        "Scikit-learn": sklearn_time
    }

    comparison_results = {}

    for model_name, (train_prob, val_prob) in probabilities.items():

        train_predictions = np.argmax(train_prob, axis=1)
        val_predictions = np.argmax(val_prob, axis=1)

        comparison_results[model_name] = {
            "train_ce": log_loss(
                y_kilter_train, train_prob, labels=np.arange(11)
            ),
            "val_ce": log_loss(
                y_kilter_val, val_prob, labels=np.arange(11)
            ),
            "train_acc": accuracy_score(
                y_kilter_train, train_predictions
            ),
            "val_acc": accuracy_score(
                y_kilter_val, val_predictions
            ),
            "time": training_times[model_name]
        }

    # Print comparison
    print("\nLOGISTIC REGRESSION COMPARISON")

    print(
        f"{'Model':<15} "
        f"{'Train CE':>12} "
        f"{'Val CE':>12} "
        f"{'Train Acc':>12} "
        f"{'Val Acc':>12} "
        f"{'Time (s)':>12}"
    )

    print("-" * 80)

    for model_name, result in comparison_results.items():

        print(
            f"{model_name:<15} "
            f"{result['train_ce']:>12.5f} "
            f"{result['val_ce']:>12.5f} "
            f"{100 * result['train_acc']:>11.2f}% "
            f"{100 * result['val_acc']:>11.2f}% "
            f"{result['time']:>12.2f}"
        )

    # Compare predictions on the validation set
    custom_pred = np.argmax(custom_val_prob, axis=1)
    sklearn_pred = np.argmax(sklearn_val_prob, axis=1)

    agreement = np.mean(custom_pred == sklearn_pred)

    print(f"\nValidation prediction agreement: {agreement * 100:.2f}%")
    print(f"Scikit-learn iterations: {sklearn_lr.n_iter_[0]}")

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

        print(f"CALCULATING {network_name}")
        n_hidden = len(architecture) - 2
        # He initialization for ReLU hidden layers
        layers = create_layers_batch(network_input_size=architecture[0], layer_output_sizes=architecture[1:], seed=2026, hidden_activation="relu")
        # ReLU hidden layers and softmax output
        activations = [relu] * n_hidden + [softmax]
        # No separate softmax derivative is needed
        activation_ders = [relu_derivative] * n_hidden + [None]

        # Train the network
        result = train_neural_network(X_kilter_train, Y_kilter_train, layers, activations, activation_ders, eta0=0.001,
                                    optimizer="adam", batch_size=128, n_epochs=40, schedule="constant", seed=2026,
                                    X_val=X_kilter_val, y_val=Y_kilter_val, regularization="l2", lmbda=1e-4, early_stopping_patience=5)
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

    plt_learning_curves(results, RESULTS_DIR_1, start_epoch=1)

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
    print(" Find the results in P3/results/part_1 ")
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
                                    optimizer="adam", batch_size=128, n_epochs=40, schedule="constant", seed=2026,
                                    X_val=X_kilter_val, y_val=Y_kilter_val, regularization="l2", lmbda=1e-4, early_stopping_patience=5)
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
                result = train_neural_network(X_kilter_train, Y_kilter_train, layers, activations, activation_ders, eta0=0.001,
                                    optimizer="adam", batch_size=128, n_epochs=40, schedule="constant", seed=2026,
                                    X_val=X_kilter_val, y_val=Y_kilter_val, regularization="l2", lmbda=1e-4, early_stopping_patience=5)
            if arch_name == "NN 2x100":
                result = train_neural_network(X_kilter_train, Y_kilter_train, layers, activations, activation_ders, eta0=0.0003,
                                    optimizer="adam", batch_size=128, n_epochs=40, schedule="constant", seed=2026,
                                    X_val=X_kilter_val, y_val=Y_kilter_val, regularization="l2", lmbda=1e-4, early_stopping_patience=5)
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

    plt_activation_architecture_heatmaps(activation_arch_results, RESULTS_DIR_3)

    print("-------------------------------------------------------------------------------")
    print(" Find the results in P3/results/part_3 ")
    print("-------------------------------------------------------------------------------")







    






