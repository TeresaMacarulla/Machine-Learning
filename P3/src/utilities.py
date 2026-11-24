
import json
from pathlib import Path
import numpy as np

# ============================================================
# Load and prepare the dataset
# ============================================================

def load_kilter_dataset(data_dir):
    """
    Load Kilter Board climbing problems from Degree45 and Degree50 folders.

    Returns:
        X: Array (N, 4761) containing flattened hold channels
           and normalized inclination.
        y: Array (N,) containing class labels 0-10 (V3-V13) corresponding to difficulty grade.
    """

    data_dir = Path(data_dir)
    folder_angles = {"Degree45": 45, "Degree50": 50}

    # Find all climbing problems in both folders
    files = []
    for folder_name, angle in folder_angles.items():
        folder = data_dir / folder_name

        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        npy_files = sorted(folder.glob("*.npy"))
        print(f"{folder_name}: {len(npy_files)} climbing problems")

        for npy_path in npy_files:
            files.append((npy_path, angle))

    if not files:
        raise ValueError("No climbing problems found.")

    # Allocate arrays for all samples
    n_samples = len(files)
    n_features = 34 * 35 * 4 + 1

    X = np.empty((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int64)

    # Load each climbing problem
    valid_count = 0
    skipped_count = 0

    for npy_path, angle in files:

        json_path = npy_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing JSON: {json_path}")

        # Load metadata first to check the grade
        with open(json_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        grade = metadata.get("grade_v")

        # Skip climbing problems without a difficulty grade
        if grade is None:
            skipped_count += 1
            continue
        grade = int(grade)

        # Check that the grade belongs to our 11 classes
        if not 3 <= grade <= 13:
            raise ValueError(f"Unexpected grade in {json_path}: V{grade}")

        # Load matrix only for valid climbing problems
        route = np.load(npy_path, allow_pickle=False)

        if route.shape != (34, 35, 10):
            raise ValueError(f"Unexpected shape: {npy_path}")

        # Flatten the four dynamic channels
        X[valid_count, :-1] = route[:, :, :4].reshape(-1)

        # Add normalized inclination
        X[valid_count, -1] = angle / 90.0

        # Encode grades V3-V13 as classes 0-10
        y[valid_count] = grade - 3

        valid_count += 1

    # Remove unused rows from the preallocated arrays
    X = X[:valid_count]
    y = y[:valid_count]

    print(f"\nValid climbing problems: {valid_count}")
    print(f"Skipped (missing grade): {skipped_count}")

    return X, y

# ============================================================
# Neural network 
# ============================================================

def create_layers_batch(network_input_size, layer_output_sizes, seed=2026, hidden_activation="sigmoid"):
    """Initialize the weights and biases of a feed-forward network."""
    rng = np.random.default_rng(seed)
    layers = []
    n_inputs = network_input_size

    for i, n_outputs in enumerate(layer_output_sizes):
        # Check whether this is the output layer
        is_output_layer = (i == len(layer_output_sizes)-1)

        if not is_output_layer and hidden_activation in ("relu", "leaky_relu"):
            # He initialization for ReLU-family hidden layers
            std = np.sqrt(2.0 / n_inputs)
        else:
            std = np.sqrt(1.0 / n_inputs)

        W = rng.normal(0.0, std, size=(n_inputs, n_outputs))
        b = np.zeros(n_outputs)
        layers.append((W, b))
        n_inputs = n_outputs

    return layers

# ============================================================
# Activation functions and their first derivatives
# ============================================================

def relu(z):
    """Rectified Linear Unit activation function."""
    return np.maximum(0.0, z)

def relu_derivative(z):
    """First derivative of ReLU (defined as 0 at z=0)."""
    return np.where(np.asarray(z) > 0, 1.0, 0.0)

# ============================================================
# Cost/loss functions and their respective gradients
# ============================================================

def softmax(z):
    """Numerically stable softmax, applied row-wise."""
    z = np.asarray(z, dtype=float)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ============================================================
# Stochastic gradient descent
# ============================================================

def sgd_learning_rate(step, eta0, schedule="constant", decay_scale=10.0):
    """
    Learning-rate schedule for stochastic optimization.
    constant:
        eta_t = eta0
    inverse_time:
        eta_t = eta0 * decay_scale / (step + decay_scale)
    The inverse-time form starts at eta0 when step = 0.
    """

    if schedule == "constant":
        return eta0
    elif schedule == "inverse_time":
        return ( eta0 * decay_scale / (step + decay_scale) )

def train_neural_network(X, y, layers, activation_funcs, activation_ders,
                                   eta0=0.01, optimizer="sgd", batch_size=32,
                                   n_epochs=300, schedule="constant", decay_scale=10.0,
                                   seed=2026, rho=0.9, beta1=0.9, beta2=0.999, eps=1e-8,
                                   X_val=None, y_val=None, regularization=None, lmbda=0.0, 
                                   task="regression", early_stopping_patience=None):
    """Train a neural network using mini-batch SGD, RMSprop or Adam."""

    n = len(y)
    rng = np.random.default_rng(seed)

    # Copy parameters to preserve the original initialization
    layers = [(W.copy(), b.copy()) for W, b in layers]

    # Optimizer states: one array per weight matrix and bias vector
    accumulators = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    first_moments = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    second_moments = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]

    def calculate_metrics(X_data, y_data):
        _, _, y_pred = feed_forward_saver_batch(X_data, layers, activation_funcs)
        if task == "classification":
            loss = multiclass_cross_entropy(y_data, y_pred)
            accuracy = classification_accuracy(y_data, y_pred)
        else:
            loss = mean_squared_error(y_data, y_pred)
            accuracy = None
        return loss, accuracy

    # Initial training loss and accuracy
    train_loss, train_accuracy = calculate_metrics(X, y)
    cost_history = [train_loss]
    objective_history = [
        train_loss + nn_regularization_penalty(layers, lmbda, regularization)
    ]
    accuracy_history = []
    if task == "classification":
        accuracy_history.append(train_accuracy)

    # Initial validation loss and accuracy
    val_cost_history = []
    val_accuracy_history = []
    if X_val is not None and y_val is not None:
        val_loss, val_accuracy = calculate_metrics(X_val, y_val)
        val_cost_history.append(val_loss)
        if task == "classification":
            val_accuracy_history.append(val_accuracy)
    update_count = 0

    # Early-stopping setup
    best_layers = None
    best_epoch = None
    best_val_ce = None
    epochs_without_improvement = 0

    if early_stopping_patience is not None:
        if X_val is None or y_val is None:
            raise ValueError("Early stopping requires validation data.")
        # Initially, the best model is the untrained network (epoch 0)
        best_layers = [(W.copy(), b.copy()) for W, b in layers]
        best_epoch = 0
        best_val_ce = val_cost_history[0]

    for epoch in range(n_epochs):
        # Shuffle observations
        indices = rng.permutation(n)

        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # Learning rate
            eta = sgd_learning_rate(update_count, eta0, schedule, decay_scale)
            # Backpropagation
            layer_grads = backpropagation_batch(X_batch, y_batch, layers, activation_funcs, activation_ders, regularization=regularization, lmbda=lmbda, task=task)
            update_count += 1

            # Update each layer's weights and biases
            for i, ((W, b), (dW, db)) in enumerate(zip(layers, layer_grads)):
                # j=0: weights, j=1: biases
                for j, (parameter, gradient) in enumerate(((W, dW), (b, db))):
                    if optimizer == "sgd":
                        update = eta * gradient
                    elif optimizer == "rmsprop":
                        update, accumulators[i][j] = rmsprop_update(gradient, accumulators[i][j], eta, rho=rho, eps=eps)
                    elif optimizer == "adam":
                        (update, first_moments[i][j], second_moments[i][j]) = adam_update(
                         gradient, first_moments[i][j], second_moments[i][j], iteration=update_count, eta=eta, beta1=beta1, beta2=beta2, eps=eps)
                    parameter -= update

        # Training metrics after the epoch
        train_loss, train_accuracy = calculate_metrics(X, y)
        cost_history.append(train_loss)
        objective_history.append(train_loss + nn_regularization_penalty(layers, lmbda, regularization))
        if task == "classification":
            accuracy_history.append(train_accuracy)

        # Validation metrics after the epoch
        if X_val is not None and y_val is not None:
            val_loss, val_accuracy = calculate_metrics(X_val, y_val)
            val_cost_history.append(val_loss)
            if task == "classification":
                val_accuracy_history.append(val_accuracy)

        # Early-stopping check, after recording validation metrics
        if early_stopping_patience is not None:
            if val_loss < best_val_ce:
                # Save the new best model
                best_val_ce = val_loss
                best_epoch = epoch + 1
                best_layers = [(W.copy(), b.copy()) for W, b in layers]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            # Stop after consecutive epochs without improvement
            if epochs_without_improvement >= early_stopping_patience:
                break

    return {
        "layers": layers,
        "cost_history": np.asarray(cost_history),
        "objective_history": np.asarray(objective_history),
        "val_cost_history": np.asarray(val_cost_history),
        "epochs": len(cost_history) - 1,
        "updates": update_count,
        "accuracy_history": np.asarray(accuracy_history),
        "val_accuracy_history": np.asarray(val_accuracy_history),
        "best_layers": best_layers,
        "best_epoch": best_epoch,
        "best_val_ce": best_val_ce,
    }
