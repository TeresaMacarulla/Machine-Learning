
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

# ============================================================
# Load and prepare the dataset
# ============================================================

def load_kilter_dataset_sparse(data_dir):
    """Load Kilter Board routes directly into a memory-efficient CSR matrix.

    X: (N, 4761) float32; y: V3..V13 mapped to integer labels 0..10.
    Missing-grade routes are skipped. The first 4760 features are hold roles;
    the last feature is inclination divided by 90.
    """

    data_dir = Path(data_dir)
    file_angle_pairs = []
    for folder_name, angle in (("Degree45", 45), ("Degree50", 50)):
        folder = data_dir / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        npy_paths = sorted(folder.glob("*.npy"))
        print(f"{folder_name}: {len(npy_paths)} climbing problems", flush=True)
        file_angle_pairs.extend((path, angle) for path in npy_paths)
    if not file_angle_pairs:
        raise ValueError("No climbing problems found")

    # Data structure for a compressed sparse row (CSR) matrix.
    indices, values, indptr, labels = [], [], [0], []
    skipped = 0
    for npy_path, angle in file_angle_pairs:
        json_path = npy_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing JSON: {json_path}")
        with json_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        grade = metadata.get("grade_v")
        if grade is None:
            skipped += 1
            continue
        grade = int(grade)
        if not 3 <= grade <= 13:
            raise ValueError(f"Unexpected grade V{grade} in {json_path}")

        route = np.load(npy_path, allow_pickle=False)
        if route.shape != (34, 35, 10):
            raise ValueError(f"Unexpected shape {route.shape} in {npy_path}")

        dynamic = route[:, :, :4].reshape(-1)
        active = np.flatnonzero(dynamic)
        indices.extend(active.tolist())
        values.extend(dynamic[active].tolist())
        indices.append(4760)          # Last feature: inclination
        values.append(angle / 90.0)
        indptr.append(len(indices))
        labels.append(grade - 3)

    X = csr_matrix(
        (np.asarray(values, dtype=np.float32),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int32)),
        shape=(len(labels), 4761), dtype=np.float32
    )
    y = np.asarray(labels, dtype=np.int64)
    print(f"Valid climbing problems: {len(y)}", flush=True)
    print(f"Skipped (missing grade): {skipped}", flush=True)
    print(f"Nonzero entries: {X.nnz:,}", flush=True)
    print(f"CSR memory: {(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1024**2:.1f} MiB", flush=True)
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

        W = rng.normal(0.0, std, size=(n_inputs, n_outputs)).astype(np.float32)
        b = np.zeros(n_outputs, dtype=np.float32)
        layers.append((W, b))
        n_inputs = n_outputs

    return layers

def feed_forward_saver_batch(X, layers, activation_funcs):
    """Forward pass that stores intermediate layer inputs and pre-activations."""

    layer_inputs = []
    zs = []
    a = X

    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation_func(z)
        zs.append(z)

    return layer_inputs, zs, a

def backpropagation_batch(X, y_true, layers, activation_funcs, activation_ders, regularization=None, lmbda=0.0):
    """Calculate gradients of all network weights and biases using backpropagation."""
    # Forward pass, saving intermediate values
    layer_inputs, zs, y_pred = feed_forward_saver_batch(X, layers, activation_funcs)

    # Store the gradients of each layer
    layer_grads = [None] * len(layers)

    # Work backwards, from the output layer to the first hidden layer
    for i in reversed(range(len(layers))):

        if i == len(layers) - 1:
            # Output layer
            # Softmax + cross-entropy
            m = X.shape[0]
            dC_dz = (y_pred - y_true) / m
        else:
            # Hidden layers: unchanged backpropagation recursion
            W_next, b_next = layers[i + 1]
            dC_da = dC_dz @ W_next.T
            dC_dz = dC_da * activation_ders[i](zs[i])

        # Gradients of weights and biases
        dC_dW = layer_inputs[i].T @ dC_dz
        dC_db = np.sum(dC_dz, axis=0)
        # Optional regularization of weights
        W, b = layers[i]

        if regularization == "l1":
            dC_dW += lmbda * np.sign(W)
        elif regularization == "l2":
            dC_dW += 2.0 * lmbda * W

        # Save this layer's gradients
        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads

# ============================================================
# Activation functions and their first derivatives
# ============================================================

def sigmoid(z):
    """Sigmoid activation function (numerically stable)."""
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

def sigmoid_derivative(z):
    """First derivative of the sigmoid function."""
    s = sigmoid(z)
    return s * (1.0 - s)

def relu(z):
    """Rectified Linear Unit activation function."""
    return np.maximum(0.0, z)

def relu_derivative(z):
    """First derivative of ReLU (defined as 0 at z=0)."""
    return np.where(np.asarray(z) > 0, 1.0, 0.0)

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.maximum(z, alpha * np.asarray(z))

def leaky_relu_derivative(z, alpha=0.01):
    """First derivative of Leaky ReLU (defined as alpha at z=0)."""
    return np.where(np.asarray(z) > 0, 1.0, alpha)

# ============================================================
# Learning rate update methods / Optimizers
# ============================================================

def rmsprop_update( gradient, accumulator, eta, rho=0.9, eps=1.0e-8,):
    """
    RMSProp update.
    """
    accumulator = ( rho * accumulator + (1.0 - rho) * gradient**2 )
    update = ( eta * gradient / (np.sqrt(accumulator) + eps) )
    return update, accumulator

def adam_update( gradient, first_moment, second_moment, iteration,
                 eta, beta1=0.9, beta2=0.999, eps=1.0e-8,):
    """
    Adam update with bias correction.
    """
    first_moment = ( beta1 * first_moment + (1.0 - beta1) * gradient )
    second_moment = ( beta2 * second_moment + (1.0 - beta2) * gradient**2 )

    first_corrected = ( first_moment / (1.0 - beta1**iteration) )
    second_corrected = ( second_moment / (1.0 - beta2**iteration) )

    update = ( eta * first_corrected / (np.sqrt(second_corrected) + eps) )
    return ( update, first_moment, second_moment, )

# ============================================================
# Cost/loss functions and their respective gradients
# ============================================================

def softmax(z):
    """Numerically stable softmax, applied row-wise."""
    z = np.asarray(z, dtype=np.float32)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def multiclass_cross_entropy(y_true, y_pred):
    """Mean cross-entropy for one-hot targets and class probabilities."""
    # Prevent log(0), which is undefined
    probabilities = np.clip(y_pred, 1e-15, 1.0)
    return -np.mean(np.sum(y_true * np.log(probabilities), axis=1))

def classification_accuracy(y_true, y_pred):
    """Classification accuracy for one-hot targets and class probabilities."""
    true_classes = np.argmax(y_true, axis=1)
    predicted_classes = np.argmax(y_pred, axis=1)
    return np.mean(true_classes == predicted_classes)

def nn_regularization_penalty(layers, lmbda=0.0, regularization=None):
    """Calculate the L1 or L2 penalty over all network weights."""

    if regularization is None or lmbda == 0.0:
        return 0.0
    if regularization == "l1":
        penalty = sum(np.sum(np.abs(W)) for W, b in layers)
    elif regularization == "l2":
        penalty = sum(np.sum(W**2) for W, b in layers)
    return lmbda * penalty

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
                                   early_stopping_patience=None):
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
        loss = multiclass_cross_entropy(y_data, y_pred)
        accuracy = classification_accuracy(y_data, y_pred)
        return loss, accuracy

    # Initial training loss and accuracy
    train_loss, train_accuracy = calculate_metrics(X, y)
    cost_history = [train_loss]
    objective_history = [
        train_loss + nn_regularization_penalty(layers, lmbda, regularization)
    ]
    accuracy_history = []
    accuracy_history.append(train_accuracy)

    # Initial validation loss and accuracy
    val_cost_history = []
    val_accuracy_history = []
    if X_val is not None and y_val is not None:
        val_loss, val_accuracy = calculate_metrics(X_val, y_val)
        val_cost_history.append(val_loss)
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
            layer_grads = backpropagation_batch(X_batch, y_batch, layers, activation_funcs, activation_ders, regularization=regularization, lmbda=lmbda)
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
        accuracy_history.append(train_accuracy)

        # Validation metrics after the epoch
        if X_val is not None and y_val is not None:
            val_loss, val_accuracy = calculate_metrics(X_val, y_val)
            val_cost_history.append(val_loss)
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
