import numpy as np
from sklearn.linear_model import Lasso
import jax
jax.config.update("jax_enable_x64", True)   # float64, so that the checks agree with numpy to 1e-15
import jax.numpy as jnp

# ============================================================
# Basic functions from project 1
# ============================================================

def runge(x):
    """Evaluate Runge's function."""
    x = np.asarray(x)
    return 1.0 / (1.0 + 25.0 * x**2)

def artificial_data(n=100, sigma=0.1, seed=2026):
    """Generate noisy observations of Runge's function."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    f = runge(x)
    noise = rng.normal(loc=0.0, scale=sigma, size=n)
    y = f + noise
    return x, y

def design_matrix(x, degree):
    """Construct X = [1, x, x^2, ..., x^degree]."""
    x = np.asarray(x, dtype=float)
    return np.vstack([x**power for power in range(degree + 1)]).T

def fit_feature_scaler(X_train):
    """Compute means/stds of non-intercept features using training data only."""
    feature_means = np.mean(X_train[:, 1:], axis=0)
    feature_stds = np.std(X_train[:, 1:], axis=0)

    if np.any(feature_stds == 0):
        raise ValueError("At least one polynomial feature has zero standard deviation.")

    return feature_means, feature_stds

def scale_design_matrix(X, feature_means, feature_stds):
    """Scale non-intercept columns using supplied training-set statistics."""
    X_scaled = X.copy()
    X_scaled[:, 1:] = (X[:, 1:] - feature_means) / feature_stds
    return X_scaled

def fit_ols(X, y):
    """Fit OLS with the Moore-Penrose pseudoinverse."""
    return np.linalg.pinv(X) @ y

def fit_ridge(X, y, lmbda):
    """
    Fit Ridge regression using the normalized cost
        (1/n) ||y - X theta||^2 + lambda theta^T P theta.
    The intercept theta_0 is not penalized.
    """

    if lmbda < 0:
        raise ValueError("lambda must be non-negative.")
    if lmbda == 0:
        return fit_ols(X, y)

    n, p = X.shape

    # Ridge penalty matrix P = diag(0, 1, ..., 1).
    # The intercept is not penalized.
    penalty = np.eye(p)
    penalty[0, 0] = 0.0

    return np.linalg.solve(
        X.T @ X + n * lmbda * penalty,
        X.T @ y,
    )

def fit_lasso(X, y, lmbda):
    if lmbda == 0:
        return fit_ols(X, y)

    model = Lasso(alpha=lmbda / 2, fit_intercept=True, max_iter=100000, tol=1e-10)
    model.fit(X[:, 1:], y)

    return np.r_[model.intercept_, model.coef_]

def predict(X, theta):
    """Return model predictions X @ theta."""
    return X @ theta

def fit_and_evaluate(x_train, x_test, y_train, y_test, degree, method="ols", lmbda=0.0,):
    """Fit and evaluate one polynomial OLS, Ridge or Lasso model."""

    X_train = design_matrix(x_train, degree)
    X_test = design_matrix(x_test, degree)

    feature_means, feature_stds = fit_feature_scaler(X_train)

    X_train_scaled = scale_design_matrix(X_train, feature_means, feature_stds)
    X_test_scaled = scale_design_matrix(X_test, feature_means, feature_stds)

    if method == "ols":
        theta = fit_ols(X_train_scaled, y_train)

    elif method == "ridge":
        theta = fit_ridge(X_train_scaled, y_train, lmbda)

    elif method == "lasso":
        theta = fit_lasso( X_train_scaled, y_train, lmbda=lmbda)

    else:
        raise ValueError( "method must be 'ols', 'ridge' or 'lasso'." )

    y_train_pred = predict(X_train_scaled, theta)
    y_test_pred = predict(X_test_scaled, theta)

    return {
        "theta": theta,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "mse_train": mean_squared_error(y_train, y_train_pred),
        "mse_test": mean_squared_error(y_test, y_test_pred),
    }

# ============================================================
# Cost/loss functions and their respective gradients
# ============================================================

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(X, y_true, theta):
    """Gradient of mean((y_true - X @ theta)**2) w.r.t. theta."""
    n = X.shape[0]

    return (2.0 / n) * (X.T @ (X @ theta - y_true))

def mse_gradient_l1(X, y_true, theta, lmbda):
    """MSE gradient plus L1 subgradient (zero chosen at theta_j=0)."""
    penalty_gradient = np.sign(theta)
    penalty_gradient[0] = 0.0

    return mse_gradient(X, y_true, theta) + lmbda * penalty_gradient

def mse_gradient_l2(X, y_true, theta, lmbda):
    """MSE gradient plus squared L2 penalty gradient."""
    penalty_gradient = 2.0 * theta
    penalty_gradient[0] = 0.0

    return mse_gradient(X, y_true, theta) + lmbda * penalty_gradient

def mse_output_derivative(y_true, y_pred):
    """Derivative of the MSE with respect to predictions."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Targets and predictions must have the same shape.")

    return (2.0 / y_true.size) * (y_pred - y_true)

# These are from weekly materials: they need to be checked 
def softmax(z):
    """Softmax of each row of the matrix z: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z/jnp.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    """Softmax of a single vector of scores."""
    e_z = jnp.exp(z - jnp.max(z))
    return e_z/jnp.sum(e_z)

# ============================================================
# Activation functions and their first derivatives
# ============================================================

def identity(z):
    """Identity activation function for regression output."""
    return z

def identity_derivative(z):
    """First derivative of the identity activation."""
    return np.ones_like(z)

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
# stochastic gradient descent
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

def optimize_regression_stochastic(X, y, layers, activation_funcs, activation_ders,
                                   eta0=0.01, batch_size=32, n_epochs=300,
                                   schedule="constant", decay_scale=10.0, seed=2026):
    """Train a neural network using mini-batch plain SGD."""
    n = len(y)
    rng = np.random.default_rng(seed)

    # Copy parameters so the original network is not modified
    layers = [(W.copy(), b.copy()) for W, b in layers]

    # Function to calculate the full training MSE
    def full_cost():
        _, _, y_pred = feed_forward_saver_batch(X, layers, activation_funcs)
        return mean_squared_error(y, y_pred)

    # Record the initial cost before training
    cost_history = [full_cost()]
    update_count = 0

    for epoch in range(n_epochs):
        # Shuffle observations at the start of each epoch
        indices = rng.permutation(n)

        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # Current learning rate
            eta = sgd_learning_rate(update_count, eta0, schedule, decay_scale)
            # Calculate gradients using backpropagation
            layer_grads = backpropagation_batch(X_batch, y_batch, layers, activation_funcs, activation_ders)

            # Update all weights and biases
            for (W, b), (dW, db) in zip(layers, layer_grads):
                W -= eta * dW
                b -= eta * db
            update_count += 1

        # Record the training MSE after each epoch
        cost_history.append(full_cost())

    return {
        "layers": layers,
        "cost_history": np.asarray(cost_history),
        "epochs": n_epochs,
        "updates": update_count
    }

# ============================================================
# Neural network 
# ============================================================

def create_layers_batch(network_input_size, layer_output_sizes, seed=2026):
    """Initialize the weights and biases of a feed-forward network."""
    rng = np.random.default_rng(seed)
    layers = []
    n_inputs = network_input_size

    for n_outputs in layer_output_sizes:
        std = np.sqrt(1.0 / n_inputs)
        W = rng.normal(0.0, std, size=(n_inputs, n_outputs))
        b = np.zeros(n_outputs)
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

def backpropagation_batch(X, y_true, layers, activation_funcs, activation_ders):
    """Calculate gradients of all network weights and biases using backpropagation."""
    # Forward pass, saving intermediate values
    layer_inputs, zs, y_pred = feed_forward_saver_batch(X, layers, activation_funcs)

    # Store the gradients of each layer
    layer_grads = [None] * len(layers)

    # Work backwards, from the last layer to the first
    for i in reversed(range(len(layers))):
        if i == len(layers) - 1:
            # Derivative of MSE with respect to the output
            dC_da = mse_output_derivative(y_true, y_pred)
        else:
            # Propagate the error backwards through the next layer
            W_next, b_next = layers[i + 1]
            dC_da = dC_dz @ W_next.T

        # Derivative with respect to the pre-activation
        dC_dz = dC_da * activation_ders[i](zs[i])

        # Gradients of weights and biases
        dC_dW = layer_inputs[i].T @ dC_dz
        dC_db = np.sum(dC_dz, axis=0)

        # Store the gradients
        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads

def gradient_check(X, y, layers, activation_funcs, activation_ders,
                   h=1e-6, n_checks=5, seed=2026):
    """Compare backpropagation gradients with central finite differences."""
    rng = np.random.default_rng(seed)

    # Analytical gradients from backpropagation
    layer_grads = backpropagation_batch(X, y, layers, activation_funcs, activation_ders)

    # Cost function evaluated with the current network parameters
    def cost():
        _, _, y_pred = feed_forward_saver_batch(X, layers, activation_funcs)
        return mean_squared_error(y, y_pred)

    max_error = 0.0
    failures = 0
    n_tested = 0

    for (W, b), (dW, db) in zip(layers, layer_grads):

        # Check both the weights and biases
        for parameter, gradient in [(W, dW), (b, db)]:

            for _ in range(min(n_checks, parameter.size)):
                # Select a random element of the parameter array
                idx = tuple(rng.integers(s) for s in parameter.shape)
                original = parameter[idx]

                # Central finite difference
                try:
                    parameter[idx] = original + h
                    cost_plus = cost()
                    parameter[idx] = original - h
                    cost_minus = cost()
                finally:
                    parameter[idx] = original

                numerical = (cost_plus - cost_minus) / (2 * h)
                analytical = gradient[idx]
                error = abs(numerical - analytical)
                max_error = max(max_error, error)
                n_tested += 1

                if not np.isclose(numerical, analytical, rtol=1e-4, atol=1e-7):
                    failures += 1

    print("Gradients tested:", n_tested)
    print(f"Maximum absolute error: {max_error:.3e}")
    print("Failed comparisons:", failures)

    return failures == 0
