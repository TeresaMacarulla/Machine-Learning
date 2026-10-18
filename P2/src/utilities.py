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

def optimize_regression_stochastic( X, y, eta0, optimizer="rmsprop", batch_size=32,
                                    n_epochs=500, schedule="inverse_time", decay_scale=10.0, accuracy_tol=1.0e-4, seed=2026, rho=0.9,
                                    beta1=0.9, beta2=0.999, eps=1.0e-8, divergence_threshold=1.0e12,):
    """
    Mini-batch stochastic optimization.
    The same update rules options are rmsprop and adam.
    Cost and relative parameter error are recorded once per epoch.
    """

    n = len(y)
    rng = np.random.default_rng(seed)

    # Optimizer states
    accumulator = np.zeros()
    first_moment = np.zeros()
    second_moment = np.zeros()

    cost_history = []
    relative_error_history = []

    converged = False
    diverged = False

    update_count = 0
    samples_processed = 0

    for epoch in range(1, n_epochs + 1):

        # Shuffle observations at the start of every epoch.
        indices = rng.permutation(n)

        for start in range(0, n, batch_size):

            batch_indices = indices[start:start + batch_size]

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Current learning rate.
            eta = sgd_learning_rate(step=update_count, eta0=eta0, schedule=schedule, decay_scale=decay_scale)
            gradient = gradient_calculator(theta, X_batch, y_batch)

            update_count += 1
            samples_processed += len(batch_indices)

            # Learning rate update methods / Optimizers
            if optimizer == "rmsprop":
                update, accumulator = rmsprop_update(gradient, accumulator, eta, rho=rho, eps=eps)

            elif optimizer == "adam":
                (update, first_moment, second_moment
                 ) = adam_update(gradient, first_moment, second_moment, update_count, eta, beta1=beta1, beta2=beta2, eps=eps)

            theta -= update

            if (not np.all(np.isfinite(theta)) or np.linalg.norm(theta) > divergence_threshold):
                diverged = True
                break

        # Evaluate the FULL cost once per epoch
        # This will be filled when we have a neural network
        cost = neural_cost()
        cost_history.append(cost)

        if diverged:
            break

    return {
        "theta": theta,
        "epochs": epoch,
        "updates": update_count,
        "samples_processed": samples_processed,
        # One epoch corresponds approximately to one complete pass through the training data.
        "data_passes": samples_processed / n,
        "converged": converged,
        "diverged": diverged,
        "cost_history": np.asarray(cost_history),
        "relative_error_history": np.asarray(relative_error_history),
    }