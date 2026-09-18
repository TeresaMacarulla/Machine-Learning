import numpy as np
from sklearn.model_selection import KFold

import jax
import jax.numpy as jnp

# JAX uses 32-bit floats by default.
# Part e asks us to verify agreement to machine precision.
jax.config.update("jax_enable_x64", True)

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
    x = np.asarray(x)
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
    penalty = ridge_penalty_matrix(p)

    return np.linalg.solve(
        X.T @ X + n * lmbda * penalty,
        X.T @ y,
    )

def ridge_penalty_matrix(n_parameters):
    """
    Ridge penalty matrix P = diag(0, 1, ..., 1).
    The intercept is not penalized.
    """
    penalty = np.eye(n_parameters)
    penalty[0, 0] = 0.0
    return penalty

def ols_cost(theta, X, y):
    """
    OLS cost:
        C(theta) = (1/n) ||X theta - y||^2
    """
    residual = X @ theta - y
    return np.mean(residual**2)

def ridge_cost(theta, X, y, lmbda):
    """
    Ridge cost using the convention adopted in this project:
        C(theta) = (1/n)||X theta - y||^2
                   + lambda * theta^T P theta

    The intercept is not penalized.
    """
    return (
        ols_cost(theta, X, y)
        + lmbda * np.sum(theta[1:]**2)
    )

def ols_gradient_analytic(theta, X, y):
    """
    Analytical gradient of the OLS cost.
    """
    n = len(y)

    return (
        2.0 / n
        * X.T @ (X @ theta - y)
    )

def ridge_gradient_analytic(theta, X, y, lmbda):
    """
    Analytical gradient of the Ridge cost.
    """
    gradient = ols_gradient_analytic(theta, X, y)

    # Do not regularize the intercept.
    gradient = gradient.copy()
    gradient[1:] += 2.0 * lmbda * theta[1:]

    return gradient

def predict(X, theta):
    """Return model predictions X @ theta."""
    return X @ theta

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """Calculate the coefficient of determination R^2."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0:
        raise ValueError("R^2 is undefined when all true values are identical.")

    return 1.0 - numerator / denominator

def fit_and_evaluate(
    x_train,
    x_test,
    y_train,
    y_test,
    degree,
    method="ols",
    lmbda=0.0,
):
    """
    Fit and evaluate one polynomial OLS or Ridge model.
    """
    X_train = design_matrix(x_train, degree)
    X_test = design_matrix(x_test, degree)

    feature_means, feature_stds = fit_feature_scaler(X_train)

    X_train_scaled = scale_design_matrix(
        X_train,
        feature_means,
        feature_stds,
    )
    X_test_scaled = scale_design_matrix(
        X_test,
        feature_means,
        feature_stds,
    )

    if method == "ols":
        theta = fit_ols(X_train_scaled, y_train)

    elif method == "ridge":
        theta = fit_ridge(
            X_train_scaled,
            y_train,
            lmbda,
        )

    else:
        raise ValueError("method must be 'ols' or 'ridge'.")

    y_train_pred = predict(X_train_scaled, theta)
    y_test_pred = predict(X_test_scaled, theta)

    return {
        "theta": theta,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "mse_train": mean_squared_error(y_train, y_train_pred),
        "mse_test": mean_squared_error(y_test, y_test_pred),
        "r2_train": r2_score(y_train, y_train_pred),
        "r2_test": r2_score(y_test, y_test_pred),
    }

def bootstrap_bias_variance(
    x_train,
    x_test,
    y_train,
    y_test,
    degrees,
    n_bootstraps=100,
    seed=2026,
):
    """
    Estimate bootstrap prediction error, squared bias and variance
    for OLS polynomial regression.
    """
    rng = np.random.default_rng(seed)

    degrees = np.asarray(degrees)
    n_train = len(x_train)

    # Generate bootstrap replicas once and reuse them
    # for all polynomial degrees.
    bootstrap_indices = rng.integers(
        0,
        n_train,
        size=(n_bootstraps, n_train),
    )

    error = np.zeros(len(degrees))
    bias2 = np.zeros(len(degrees))
    variance = np.zeros(len(degrees))

    for degree_index, degree in enumerate(degrees):

        y_pred = np.empty(
            (len(y_test), n_bootstraps)
        )

        for b in range(n_bootstraps):

            indices = bootstrap_indices[b]

            x_boot = x_train[indices]
            y_boot = y_train[indices]

            result = fit_and_evaluate(
                x_train=x_boot,
                x_test=x_test,
                y_train=y_boot,
                y_test=y_test,
                degree=degree,
                method="ols",
            )

            y_pred[:, b] = result["y_test_pred"]

        mean_prediction = np.mean(
            y_pred,
            axis=1,
        )

        error[degree_index] = np.mean(
            (y_test[:, None] - y_pred) ** 2
        )

        bias2[degree_index] = np.mean(
            (y_test - mean_prediction) ** 2
        )

        variance[degree_index] = np.mean(
            np.var(y_pred, axis=1)
        )

    return {
        "degrees": degrees,
        "error": error,
        "bias2": bias2,
        "variance": variance,
    }

def cross_validation_mse(
    x,
    y,
    degrees,
    n_splits=5,
    seed=2026,
):
    """
    Estimate OLS prediction error with k-fold cross-validation.

    Scaling is fitted independently inside each training fold.
    """
    degrees = np.asarray(degrees)

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    mean_mse = np.zeros(len(degrees))
    std_mse = np.zeros(len(degrees))

    for degree_index, degree in enumerate(degrees):

        fold_mse = []

        for train_indices, val_indices in kfold.split(x):

            x_train_fold = x[train_indices]
            y_train_fold = y[train_indices]

            x_val_fold = x[val_indices]
            y_val_fold = y[val_indices]

            result = fit_and_evaluate(
                x_train=x_train_fold,
                x_test=x_val_fold,
                y_train=y_train_fold,
                y_test=y_val_fold,
                degree=degree,
                method="ols",
            )

            fold_mse.append(result["mse_test"])

        mean_mse[degree_index] = np.mean(fold_mse)
        std_mse[degree_index] = np.std(fold_mse)

    return {
        "degrees": degrees,
        "mean_mse": mean_mse,
        "std_mse": std_mse,
    }

# ============================================================
# Automatic differentiation with JAX
# ============================================================

def _ols_cost_jax(theta, X, y):
    return jnp.mean((X @ theta - y)**2)

def _ridge_cost_jax(theta, X, y, lmbda):
    # theta[0] is the intercept and is not penalized.
    return (
        jnp.mean((X @ theta - y)**2)
        + lmbda * jnp.sum(theta[1:]**2)
    )

_ols_gradient_ad = jax.jit(
    jax.grad(_ols_cost_jax, argnums=0)
)

_ridge_gradient_ad = jax.jit(
    jax.grad(_ridge_cost_jax, argnums=0)
)

def regression_gradient_autodiff(
    theta,
    X,
    y,
    method="ols",
    lmbda=0.0,
):
    """
    Compute the regression gradient using JAX automatic
    differentiation.
    """
    theta_jax = jnp.asarray(theta)
    X_jax = jnp.asarray(X)
    y_jax = jnp.asarray(y)

    if method == "ols":
        gradient = _ols_gradient_ad(
            theta_jax,
            X_jax,
            y_jax,
        )

    elif method == "ridge":
        gradient = _ridge_gradient_ad(
            theta_jax,
            X_jax,
            y_jax,
            lmbda,
        )

    else:
        raise ValueError(
            "method must be 'ols' or 'ridge'."
        )

    return np.asarray(gradient)

# ============================================================
# Gradient-check function
# ============================================================

def compare_gradients(
    theta,
    X,
    y,
    method="ols",
    lmbda=0.0,
):
    """
    Return the maximum absolute difference between the
    analytical and automatically differentiated gradients.
    """

    if method == "ols":
        gradient_analytic = ols_gradient_analytic(
            theta,
            X,
            y,
        )

    elif method == "ridge":
        gradient_analytic = ridge_gradient_analytic(
            theta,
            X,
            y,
            lmbda,
        )

    else:
        raise ValueError(
            "method must be 'ols' or 'ridge'."
        )

    gradient_ad = regression_gradient_autodiff(
        theta,
        X,
        y,
        method=method,
        lmbda=lmbda,
    )

    return np.max(
        np.abs(
            gradient_analytic - gradient_ad
        )
    )

# ============================================================
# Hessian/learning-rate calculation
# ============================================================

def regression_hessian(
    X,
    method="ols",
    lmbda=0.0,
):
    """
    Return the Hessian of the OLS or Ridge cost.
    """
    n = X.shape[0]

    H = (
        2.0 / n
        * X.T @ X
    )

    if method == "ridge":
        H += (
            2.0
            * lmbda
            * ridge_penalty_matrix(X.shape[1])
        )

    elif method != "ols":
        raise ValueError(
            "method must be 'ols' or 'ridge'."
        )

    return H

def learning_rate_information(
    X,
    method="ols",
    lmbda=0.0,
):
    """
    Compute Hessian eigenvalues, condition number,
    theoretical optimal learning rate and stability limit.
    """
    H = regression_hessian(
        X,
        method=method,
        lmbda=lmbda,
    )

    eigenvalues = np.linalg.eigvalsh(H)

    lambda_max = eigenvalues[-1]

    # Ignore possible tiny round-off eigenvalues.
    tolerance = (
        np.finfo(float).eps
        * lambda_max
    )

    positive_eigenvalues = eigenvalues[
        eigenvalues > tolerance
    ]

    lambda_min = positive_eigenvalues[0]

    eta_max = 2.0 / lambda_max

    eta_opt = (
        2.0
        / (lambda_max + lambda_min)
    )

    condition_number = (
        lambda_max / lambda_min
    )

    return {
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "condition_number": condition_number,
        "eta_max": eta_max,
        "eta_opt": eta_opt,
    }

# ============================================================
# Gradient-descent function
# ============================================================

def gradient_descent(
    X,
    y,
    eta,
    method="ols",
    lmbda=0.0,
    gradient_source="analytic",
    theta0=None,
    max_iter=100000,
    tol=1.0e-8,
    seed=2026,
    divergence_threshold=1.0e12,
):
    """
    Plain gradient descent with a fixed learning rate.

    gradient_source:
        "analytic"   -> use the hand-derived gradient
        "autodiff"   -> use JAX automatic differentiation
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    rng = np.random.default_rng(seed)

    if theta0 is None:
        theta = rng.normal(
            size=X.shape[1]
        )
    else:
        theta = np.asarray(
            theta0,
            dtype=float,
        ).copy()

    cost_history = []
    converged = False
    diverged = False

    for iteration in range(1, max_iter + 1):

        # --------------------------------------------
        # Gradient
        # --------------------------------------------

        if gradient_source == "analytic":

            if method == "ols":
                gradient = ols_gradient_analytic(
                    theta,
                    X,
                    y,
                )

            elif method == "ridge":
                gradient = ridge_gradient_analytic(
                    theta,
                    X,
                    y,
                    lmbda,
                )

            else:
                raise ValueError(
                    "method must be 'ols' or 'ridge'."
                )

        elif gradient_source == "autodiff":

            gradient = regression_gradient_autodiff(
                theta,
                X,
                y,
                method=method,
                lmbda=lmbda,
            )

        else:
            raise ValueError(
                "gradient_source must be "
                "'analytic' or 'autodiff'."
            )

        # --------------------------------------------
        # Cost
        # --------------------------------------------

        if method == "ols":
            cost = ols_cost(
                theta,
                X,
                y,
            )

        else:
            cost = ridge_cost(
                theta,
                X,
                y,
                lmbda,
            )

        cost_history.append(cost)

        gradient_norm = np.linalg.norm(
            gradient
        )

        # --------------------------------------------
        # Convergence check
        # --------------------------------------------

        if gradient_norm < tol:
            converged = True
            break

        # --------------------------------------------
        # Gradient descent update
        # --------------------------------------------

        theta -= eta * gradient

        # Stop clearly divergent runs.
        if (
            not np.all(np.isfinite(theta))
            or np.linalg.norm(theta)
            > divergence_threshold
        ):
            diverged = True
            break

    return {
        "theta": theta,
        "iterations": iteration,
        "converged": converged,
        "diverged": diverged,
        "gradient_norm": gradient_norm,
        "cost_history": np.asarray(cost_history),
    }

# ============================================================
# PART F: Optimizer update rules
# ============================================================

def momentum_update(
    gradient,
    velocity,
    eta,
    gamma=0.9,
):
    """
    Momentum update.

    v_t = gamma * v_{t-1} + eta * gradient
    theta <- theta - v_t
    """
    velocity = ( gamma * velocity + eta * gradient )

    update = velocity
    return update, velocity

def adagrad_update(
    gradient,
    accumulator,
    eta,
    eps=1.0e-8,
):
    """
    AdaGrad update.
    """
    accumulator = ( accumulator + gradient**2 )

    update = ( eta * gradient / (np.sqrt(accumulator) + eps) )
    return update, accumulator

def rmsprop_update(
    gradient,
    accumulator,
    eta,
    rho=0.9,
    eps=1.0e-8,
):
    """
    RMSProp update.
    """
    accumulator = ( rho * accumulator + (1.0 - rho) * gradient**2 )

    update = ( eta * gradient / (np.sqrt(accumulator) + eps) )
    return update, accumulator

def adam_update(
    gradient,
    first_moment,
    second_moment,
    iteration,
    eta,
    beta1=0.9,
    beta2=0.999,
    eps=1.0e-8,
):
    """
    Adam update with bias correction.
    """
    first_moment = ( beta1 * first_moment + (1.0 - beta1) * gradient )
    second_moment = ( beta2 * second_moment + (1.0 - beta2) * gradient**2 )

    first_corrected = ( first_moment / (1.0 - beta1**iteration) )
    second_corrected = ( second_moment / (1.0 - beta2**iteration) )

    update = ( eta * first_corrected / (np.sqrt(second_corrected) + eps) )
    return ( update, first_moment, second_moment, )

def optimize_regression(
    X,
    y,
    eta,
    optimizer="plain",
    method="ols",
    lmbda=0.0,
    gradient_source="analytic",
    theta0=None,
    theta_reference=None,
    accuracy_tol=1.0e-4,
    max_iter=50000,
    seed=2026,
    gamma=0.9,
    rho=0.9,
    beta1=0.9,
    beta2=0.999,
    eps=1.0e-8,
    divergence_threshold=1.0e12,
):
    """
    Optimize OLS or Ridge using one of:

        plain
        momentum
        adagrad
        rmsprop
        adam

    If theta_reference is supplied, convergence is defined by

        ||theta - theta_reference|| / ||theta_reference||
            < accuracy_tol.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    rng = np.random.default_rng(seed)

    if theta0 is None:
        theta = rng.normal(
            size=X.shape[1]
        )
    else:
        theta = np.asarray(
            theta0,
            dtype=float,
        ).copy()

    # ------------------------------------------------------------
    # Optimizer states
    # ------------------------------------------------------------

    velocity = np.zeros_like(theta)

    accumulator = np.zeros_like(theta)

    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)

    cost_history = []
    relative_error_history = []

    converged = False
    diverged = False

    if theta_reference is not None:

        theta_reference = np.asarray( theta_reference, dtype=float, )
        reference_norm = max( np.linalg.norm(theta_reference), eps, )

    # ------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------

    for iteration in range( 1, max_iter + 1, ):

        # --------------------------------------------------------
        # Gradient
        # --------------------------------------------------------

        if gradient_source == "analytic":
            if method == "ols":
                gradient = ols_gradient_analytic( theta, X, y, )
            elif method == "ridge":
                gradient = ridge_gradient_analytic( theta, X, y, lmbda, )
            else:
                raise ValueError( "method must be 'ols' or 'ridge'." )

        elif gradient_source == "autodiff":
            gradient = regression_gradient_autodiff( theta, X, y, method=method, lmbda=lmbda, )

        else:
            raise ValueError( "gradient_source must be 'analytic' or 'autodiff'." )

        # --------------------------------------------------------
        # Cost
        # --------------------------------------------------------

        if method == "ols":
            cost = ols_cost( theta, X, y,)
        else:
            cost = ridge_cost( theta, X, y, lmbda,)

        cost_history.append(cost)

        # --------------------------------------------------------
        # Error relative to closed-form solution
        # --------------------------------------------------------

        if theta_reference is not None:

            relative_error = (
                np.linalg.norm(
                    theta - theta_reference
                )
                / reference_norm
            )

            relative_error_history.append(
                relative_error
            )

            if relative_error < accuracy_tol:
                converged = True
                break

        # --------------------------------------------------------
        # Optimizer update
        # --------------------------------------------------------

        if optimizer == "plain":

            update = eta * gradient

        elif optimizer == "momentum":

            update, velocity = momentum_update(
                gradient,
                velocity,
                eta,
                gamma=gamma,
            )

        elif optimizer == "adagrad":

            update, accumulator = adagrad_update(
                gradient,
                accumulator,
                eta,
                eps=eps,
            )

        elif optimizer == "rmsprop":

            update, accumulator = rmsprop_update(
                gradient,
                accumulator,
                eta,
                rho=rho,
                eps=eps,
            )

        elif optimizer == "adam":

            (
                update,
                first_moment,
                second_moment,
            ) = adam_update(
                gradient,
                first_moment,
                second_moment,
                iteration,
                eta,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
            )

        else:
            raise ValueError(
                "optimizer must be 'plain', "
                "'momentum', 'adagrad', "
                "'rmsprop' or 'adam'."
            )

        theta -= update

        # --------------------------------------------------------
        # Divergence check
        # --------------------------------------------------------

        if (
            not np.all(np.isfinite(theta))
            or np.linalg.norm(theta)
            > divergence_threshold
        ):
            diverged = True
            break

    # ------------------------------------------------------------
    # Final relative error
    # ------------------------------------------------------------

    if theta_reference is not None:

        relative_error = (
            np.linalg.norm(
                theta - theta_reference
            )
            / reference_norm
        )

    else:
        relative_error = np.nan

    return {
        "theta": theta,
        "iterations": iteration,
        "converged": converged,
        "diverged": diverged,
        "relative_theta_error": relative_error,
        "cost_history": np.asarray(cost_history),
        "relative_error_history":
            np.asarray(relative_error_history),
    }

def optimizer_learning_rate_sweep(
    X_train,
    y_train,
    X_test,
    y_test,
    theta_reference,
    eta_values_by_optimizer,
    method="ols",
    lmbda=0.0,
    theta0=None,
    accuracy_tol=1.0e-4,
    max_iter=50000,
    seed=2026,
):
    """
    Run several learning rates for each optimizer.
    """

    results = {}

    for optimizer, eta_values in (
        eta_values_by_optimizer.items()
    ):

        results[optimizer] = {}

        for eta in eta_values:

            result = optimize_regression(
                X=X_train,
                y=y_train,
                eta=eta,
                optimizer=optimizer,
                method=method,
                lmbda=lmbda,
                gradient_source="analytic",
                theta0=theta0,
                theta_reference=theta_reference,
                accuracy_tol=accuracy_tol,
                max_iter=max_iter,
                seed=seed,
            )

            y_test_pred = predict(
                X_test,
                result["theta"],
            )

            result["test_mse"] = (
                mean_squared_error(
                    y_test,
                    y_test_pred,
                )
            )

            results[optimizer][eta] = result

    return results

def select_best_optimizer_runs(
    sweep_results,
):
    """
    Select the converged learning rate requiring the
    fewest iterations for each optimizer.
    """

    best_results = {}

    for optimizer, runs in sweep_results.items():

        converged_runs = [
            (eta, result)
            for eta, result in runs.items()
            if result["converged"]
        ]

        if len(converged_runs) == 0:

            best_results[optimizer] = None
            continue

        best_eta, best_result = min(
            converged_runs,
            key=lambda item:
                item[1]["iterations"],
        )

        best_results[optimizer] = {
            "eta": best_eta,
            "result": best_result,
        }

    return best_results