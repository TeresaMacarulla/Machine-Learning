import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

"""
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})
"""

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
        (1/n) ||y - X theta||^2 + lambda ||theta||^2.
    The intercept theta_0 is not penalized.
    """
    if lmbda < 0:
        raise ValueError("lambda must be non-negative.")
    if lmbda == 0:
        return fit_ols(X, y)
    n, p = X.shape

    # Do not penalize the intercept.
    penalty = np.eye(p)
    penalty[0, 0] = 0.0

    return np.linalg.solve(
        X.T @ X + n * lmbda * penalty,
        X.T @ y,
    )

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