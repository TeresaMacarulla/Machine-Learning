import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

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

def plot_fits(
    x,
    y,
    fit_results,
    n_plot_points=500,
    save_path=None,
    method="OLS",
    lmbda=None,
):
    """
    Plot Runge's function, the generated data, and several
    polynomial approximations in the same figure.

    fit_results must be a dictionary of the form
        degree: result

    where result is the dictionary returned by fit_and_evaluate().
    """
    x_plot = np.linspace(-1.0, 1.0, n_plot_points)
    f_plot = runge(x_plot)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        x_plot,
        f_plot,
        linewidth=2,
        label="Exact Runge function",
    )

    ax.scatter(
        x,
        y,
        s=20,
        alpha=0.5,
        label="Generated data",
    )

    for degree, result in sorted(fit_results.items()):

        X_plot = design_matrix(x_plot, degree)

        X_plot_scaled = scale_design_matrix(
            X_plot,
            result["feature_means"],
            result["feature_stds"],
        )

        y_plot_pred = predict(
            X_plot_scaled,
            result["theta"],
        )

        ax.plot(
            x_plot,
            y_plot_pred,
            linestyle="--",
            label=rf"$d={degree}$",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if method.lower() == "ridge" and lmbda is not None:
        ax.set_title(
            rf"Ridge approximation of Runge's function "
            rf"($\lambda={lmbda:.0e}$)"
        )
    else:
        ax.set_title(
            f"{method} approximation of Runge's function"
        )

    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig, ax

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

def plot_ridge_results(
    degrees,
    ridge_lambdas,
    ridge_results,
    mse_test_ols,
    r2_test_ols,
    theta_ols_by_degree,
    theta_ridge_by_degree,
    comparison_degree,
    baseline_n,
    baseline_sigma,
    save_dir,
):
    """
    Generate the main Ridge result figures:

    1. OLS vs Ridge test MSE
    2. OLS vs Ridge test R^2
    3. OLS vs Ridge coefficients at one polynomial degree
    4. Ridge coefficients only at the same degree
    """

    # Find the index corresponding to comparison_degree.
    degree_index = np.where(
        degrees == comparison_degree
    )[0][0]

    # ============================================================
    # OLS vs Ridge: test MSE
    # ============================================================

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        degrees,
        mse_test_ols,
        marker="o",
        linewidth=2,
        label="OLS",
    )

    for lmbda in ridge_lambdas:
        ax.plot(
            degrees,
            ridge_results[lmbda]["mse_test"],
            marker="o",
            label=rf"Ridge $\lambda={lmbda:.0e}$",
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Test MSE")
    ax.set_title(
        rf"OLS and Ridge regression "
        rf"($n={baseline_n}$, $\sigma={baseline_sigma}$)"
    )

    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        save_dir / "Ridge_MSE_vs_degree.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    # ============================================================
    # OLS vs Ridge: test R^2
    # ============================================================

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        degrees,
        r2_test_ols,
        marker="o",
        linewidth=2,
        label="OLS",
    )

    for lmbda in ridge_lambdas:
        ax.plot(
            degrees,
            ridge_results[lmbda]["r2_test"],
            marker="o",
            label=rf"Ridge $\lambda={lmbda:.0e}$",
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"Test $R^2$")
    ax.set_title(
        rf"OLS and Ridge regression "
        rf"($n={baseline_n}$, $\sigma={baseline_sigma}$)"
    )

    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        save_dir / "Ridge_R2_vs_degree.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    # ============================================================
    # OLS + Ridge coefficient comparison
    # ============================================================

    ols_theta = theta_ols_by_degree[degree_index]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        np.arange(1, len(ols_theta)),
        ols_theta[1:],
        marker="o",
        linewidth=2,
        label="OLS",
    )

    for lmbda in ridge_lambdas:

        theta = theta_ridge_by_degree[lmbda][degree_index]

        ax.plot(
            np.arange(1, len(theta)),
            theta[1:],
            marker="o",
            label=rf"Ridge $\lambda={lmbda:.0e}$",
        )

    ax.set_xlabel(r"Coefficient index $j$")
    ax.set_ylabel(r"$\theta_j$")
    ax.set_title(
        rf"OLS and Ridge coefficients, "
        rf"degree {comparison_degree}"
    )

    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        save_dir
        / f"Ridge_OLS_coefficients_degree{comparison_degree}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    # ============================================================
    # Ridge coefficients only
    # ============================================================

    fig, ax = plt.subplots(figsize=(9, 6))

    for lmbda in ridge_lambdas:

        theta = theta_ridge_by_degree[lmbda][degree_index]

        ax.plot(
            np.arange(1, len(theta)),
            theta[1:],
            marker="o",
            label=rf"$\lambda={lmbda:.0e}$",
        )

    ax.set_xlabel(r"Coefficient index $j$")
    ax.set_ylabel(r"$\theta_j$")
    ax.set_title(
        rf"Ridge coefficient shrinkage, "
        rf"degree {comparison_degree}"
    )

    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        save_dir
        / f"Ridge_coefficients_degree{comparison_degree}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)