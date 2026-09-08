import numpy as np
import matplotlib.pyplot as plt

from utilities import (
    design_matrix,
    scale_design_matrix,
    runge,
    predict,
)

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

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

    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig, ax

def plot_OLS_degree_results(
    degrees, 
    mse_train_degree,
    mse_test_degree,
    baseline_n,
    baseline_sigma,
    r2_train_degree,
    r2_test_degree,
    save_path=None,
):
    """
    Generate the main OLS study 1 result figures:

    1. MSE as a function of polynomial degree
    2. R^2 as a function of polynomial degree
    """

    # ============================================================
    # MSE as a function of polynomial degree
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(degrees, mse_train_degree, marker="o", label="Training MSE")
    ax.plot(degrees, mse_test_degree, marker="o", label="Test MSE")

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE")
    ax.set_title(
        rf"OLS: MSE vs polynomial degree "
        rf"($n={baseline_n}$, $\sigma={baseline_sigma}$)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path/ "OLS_MSE_vs_degree.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    # ============================================================
    # R^2 as a function of polynomial degree
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(degrees, r2_train_degree, marker="o", label="Training $R^2$")
    ax.plot(degrees, r2_test_degree, marker="o", label="Test $R^2$")

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(
        rf"OLS: $R^2$ vs polynomial degree "
        rf"($n={baseline_n}$, $\sigma={baseline_sigma}$)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path/ "OLS_R2_vs_degree.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

def plot_OLS_n_results(
    n_values,
    degrees,
    mse_test_by_n,
    baseline_sigma,
    r2_test_by_n,
    save_path=None,
):
    """
    Generate the main OLS study 2 result figures:

    1. Test MSE: compare different sample sizes
    2. Test R^2: compare different sample sizes
    """
     
    # ============================================================
    #Test MSE: compare different sample sizes
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for n in n_values:
        ax.plot(
            degrees,
            mse_test_by_n[n],
            marker="o",
            label=rf"$n={n}$",
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Test MSE")
    ax.set_title(
        rf"Effect of sample size on OLS "
        rf"($\sigma={baseline_sigma}$)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path/ "OLS_MSE_vs_degree_different_n.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    # ============================================================
    #Test R^2: compare different sample sizes
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for n in n_values:
        ax.plot(
            degrees,
            r2_test_by_n[n],
            marker="o",
            label=rf"$n={n}$",
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"Test $R^2$")
    ax.set_title(
        rf"Effect of sample size on OLS "
        rf"($\sigma={baseline_sigma}$)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path/ "OLS_R2_vs_degree_different_n.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

def plot_OLS_sigma_results(
    sigma_values,
    degrees,
    mse_test_by_sigma,
    baseline_n,
    r2_test_by_sigma,
    save_path=None,
):
    """
    Generate the main OLS study 3 result figures:

    1. Test MSE: compare different noise levels
    2. Test R^2: compare different noise levels
    """

    # ============================================================
    #Test MSE: compare different noise levels
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for sigma in sigma_values:
        ax.plot(
            degrees,
            mse_test_by_sigma[sigma],
            marker="o",
            label=rf"$\sigma={sigma}$",
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Test MSE")
    ax.set_title(
        rf"Effect of noise on OLS "
        rf"($n={baseline_n}$)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path/ "OLS_MSE_vs_degree_different_sigma.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    # ============================================================
    #Test R^2: compare different noise levels
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for sigma in sigma_values:
        ax.plot(
            degrees,
            r2_test_by_sigma[sigma],
            marker="o",
            label=rf"$\sigma={sigma}$",
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"Test $R^2$")
    ax.set_title(
        rf"Effect of noise on OLS "
        rf"($n={baseline_n}$)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path/ "OLS_R2_vs_degree_different_sigma.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

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
            linewidth=1.5,
            marker="+",
            label=rf"Ridge $\lambda={lmbda:.0e}$",
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Test MSE")
    ax.set_title(
        rf"OLS and Ridge regression "
        rf"($n={baseline_n}$, $\sigma={baseline_sigma}$)"
    )

    ax.legend()
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
            linewidth=1.5,
            marker="+",
            label=rf"Ridge $\lambda={lmbda:.0e}$",
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"Test $R^2$")
    ax.set_title(
        rf"OLS and Ridge regression "
        rf"($n={baseline_n}$, $\sigma={baseline_sigma}$)"
    )

    ax.legend()
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
            marker="+",
            linewidth=1.5,
            label=rf"Ridge $\lambda={lmbda:.0e}$",
        )

    ax.set_xlabel(r"Coefficient index $j$")
    ax.set_ylabel(r"$\theta_j$")
    ax.set_title(
        rf"OLS and Ridge coefficients, "
        rf"degree {comparison_degree}"
    )

    ax.legend()
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
            marker="+",
            label=rf"$\lambda={lmbda:.0e}$",
        )

    ax.set_xlabel(r"Coefficient index $j$")
    ax.set_ylabel(r"$\theta_j$")
    ax.set_title(
        rf"Ridge coefficient shrinkage, "
        rf"degree {comparison_degree}"
    )

    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        save_dir
        / f"Ridge_coefficients_degree{comparison_degree}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)