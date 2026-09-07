import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

from utilities import (
    artificial_data,
    fit_and_evaluate_ols,
    plot_ols_fit,
)


# Path to the root P1 folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the plots folder
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ============================================================
    # General experiment settings
    # ============================================================

    seed = 2026
    test_size = 0.20

    # Polynomial degrees required for the main Part a) study.
    degrees = np.arange(1, 16)

    # Baseline values suggested by the project description.
    baseline_n = 100
    baseline_sigma = 0.1

    # Values chosen to study the effect of sample size and noise.
    n_values = [25, 100, 200]
    sigma_values = [0.05, 0.1, 0.2, 0.3]


    # ============================================================
    # STUDY 1: Polynomial degree
    # Keep n and sigma fixed.
    # ============================================================

    # Generate ONE data set and ONE train/test split.
    # All polynomial degrees are evaluated on exactly the same data.
    x, y = artificial_data(
        n=baseline_n,
        sigma=baseline_sigma,
        seed=seed,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
    )

    mse_train_degree = []
    mse_test_degree = []
    r2_train_degree = []
    r2_test_degree = []
    theta_by_degree = []

    # Save fitted-function plots only for a few representative degrees.
    representative_degrees = [2, 5, 10, 13, 14, 15]

    for degree in degrees:
        result = fit_and_evaluate_ols(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            degree=degree,
        )

        mse_train_degree.append(result["mse_train"])
        mse_test_degree.append(result["mse_test"])
        r2_train_degree.append(result["r2_train"])
        r2_test_degree.append(result["r2_test"])
        theta_by_degree.append(result["theta"])

        # A few representative fitted curves are useful for the report.
        if degree in representative_degrees:
            fig, _ = plot_ols_fit(
                x=x,
                y=y,
                degree=degree,
                theta=result["theta"],
                feature_means=result["feature_means"],
                feature_stds=result["feature_stds"],
                n_plot_points=500,
                save_path=PLOTS_DIR
                / f"OLS_baseline_n{baseline_n}_sig{baseline_sigma}_deg{degree}.pdf",
            )
            plt.close(fig)

    mse_train_degree = np.asarray(mse_train_degree)
    mse_test_degree = np.asarray(mse_test_degree)
    r2_train_degree = np.asarray(r2_train_degree)
    r2_test_degree = np.asarray(r2_test_degree)


    # ------------------------------------------------------------
    # MSE as a function of polynomial degree
    # ------------------------------------------------------------
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

    fig.savefig(
        PLOTS_DIR / "OLS_MSE_vs_degree.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


    # ------------------------------------------------------------
    # R^2 as a function of polynomial degree
    # ------------------------------------------------------------
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

    fig.savefig(
        PLOTS_DIR / "OLS_R2_vs_degree.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


    # ------------------------------------------------------------
    # OLS parameters theta as polynomial degree increases
    # ------------------------------------------------------------

    # Each degree has a different number of coefficients.
    # We therefore create a rectangular array and fill missing
    # coefficients with NaN so Matplotlib does not draw them.
    max_degree = int(np.max(degrees))
    theta_matrix = np.full(
        (len(degrees), max_degree + 1),
        np.nan,
    )

    for row, theta in enumerate(theta_by_degree):
        theta_matrix[row, : len(theta)] = theta

    fig, ax = plt.subplots(figsize=(9, 6))

    for j in range(max_degree + 1):
        ax.plot(
            degrees,
            theta_matrix[:, j],
            marker="o",
            markersize=3,
            label=rf"$\theta_{j}$",
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"OLS parameter $\theta_j$")
    ax.set_title(
        rf"OLS parameters vs polynomial degree "
        rf"($n={baseline_n}$, $\sigma={baseline_sigma}$)"
    )
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / "OLS_theta_vs_degree.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


    # ============================================================
    # STUDY 2: Number of data points n
    # Keep sigma fixed and repeat the degree sweep.
    # ============================================================

    mse_test_by_n = {}
    r2_test_by_n = {}

    for n in n_values:
        x_n, y_n = artificial_data(
            n=n,
            sigma=baseline_sigma,
            seed=seed,
        )

        x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(
            x_n,
            y_n,
            test_size=test_size,
            random_state=seed,
        )

        mse_curve = []
        r2_curve = []

        # Use the same data and split for every degree at this n.
        for degree in degrees:
            result = fit_and_evaluate_ols(
                x_train=x_train_n,
                x_test=x_test_n,
                y_train=y_train_n,
                y_test=y_test_n,
                degree=degree,
            )

            mse_curve.append(result["mse_test"])
            r2_curve.append(result["r2_test"])

        mse_test_by_n[n] = np.asarray(mse_curve)
        r2_test_by_n[n] = np.asarray(r2_curve)


    # ------------------------------------------------------------
    # Test MSE: compare different sample sizes
    # ------------------------------------------------------------
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

    fig.savefig(
        PLOTS_DIR / "OLS_MSE_vs_degree_different_n.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


    # ------------------------------------------------------------
    # Test R^2: compare different sample sizes
    # ------------------------------------------------------------
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

    fig.savefig(
        PLOTS_DIR / "OLS_R2_vs_degree_different_n.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


    # ============================================================
    # STUDY 3: Noise level sigma
    # Keep n fixed and repeat the degree sweep.
    # ============================================================

    mse_test_by_sigma = {}
    r2_test_by_sigma = {}

    for sigma in sigma_values:
        # Because artificial_data uses the same seed and n here,
        # the x-values and underlying standard-normal random sample
        # are the same; sigma changes only the noise amplitude.
        x_sigma, y_sigma = artificial_data(
            n=baseline_n,
            sigma=sigma,
            seed=seed,
        )

        (
            x_train_sigma,
            x_test_sigma,
            y_train_sigma,
            y_test_sigma,
        ) = train_test_split(
            x_sigma,
            y_sigma,
            test_size=test_size,
            random_state=seed,
        )

        mse_curve = []
        r2_curve = []

        for degree in degrees:
            result = fit_and_evaluate_ols(
                x_train=x_train_sigma,
                x_test=x_test_sigma,
                y_train=y_train_sigma,
                y_test=y_test_sigma,
                degree=degree,
            )

            mse_curve.append(result["mse_test"])
            r2_curve.append(result["r2_test"])

        mse_test_by_sigma[sigma] = np.asarray(mse_curve)
        r2_test_by_sigma[sigma] = np.asarray(r2_curve)


    # ------------------------------------------------------------
    # Test MSE: compare different noise levels
    # ------------------------------------------------------------
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

    fig.savefig(
        PLOTS_DIR / "OLS_MSE_vs_degree_different_sigma.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


    # ------------------------------------------------------------
    # Test R^2: compare different noise levels
    # ------------------------------------------------------------
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

    fig.savefig(
        PLOTS_DIR / "OLS_R2_vs_degree_different_sigma.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


    # ============================================================
    # Print a compact numerical summary
    # ============================================================

    best_degree_baseline = degrees[np.argmin(mse_test_degree)]

    print("Baseline study")
    print("--------------")
    print(f"n = {baseline_n}")
    print(f"sigma = {baseline_sigma}")
    print(f"Best degree by test MSE: {best_degree_baseline}")
    print(
        f"Minimum test MSE: "
        f"{np.min(mse_test_degree):.6f}"
    )
    print()

    print("Sample-size study")
    print("-----------------")
    for n in n_values:
        best_degree = degrees[np.argmin(mse_test_by_n[n])]
        print(
            f"n={n:3d}: best degree={best_degree:2d}, "
            f"test MSE={np.min(mse_test_by_n[n]):.6f}"
        )
    print()

    print("Noise study")
    print("-----------")
    for sigma in sigma_values:
        best_degree = degrees[np.argmin(mse_test_by_sigma[sigma])]
        print(
            f"sigma={sigma:4.2f}: best degree={best_degree:2d}, "
            f"test MSE={np.min(mse_test_by_sigma[sigma]):.6f}"
        )


if __name__ == "__main__":
    main()
