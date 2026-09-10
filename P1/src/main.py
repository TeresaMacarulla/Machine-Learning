import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

from utilities import (
    artificial_data,
    design_matrix,
    fit_feature_scaler,
    scale_design_matrix,
    fit_and_evaluate,
    bootstrap_bias_variance,
    cross_validation_mse,
)

from plot_generator import (
    plot_fits,
    plot_ridge_results,
    plot_OLS_degree_results,
    plot_OLS_n_results,
    plot_OLS_sigma_results,
    plot_bias_variance_errors,
    plot_bootstrap_bias_variance,
    plot_cross_validation_comparison,
)

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

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

    # Polynomial degrees required for the main Part a) and Part b) study.
    degrees = np.arange(1, 16)

    # Baseline values suggested by the project description.
    baseline_n = 100
    baseline_sigma = 0.1

    # Values chosen to study the effect of sample size and noise.
    n_values = [100, 200, 400]
    sigma_values = [0.05, 0.1, 0.2, 0.3]

    # ============================================================
    # PART A STUDY 1 OLS: Polynomial degree
    # Keep n and sigma fixed.
    # ============================================================

    # Generate ONE data set and ONE train/test split.
    # All polynomial degrees are evaluated on exactly the same data.
    x, y = artificial_data( n=baseline_n, sigma=baseline_sigma, seed=seed, )

    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=test_size, random_state=seed, )

    mse_train_degree = []
    mse_test_degree = []
    r2_train_degree = []
    r2_test_degree = []
    theta_by_degree = []

    # Save fitted-function plots only for a few representative degrees.
    representative_degrees = [2, 5, 10, 15]
    ols_representative_fits = {}

    for degree in degrees:
        result = fit_and_evaluate( x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, 
                                  degree=degree, method="ols" )

        mse_train_degree.append(result["mse_train"])
        mse_test_degree.append(result["mse_test"])
        r2_train_degree.append(result["r2_train"])
        r2_test_degree.append(result["r2_test"])
        theta_by_degree.append(result["theta"])

        # A few representative fitted curves are useful for the report.
        if degree in representative_degrees:
            ols_representative_fits[degree] = result

    fig, _ = plot_fits( x=x, y=y, fit_results=ols_representative_fits, n_plot_points=500,
                        save_path=PLOTS_DIR / "part_a/OLS_representative_fits.pdf", method="OLS", )
    plt.close(fig)

    mse_train_degree = np.asarray(mse_train_degree)
    mse_test_degree = np.asarray(mse_test_degree)
    r2_train_degree = np.asarray(r2_train_degree)
    r2_test_degree = np.asarray(r2_test_degree)

    # ------------------------------------------------------------
    # Main Study 1 OLS result plots
    # ------------------------------------------------------------

    plot_OLS_degree_results( degrees, mse_train_degree, mse_test_degree, baseline_n, baseline_sigma,
                            r2_train_degree, r2_test_degree, save_path=PLOTS_DIR / "part_a", )

    # ------------------------------------------------------------
    # OLS parameters theta as polynomial degree increases
    # ------------------------------------------------------------

    # Each degree has a different number of coefficients.
    # We therefore create a rectangular array and fill missing
    # coefficients with NaN so Matplotlib does not draw them.
    max_degree = int(np.max(degrees))
    theta_matrix = np.full((len(degrees), max_degree + 1), np.nan, )

    for row, theta in enumerate(theta_by_degree):
        theta_matrix[row, : len(theta)] = theta

    fig, ax = plt.subplots(figsize=(9, 6))

    for j in range(max_degree + 1):
        ax.plot( degrees, theta_matrix[:, j], marker="o", markersize=3, label=rf"$\theta_{j}$", )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"OLS parameter $\theta_j$")
    ax.set_title( rf"OLS parameters vs polynomial degree " rf"($n={baseline_n}$, $\sigma={baseline_sigma}$)" )
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()

    fig.savefig( PLOTS_DIR / "part_a/OLS_theta_vs_degree.pdf", bbox_inches="tight", )
    plt.close(fig)


    # ============================================================
    # PART A STUDY 2 OLS: Number of data points n
    # Keep sigma fixed and repeat the degree sweep.
    # ============================================================
    
    # From now on we change the baseline n to 200 to avoid the great peaks in n=100 case
    baseline_n = 200
    mse_test_by_n = {}
    r2_test_by_n = {}

    for n in n_values:
        x_n, y_n = artificial_data( n=n, sigma=baseline_sigma, seed=seed, )

        x_train_n, x_test_n, y_train_n, y_test_n = train_test_split( x_n, y_n, test_size=test_size, 
                                                                    random_state=seed, )

        mse_curve = []
        r2_curve = []

        # Use the same data and split for every degree at this n.
        for degree in degrees:
            result = fit_and_evaluate( x_train=x_train_n, x_test=x_test_n, y_train=y_train_n, y_test=y_test_n,
                                        degree=degree, method="ols" )

            mse_curve.append(result["mse_test"])
            r2_curve.append(result["r2_test"])

        mse_test_by_n[n] = np.asarray(mse_curve)
        r2_test_by_n[n] = np.asarray(r2_curve)

    # ------------------------------------------------------------
    # Main Study 2 OLS result plots
    # ------------------------------------------------------------

    plot_OLS_n_results( n_values, degrees, mse_test_by_n, baseline_sigma, r2_test_by_n, save_path=PLOTS_DIR / "part_a", )


    # ============================================================
    # PART A STUDY 3 OLS: Noise level sigma
    # Keep n fixed and repeat the degree sweep.
    # ============================================================

    mse_test_by_sigma = {}
    r2_test_by_sigma = {}

    for sigma in sigma_values:
        # Because artificial_data uses the same seed and n here,
        # the x-values and underlying standard-normal random sample
        # are the same; sigma changes only the noise amplitude.
        x_sigma, y_sigma = artificial_data( n=baseline_n, sigma=sigma, seed=seed, )

        ( x_train_sigma, x_test_sigma, y_train_sigma, y_test_sigma, 
        ) = train_test_split( x_sigma, y_sigma, test_size=test_size, random_state=seed, )

        mse_curve = []
        r2_curve = []

        for degree in degrees:
            result = fit_and_evaluate( x_train=x_train_sigma, x_test=x_test_sigma, 
                                      y_train=y_train_sigma, y_test=y_test_sigma,
                                      degree=degree, method="ols" )

            mse_curve.append(result["mse_test"])
            r2_curve.append(result["r2_test"])

        mse_test_by_sigma[sigma] = np.asarray(mse_curve)
        r2_test_by_sigma[sigma] = np.asarray(r2_curve)


    # ------------------------------------------------------------
    # Main Study 3 OLS result plots
    # ------------------------------------------------------------

    plot_OLS_sigma_results( sigma_values, degrees, mse_test_by_sigma, baseline_n, 
                           r2_test_by_sigma, save_path=PLOTS_DIR / "part_a", )

    # ------------------------------------------------------------
    # Print a compact numerical summary for OLS
    # ------------------------------------------------------------

    best_degree_baseline = degrees[np.argmin(mse_test_degree)]

    print("Baseline study OLS")
    print("--------------")
    print(f"n = 100")
    print(f"sigma = {baseline_sigma}")
    print(f"Best degree by test MSE: {best_degree_baseline}")
    print(
        f"Minimum test MSE: "
        f"{np.min(mse_test_degree):.6f}"
    )
    print()

    print("Sample-size study OLS")
    print("-----------------")
    for n in n_values:
        best_degree = degrees[np.argmin(mse_test_by_n[n])]
        print(
            f"n={n:3d}: best degree={best_degree:2d}, "
            f"test MSE={np.min(mse_test_by_n[n]):.6f}"
        )
    print()

    print("Noise study OLS")
    print("-----------")
    for sigma in sigma_values:
        best_degree = degrees[np.argmin(mse_test_by_sigma[sigma])]
        print(
            f"sigma={sigma:4.2f}: best degree={best_degree:2d}, "
            f"test MSE={np.min(mse_test_by_sigma[sigma]):.6f}"
        )


    # ============================================================
    # PART B: Ridge regression
    # ============================================================

    # We need to run again OLS but for n=200 case to make later comparisons
    # Generate ONE data set and ONE train/test split.
    # All polynomial degrees are evaluated on exactly the same data.
    x, y = artificial_data( n=baseline_n, sigma=baseline_sigma, seed=seed, )

    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=test_size, random_state=seed, )

    mse_train_degree = []
    mse_test_degree = []
    r2_train_degree = []
    r2_test_degree = []
    theta_by_degree = []

    for degree in degrees:
        result = fit_and_evaluate( x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, 
                                   degree=degree, method="ols" )

        mse_train_degree.append(result["mse_train"])
        mse_test_degree.append(result["mse_test"])
        r2_train_degree.append(result["r2_train"])
        r2_test_degree.append(result["r2_test"])
        theta_by_degree.append(result["theta"])

    mse_train_degree = np.asarray(mse_train_degree)
    mse_test_degree = np.asarray(mse_test_degree)
    r2_train_degree = np.asarray(r2_train_degree)
    r2_test_degree = np.asarray(r2_test_degree)
    
    ridge_lambdas = np.logspace(-6, 0, 7)
    representative_lmbdas = ridge_lambdas[[0, 3, 6]]
    ridge_representative_fits = {
        lmbda: {} for lmbda in representative_lmbdas
    }

    ridge_results = {}
    ridge_theta_by_degree = {}

    for lmbda in ridge_lambdas:

        mse_train = []
        mse_test = []
        r2_train = []
        r2_test = []
        theta_list = []

        for degree in degrees:

            result = fit_and_evaluate( x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                       degree=degree, method="ridge", lmbda=lmbda, )

            mse_train.append(result["mse_train"])
            mse_test.append(result["mse_test"])
            r2_train.append(result["r2_train"])
            r2_test.append(result["r2_test"])
            theta_list.append(result["theta"])

            if (
                degree in representative_degrees
                and lmbda in representative_lmbdas
            ):
                ridge_representative_fits[lmbda][degree] = result

        ridge_results[lmbda] = {
            "mse_train": np.asarray(mse_train),
            "mse_test": np.asarray(mse_test),
            "r2_train": np.asarray(r2_train),
            "r2_test": np.asarray(r2_test),
        }

        ridge_theta_by_degree[lmbda] = theta_list

    for lmbda in representative_lmbdas:

        fig, _ = plot_fits( x=x, y=y, fit_results=ridge_representative_fits[lmbda], n_plot_points=500,
                            save_path=PLOTS_DIR / f"part_b/Ridge_representative_fits_lambda{lmbda:.0e}.pdf",
                            method="Ridge", lmbda=lmbda, )

        plt.close(fig)

    # ------------------------------------------------------------
    # Main Ridge result plots
    # ------------------------------------------------------------

    plot_ridge_results( degrees=degrees, ridge_lambdas=ridge_lambdas, ridge_results=ridge_results,
                        mse_test_ols=mse_test_degree, r2_test_ols=r2_test_degree,
                        theta_ols_by_degree=theta_by_degree, theta_ridge_by_degree=ridge_theta_by_degree,
                        comparison_degree=15, baseline_n=baseline_n, baseline_sigma=baseline_sigma,
                        save_dir=PLOTS_DIR / "part_b", )

    # ------------------------------------------------------------
    # Ridge shrinkage factors from singular values
    # ------------------------------------------------------------

    shrinkage_degree = 15

    X_train_shrink = design_matrix( x_train, shrinkage_degree, )

    feature_means, feature_stds = fit_feature_scaler( X_train_shrink )

    X_train_shrink = scale_design_matrix( X_train_shrink, feature_means, feature_stds, )

    # Exclude the intercept because it is not regularized.
    X_features = X_train_shrink[:, 1:]

    singular_values = np.linalg.svd( X_features, compute_uv=False, )

    n_train = len(y_train)

    fig, ax = plt.subplots(figsize=(8, 5))

    for lmbda in ridge_lambdas:

        shrinkage = ( singular_values**2 / (singular_values**2 + n_train * lmbda) )

        ax.plot( singular_values, shrinkage, marker="+", label=rf"$\lambda={lmbda:.0e}$", )

    ax.set_xscale("log")
    ax.set_xlabel(r"Singular value $\sigma_i$")
    ax.set_ylabel(
        r"Shrinkage factor "
        r"$\sigma_i^2/(\sigma_i^2+n\lambda)$"
    )
    ax.set_title(
        rf"Ridge singular-value shrinkage "
        rf"(degree {shrinkage_degree})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / "part_b/Ridge_shrinkage_factors.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Print a compact numerical summary for Ridge
    # ------------------------------------------------------------

    print()
    print("Ridge study")
    print("-----------")

    for lmbda in ridge_lambdas:

        mse_curve = ridge_results[lmbda]["mse_test"]

        best_index = np.argmin(mse_curve)
        best_degree = degrees[best_index]

        print(
            f"lambda={lmbda:.0e}: "
            f"best degree={best_degree:2d}, "
            f"test MSE={mse_curve[best_index]:.6f}, "
            f"R2={ridge_results[lmbda]['r2_test'][best_index]:.6f}"
        )


    # ============================================================
    # PART C: Bias-variance trade-off
    # Training and test error vs model complexity
    # ============================================================

    bias_variance_n_values = [100,200, 400]
    bias_variance_sigma = 0.1

    bias_variance_degrees = {
        100: np.arange(1, 16),
        200: np.arange(1, 26),
        400: np.arange(1, 41),
    }

    mse_train_bias_variance = {}
    mse_test_bias_variance = {}
    data_splits_by_n = {}

    for n in bias_variance_n_values:

        # Generate one data set for this sample size.
        x_n, y_n = artificial_data( n=n, sigma=bias_variance_sigma, seed=seed, )

        # Keep the same train/test split for every polynomial degree.
        split = train_test_split( x_n, y_n, test_size=test_size, random_state=seed, )
        x_train_n, x_test_n, y_train_n, y_test_n = split
        data_splits_by_n[n] = split

        mse_train_curve = []
        mse_test_curve = []

        for degree in bias_variance_degrees[n]:

            result = fit_and_evaluate( x_train=x_train_n, x_test=x_test_n, y_train=y_train_n, y_test=y_test_n,
                                       degree=degree, method="ols", )

            mse_train_curve.append(result["mse_train"])
            mse_test_curve.append(result["mse_test"])

        mse_train_bias_variance[n] = np.asarray( mse_train_curve )

        mse_test_bias_variance[n] = np.asarray( mse_test_curve )

    plot_bias_variance_errors( n_values=bias_variance_n_values, mse_train_by_n=mse_train_bias_variance,
                               mse_test_by_n=mse_test_bias_variance, sigma=bias_variance_sigma,
                               save_path=PLOTS_DIR / "part_c", )


    # ============================================================
    # PART C: Bootstrap bias-variance decomposition
    # PART D: k-fold cross-validation
    # ============================================================

    n_bootstraps = 100
    bootstrap_results_by_n = {}

    cv5_results_by_n = {}
    cv10_results_by_n = {}

    for n in bias_variance_n_values:

        ( x_train_n, x_test_n, y_train_n, y_test_n, ) = data_splits_by_n[n]

        bootstrap_results_by_n[n] = bootstrap_bias_variance( x_train=x_train_n, x_test=x_test_n,
                                                             y_train=y_train_n, y_test=y_test_n,
                                                             degrees=bias_variance_degrees[n],
                                                             n_bootstraps=n_bootstraps, seed=seed, )

        degrees_n = bias_variance_degrees[n]

        # 5-fold cross-validation on the training data.
        cv5_results_by_n[n] = cross_validation_mse( x=x_train_n, y=y_train_n, degrees=degrees_n, n_splits=5, seed=seed, )

        # 10-fold cross-validation on the same training data.
        cv10_results_by_n[n] = cross_validation_mse( x=x_train_n, y=y_train_n, degrees=degrees_n, n_splits=10, seed=seed, )

    plot_bootstrap_bias_variance( results_by_n=bootstrap_results_by_n, sigma=bias_variance_sigma, save_path=PLOTS_DIR / "part_c", )

    plot_cross_validation_comparison( bootstrap_results_by_n=bootstrap_results_by_n, cv5_results_by_n=cv5_results_by_n,
                                      cv10_results_by_n=cv10_results_by_n, sigma=bias_variance_sigma, save_path=PLOTS_DIR / "part_d", )

    print()
    print("Cross-validation study OLS")
    print("--------------------------")

    for n in bias_variance_n_values:

        degrees_n = bias_variance_degrees[n]

        cv5 = cv5_results_by_n[n]["mean_mse"]
        cv10 = cv10_results_by_n[n]["mean_mse"]

        best_5 = np.argmin(cv5)
        best_10 = np.argmin(cv10)

        print(
            f"n={n:3d}: "
            f"5-fold -> d={degrees_n[best_5]:2d}, "
            f"MSE={cv5[best_5]:.6f}; "
            f"10-fold -> d={degrees_n[best_10]:2d}, "
            f"MSE={cv10[best_10]:.6f}"
        )


if __name__ == "__main__":
    main()
