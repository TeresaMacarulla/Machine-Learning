import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

from utilities import (
    artificial_data,
    design_matrix,
    fit_feature_scaler,
    scale_design_matrix,
    fit_and_evaluate,
    fit_ols,
    fit_ridge,
    predict,
    mean_squared_error,
    bootstrap_bias_variance,
    cross_validation_mse,
    compare_gradients,
    learning_rate_information,
    gradient_descent,
    ols_cost,
    ridge_cost,
    optimizer_learning_rate_sweep,
    select_best_optimizer_runs,
)

from plot_generator import (
    plot_fits,
    plot_ridge_results,
    plot_OLS_degree_results,
    plot_OLS_n_results,
    plot_OLS_sigma_results,
    plot_bias_variance_errors,
    plot_bootstrap_bias_variance,
    plot_validation_comparison,
    plot_gd_learning_rate_study,
    plot_optimizer_convergence,
    plot_optimizer_eta_sensitivity,
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

# ============================================================
# PART A: General experiment settings
# ============================================================
def run_part_a( seed=2026, test_size=0.20, ):

    print("\nRunning Part A...\n")

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
def run_part_b( seed=2026, test_size=0.20, ):

    print("\nRunning Part B...\n")

    representative_degrees = [2, 5, 10, 15]
    baseline_n = 200
    baseline_sigma = 0.1
    degrees = np.arange(1, 16)

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
# PART C and D: Preparation
# ============================================================
def prepare_part_cd_data( seed=2026, test_size=0.20, ):
    """
    Generate the common data and ordinary OLS results
    required by Parts C and D.
    """

    n_values = [100, 200, 400]
    sigma = 0.1

    degrees_by_n = { 100: np.arange(1, 16), 200: np.arange(1, 26), 400: np.arange(1, 41), }

    data_splits_by_n = {}
    data_nosplit_by_n = {}

    mse_train_by_n = {}
    mse_test_by_n = {}

    for n in n_values:

        x_n, y_n = artificial_data( n=n, sigma=sigma, seed=seed, )

        split = train_test_split( x_n, y_n, test_size=test_size, random_state=seed, )

        ( x_train_n, x_test_n, y_train_n, y_test_n, ) = split

        data_splits_by_n[n] = split
        data_nosplit_by_n[n] = (x_n, y_n)

        mse_train_curve = []
        mse_test_curve = []

        for degree in degrees_by_n[n]:

            result = fit_and_evaluate( x_train=x_train_n, x_test=x_test_n, y_train=y_train_n, y_test=y_test_n,
                                       degree=degree, method="ols", )

            mse_train_curve.append( result["mse_train"] )

            mse_test_curve.append( result["mse_test"] )

        mse_train_by_n[n] = np.asarray( mse_train_curve )

        mse_test_by_n[n] = np.asarray( mse_test_curve )

    return {
        "n_values": n_values,
        "sigma": sigma,
        "degrees_by_n": degrees_by_n,
        "data_splits_by_n": data_splits_by_n,
        "data_nosplit_by_n": data_nosplit_by_n,
        "mse_train_by_n": mse_train_by_n,
        "mse_test_by_n": mse_test_by_n,
    }


# ============================================================
# PART C: Bias-variance trade-off
# Training and test error vs model complexity
# Bootstrap bias-variance decomposition
# ============================================================
def run_part_c( seed=2026, test_size=0.20, ):
    print("\nRunning Part C...\n")

    data = prepare_part_cd_data( seed=seed, test_size=test_size, )

    n_values = data["n_values"]
    sigma = data["sigma"]
    degrees_by_n = data["degrees_by_n"]

    plot_bias_variance_errors(
        n_values=n_values,
        mse_train_by_n=data["mse_train_by_n"],
        mse_test_by_n=data["mse_test_by_n"],
        sigma=sigma,
        save_path=PLOTS_DIR / "part_c",
    )

    # Bootstrap
    bootstrap_results_by_n = {}

    for n in n_values:

        ( x_train, x_test, y_train, y_test, ) = data["data_splits_by_n"][n]

        bootstrap_results_by_n[n] = (
            bootstrap_bias_variance(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                degrees=degrees_by_n[n],
                n_bootstraps=100,
                seed=seed,
            )
        )

    plot_bootstrap_bias_variance(
        results_by_n=bootstrap_results_by_n,
        sigma=sigma,
        save_path=PLOTS_DIR / "part_c",
    )


# ============================================================
# PART D: k-fold cross-validation and compare all methods
# ============================================================
def run_part_d( seed=2026, test_size=0.20, ):
    print("\nRunning Part D...\n")

    data = prepare_part_cd_data( seed=seed, test_size=test_size, )

    n_values = data["n_values"]
    sigma = data["sigma"]
    degrees_by_n = data["degrees_by_n"]

    ridge_lambdas = np.logspace(-6, 0, 7)
    representative_lmbdas = ridge_lambdas[[0, 3, 6]]

    bootstrap_results_by_n = {}
    cv5_results_by_n = {}
    cv10_results_by_n = {}

    ridge_results_by_n = { n: {} for n in n_values }

    for n in n_values:

        ( x_train, x_test, y_train, y_test, ) = data["data_splits_by_n"][n]

        x_all, y_all = data["data_nosplit_by_n"][n]

        degrees_n = degrees_by_n[n]

        # Ridge
        for lmbda in representative_lmbdas:

            mse_test_ridge = []

            for degree in degrees_n:

                result = fit_and_evaluate( x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                           degree=degree, method="ridge", lmbda=lmbda, )

                mse_test_ridge.append( result["mse_test"] )

            ridge_results_by_n[n][lmbda] = {
                "mse_test":
                    np.asarray(mse_test_ridge)
            }

        # Bootstrap
        bootstrap_results_by_n[n] = (
            bootstrap_bias_variance( x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                     degrees=degrees_n, n_bootstraps=100, seed=seed, )
        )

        # Cross-validation uses all available data
        cv5_results_by_n[n] = cross_validation_mse( x=x_all, y=y_all, degrees=degrees_n,
                                                    n_splits=5, seed=seed, )

        cv10_results_by_n[n] = cross_validation_mse( x=x_all, y=y_all, degrees=degrees_n,
                                                     n_splits=10, seed=seed, )

    plot_validation_comparison(
        bootstrap_results_by_n=bootstrap_results_by_n,
        cv5_results_by_n=cv5_results_by_n,
        cv10_results_by_n=cv10_results_by_n,
        mse_test_by_n=data["mse_test_by_n"],
        ridge_lambdas=representative_lmbdas,
        ridge_results=ridge_results_by_n,
        sigma=sigma,
        save_path=PLOTS_DIR / "part_d",
    )


# ============================================================
# PART E: Plain gradient descent Analytical vs automatic differentiation
# ============================================================
def run_part_e( seed=2026, test_size=0.20, ):

    print("\nRunning Part E...\n")

    gd_n = 200
    gd_sigma = 0.1
    gd_degree = 5
    gd_lmbda = 1.0e-2

    # ------------------------------------------------------------
    # Generate the data and use the same preprocessing convention as in Parts A and B.
    # ------------------------------------------------------------

    x_gd, y_gd = artificial_data( n=gd_n, sigma=gd_sigma, seed=seed, )

    ( x_train_gd, x_test_gd, y_train_gd, y_test_gd, 
     ) = train_test_split( x_gd, y_gd, test_size=test_size, random_state=seed, )

    X_train_gd = design_matrix( x_train_gd, gd_degree, )

    X_test_gd = design_matrix( x_test_gd, gd_degree, )

    feature_means, feature_stds = fit_feature_scaler( X_train_gd )

    X_train_gd = scale_design_matrix( X_train_gd, feature_means, feature_stds, )

    X_test_gd = scale_design_matrix( X_test_gd, feature_means, feature_stds, )

    # Use exactly the same initial theta for every GD run.
    rng_gd = np.random.default_rng(seed)

    theta0 = rng_gd.normal( size=X_train_gd.shape[1] )

    # ------------------------------------------------------------
    # 1. Analytical gradient vs automatic differentiation
    # ------------------------------------------------------------

    ols_gradient_difference = compare_gradients( theta=theta0, X=X_train_gd,
                                                 y=y_train_gd, method="ols", )

    ridge_gradient_difference = compare_gradients( theta=theta0, X=X_train_gd, 
                                                   y=y_train_gd, method="ridge",
                                                   lmbda=gd_lmbda, )

    print()
    print("Gradient check")
    print("--------------")
    print(
        "OLS   max |AD - analytic| = "
        f"{ols_gradient_difference:.3e}"
    )
    print(
        "Ridge max |AD - analytic| = "
        f"{ridge_gradient_difference:.3e}"
    )

    # ------------------------------------------------------------
    # 2. Closed-form solutions for reference
    # ------------------------------------------------------------

    theta_ols_closed = fit_ols( X_train_gd, y_train_gd, )

    theta_ridge_closed = fit_ridge( X_train_gd, y_train_gd, gd_lmbda, )

    # ------------------------------------------------------------
    # 3. Hessian and theoretical learning-rate limits
    # ------------------------------------------------------------

    ols_lr_info = learning_rate_information( X_train_gd, method="ols", )

    ridge_lr_info = learning_rate_information( X_train_gd, method="ridge", lmbda=gd_lmbda, )

    print()
    print("Learning-rate information")
    print("-------------------------")

    print(
        "OLS: "
        f"kappa={ols_lr_info['condition_number']:.3e}, "
        f"eta_opt={ols_lr_info['eta_opt']:.6f}, "
        f"eta_max={ols_lr_info['eta_max']:.6f}"
    )

    print(
        "Ridge: "
        f"kappa={ridge_lr_info['condition_number']:.3e}, "
        f"eta_opt={ridge_lr_info['eta_opt']:.6f}, "
        f"eta_max={ridge_lr_info['eta_max']:.6f}"
    )

    # ------------------------------------------------------------
    # 4. Run GD using BOTH gradient implementations and compare with Parts A and B
    # ------------------------------------------------------------

    print()
    print("Gradient descent: analytical vs AD")
    print("----------------------------------")

    gd_solution_results = {}

    part_ab_results = {
    "ols": fit_and_evaluate( x_train=x_train_gd, x_test=x_test_gd, y_train=y_train_gd, y_test=y_test_gd,
                             degree=gd_degree, method="ols", ),

    "ridge": fit_and_evaluate( x_train=x_train_gd, x_test=x_test_gd, y_train=y_train_gd, y_test=y_test_gd,
                               degree=gd_degree, method="ridge", lmbda=gd_lmbda, ),
    }

    for method, lmbda, theta_closed, lr_info in [
        ( "ols", 0.0, theta_ols_closed, ols_lr_info, ),
        ( "ridge", gd_lmbda, theta_ridge_closed, ridge_lr_info, ),
    ]:

        for gradient_source in [ "analytic", "autodiff", ]:

            result = gradient_descent(
                X=X_train_gd,
                y=y_train_gd,
                eta=lr_info["eta_opt"],
                method=method,
                lmbda=lmbda,
                gradient_source=gradient_source,
                theta0=theta0,
                max_iter=100000,
                tol=1.0e-8,
                seed=seed,
            )

            theta_error = np.linalg.norm( result["theta"] - theta_closed )

            y_test_pred = predict( X_test_gd, result["theta"], )
            test_mse = mean_squared_error( y_test_gd, y_test_pred, )

            if gradient_source == "analytic":
                gd_solution_results[method] = { "theta": result["theta"].copy(), "test_mse": test_mse, }

            print(
                f"{method.upper():5s} "
                f"{gradient_source:8s}: "
                f"iterations={result['iterations']:6d}, "
                f"converged={result['converged']}, "
                f"|theta-theta_closed|={theta_error:.3e}, "
                f"test MSE={test_mse:.6f}"
            )

        theta_difference = np.linalg.norm( gd_solution_results[method]["theta"] - part_ab_results[method]["theta"] )
        mse_difference = abs( gd_solution_results[method]["test_mse"] - part_ab_results[method]["mse_test"] )   

        print(
            f"{method.upper():5s}: "
            f"Part A/B MSE={part_ab_results[method]['mse_test']:.9f}, "
            f"GD MSE={gd_solution_results[method]['test_mse']:.9f}, "
            f"|delta MSE|={mse_difference:.3e}, "
            f"|delta theta|={theta_difference:.3e}"
        )
        
   # ------------------------------------------------------------
   # 5. Learning-rate study
   # ------------------------------------------------------------

    eta_factors = [ 0.10, 0.50, 0.90, 0.99, 1.01, ]

    print()
    print("Learning-rate study")
    print("-------------------")

    learning_rate_results = {
        "ols": {},
        "ridge": {},
    }

    for method, lmbda, theta_closed, lr_info in [
        ( "ols", 0.0, theta_ols_closed, ols_lr_info, ),
        ( "ridge", gd_lmbda, theta_ridge_closed, ridge_lr_info, ),
    ]:

        print()
        print(method.upper())

        for factor in eta_factors:

            eta = ( factor * lr_info["eta_max"] )

            result = gradient_descent(
                X=X_train_gd,
                y=y_train_gd,
                eta=eta,
                method=method,
                lmbda=lmbda,
                gradient_source="analytic",
                theta0=theta0,
                max_iter=100000,
                tol=1.0e-8,
                seed=seed,
            )

            theta_error = np.linalg.norm( result["theta"] - theta_closed )

            learning_rate_results[method][factor] = {
                "eta": eta,
                "result": result,
                "theta_error": theta_error,
            }

            print(
                f"eta/eta_max={factor:4.2f}: "
                f"eta={eta:.6f}, "
                f"iterations={result['iterations']:6d}, "
                f"converged={result['converged']}, "
                f"diverged={result['diverged']}, "
                f"|theta-theta_closed|={theta_error:.3e}"
            )

    minimum_costs = {
        "ols": ols_cost( theta_ols_closed, X_train_gd, y_train_gd, ),
        "ridge": ridge_cost( theta_ridge_closed, X_train_gd, y_train_gd, gd_lmbda, ),
    }

    plot_gd_learning_rate_study( learning_rate_results=learning_rate_results,
                                 minimum_costs=minimum_costs,
                                 save_path=PLOTS_DIR / "part_e", )



# ============================================================
# PART F: Momentum and adaptive optimizers
# ============================================================

def run_part_f( seed=2026, test_size=0.20, ):

    print("\nRunning Part F...\n")

    opt_n = 200
    opt_sigma = 0.1
    opt_degree = 5
    opt_lmbda = 1.0e-2

    accuracy_tol = 1.0e-4
    max_iter = 50000

    output_dir = PLOTS_DIR / "part_f"

    output_dir.mkdir( parents=True, exist_ok=True, )

    # ------------------------------------------------------------
    # Same data setup as Part E
    # ------------------------------------------------------------

    x, y = artificial_data( n=opt_n, sigma=opt_sigma, seed=seed, )

    ( x_train, x_test, y_train, y_test,
     ) = train_test_split( x, y, test_size=test_size, random_state=seed, )

    X_train = design_matrix( x_train, opt_degree, )
    X_test = design_matrix( x_test, opt_degree,)

    feature_means, feature_stds = (
        fit_feature_scaler(X_train)
    )

    X_train = scale_design_matrix(
        X_train,
        feature_means,
        feature_stds,
    )

    X_test = scale_design_matrix(
        X_test,
        feature_means,
        feature_stds,
    )

    # Same initial theta for every optimizer.
    rng = np.random.default_rng(seed)

    theta0 = rng.normal(
        size=X_train.shape[1]
    )

    # ------------------------------------------------------------
    # Closed-form reference solutions
    # ------------------------------------------------------------

    theta_closed = {
        "ols": fit_ols(
            X_train,
            y_train,
        ),

        "ridge": fit_ridge(
            X_train,
            y_train,
            opt_lmbda,
        ),
    }

    minimum_cost = {
        "ols": ols_cost(
            theta_closed["ols"],
            X_train,
            y_train,
        ),

        "ridge": ridge_cost(
            theta_closed["ridge"],
            X_train,
            y_train,
            opt_lmbda,
        ),
    }

    # ------------------------------------------------------------
    # Run OLS and Ridge separately
    # ------------------------------------------------------------

    for method in [
        "ols",
        "ridge",
    ]:

        if method == "ols":

            lmbda = 0.0

        else:

            lmbda = opt_lmbda

        # Plain-GD information is useful for choosing
        # the search interval.
        lr_info = learning_rate_information(
            X_train,
            method=method,
            lmbda=lmbda,
        )

        # --------------------------------------------------------
        # Learning-rate search ranges
        #
        # They deliberately differ by optimizer because eta does
        # not have the same meaning for adaptive methods.
        # --------------------------------------------------------

        eta_values_by_optimizer = {

            "plain": np.unique(
                np.array([
                    0.01,
                    0.03,
                    0.10,
                    0.20,
                    0.30,
                    lr_info["eta_opt"],
                ])
            ),

            "momentum": np.array([
                0.003,
                0.01,
                0.03,
                0.10,
                0.20,
                0.30,
            ]),

            "adagrad": np.array([
                0.03,
                0.10,
                0.30,
                0.50,
                0.70,
                1.00,
            ]),

            "rmsprop": np.array([
                1.0e-5,
                3.0e-5,
                5.0e-5,
                8.0e-5,
                1.0e-4,
                3.0e-4,
            ]),

            "adam": np.array([
                0.001,
                0.003,
                0.01,
                0.03,
                0.10,
                0.30,
            ]),
        }

        sweep_results = (
            optimizer_learning_rate_sweep(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                theta_reference=
                    theta_closed[method],
                eta_values_by_optimizer=
                    eta_values_by_optimizer,
                method=method,
                lmbda=lmbda,
                theta0=theta0,
                accuracy_tol=accuracy_tol,
                max_iter=max_iter,
                seed=seed,
            )
        )

        best_results = (
            select_best_optimizer_runs(
                sweep_results
            )
        )

        # --------------------------------------------------------
        # Numerical summary
        # --------------------------------------------------------

        print()
        print(
            f"Optimizer comparison {method.upper()}"
        )
        print(
            "--------------------------------"
        )

        for optimizer, data in (
            best_results.items()
        ):

            if data is None:

                print(
                    f"{optimizer:8s}: "
                    "no learning rate reached "
                    "the target accuracy"
                )

                continue

            eta = data["eta"]
            result = data["result"]

            print(
                f"{optimizer:8s}: "
                f"eta={eta:.3e}, "
                f"iterations={result['iterations']:6d}, "
                f"relative error="
                f"{result['relative_theta_error']:.3e}, "
                f"test MSE="
                f"{result['test_mse']:.6f}"
            )

        # --------------------------------------------------------
        # Plots
        # --------------------------------------------------------

        plot_optimizer_convergence(
            best_results=best_results,
            minimum_cost=minimum_cost[method],
            method=method,
            save_path=output_dir,
        )

        plot_optimizer_eta_sensitivity(
            sweep_results=sweep_results,
            method=method,
            max_iter=max_iter,
            accuracy_tol=accuracy_tol,
            save_path=output_dir,
        )


# ============================================================
# MAIN PART
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description=( "Run selected parts of Machine Learning Project 1." )
    )

    parser.add_argument(
        "part",
        choices=[ "a", "b", "c", "d", "e", "f", ],
        help=( "Project part to run: "
            "a, b, c, d, e, f." ),
    )

    args = parser.parse_args()

    seed = 2026
    test_size = 0.20

    if args.part == "a":
        run_part_a( seed=seed, test_size=test_size, )

    elif args.part == "b":
        run_part_b( seed=seed, test_size=test_size, )

    elif args.part == "c":
        run_part_c( seed=seed, test_size=test_size, )

    elif args.part == "d":
        run_part_d( seed=seed, test_size=test_size, )

    elif args.part == "e":
        run_part_e( seed=seed, test_size=test_size, )

    elif args.part == "f":
        run_part_f( seed=seed, test_size=test_size, )

    print("\nSee results in P1/plots \n")

if __name__ == "__main__":
    main()