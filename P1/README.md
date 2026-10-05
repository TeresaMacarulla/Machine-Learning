# Project 1 — Regression Analysis, Resampling Methods and Gradient Descent

This repository contains the code and figures for **Project 1** in FYS-STK3155 Applied Data Analytics and Machine Learning at the University of Oslo.

The project studies polynomial regression of noisy samples of **Runge's function**, with emphasis on model complexity, regularization, resampling, and numerical optimization. The implemented methods include OLS, Ridge and Lasso regression, bootstrap bias--variance analysis, K-fold cross-validation, gradient descent, Momentum, AdaGrad, RMSProp, Adam, and mini-batch stochastic gradient descent.

## Repository structure

```text
.
├── README.md
├── environment.yml
├── plots/                  # Generated figures, grouped by project part
│   ├── part_a/
│   ├── part_b/
│   ├── part_c/
│   ├── part_d/
│   ├── part_e/
│   ├── part_f/
│   ├── part_g/
│   ├── part_h/
│   └── part_i/
├── report/
│   └── figures/
└── src/
    ├── main.py             # Runs Parts a--i
    ├── utilities.py        # Regression, resampling and optimization routines
    └── plot_generator.py   # Plotting functions
```

## Environment

Create the Conda environment from the supplied file:

```bash
conda env create -f environment.yml
conda activate fysstk3155-p1
```

The environment uses Python 3.10 together with NumPy, Matplotlib, scikit-learn and JAX.

## Running the project

Each project part can be run independently from the repository root:

```bash
python src/main.py a
python src/main.py b
python src/main.py c
python src/main.py d
python src/main.py e
python src/main.py f
python src/main.py g
python src/main.py h
python src/main.py i
```

The corresponding figures are written automatically to `plots/part_<letter>/`.

### Project parts

- **a:** OLS polynomial regression; dependence on degree, sample size and noise.
- **b:** Ridge regression and coefficient/singular-value shrinkage.
- **c:** Training/test error and bootstrap bias--variance analysis.
- **d:** Bootstrap and 5-/10-fold cross-validation comparison.
- **e:** Plain gradient descent, analytical vs automatic differentiation, and learning-rate stability.
- **f:** Momentum, AdaGrad, RMSProp and Adam.
- **g:** Lasso regression, coordinate descent and sparsity.
- **h:** Mini-batch stochastic optimization, batch-size dependence and learning-rate schedules.
- **i:** Final 5-fold cross-validated model selection for OLS, Ridge and Lasso.

A fixed random seed (`2026`) is used throughout the experiments for reproducibility.
