# FYS-STK3155 — Project 2: Regression and MNIST Classification

A NumPy implementation of feed-forward neural networks (NN) and manual backpropagation for two tasks: fitting noisy Runge-function data and classifying handwritten digits from MNIST. The experiments compare optimizers, activation functions, NN width/depth and $L_1$/ $L_2$ penalties.

## Setup

From the **project root** (the folder containing `environment.yml`), create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate fysstk3155-p2
```

If an older environment already exists, update it with `conda env update -f environment.yml --prune`. The environment includes Python 3.10, NumPy, Matplotlib, scikit-learn and JupyterLab. 

The full MNIST archive is placed at `data/mnist.npz` and is used in classification experiments. The regression experiments generate their own Runge data (seed 2026).

## Run

```bash
python src/main.py
```

Choose a top-level experiment:

| Option | Experiment | Plot directory |
|---|---|---|
| `1` | Optimizers, learning rates and OLS versus regression NNs | `plots/part_1/` |
| `2` | Runge regression: activation, width and depth | `plots/part_2/` |
| `3` | Runge regression: $L_1$/ $L_2$ regularization and Ridge/Lasso comparison | `plots/part_3/` |
| `4` | MNIST classification (opens submenu options 1–7) | `plots/part_4/` |

For MNIST, submenu choices 1–3 cover initial networks, learning rates and architecture/activation comparisons; choices 4–6 cover penalties, early stopping and the scikit-learn validation reference; choice 7 retrains the **fixed, validation-selected** custom classifiers on all 60,000 official training images and reports test performance and the confusion matrix. Submenu options 2–3 automatically rerun the baseline models they require. 

Some training grids can take time to complete, especially the 96-run Runge regularization search and full MNIST experiments.

## Repository layout

```text
.
├── data/mnist.npz          # Full MNIST dataset 
├── environment.yml        # Conda environment: fysstk3155-p2
├── src/
│   ├── main.py            # Interactive experiments 
│   ├── utilities.py       # Models, losses, gradients, optimizers and training
│   └── plot_generator.py  # PDF figures generators
├── plots/
│   ├── part_1/            # Optimizer/OLS comparisons
│   ├── part_2/            # Activation, depth and width comparisons
│   ├── part_3/            # Regression regularization comparisons
│   └── part_4/            # MNIST learning curves, heatmaps, confusion matrix
└── report/                # LaTeX report
```

Training uses a shared function, `train_neural_network`, in `src/utilities.py` for both regression (MSE/identity output) and classification (cross-entropy/softmax output). Figures are written as PDF files; the report in `report/` documents the methods, results and limitations. 
