# data set from: https://huggingface.co/datasets/Ken1m/KilterBoard
# good data set from https://github.com/Pa-Sto/kilterboardie/tree/main

from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split

from utilities import (
    load_kilter_dataset
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# Load and prepare the dataset
# ============================================================

# Load and prepare Kilter Board dataset
DATA_DIR = PROJECT_ROOT / "data"

X_kilter, y_kilter = load_kilter_dataset(DATA_DIR)

print("\nDataset summary")
print("X shape:", X_kilter.shape)
print("y shape:", y_kilter.shape)

# Inspect the number of samples per grade
grades, counts = np.unique(y_kilter + 3, return_counts=True)

for grade, count in zip(grades, counts):
    print(f"V{grade}: {count} climbing problems")

# First split: 70% training, 30% validation + test
X_kilter_train, X_temp, y_kilter_train, y_temp = train_test_split(X_kilter, y_kilter, test_size=0.30, random_state=2026, stratify=y_kilter)

# Second split: divide the remaining 30% equally into 15% validation and 15% test
X_kilter_val, X_kilter_test, y_kilter_val, y_kilter_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=2026, stratify=y_temp)

print("\nDataset split")
print("Training:", X_kilter_train.shape, y_kilter_train.shape)
print("Validation:", X_kilter_val.shape, y_kilter_val.shape)
print("Test:", X_kilter_test.shape, y_kilter_test.shape)










    






