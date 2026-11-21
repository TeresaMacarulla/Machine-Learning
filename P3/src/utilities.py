
import json
from pathlib import Path
import numpy as np

# ============================================================
# Load and prepare the dataset
# ============================================================

def load_kilter_dataset(data_dir):
    """
    Load Kilter Board climbing problems from Degree45 and Degree50 folders.

    Returns:
        X: Array (N, 4761) containing flattened hold channels
           and normalized inclination.
        y: Array (N,) containing class labels 0-10 (V3-V13) corresponding to difficulty grade.
    """

    data_dir = Path(data_dir)
    folder_angles = {"Degree45": 45, "Degree50": 50}

    # Find all climbing problems in both folders
    files = []
    for folder_name, angle in folder_angles.items():
        folder = data_dir / folder_name

        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        npy_files = sorted(folder.glob("*.npy"))
        print(f"{folder_name}: {len(npy_files)} climbing problems")

        for npy_path in npy_files:
            files.append((npy_path, angle))

    if not files:
        raise ValueError("No climbing problems found.")

    # Allocate arrays for all samples
    n_samples = len(files)
    n_features = 34 * 35 * 4 + 1

    X = np.empty((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int64)

    # Load each climbing problem
    valid_count = 0
    skipped_count = 0

    for npy_path, angle in files:

        json_path = npy_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing JSON: {json_path}")

        # Load metadata first to check the grade
        with open(json_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        grade = metadata.get("grade_v")

        # Skip climbing problems without a difficulty grade
        if grade is None:
            skipped_count += 1
            continue
        grade = int(grade)

        # Check that the grade belongs to our 11 classes
        if not 3 <= grade <= 13:
            raise ValueError(f"Unexpected grade in {json_path}: V{grade}")

        # Load matrix only for valid climbing problems
        route = np.load(npy_path, allow_pickle=False)

        if route.shape != (34, 35, 10):
            raise ValueError(f"Unexpected shape: {npy_path}")

        # Flatten the four dynamic channels
        X[valid_count, :-1] = route[:, :, :4].reshape(-1)

        # Add normalized inclination
        X[valid_count, -1] = angle / 90.0

        # Encode grades V3-V13 as classes 0-10
        y[valid_count] = grade - 3

        valid_count += 1

    # Remove unused rows from the preallocated arrays
    X = X[:valid_count]
    y = y[:valid_count]

    print(f"\nValid climbing problems: {valid_count}")
    print(f"Skipped (missing grade): {skipped_count}")

    return X, y



