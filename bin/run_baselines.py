from pathlib import Path

import numpy as np
import pandas as pd
from fastcore.script import call_parse
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from metarep.constants import NUM_THINGS_CATEGORIES


@call_parse
def main(
    backbone: str = "dinov2_vitb14_reg",
    force: bool = False,
    n_splits: int = 10,
):
    "Run baseline linear model benchmarks with cross-validation"
    baseline_path = Path("data/baselines")
    baseline_path.mkdir(parents=True, exist_ok=True)
    save_name = baseline_path / f"{backbone}_baselines.csv"
    if save_name.exists() and not force:
        print(f"File {save_name} already exists. Use --force to overwrite.")
        return

    img_names = sorted(Path("data/external/THINGS").glob("*/*.jpg"))
    unique_ids = [line.strip() for line in open("data/external/unique_id.txt", "r")]

    assert all(img_name.parent.name in unique_ids for img_name in img_names), "Not all parent folders in img_names are in unique_ids"

    img_idx = []
    parent_name_list = [img_name.parent.name for img_name in img_names]
    for unique_id in unique_ids:
        img_idx.append(parent_name_list.index(unique_id))

    assert 0 in img_idx, "The first image must be in there"
    assert len(img_idx) == NUM_THINGS_CATEGORIES,  f"Expected {NUM_THINGS_CATEGORIES} categories, found {len(img_idx)}"

    representations = np.load(f"data/backbone_reps/{backbone}.npz")
    X = np.hstack([representations[key] for key in representations.keys()])[img_idx]
    Y = np.loadtxt("data/external/spose_embedding_66d_sorted.txt")

    results = dict(dimension=[], r2_mean=[], r2_std=[], train_ratio=[], estimator=[])
    estimator = BayesianRidge()
    for dimension in tqdm(range(Y.shape[1])):
        y = Y[:, dimension]
        for train_ratio in tqdm(np.logspace(-1.2, -0.01, 10), desc=f"Dimension {dimension}", leave=False):
            cv = ShuffleSplit(n_splits=n_splits, train_size=train_ratio, random_state=1234)
            scores = []
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                scores.append(r2_score(y_test, y_pred))

            results["dimension"].append(dimension)
            results["r2_mean"].append(np.mean(scores))
            results["r2_std"].append(np.std(scores))
            results["train_ratio"].append(train_ratio)
            results["estimator"].append(estimator.__class__.__name__)

    df = pd.DataFrame(results)
    df.to_csv(save_name, index=False)