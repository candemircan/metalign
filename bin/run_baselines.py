from pathlib import Path

import numpy as np
import pandas as pd
from fastcore.script import call_parse
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from metarep.data import prepare_things_spose


@call_parse
def main(
    backbone: str = "dinov2_vitb14_reg", # backbones from which the representations are taken. different token types will be concatenated
    tokens: str = "all", # which tokens to use. if all, all tokens will be concatenated. otherwise needs to be one of cls, patch, or register (if avail)
    force: bool = False, # if True, overwrite existing files, otherwise skip if the file already exists
    n_splits: int = 10, # number of splits for cross-validation
    n_components: int = None # number of components to use for dimensionality reduction. If None, use the original data.
):
    """
    Do linear modelling of hebart_features ~ model_representations with cross-validation.
    Vary the train-test split ratio, and save R2 scores for each dimension and split ratio.
    The results are saved in data/baselines/{backbone}_baselines.csv.
    """
    baseline_path = Path("data/baselines")
    baseline_path.mkdir(parents=True, exist_ok=True)
    n_components_str = f"_pca{n_components}" if n_components is not None else ""
    tokens_str = f"_{tokens}" if tokens != "all" else ""
    save_name = baseline_path / f"{backbone}{n_components_str}{tokens_str}.csv"
    if save_name.exists() and not force:
        print(f"File {save_name} already exists. Use --force to overwrite.")
        return

    representations = np.load(f"data/backbone_reps/{backbone}.npz")
    X, Y = prepare_things_spose(representations=representations, return_tensors="np", tokens=tokens)

    if n_components is not None:
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)


    results = dict(dimension=[], r2=[], train_ratio=[], estimator=[], fold=[])
    estimator = BayesianRidge()
    for dimension in tqdm(range(Y.shape[1])):
        y = Y[:, dimension]
        for train_ratio in tqdm(np.logspace(-1.2, -0.01, 10), desc=f"Dimension {dimension}", leave=False):
            cv = ShuffleSplit(n_splits=n_splits, train_size=train_ratio, random_state=1234)
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                score = r2_score(y_test, y_pred)

                results["dimension"].append(dimension)
                results["r2"].append(score)
                results["train_ratio"].append(train_ratio)
                results["estimator"].append(estimator.__class__.__name__)
                results["fold"].append(fold_idx)

    df = pd.DataFrame(results)
    df.to_csv(save_name, index=False)