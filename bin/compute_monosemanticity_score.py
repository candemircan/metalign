from pathlib import Path

import h5py
import numpy as np
from fastcore.script import call_parse

from metarep.data import h5_to_numpy
from metarep.metrics import monosemanticity_score


@call_parse
def main(
    model_name: str = "sae-top_k-64-cls_only-layer_11-hook_resid_post",  # model name, meant to match the name of the HF repo (without the org)
    data_root: Path = Path("data/sae"),  # where you keep your SAE activations
    min_nonzero: int = 100,  # minimum number of non-zero activations per column to keep it in the final array
    raw_only: bool = False, # if True, compute the monosemanticity score for the neurons in the raw activations
    batch_size: int = 26107,  # batch size for processing similarity matrix to reduce memory usage
    force: bool = False  # if True, overwrite existing files
):
    """
    Compute monosemanticity score for a given SAE model using both things and coco datasets.
    """
    raw_str = "raw_" if raw_only else ""
    save_path = data_root / "monosemanticity" / f"{raw_str}{model_name}_min_nonzero_{min_nonzero}.npy"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and not force:
        print(f"File {save_path} already exists. Use --force to overwrite.")
        return

    # load SAE activations from both datasets
    things_sae = h5_to_numpy(f"things_{model_name}", data_root, 0)
    coco_sae = h5_to_numpy(f"coco_{model_name}", data_root, 0)
    sae_activations = np.concatenate([things_sae, coco_sae], axis=0)

    # now filter for min_nonzero, such that any column has at least `min_nonzero` non-zero values
    non_zero_counts = np.count_nonzero(sae_activations, axis=0)
    sae_activations = sae_activations[:, non_zero_counts >= min_nonzero]

    # Load raw activations from both datasets
    raw_activations = []
    for dataset in ["things", "coco"]:
        h5_file_path = data_root / f"{dataset}_{model_name}.h5"
        with h5py.File(h5_file_path, 'r') as f:
            for k in f.keys():
                raw_activations.append(f[k]["raw"][:])

    raw_activations = np.array(raw_activations)
    mscore = monosemanticity_score(raw_activations, raw_activations if raw_only else sae_activations, batch_size=batch_size)

    np.save(save_path, mscore)