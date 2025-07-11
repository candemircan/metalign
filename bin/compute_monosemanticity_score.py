from pathlib import Path

import h5py
import numpy as np
from fastcore.script import call_parse

from metarep.data import h5_to_numpy
from metarep.metrics import monosemanticity_score


@call_parse
def main(
    model_name: str = "sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05",  # model name, meant to match the name of the HF repo (without the org)
    data_root: Path = Path("data/sae"),  # where you keep your SAE activations
    min_nonzero: int = 0,  # minimum number of non-zero activations per column to keep it in the final array
    raw_only: bool = False, # if True, compute the monosemanticity score for the neurons in the raw activations
    batch_size: int = 26107,  # batch size for processing similarity matrix to reduce memory usage
    force: bool = False  # if True, overwrite existing files
):
    """
    Compute monosemanticity score for a given SAE model.
    """
    raw_str = "raw_" if raw_only else ""
    save_path  = data_root / "monosemanticity" / f"{raw_str}{model_name}_min_nonzero_{min_nonzero}.npy"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and not force:
        print(f"File {save_path} already exists. Use --force to overwrite.")
        return
    sae_activations = h5_to_numpy(model_name, data_root, min_nonzero)
    h5_file_path = data_root / f"{model_name}.h5"
    raw_activations = []
    with h5py.File(h5_file_path, 'r') as f:
        for k in f.keys():
            raw_activations.append(f[k]["raw"][:])

    raw_activations = np.array(raw_activations)
    mscore = monosemanticity_score(raw_activations, raw_activations if raw_only else sae_activations, batch_size=batch_size) 

    np.save(save_path, mscore) 
