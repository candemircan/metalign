"""
In-context learning eval: per-position accuracy and MSE over the context window.

Compares standard (embed) vs no-embed Transformer models.
Samples multiple episodes per function to capture variability.

Output: h5 with datasets of shape (n_functions, episodes_per_function, seq_len)
         for accuracy and MSE of each model.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from fastcore.script import call_parse
from torch.nn import functional as F
from tqdm import tqdm

from metalign.data import FunctionDataset, load_backbone
from metalign.model import get_model
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)


def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = get_model(c=ckpt['config'])
    model.load_state_dict(fix_state_dict(ckpt['state_dict']))
    model.eval()
    return model.to(device)


def _eval_episodes(model, X_batch: torch.Tensor, Y_bin_batch: torch.Tensor,
                   Y_mag_batch: torch.Tensor, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run a batch of episodes through a sequence model.

    Args:
        X_batch (torch.Tensor): Input features (episodes, sl, feat_dim).
        Y_bin_batch (torch.Tensor): Binary labels (episodes, sl).
        Y_mag_batch (torch.Tensor): Magnitude values (episodes, sl).
        device (str): Device string.

    Returns:
        acc (np.ndarray): Per-position accuracy (episodes, sl).
        mse (np.ndarray): Per-position MSE (episodes, sl).
    """
    X   = X_batch.to(device)
    Y_m = Y_mag_batch.to(device)
    Y_b = Y_bin_batch.to(device)

    bin_logits, mag_preds = model(X, Y_m)
    bin_logits = bin_logits.squeeze(-1)  # (episodes, sl)
    mag_preds  = mag_preds.squeeze(-1)   # (episodes, sl)

    acc = ((torch.sigmoid(bin_logits) > 0.5).float() == Y_b).float().cpu().numpy()
    mse = F.mse_loss(mag_preds, Y_m, reduction='none').cpu().numpy()
    return acc, mse


def _eval_seq_models(models: dict, ds: FunctionDataset, sl: int,
                     episodes_per_function: int, device: str) -> dict:
    """Evaluate sequence models (Transformer/LSTM) on episode-based tasks.

    Args:
        models: {name: model} dict of sequence models.
        ds: FunctionDataset with eval data.
        sl: Sequence length.
        episodes_per_function: Number of episodes per function.
        device: Device string.

    Returns:
        {name_acc: (n_functions, episodes, sl), name_mse: (n_functions, episodes, sl)} arrays.
    """
    n_functions = ds.num_functions
    dim_indices = list(range(n_functions))
    results = {}
    for name in models:
        results[f'{name}_acc'] = np.zeros((n_functions, episodes_per_function, sl), dtype=np.float32)
        results[f'{name}_mse'] = np.zeros_like(results[f'{name}_acc'])

    print(f"Evaluating {n_functions} functions × {episodes_per_function} episodes × {len(models)} models...")
    for f_i, dim_idx in tqdm(enumerate(dim_indices)):
        episodes    = [ds._sample_episode(dim_idx) for _ in range(episodes_per_function)]
        ep_sl       = episodes[0][0].shape[0]
        X_batch     = torch.stack([e[0] for e in episodes])  # (ep, ep_sl, feat)
        Y_bin_batch = torch.stack([e[1] for e in episodes])  # (ep, ep_sl)
        Y_mag_batch = torch.stack([e[2] for e in episodes])  # (ep, ep_sl)

        for name, model in models.items():
            acc, mse = _eval_episodes(model, X_batch, Y_bin_batch, Y_mag_batch, device)
            results[f'{name}_acc'][f_i, :, :ep_sl] = acc
            results[f'{name}_mse'][f_i, :, :ep_sl] = mse

    results['dim_indices'] = np.array(dim_indices)
    results['sl'] = sl
    return results


@call_parse
def main(
    eval_features:str = "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",  # features stem under data/sae/
    min_nonzero:int  = 120,      # min non-zero activations per column in eval features
    episodes_per_function:int = 20,  # episodes per function
    seed:int         = 1234,       # random seed
    output_path:str  = "data/evals/coco_incontext",  # directory to save results
):
    "Evaluate per-position in-context learning accuracy and MSE for all backbones."
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    p_reps       = Path("data/backbone_reps")
    backbone_dict = json.load((p_reps / "backbones.json").open())
    best_models  = json.load((Path("data/checkpoints") / "best_models.json").open())
    features_path = Path(f"data/sae/{eval_features}.h5")
    out_dir      = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_keys     = ['main', 'noembed']

    for backbone_name in backbone_dict:
        out_file    = out_dir / f"{backbone_name}.h5"
        eval_inputs = load_backbone(f"{p_reps}/coco_eval_{backbone_dict[backbone_name]}.h5")

        if out_file.exists():
            with h5py.File(out_file, 'r') as f:
                results = {k: f[k][()] for k in f.keys()} if all(f'{k}_acc' in f for k in seq_keys) else None
        else:
            results = None

        if results is None:
            models = {
                'main':    _load_model(best_models["[main]"][backbone_name], device),
                'noembed': _load_model(f"data/checkpoints/[noemb]_{backbone_name}/model.pt", device),
            }
            sl      = models['main'].c.sl
            ds      = FunctionDataset(inputs=eval_inputs, features_path=features_path,
                                      seq_len=sl, min_nonzero=min_nonzero)
            results = _eval_seq_models(models, ds, sl, episodes_per_function, device)

            with h5py.File(out_file, 'w') as f:
                for k, v in results.items():
                    compress = {'compression': 'gzip'} if np.asarray(v).ndim > 0 else {}
                    f.create_dataset(k, data=v, **compress)
            print(f"Saved to {out_file}")

        print(f"\n{backbone_name}")
        print(f"{'':12}  {'acc':>6}  {'mse':>8}")
        for k in seq_keys:
            print(f"{k:12}  {results[f'{k}_acc'].mean():>6.3f}  {results[f'{k}_mse'].mean():>8.4f}")
