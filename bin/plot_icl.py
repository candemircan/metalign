"""Per-position in-context learning curves: accuracy and MSE over the context window.

Loads h5 outputs from eval_icl.py and plots per-position accuracy (top) and MSE (bottom)
for each backbone (columns), comparing models with and without linear embeddings.
"""
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

BACKBONES       = ['siglip2', 'mae', 'dinov3', 'clip']
BACKBONE_LABELS = {'clip': 'CLIP', 'dinov3': 'DINOv3', 'mae': 'MAE', 'siglip2': 'SigLIP2'}
DATA_PATH       = Path("data/evals/coco_incontext")


def _load(bb):
    p = DATA_PATH / f"{bb}.h5"
    if not p.exists(): return None
    with h5py.File(p, 'r') as f:
        return {k: f[k][()] for k in f.keys()}


def _curve_stats(arr):
    """Mean ± SE per context position, averaging over episodes first.

    Args:
        arr (np.ndarray): Per-position values (n_functions, episodes, sl).

    Returns:
        mean (np.ndarray): Shape (sl,).
        se   (np.ndarray): Shape (sl,).
    """
    per_fn = arr.mean(1)                              # (n_functions, sl)
    return per_fn.mean(0), per_fn.std(0) / np.sqrt(per_fn.shape[0])


def main():
    _colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    c_main  = _colors[1]
    c_noemb = _colors[-1]
    models  = [('main',    c_main,  'meta-learned (with linear embedding)'),
               ('noembed', c_noemb, 'meta-learned (without linear embedding)')]
    metrics = [('acc', 'Accuracy'), ('mse', 'MSE')]

    fig, axes = plt.subplots(2, len(BACKBONES), figsize=(4 * len(BACKBONES), 6), sharey='row')

    for col, bb in enumerate(BACKBONES):
        res = _load(bb)
        if res is None:
            for row in range(2): axes[row, col].set_visible(False)
            continue

        x = np.arange(res['main_acc'].shape[-1])
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for key, color, label in models:
                mean, se = _curve_stats(res[f'{key}_{metric}'])
                ax.plot(x, mean, color=color, label=label if (col == 0 and row == 0) else None)
                ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=color)
            if col == 0:
                ax.set_ylabel(ylabel)
                ax.text(-0.15, 1.05, 'AB'[row], transform=ax.transAxes,
                        fontsize=18, fontweight='bold', va='top')
            if row == 0: ax.set_title(BACKBONE_LABELS[bb])
            ax.set_xlabel("Context position")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=2, frameon=False)

    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"figures/icl.{ext}", bbox_inches='tight',
                    **({"dpi": 300} if ext == "png" else {}))


if __name__ == "__main__":
    main()
