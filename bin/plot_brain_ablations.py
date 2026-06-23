"""ROI bar plots comparing main/raw/midsae feature distributions, one row per backbone."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from plot_brain import (
    BACKBONE_LABELS,
    BACKBONES,
    EVALS_PATH,
    PARTICIPANTS,
    fdr_bh,
    find_common_rois,
    fisher_pval,
    get_ordered_rois,
    merge_lr_rois,
    sign_flip_pval,
)

EXPERIMENTS = ['main', 'raw', 'midsae']
EXP_LABELS  = {'main': 'SAE (L11)', 'raw': 'Raw', 'midsae': 'SAE (L6)'}


def load_ablation_data(bb, common_rois, roi_map, roi_names):
    rows              = []
    fold_diffs_raw    = {roi_name: [] for roi_name in roi_names}  # main - raw per participant
    fold_diffs_midsae = {roi_name: [] for roi_name in roi_names}  # main - midsae per participant

    for p_idx, p in enumerate(PARTICIPANTS):
        fold_data = {}  # exp -> {roi_name -> r2_meta_folds}
        for exp in EXPERIMENTS:
            npy = EVALS_PATH / f"{p_idx + 1}_{exp}_{bb}.npy"
            if not npy.exists(): continue
            res         = np.load(npy, allow_pickle=True).item()
            roi_results = res['roi_results']
            fold_data[exp] = {}
            for roi_name in roi_names:
                lr_rois  = [c for c in roi_map[roi_name] if c in common_rois]
                matching = [r for r in lr_rois if r in roi_results]
                if not matching: continue
                r2_meta = np.mean([roi_results[r]['r2_meta'] for r in matching])
                rows.append({'ROI': roi_name, 'Experiment': EXP_LABELS[exp], 'R2': r2_meta, 'Participant': p})
                fold_data[exp][roi_name] = np.mean([roi_results[r]['r2_meta_folds'] for r in matching], axis=0)

        for roi_name in roi_names:
            if 'main' in fold_data and 'raw' in fold_data:
                if roi_name in fold_data['main'] and roi_name in fold_data['raw']:
                    fold_diffs_raw[roi_name].append(fold_data['main'][roi_name] - fold_data['raw'][roi_name])
            if 'main' in fold_data and 'midsae' in fold_data:
                if roi_name in fold_data['main'] and roi_name in fold_data['midsae']:
                    fold_diffs_midsae[roi_name].append(fold_data['main'][roi_name] - fold_data['midsae'][roi_name])

    def _pvals(fold_diffs):
        raw = {}
        for roi_name in roi_names:
            if not fold_diffs[roi_name]: continue
            raw[roi_name] = fisher_pval([sign_flip_pval(d) for d in fold_diffs[roi_name]])
        return fdr_bh(raw) if raw else {}

    return pd.DataFrame(rows), _pvals(fold_diffs_raw), _pvals(fold_diffs_midsae)


def main():
    common_rois  = find_common_rois()
    roi_map, roi_names = merge_lr_rois(common_rois)
    ordered_rois = get_ordered_rois(roi_names)
    Path("figures").mkdir(exist_ok=True)

    colors         = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    palette        = {EXP_LABELS['main']: colors[1], EXP_LABELS['raw']: colors[2], EXP_LABELS['midsae']: colors[3]}
    hue_order      = [EXP_LABELS[e] for e in EXPERIMENTS]
    n_exp          = len(EXPERIMENTS)
    dodge          = 0.8 / n_exp

    fig, axes = plt.subplots(len(BACKBONES), 1, figsize=(6.99866, 3 * len(BACKBONES)), sharex=True)
    for ax, bb in zip(axes, BACKBONES):
        df, roi_pvals_raw, roi_pvals_midsae = load_ablation_data(bb, common_rois, roi_map, roi_names)
        if df.empty: continue
        sns.barplot(data=df, x='ROI', y='R2', hue='Experiment',
                    ax=ax, order=ordered_rois, hue_order=hue_order,
                    palette=palette, alpha=0.8, edgecolor='black', linewidth=1, saturation=1, errorbar=None)
        ax.set_ylabel(r"$R^2$")
        ax.set_title(BACKBONE_LABELS[bb], fontweight='bold')
        ax.get_legend().remove()

        for p in PARTICIPANTS:
            for i, roi in enumerate(ordered_rois):
                pts = df[(df.ROI == roi) & (df.Participant == p)]
                if pts.empty: continue
                xs, ys = [], []
                for j, exp_label in enumerate(hue_order):
                    row = pts[pts.Experiment == exp_label]
                    if row.empty: continue
                    xs.append(i + (j - (n_exp - 1) / 2) * dodge)
                    ys.append(row['R2'].values[0])
                if len(xs) < 2: continue
                ax.plot(xs, ys, color='black', linewidth=1.2, zorder=4, alpha=0.7)
                ax.scatter(xs, ys, color='black', s=10, zorder=5, alpha=0.7, edgecolor='none')

        # Colored stars: Raw color = main > raw; midsae color = main > midsae
        # Two-sided p-value + direction check (only show star when main actually wins)
        roi_means = df.groupby(['ROI', 'Experiment'])['R2'].mean()
        for i, roi in enumerate(ordered_rois):
            y_cursor = df[df.ROI == roi]['R2'].max() + 0.005
            for pvals, color, comp in [(roi_pvals_raw, colors[2], 'raw'), (roi_pvals_midsae, colors[3], 'midsae')]:
                pval = pvals.get(roi)
                if pval is None or pval >= 0.05: continue
                main_r2 = roi_means.get((roi, EXP_LABELS['main']), 0)
                comp_r2 = roi_means.get((roi, EXP_LABELS[comp]),   0)
                if main_r2 <= 0 or main_r2 <= comp_r2: continue  # negative or wrong direction
                star = '***' if pval < 0.001 else ('**' if pval < 0.01 else '*')
                ax.text(i, y_cursor, star, ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
                y_cursor += 0.025

    axes[-1].legend(title='', frameon=False, loc='upper right', ncol=3)
    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(f"figures/brain_ablations.{ext}", bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/brain_ablations.{png,pdf,svg}")


if __name__ == '__main__':
    main()
