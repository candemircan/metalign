"""Plot brain encoding results: cortical flatmap + ROI bar plot per backbone."""
from pathlib import Path

import cortex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.image import load_img, new_img_like
from scipy.stats import chi2

DATA_PATH = Path("data/external/brain_data")
EVALS_PATH = Path("data/evals/brain")

ROIS = ['lEBA', "rEBA", "lFFA", "rFFA", "lOFA", "rOFA", "lSTS", "rSTS",
        "lPPA", "rPPA", "lRSC", "rRSC", "lTOS", "rTOS", 'lLOC', "rLOC",
        "V1", "V2", "V3", "hV4"]
PARTICIPANTS = ['01', '02', '03']
BACKBONES = ['siglip2', 'mae', 'dinov3', 'clip']
BACKBONE_LABELS = {'clip': 'CLIP', 'dinov3': 'DINOv3', 'mae': 'MAE', 'siglip2': 'SigLIP2'}

def load_brain_diff(bb, participant_idx=1):
    "Load brain difference map for a single participant"
    p = PARTICIPANTS[participant_idx - 1]
    vox_df = pd.read_csv(DATA_PATH / f"sub-{p}_VoxelMetadata.csv")
    brain_mask = load_img(DATA_PATH / f"masks/sub-{p}_space-T1w_brainmask.nii.gz")
    results = np.load(EVALS_PATH / f"{participant_idx}_main_{bb}.npy", allow_pickle=True).item()

    if 'r2_base' not in results or 'r2_meta' not in results:
        return None

    diffs = (results['r2_meta'] - results['r2_base']).mean(0)
    new_brain = np.zeros_like(brain_mask.get_fdata())

    for col_no, voxel in enumerate(results["voxel_idxs"]):
        voxel_info = vox_df[vox_df.voxel_id == voxel][["voxel_x", "voxel_y", "voxel_z"]].to_numpy()
        x, y, z = voxel_info[0]
        new_brain[x, y, z] = diffs[col_no]

    return new_img_like(ref_niimg=brain_mask, data=new_brain)

def find_common_rois():
    "Find ROIs present in all participants"
    all_participant_rois = []
    for p in PARTICIPANTS:
        vox_df = pd.read_csv(DATA_PATH / f"sub-{p}_VoxelMetadata.csv")
        available = set(vox_df.columns).intersection(ROIS)
        all_participant_rois.append(available)
    return set.intersection(*all_participant_rois)

def merge_lr_rois(common_rois):
    "Merge left/right ROIs into combined ROIs"
    roi_map, roi_names = {}, []
    for r in common_rois:
        if (r.startswith('l') or r.startswith('r')) and len(r) > 1 and r[1:].isalpha():
            base = r[1:]
            if base not in roi_map:
                roi_map[base] = []
                roi_names.append(base)
            roi_map[base].append(r)
        else:
            if r not in roi_map:
                roi_map[r] = [r]
                roi_names.append(r)
    return roi_map, roi_names

def sign_flip_pval(d):
    "Exact sign-flip p-value for 1D array of n fold differences (exhaustive over 2^n patterns)."
    n = len(d)
    bits  = np.arange(2**n)
    # this works well with n=12, but otherwise don't try at home
    # uses bit-shifting to turn integers 0 to 2^n-1 into binary patterns
    signs = 2 * ((bits[:, None] >> np.arange(n)) & 1) - 1  # (2**n, n)
    null  = (signs * d).mean(1)
    return (np.abs(null) >= np.abs(d.mean())).mean()

def fisher_pval(pvals):
    "Combine independent p-values via Fisher's method"
    # prevent log(0) if there is
    pvals = np.clip(pvals, 1e-16, 1.0)
    stat = -2 * np.sum(np.log(pvals))
    return chi2.sf(stat, df=2 * len(pvals))

def fdr_bh(pvals_dict):
    "Benjamini-Hochberg FDR correction on a {roi: pval} dict"
    keys = list(pvals_dict.keys())
    p    = np.array([pvals_dict[k] for k in keys])
    n    = len(p)
    order     = np.argsort(p)
    corrected = np.minimum(1, p[order] * n / np.arange(1, n + 1))
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    result    = np.empty(n)
    result[order] = corrected
    return dict(zip(keys, result))

def load_roi_data(bb, common_rois, roi_map, roi_names):
    "Load ROI R2 data for all participants for a given backbone"
    rows       = []
    fold_diffs = {roi_name: [] for roi_name in roi_names}

    for p_idx, p in enumerate(PARTICIPANTS):
        npy = EVALS_PATH / f"{p_idx + 1}_main_{bb}.npy"
        if not npy.exists(): continue
        res         = np.load(npy, allow_pickle=True).item()
        roi_results = res['roi_results']

        for roi_name in roi_names:
            lr_rois  = [c for c in roi_map[roi_name] if c in common_rois]
            matching = [r for r in lr_rois if r in roi_results]
            if not matching: continue
            r2_base = np.mean([roi_results[r]['r2_base'] for r in matching])
            r2_meta = np.mean([roi_results[r]['r2_meta'] for r in matching])
            for model, r2 in [('base', r2_base), ('meta-learned', r2_meta)]:
                rows.append({'ROI': roi_name, 'Model': model, 'R2': r2, 'Participant': p})
            d = np.mean([roi_results[r]['r2_meta_folds'] - roi_results[r]['r2_base_folds'] for r in matching], axis=0)
            fold_diffs[roi_name].append(d)

    # Sign-flip test per participant, Fisher-combine across participants, FDR-correct across ROIs
    raw_pvals = {}
    for roi_name in roi_names:
        if not fold_diffs[roi_name]: continue
        raw_pvals[roi_name] = fisher_pval([sign_flip_pval(d) for d in fold_diffs[roi_name]])
    roi_pvals = fdr_bh(raw_pvals) if raw_pvals else {}

    return pd.DataFrame(rows), roi_pvals

def get_ordered_rois(roi_names):
    "Order ROIs based on original ROIS list"
    ordered = []
    for r in ROIS:
        if r.startswith('l') or r.startswith('r'):
            base = r[1:]
            if base in roi_names and base not in ordered:
                ordered.append(base)
        else:
            if r in roi_names and r not in ordered:
                ordered.append(r)
    return ordered

def main():
    common_rois = find_common_rois()
    roi_map, roi_names = merge_lr_rois(common_rois)
    ordered_rois = get_ordered_rois(roi_names)
    Path("figures").mkdir(exist_ok=True)

    for bb in BACKBONES:
        diff_img = load_brain_diff(bb, 1)
        plot_df, roi_pvals = load_roi_data(bb, common_rois, roi_map, roi_names)
        if plot_df.empty: continue
        has_flatmap = diff_img is not None

        if has_flatmap:
            mosaic = [['A', 'A', 'A'], ['B', 'B', 'B']]
            fig, axd = plt.subplot_mosaic(mosaic, figsize=(6.99866, 9.6), height_ratios=[6, 1])

            abs_max = np.max(np.abs(diff_img.get_fdata()))
            vol_data = cortex.Volume(
                np.swapaxes(diff_img.get_fdata(), 0, -1), 'S1', 'align_auto',
                cmap='twilight', vmin=-abs_max, vmax=abs_max
            )
            cortex.quickshow(
                vol_data, pixelwise=True, nanmean=True, with_colorbar=False,
                colorbar_location=(.4, .45, .2, .04), fig=axd["A"], with_borders=True,
                roi_list=["EBA", "FFA", "OFA", "STS", "PPA", "RSC", "TOS", "LOC",
                          "V1", "V2", "V3", "hV4", "pSTS", "MPA"]
            )
            axd["A"].set_zorder(2)
            ax_bar = axd["B"]
        else:
            fig, ax_bar = plt.subplots(figsize=(6.99866, 3))

        sns.barplot(data=plot_df, x='ROI', y='R2', hue='Model',
                    ax=ax_bar, order=ordered_rois, hue_order=['base', 'meta-learned'],
                    alpha=0.8, edgecolor="black", linewidth=1, saturation=1, errorbar=None)
        ax_bar.set_ylabel(r"$R^2$")
        ax_bar.set_zorder(1)

        # Overlay individual participant means with connecting lines
        palette   = sns.color_palette()
        dot_color = palette[-1]
        hue_order = ['base', 'meta-learned']
        dodge     = 0.4
        for p in PARTICIPANTS:
            for i, roi in enumerate(ordered_rois):
                pts = plot_df[(plot_df.ROI == roi) & (plot_df.Participant == p)]
                if len(pts) < 2: continue
                xs, ys = [], []
                for j, model in enumerate(hue_order):
                    row = pts[pts.Model == model]
                    if row.empty: continue
                    xs.append(i + (j - 0.5) * dodge)
                    ys.append(row['R2'].values[0])
                ax_bar.plot(xs, ys, color='black',     linewidth=1.4, zorder=4, alpha=0.8)
                ax_bar.plot(xs, ys, color=dot_color,   linewidth=0.8, zorder=5, alpha=0.8)
                ax_bar.scatter(xs, ys, color=dot_color, s=10, zorder=6, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Annotate significant ROIs; skip if both model means are negative
        for i, roi in enumerate(ordered_rois):
            pval = roi_pvals.get(roi)
            if pval is None or pval >= 0.05: continue
            roi_means = plot_df[plot_df.ROI == roi].groupby('Model')['R2'].mean()
            if (roi_means < 0).all(): continue
            star     = '***' if pval < 0.001 else ('**' if pval < 0.01 else '*')
            roi_vals = plot_df[plot_df.ROI == roi]['R2']
            ax_bar.text(i, roi_vals.max() + 0.005, star, ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax_bar.legend(title='', frameon=False, loc='upper center',
                      bbox_to_anchor=(0.88, 1.), ncol=2, columnspacing=0.5)

        if has_flatmap:
            axd["A"].text(-0.09, 1.0, 'A', transform=axd["A"].transAxes, fontsize=16,
                          fontweight='bold', va='top', ha='right')
            ax_bar.text(-0.09, 1.03, 'B', transform=ax_bar.transAxes, fontsize=16,
                        fontweight='bold', va='top', ha='right')

            plt.tight_layout()
            plt.subplots_adjust(hspace=-0.7)

            cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap='twilight', norm=plt.Normalize(vmin=-abs_max, vmax=abs_max)),
                ax=axd["A"], orientation='vertical', fraction=0.02, pad=0.02
            )
            cbar.set_label(r'$R^2_{\mathrm{meta\text{-}learned}}$ $-$ $R^2_{\mathrm{base}}$')
            cbar.locator = plt.MaxNLocator(nbins=6)
            cbar.update_ticks()
            cbar.ax.set_yticklabels([
                f'+{t.get_text()}' if not t.get_text().startswith(('-', '\u2212'))
                and t.get_text().replace('.', '', 1).replace('0', '') != ''
                else t.get_text() for t in cbar.ax.get_yticklabels()
            ])
            cbar.ax.set_zorder(3)
        else:
            plt.tight_layout()

        for ext in ["png", "pdf", "svg"]:
            fig.savefig(f"figures/brain_main_{bb}.{ext}", bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figures/brain_main_{bb}.{{png,pdf,svg}}")

if __name__ == "__main__":
    main()
