"""ROI bar plots for all backbones in one figure, one row per backbone."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from plot_brain import (
    BACKBONE_LABELS,
    BACKBONES,
    PARTICIPANTS,
    find_common_rois,
    get_ordered_rois,
    load_roi_data,
    merge_lr_rois,
)


def plot_roi_panel(ax, plot_df, roi_pvals, ordered_rois):
    hue_order = ['base', 'meta-learned']
    palette   = sns.color_palette()
    dot_color = palette[-1]
    dodge     = 0.4
    sns.barplot(data=plot_df, x='ROI', y='R2', hue='Model',
                ax=ax, order=ordered_rois, hue_order=hue_order,
                alpha=0.8, edgecolor='black', linewidth=1, saturation=1, errorbar=None)
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
            ax.plot(xs, ys, color='black',   linewidth=1.4, zorder=4, alpha=0.8)
            ax.plot(xs, ys, color=dot_color, linewidth=0.8, zorder=5, alpha=0.8)
            ax.scatter(xs, ys, color=dot_color, s=10, zorder=6, alpha=0.8, edgecolor='black', linewidth=0.5)
    for i, roi in enumerate(ordered_rois):
        pval = roi_pvals.get(roi)
        if pval is None or pval >= 0.05: continue
        roi_means = plot_df[plot_df.ROI == roi].groupby('Model')['R2'].mean()
        if (roi_means < 0).all(): continue
        star     = '***' if pval < 0.001 else ('**' if pval < 0.01 else '*')
        roi_vals = plot_df[plot_df.ROI == roi]['R2']
        ax.text(i, roi_vals.max() + 0.005, star, ha='center', va='bottom', fontsize=10, fontweight='bold')


def main():
    common_rois  = find_common_rois()
    roi_map, roi_names = merge_lr_rois(common_rois)
    ordered_rois = get_ordered_rois(roi_names)
    Path("figures").mkdir(exist_ok=True)

    fig, axes = plt.subplots(len(BACKBONES), 1, figsize=(6.99866, 3 * len(BACKBONES)), sharex=True)
    for ax, bb in zip(axes, BACKBONES):
        plot_df, roi_pvals = load_roi_data(bb, common_rois, roi_map, roi_names)
        if plot_df.empty: continue
        plot_roi_panel(ax, plot_df, roi_pvals, ordered_rois)
        ax.set_ylabel(r"$R^2$")
        ax.set_title(BACKBONE_LABELS[bb], fontweight='bold')
        ax.get_legend().remove()

    axes[-1].legend(title='', frameon=False, loc='upper right', ncol=2)
    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(f"figures/brain_combined.{ext}", bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/brain_combined.{png,pdf,svg}")


if __name__ == '__main__':
    main()
