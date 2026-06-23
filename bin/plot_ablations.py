import string
import sys
from pathlib import Path

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from groupBMC import GroupBMC
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from plot_evals import get_noise_ceilings


def get_3way_pxp():
    """Run 3-way Group BMC (main vs raw vs midsae) for each dataset/subset/backbone."""
    backbone_map = {'clip': 'CLIP', 'dinov3': 'DINOv3', 'mae': 'MAE', 'siglip2': 'SigLIP2'}
    eval_paths = {
        "things": "data/evals/thingso1o",
        "levels": "data/evals/levelso1o",
        "rewardlearning": "data/evals/rewardlearning",
        "categorylearning": "data/evals/categorylearning",
    }
    rows = {}
    for ds, dir_path in eval_paths.items():
        for bb_key, bb_label in backbone_map.items():
            paths = {exp: Path(f"{dir_path}/{exp}_{bb_key}_trials.h5") for exp in ("main", "raw", "midsae")}
            if not all(p.exists() for p in paths.values()): continue
            dfs = {exp: pd.read_hdf(p, "trials") for exp, p in paths.items()}

            def _run(subframes):
                p_nlls = {}
                for exp, sdf in subframes.items():
                    p_nlls[exp] = sdf.groupby("participant_id")["meta_nll"].sum()
                merged = pd.DataFrame(p_nlls).dropna()
                L = -merged[["main", "raw", "midsae"]].values.T
                bmc = GroupBMC(L)
                res = bmc.get_result()
                return res.protected_exceedance_probability[0]  # PXP for main (index 0)

            # per triplet_type subsets
            if "triplet_type" in dfs["main"].columns:
                for tt in sorted(dfs["main"]["triplet_type"].unique()):
                    sub_dfs = {exp: df[df.triplet_type == tt] for exp, df in dfs.items()}
                    rows[(ds, tt, bb_label)] = _run(sub_dfs)
            rows[(ds, "all", bb_label)] = _run(dfs)
    return rows

def get_summary_data(experiments=("main", "raw", "midsae")):
    backbone_map = {'clip': 'CLIP', 'dinov3': 'DINOv3', 'mae': 'MAE', 'siglip2': 'SigLIP2'}
    eval_paths = {
        "things": ("data/evals/thingso1o", 3),
        "levels": ("data/evals/levelso1o", 3),
        "rewardlearning": ("data/evals/rewardlearning", 2),
        "categorylearning": ("data/evals/categorylearning", 2),
    }

    rows = []
    for ds, (dir_path, n_choices) in eval_paths.items():
        null_nll = np.log(n_choices)
        for exp in experiments:
            for bb_key, bb_label in backbone_map.items():
                path = Path(f"{dir_path}/{exp}_{bb_key}_trials.h5")
                if not path.exists(): continue
                df = pd.read_hdf(path, "trials")
                meta_r2 = 1 - df.meta_nll.values / null_nll

                def _summarise(vals):
                    n = len(vals)
                    return dict(n=n, meta_r2_mean=vals.mean(), meta_r2_sem=vals.std() / np.sqrt(n))

                if "triplet_type" in df.columns:
                    for tt, idx in df.groupby("triplet_type").groups.items():
                        idx = idx.values if hasattr(idx, 'values') else np.array(idx)
                        rows.append(dict(dataset=ds, experiment=exp, backbone=bb_label, subset=tt,
                                         **_summarise(meta_r2[idx])))
                rows.append(dict(dataset=ds, experiment=exp, backbone=bb_label, subset="all",
                                 **_summarise(meta_r2)))
    return pd.DataFrame(rows)

def load_single_feature(feature_path, feature_dim):
    """Load values for a single feature dimension across all images, without materializing full matrix."""
    with h5py.File(feature_path, 'r') as f:
        n = len(f)
        first = f[str(0)]
        if isinstance(first, h5py.Group) and "activations" in first:
            # sparse SAE format
            vals = np.zeros(n, dtype=np.float32)
            for i in range(n):
                g = f[str(i)]
                indices = g["indices"][:]
                mask = indices == feature_dim
                if mask.any():
                    vals[i] = g["activations"][:][mask][0]
        else:
            # dense format - read only the target index from each row
            vals = np.array([f[str(i)][feature_dim] for i in range(n)], dtype=np.float32)
    return vals

def load_feature_examples(image_paths, feature_path, feature_dim, n=3, seed=0):
    vals = load_single_feature(feature_path, feature_dim)
    order = np.argsort(vals)

    # bottom: if tied at the low end, sample randomly among tied values
    bot_val = vals[order[0]]
    tied = np.where(vals[order] == bot_val)[0]
    if len(tied) > n:
        rng = np.random.default_rng(seed)
        bot_idx = order[rng.choice(tied, n, replace=False)]
    else:
        bot_idx = order[:n]

    # top: ascending order (smallest to largest)
    top_idx = order[-n:]
    return (
        [image_paths[i] for i in bot_idx], vals[bot_idx],
        [image_paths[i] for i in top_idx], vals[top_idx],
    )

def read_square(path, size=255):
    """Load image, center-crop to square, resize."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
    return np.asarray(img)

def plot_image_rows(fig, gs, image_paths, feature_configs):
    panel_axes = []   # [(all_axes, color, func_axes_groups)]
    for col, (label, feat_path, feat_dims, color) in enumerate(feature_configs):
        n_funcs = len(feat_dims)
        inner = gs[0, col].subgridspec(n_funcs * 2, 6, wspace=0.05, hspace=0.08)
        all_axes, func_groups = [], []
        for fi, feat_dim in enumerate(feat_dims):
            bot_imgs, bot_vals, top_imgs, top_vals = load_feature_examples(
                image_paths, feat_path, feat_dim, n=6, seed=col * n_funcs + fi + 1)

            func_axes_low, func_axes_high = [], []
            r_low, r_high = fi * 2, fi * 2 + 1
            for j, (img_path, val) in enumerate(zip(bot_imgs, bot_vals)):
                ax = fig.add_subplot(inner[r_low, j])
                ax.imshow(read_square(img_path))
                ax.set_axis_off()
                ax.text(0.5, -0.05, f"{val:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=9)
                all_axes.append(ax); func_axes_low.append(ax)

            for j, (img_path, val) in enumerate(zip(top_imgs, top_vals)):
                ax = fig.add_subplot(inner[r_high, j])
                ax.imshow(read_square(img_path))
                ax.set_axis_off()
                ax.text(0.5, -0.05, f"{val:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=9)
                all_axes.append(ax); func_axes_high.append(ax)

            func_groups.append((feat_dim, func_axes_low, func_axes_high))

        panel_axes.append((all_axes, color, func_groups))

    fig.canvas.draw()

    # colored background boxes + title above each panel
    title_texts = []
    for pi, (all_axes, color, func_groups) in enumerate(panel_axes):
        positions = [ax.get_position() for ax in all_axes]
        x0 = min(p.x0 for p in positions) - 0.006
        y0 = min(p.y0 for p in positions) - 0.018
        x1 = max(p.x1 for p in positions) + 0.006
        y1 = max(p.y1 for p in positions) + 0.008
        fill_color = mcolors.to_rgba(color, alpha=0.1)
        edge_color = mcolors.to_rgba(color, alpha=0.8)
        rect = FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005", edgecolor=edge_color, linewidth=5, facecolor=fill_color,
            transform=fig.transFigure, zorder=0)
        fig.patches.append(rect)

        # title above the colored box, shifted right to leave room for bar
        label = feature_configs[pi][0]
        title_texts.append((
            fig.text((x0 + x1) / 2, y1 + 0.015, label, ha="center", va="bottom",
                     fontsize=14, fontweight="bold"),
            color))

        # dashed rounded rects per function with feature label between rows
        for feat_dim, axes_low, axes_high in func_groups:
            fpos = [ax.get_position() for ax in axes_low + axes_high]
            fx0 = min(p.x0 for p in fpos) - 0.004
            fy0 = min(p.y0 for p in fpos) - 0.012
            fx1 = max(p.x1 for p in fpos) + 0.004
            fy1 = max(p.y1 for p in fpos) + 0.004
            frect = FancyBboxPatch(
                (fx0, fy0), fx1 - fx0, fy1 - fy0,
                boxstyle="round,pad=0.003", edgecolor=mcolors.to_rgba(color, alpha=0.5),
                linewidth=1.5, linestyle="--", facecolor="none",
                transform=fig.transFigure, zorder=1)
            fig.patches.append(frect)

            # feature index label centered between the low and high rows
            low_pos = [ax.get_position() for ax in axes_low]
            high_pos = [ax.get_position() for ax in axes_high]
            gap_y = (min(p.y0 for p in high_pos) + max(p.y1 for p in low_pos)) / 2
            fig.text((fx0 + fx1) / 2, gap_y, rf"$f_{{{feat_dim}}}$",
                     ha="center", va="center", fontsize=13,
                     color=mcolors.to_rgba(color, alpha=0.8))

    # shared "low" / "high" labels on the far left, aligned to each row pair
    first_groups = panel_axes[0][2]  # func_groups of first panel
    left_x = min(ax.get_position().x0 for ax in panel_axes[0][0]) - 0.015
    for feat_dim, axes_low, axes_high in first_groups:
        low_pos = [ax.get_position() for ax in axes_low]
        high_pos = [ax.get_position() for ax in axes_high]
        low_mid_y = (min(p.y0 for p in low_pos) + max(p.y1 for p in low_pos)) / 2
        high_mid_y = (min(p.y0 for p in high_pos) + max(p.y1 for p in high_pos)) / 2
        fig.text(left_x, low_mid_y, "low", ha="right", va="center",
                 fontsize=11, fontstyle="italic", color="0.4", rotation=90)
        fig.text(left_x, high_mid_y, "high", ha="right", va="center",
                 fontsize=11, fontstyle="italic", color="0.4", rotation=90)

    # "A" label
    first_ax = panel_axes[0][0][0]
    top_y = first_ax.get_position().y1
    fig.text(first_ax.get_position().x0 - 0.02, top_y + 0.025,
             "A", fontsize=18, fontweight='bold', va='bottom', ha='right')

    # small colored bars next to each panel title (legend for bar charts)
    fig.canvas.draw()
    bar_w, bar_h = 0.018, 0.012
    for txt, color in title_texts:
        bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        bb_fig = bb.transformed(fig.transFigure.inverted())
        bx = bb_fig.x0 - bar_w - 0.008
        by = bb_fig.y0 + (bb_fig.height - bar_h) / 2
        bar = Rectangle((bx, by), bar_w, bar_h, facecolor=mcolors.to_rgba(color, alpha=0.8),
                         edgecolor="black", linewidth=1, transform=fig.transFigure, zorder=2)
        fig.patches.append(bar)

def main():
    summary = get_summary_data()
    noise_ceilings = get_noise_ceilings()
    pxp_main = get_3way_pxp()
    colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    c_meta, c_raw, c_midsae = colors[1], colors[2], colors[3]

    image_paths = sorted(Path("data/external/THINGSplus").glob("*.jpg"))
    image_paths = [str(p) for p in image_paths]

    feature_configs = [
        ("Disentangled & High-Level Tasks\n(CLIP Layer 11 SAE Latents)", "data/sae/thingsplus_sae-top_k-64-cls_only-layer_11-hook_resid_post.h5", [25264, 836], c_meta),
        ("Entangled Tasks\n(CLIP Layer 11 Activations)", "data/backbone_reps/thingsplus_CLIP-ViT-B-32-DataComp.XL-s13B-b90K_sae-top_k-64-cls_only-layer_11-hook_resid_post_raw.h5", [66, 161], c_raw),
        ("Mid-Level Tasks\n(CLIP Layer 6 SAE Latents)", "data/sae/thingsplus_sae-top_k-64-cls_only-layer_6-hook_resid_post.h5", [20158, 28491], c_midsae),
    ]

    tasks = [
        ("things", "all", "THINGS"),
        ("levels", "within_class", "Levels Within Class"),
        ("levels", "between_class", "Levels Between Class"),
        ("levels", "class_border", "Levels Class Border"),
        ("categorylearning", "all", "Category Learning"),
        ("rewardlearning", "all", "Reward Learning"),
    ]

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 6, height_ratios=[2.5, 1.5], hspace=0.15, wspace=0.25)

    # top row: feature examples (3 panels, each spanning 2 columns)
    gs_top = gs[0, :].subgridspec(1, 3, wspace=0.15)
    plot_image_rows(fig, gs_top, image_paths, feature_configs)

    # bottom row: horizontal bar charts (raw R² values, no legend—colors match top row)
    w = 0.25
    backbones = ['CLIP', 'DINOv3', 'MAE', 'SigLIP2']
    experiments = [("midsae", c_midsae), ("raw", c_raw), ("main", c_meta)]
    edge_color = "black"  # last color of the palette for borders

    for col, (ds, sub, title) in enumerate(tasks):
        ax = fig.add_subplot(gs[1, col])
        mask = (summary.dataset == ds) & (summary.subset == sub)
        df = summary[mask]
        nc = noise_ceilings.get((ds, sub))
        nc = nc if (nc is not None and not np.isnan(nc)) else None

        y = np.arange(len(backbones))
        main_means, main_sems = [], []
        for i, (exp, color) in enumerate(experiments):
            means, sems = [], []
            for bb in backbones:
                row = df[(df.experiment == exp) & (df.backbone == bb)]
                if row.empty:
                    means.append(0); sems.append(0)
                    continue
                m, s = row.meta_r2_mean.values[0], row.meta_r2_sem.values[0]
                if nc is not None: m, s = m / nc, s / nc
                means.append(m); sems.append(s)

            ax.barh(y + (i - 1) * w, means, w, xerr=sems, capsize=3, alpha=0.8,
                    color=color, edgecolor=edge_color, linewidth=1)
            if exp == "main": main_means, main_sems = means, sems

        # annotate PXP for "Disentangled & Abstract" on the main bars
        for j, bb in enumerate(backbones):
            pxp = pxp_main.get((ds, sub, bb))
            if pxp is None: continue
            pxp_str = "PXP>.99" if pxp >= 0.995 else f"PXP={pxp:.2f}"
            bar_right = main_means[j] + main_sems[j] + 0.005
            ax.text(bar_right, y[j] + w, pxp_str, ha="left", va="center", fontsize=11)

        ax.set_yticks(y)
        ax.set_yticklabels(backbones if col == 0 else [])
        ax.set_title(title)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.text(-0.15, 1.05, string.ascii_uppercase[1 + col], transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
        ax.set_xlabel(r"McFadden's $R^2$")
    fig.savefig("figures/ablations.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/ablations.pdf", bbox_inches="tight")
    plt.savefig("figures/ablations.svg", bbox_inches="tight")

if __name__ == "__main__":
    main()
