import pickle
import string
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter


def get_noise_ceilings():
    # THINGS: find repeated (subject, triplet) pairs, compute entropy of choice distribution
    _things = pd.read_csv("data/external/THINGS_triplets.csv", sep="\t",
                          usecols=["subject_id", "image1", "image2", "image3", "choice"])
    _things["tid"] = _things.groupby(["subject_id", "image1", "image2", "image3"]).ngroup()
    _tcounts = _things.groupby("tid").size()
    _rep = _things[_things.tid.isin(_tcounts[_tcounts > 1].index)]
    _cc = _rep.groupby(["tid", "choice"]).size().unstack(fill_value=0)
    _probs = _cc.div(_cc.sum(axis=1), axis=0)
    _ent = -(_probs * np.log(_probs.clip(1e-10))).sum(axis=1)
    things_nc_r2 = 1 - _ent.mean() / np.log(3)

    # LEVELS: find repeated (participant, sorted triplet) pairs per triplet_type
    with open("data/external/levels.pkl", "rb") as f:
        _levels = pickle.load(f)
    _lrows = []
    for _pid, _pdata in _levels.items():
        for _t in _pdata:
            if _t.get("exp_trial_type") != "exp_trial": continue
            _imgs = tuple(sorted([_t["image1Path"], _t["image2Path"], _t["image3Path"]]))
            _lrows.append((_pid, _imgs, _t["selected_image"], _t.get("triplet_type", "unknown")))
    _ldf = pd.DataFrame(_lrows, columns=["pid", "triplet", "selected", "triplet_type"])

    _levels_nc = {}
    for _tt in ["within_class", "between_class", "class_border", "all"]:
        _sub = _ldf if _tt == "all" else _ldf[_ldf.triplet_type == _tt]
        _sub = _sub.copy()
        _sub["tid"] = _sub.groupby(["pid", "triplet"]).ngroup()
        _tc = _sub.groupby("tid").size()
        _rep_sub = _sub[_sub.tid.isin(_tc[_tc > 1].index)]
        if len(_rep_sub) == 0:
            _levels_nc[_tt] = np.nan
            continue
        _cc = _rep_sub.groupby(["tid", "selected"]).size().unstack(fill_value=0)
        _p = _cc.div(_cc.sum(axis=1), axis=0)
        _e = -(_p * np.log(_p.clip(1e-10))).sum(axis=1)
        _levels_nc[_tt] = 1 - _e.mean() / np.log(3)

    return {
        ("things", "all"): things_nc_r2,
        ("levels", "within_class"): _levels_nc["within_class"],
        ("levels", "between_class"): _levels_nc["between_class"],
        ("levels", "class_border"): _levels_nc["class_border"],
        ("levels", "all"): _levels_nc["all"],
    }

def get_summary_data():
    backbone_map = {'clip': 'CLIP', 'dinov3': 'DINOv3', 'mae': 'MAE', 'siglip2': 'SigLIP2'}
    backbones = list(backbone_map.keys())
    eval_paths = {
        "things": ("data/evals/thingso1o", 3),
        "levels": ("data/evals/levelso1o", 3),
        "rewardlearning": ("data/evals/rewardlearning", 2),
        "categorylearning": ("data/evals/categorylearning", 2),
    }

    rows = []
    for _dataset, (_dir, _n_choices) in eval_paths.items():
        _null_nll = np.log(_n_choices)
        for _bb in backbones:
            _path = Path(f"{_dir}/main_{_bb}_trials.h5")
            if not _path.exists(): continue
            _df = pd.read_hdf(_path, "trials")
            _base_r2 = 1 - _df.base_nll.values / _null_nll
            _meta_r2 = 1 - _df.meta_nll.values / _null_nll
            
            def _summarise(_b, _m):
                _n = len(_b)
                return dict(n=_n, base_r2_mean=_b.mean(), base_r2_sem=_b.std() / np.sqrt(_n),
                            meta_r2_mean=_m.mean(), meta_r2_sem=_m.std() / np.sqrt(_n))

            if "triplet_type" in _df.columns:
                for _tt, _idx in _df.groupby("triplet_type").groups.items():
                    _idx = _idx.values if hasattr(_idx, 'values') else np.array(_idx)
                    rows.append(dict(dataset=_dataset, backbone=backbone_map[_bb], subset=_tt,
                                     **_summarise(_base_r2[_idx], _meta_r2[_idx])))
            rows.append(dict(dataset=_dataset, backbone=backbone_map[_bb], subset="all",
                             **_summarise(_base_r2, _meta_r2)))
    return pd.DataFrame(rows)

def get_all_stats():
    _o1o = pd.read_csv("data/evals/o1o_stats.csv")
    _o1o = _o1o[(_o1o.comparison == "meta_vs_base") & (_o1o.experiment == "main")].reset_index(drop=True)
    _rl = pd.read_csv("data/evals/rewardlearning/stats.csv")
    _rl = _rl[(_rl.comparison == "meta_vs_base") & (_rl.experiment == "main")].reset_index(drop=True)
    _rl["dataset"], _rl["subset"] = "rewardlearning", "all"
    _cl = pd.read_csv("data/evals/categorylearning/stats.csv")
    _cl = _cl[(_cl.comparison == "meta_vs_base") & (_cl.experiment == "main")].reset_index(drop=True)
    _cl["dataset"], _cl["subset"] = "categorylearning", "all"
    return pd.concat([_o1o, _rl, _cl], ignore_index=True)

def _add_img(ax, path, bounds):
    """Add image to ax using inset_axes with bounds [x, y, w, h] in axes coords."""
    if not path.exists(): return
    _imax = ax.inset_axes(bounds, transform=ax.transAxes)
    _imax.imshow(mpimg.imread(path)); _imax.set_axis_off()


def draw_category_schematic(ax, things_dir, sel_color):
    """Draw category learning task schematic: image -> F/J keys -> feedback."""
    ax.set_xlim(0, 14); ax.set_ylim(0, 20); ax.set_axis_off(); ax.set_aspect('equal')

    # top panel
    ax.add_patch(Rectangle((0.3, 10), 13.4, 9.5, lw=1.5, ec='grey', fc='white'))
    _add_img(ax, things_dir / "bread.jpg", [0.2, 0.62, 0.6, 0.28])
    # F and J keys
    for _x, _lbl, _c in [(4.5, "F", "grey"), (8, "J", sel_color)]:
        ax.add_patch(Rectangle((_x, 10.5), 1.4, 1.4, lw=1.5, ec='grey', fc='white'))
        ax.text(_x + 0.7, 11.1, _lbl, ha='center', va='center', fontsize=18, fontweight='bold', color=_c)

    # bottom panel (feedback)
    ax.add_patch(Rectangle((0.3, 0.3), 13.4, 9, lw=1.5, ec='grey', fc='white'))
    ax.text(7, 6.2, "Correct!", ha='center', va='center', fontsize=20, fontweight='bold', color='olive')
    ax.text(7, 3.5, "Folty loves the image!", ha='center', va='center', fontsize=14, color='grey')


def draw_reward_schematic(ax, things_dir, sel_color):
    """Draw reward learning task schematic: two images -> arrows -> reward feedback."""
    ax.set_xlim(0, 14); ax.set_ylim(0, 20); ax.set_axis_off(); ax.set_aspect('equal')
    _imgs = [things_dir / "screw.jpg", things_dir / "soy_sauce.jpg"]

    # top panel
    ax.add_patch(Rectangle((0.3, 10), 13.4, 9.5, lw=1.5, ec='grey', fc='white'))
    _add_img(ax, _imgs[0], [0.05, 0.62, 0.38, 0.28])
    _add_img(ax, _imgs[1], [0.57, 0.62, 0.38, 0.28])
    # arrow boxes + arrows
    for _x, _dx, _c in [(4.5, -1.2, sel_color), (7.8, 1.2, 'grey')]:
        ax.add_patch(Rectangle((_x, 10.5), 2.0, 1.4, lw=1.5, ec='grey', fc='white', zorder=0))
        ax.annotate("", xy=(_x + 1.0 + _dx * 0.5, 11.2), xytext=(_x + 1.0 - _dx * 0.5, 11.2),
                     arrowprops=dict(arrowstyle="-|>", color=_c, lw=3, mutation_scale=25))

    # bottom panel (feedback with rewards)
    ax.add_patch(Rectangle((0.3, 0.3), 13.4, 9, lw=1.5, ec='grey', fc='white'))
    _add_img(ax, _imgs[0], [0.05, 0.15, 0.38, 0.28])
    _add_img(ax, _imgs[1], [0.57, 0.15, 0.38, 0.28])
    ax.text(3.3, 1.5, "93", ha='center', va='center', fontsize=18, fontweight='bold', color=sel_color)
    ax.text(10.7, 1.5, "8", ha='center', va='center', fontsize=18, fontweight='bold', color='grey')


def plot_r2_bar(summary, stats, dataset, subset, ax, title, noise_ceiling=None, show_yticks=True, show_xlabel=True, xlim_pad=0):
    _w = 0.35
    _bb_reverse = {'CLIP': 'clip', 'DINOv3': 'dinov3', 'MAE': 'mae', 'SigLIP2': 'siglip2'}
    _colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    _c_base, _c_meta = _colors[0], _colors[1]
    _edge_color = "black"
    
    _mask = (summary.dataset == dataset) & (summary.subset == subset)
    _df = summary[_mask].copy()
    _bbs = sorted(_df.backbone.unique())
    _nc = noise_ceiling if (noise_ceiling is not None and not np.isnan(noise_ceiling)) else None

    _stat_mask = (stats.dataset == dataset) & (stats.subset == subset)
    _stat_df = stats[_stat_mask].copy()
    _stat_df["backbone_label"] = _stat_df.backbone.map({v: k for k, v in _bb_reverse.items()})

    _base_means, _base_sems = [], []
    _meta_means, _meta_sems = [], []
    _annotations = []
    for _bb in _bbs:
        _row = _df[_df.backbone == _bb].iloc[0]
        _bm, _bs = _row.base_r2_mean, _row.base_r2_sem
        _mm, _ms = _row.meta_r2_mean, _row.meta_r2_sem
        if _nc is not None:
            _bm, _bs = _bm / _nc, _bs / _nc
            _mm, _ms = _mm / _nc, _ms / _nc
        _base_means.append(_bm); _base_sems.append(_bs)
        _meta_means.append(_mm); _meta_sems.append(_ms)

        _srow = _stat_df[_stat_df.backbone_label == _bb]
        if len(_srow):
            _pxp_val = _srow.pxp_meta.values[0]
            _pxp = "PXP>.99" if _pxp_val >= 0.995 else f"PXP={_pxp_val:.2f}"
            _ann =  _pxp
            _annotations.append(_ann)
        else: _annotations.append("")

    _y = np.arange(len(_bbs))
    ax.barh(_y + _w/2, _base_means, _w, xerr=_base_sems, capsize=3, alpha=0.8, color=_c_base, edgecolor=_edge_color, linewidth=1, label="base")
    ax.barh(_y - _w/2, _meta_means, _w, xerr=_meta_sems, capsize=3, alpha=0.8, color=_c_meta, edgecolor=_edge_color, linewidth=1, label="meta-learned")
    ax.set_yticks(_y)
    ax.set_yticklabels(_bbs if show_yticks else [])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    for _i, _ann in enumerate(_annotations):
        if _ann:
            _bar_right = _meta_means[_i] + _meta_sems[_i] + 0.005
            ax.text(_bar_right, _y[_i] - _w/2, _ann, ha="left", va="center", fontsize=11)

    if xlim_pad > 0 and _base_means:
        _max_val = max([x + e for x, e in zip(_base_means, _base_sems)] + [x + e for x, e in zip(_meta_means, _meta_sems)])
        ax.set_xlim(0, _max_val + xlim_pad)

    ax.set_title(title, y=.95)
    ax.set_xlabel("McFadden's $R^2$" if show_xlabel else "")
    return _bbs

def main():
    noise_ceilings = get_noise_ceilings()
    trial_summary = get_summary_data()
    all_stats = get_all_stats()
    sel_color = [c['color'] for c in plt.rcParams['axes.prop_cycle']][4]
    things_dir = Path("data/external/THINGSplus")
    
    example_images = {
        "THINGS": ["lamp.jpg", "seahorse.jpg", "dog.jpg"],
        "Levels Within Class": ["rooster.jpg", "bird.jpg", "parrot.jpg"],
        "Levels Between Class": ["television.jpg", "whipped_cream.jpg", "keyboard.jpg"],
        "Levels Class Border": ["gum.jpg", "gumball.jpg", "hippopotamus.jpg"],
    }
    selected_idx = {"THINGS": 0, "Levels Within Class": 1, "Levels Between Class": 1, "Levels Class Border": 2}
    o1o_subplots = [("things", "all", "THINGS"), ("levels", "within_class", "Levels Within Class"),
                    ("levels", "between_class", "Levels Between Class"), ("levels", "class_border", "Levels Class Border")]

    fig = plt.figure(figsize=(20, 10))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[4, 3], hspace=0.2)
    gs_top = gs_outer[0].subgridspec(2, 4, height_ratios=[1, 3], hspace=0.12, wspace=0.2)

    _first_ax = None
    for col, (ds, sub, title) in enumerate(o1o_subplots):
        imgs, sel = example_images[title], selected_idx[title]
        gs_img = gs_top[0, col].subgridspec(1, len(imgs), wspace=0.05)
        for j, img_path in enumerate(imgs):
            ax_img = fig.add_subplot(gs_img[0, j])
            if (things_dir / img_path).exists():
                ax_img.imshow(mpimg.imread(things_dir / img_path))
            ax_img.set_axis_off()
            if j == sel:
                ax_img.add_patch(Rectangle((0, 0), 1, 1, linewidth=5, edgecolor=sel_color, facecolor="none", transform=ax_img.transAxes, clip_on=False))
            if j == 0:
                ax_img.text(-0.25 if col == 0 else -0.1, 1.1, string.ascii_uppercase[col], transform=ax_img.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

        ax = fig.add_subplot(gs_top[1, col])
        plot_r2_bar(trial_summary, all_stats, ds, sub, ax, title, noise_ceiling=noise_ceilings.get((ds, sub)), show_yticks=(col == 0), xlim_pad=0.1)
        if col == 0: _first_ax = ax

    handles, labels = _first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, .93), ncol=2, frameon=False)

    gs_bot = gs_outer[1].subgridspec(1, 4, wspace=0.2, width_ratios=[1, 2, 1, 2])
    for i, (col, draw_fn, title) in enumerate([(0, draw_category_schematic, "Category Learning"), (2, draw_reward_schematic, "Reward Learning")]):
        ax = fig.add_subplot(gs_bot[0, col])
        draw_fn(ax, things_dir, sel_color)
        ax.set_title(title, y=.98)
        ax.text(-0.1, 1.05, string.ascii_uppercase[4 + i], transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

    for i, (ds, sub, title) in enumerate([("categorylearning", "all", "Category Learning"), ("rewardlearning", "all", "Reward Learning")]):
        ax = fig.add_subplot(gs_bot[0, i * 2 + 1])
        plot_r2_bar(trial_summary, all_stats, ds, sub, ax, "", show_yticks=False, xlim_pad=0.05)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("figures/evals.png", dpi=300, bbox_inches='tight')
    fig.savefig("figures/evals.pdf", bbox_inches='tight')
    plt.savefig("figures/evals.svg", bbox_inches='tight')

if __name__ == "__main__":
    main()
