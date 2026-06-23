"""Plot meta-learned transformer vs static linear baseline: behavioral + brain."""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import chi2

DATA_PATH  = Path("data/external/brain_data")
EVALS_PATH = Path("data/evals")
BRAIN_PATH = Path("data/evals/brain")

ROIS = ['lEBA', "rEBA", "lFFA", "rFFA", "lOFA", "rOFA", "lSTS", "rSTS",
        "lPPA", "rPPA", "lRSC", "rRSC", "lTOS", "rTOS", 'lLOC', "rLOC",
        "V1", "V2", "V3", "hV4"]
PARTICIPANTS    = ['01', '02', '03']
BACKBONES       = ['siglip2', 'mae', 'dinov3', 'clip']
BACKBONE_LABELS = {'clip': 'CLIP', 'dinov3': 'DINOv3', 'mae': 'MAE', 'siglip2': 'SigLIP2'}


def get_noise_ceilings():
    # THINGS: find repeated (subject, triplet) pairs, compute entropy of choice distribution
    _things = pd.read_csv("data/external/THINGS_triplets.csv", sep="\t",
                          usecols=["subject_id", "image1", "image2", "image3", "choice"])
    _things["tid"] = _things.groupby(["subject_id", "image1", "image2", "image3"]).ngroup()
    _tcounts = _things.groupby("tid").size()
    _rep     = _things[_things.tid.isin(_tcounts[_tcounts > 1].index)]
    _cc      = _rep.groupby(["tid", "choice"]).size().unstack(fill_value=0)
    _probs   = _cc.div(_cc.sum(axis=1), axis=0)
    _ent     = -(_probs * np.log(_probs.clip(1e-10))).sum(axis=1)
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
        _tc      = _sub.groupby("tid").size()
        _rep_sub = _sub[_sub.tid.isin(_tc[_tc > 1].index)]
        if len(_rep_sub) == 0:
            _levels_nc[_tt] = np.nan
            continue
        _cc = _rep_sub.groupby(["tid", "selected"]).size().unstack(fill_value=0)
        _p  = _cc.div(_cc.sum(axis=1), axis=0)
        _e  = -(_p * np.log(_p.clip(1e-10))).sum(axis=1)
        _levels_nc[_tt] = 1 - _e.mean() / np.log(3)

    return {
        ("things", "all"):           things_nc_r2,
        ("levels", "within_class"):  _levels_nc["within_class"],
        ("levels", "between_class"): _levels_nc["between_class"],
        ("levels", "class_border"):  _levels_nc["class_border"],
        ("levels", "all"):           _levels_nc["all"],
    }


def get_summary_data():
    """Load main and static_linear R² from static_linear experiment h5 files.

    The static_linear h5 files contain both meta_nll (static linear model) and
    main_nll (main transformer), so a single load suffices per backbone.
    """
    backbone_map = {'clip': 'CLIP', 'dinov3': 'DINOv3', 'mae': 'MAE', 'siglip2': 'SigLIP2'}
    eval_paths   = {
        "things":           (EVALS_PATH / "thingso1o",       3),
        "levels":           (EVALS_PATH / "levelso1o",       3),
        "rewardlearning":   (EVALS_PATH / "rewardlearning",  2),
        "categorylearning": (EVALS_PATH / "categorylearning",2),
    }

    rows = []
    for _dataset, (_dir, _n_choices) in eval_paths.items():
        _null_nll = np.log(_n_choices)
        for _bb in BACKBONES:
            _path = _dir / f"static_linear_{_bb}_trials.h5"
            if not _path.exists(): continue
            _df      = pd.read_hdf(_path, "trials")
            _sl_r2   = 1 - _df.meta_nll.values / _null_nll
            _main_r2 = 1 - _df.main_nll.values / _null_nll

            def _summarise(_sl, _m):
                _n = len(_sl)
                return dict(n=_n,
                            sl_r2_mean=_sl.mean(),    sl_r2_sem=_sl.std() / np.sqrt(_n),
                            main_r2_mean=_m.mean(), main_r2_sem=_m.std() / np.sqrt(_n))

            if "triplet_type" in _df.columns:
                for _tt, _idx in _df.groupby("triplet_type").groups.items():
                    _idx = _idx.values if hasattr(_idx, 'values') else np.array(_idx)
                    rows.append(dict(dataset=_dataset, backbone=backbone_map[_bb], subset=_tt,
                                     **_summarise(_sl_r2[_idx], _main_r2[_idx])))
            rows.append(dict(dataset=_dataset, backbone=backbone_map[_bb], subset="all",
                             **_summarise(_sl_r2, _main_r2)))
    return pd.DataFrame(rows)


def get_all_stats():
    _o1o = pd.read_csv(EVALS_PATH / "o1o_stats.csv")
    _o1o = _o1o[(_o1o.comparison == "meta_vs_main") & (_o1o.experiment == "static_linear")].reset_index(drop=True)
    _rl  = pd.read_csv(EVALS_PATH / "rewardlearning/stats.csv")
    _rl  = _rl[(_rl.comparison  == "meta_vs_main") & (_rl.experiment  == "static_linear")].reset_index(drop=True)
    _rl["dataset"], _rl["subset"] = "rewardlearning", "all"
    _cl  = pd.read_csv(EVALS_PATH / "categorylearning/stats.csv")
    _cl  = _cl[(_cl.comparison  == "meta_vs_main") & (_cl.experiment  == "static_linear")].reset_index(drop=True)
    _cl["dataset"], _cl["subset"] = "categorylearning", "all"
    return pd.concat([_o1o, _rl, _cl], ignore_index=True)


def plot_r2_bar(summary, stats, dataset, subset, ax, title, noise_ceiling=None, show_yticks=True, xlim_pad=0):
    _w          = 0.35
    _bb_reverse = {'CLIP': 'clip', 'DINOv3': 'dinov3', 'MAE': 'mae', 'SigLIP2': 'siglip2'}
    _colors     = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    _c_sl, _c_main = _colors[-3], _colors[1]

    _mask   = (summary.dataset == dataset) & (summary.subset == subset)
    _df     = summary[_mask].copy()
    _bbs    = sorted(_df.backbone.unique())
    _nc     = noise_ceiling if (noise_ceiling is not None and not np.isnan(noise_ceiling)) else None

    _stat_mask = (stats.dataset == dataset) & (stats.subset == subset)
    _stat_df   = stats[_stat_mask].copy()
    _stat_df["backbone_label"] = _stat_df.backbone.map({v: k for k, v in _bb_reverse.items()})

    _sl_means,   _sl_sems   = [], []
    _main_means, _main_sems = [], []
    _annotations = []
    for _bb in _bbs:
        _row = _df[_df.backbone == _bb].iloc[0]
        _sm, _ss = _row.sl_r2_mean,   _row.sl_r2_sem
        _mm, _ms = _row.main_r2_mean, _row.main_r2_sem
        if _nc is not None:
            _sm, _ss = _sm / _nc, _ss / _nc
            _mm, _ms = _mm / _nc, _ms / _nc
        _sl_means.append(_sm);   _sl_sems.append(_ss)
        _main_means.append(_mm); _main_sems.append(_ms)

        _srow = _stat_df[_stat_df.backbone_label == _bb]
        if len(_srow):
            _pxp_val = _srow.pxp_main.values[0]
            _pxp     = "PXP>.99" if _pxp_val >= 0.995 else f"PXP={_pxp_val:.2f}"
            _annotations.append(_pxp)
        else: _annotations.append("")

    _y = np.arange(len(_bbs))
    ax.barh(_y + _w/2, _sl_means,   _w, xerr=_sl_sems,   capsize=3, alpha=0.8,
            color=_c_sl,   edgecolor="black", linewidth=1, label="multitask-learned")
    ax.barh(_y - _w/2, _main_means, _w, xerr=_main_sems, capsize=3, alpha=0.8,
            color=_c_main, edgecolor="black", linewidth=1, label="meta-learned")
    ax.set_yticks(_y)
    ax.set_yticklabels(_bbs if show_yticks else [])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel("McFadden's $R^2$")

    for _i, _ann in enumerate(_annotations):
        if _ann:
            _bar_right = _main_means[_i] + _main_sems[_i] + 0.005
            ax.text(_bar_right, _y[_i] - _w/2, _ann, ha="left", va="center", fontsize=11)

    if xlim_pad > 0 and _sl_means:
        _max_val = max([x + e for x, e in zip(_sl_means, _sl_sems)] +
                       [x + e for x, e in zip(_main_means, _main_sems)])
        ax.set_xlim(0, _max_val + xlim_pad)

    ax.set_title(title, y=.99)
    return _bbs


def find_common_rois():
    all_participant_rois = []
    for p in PARTICIPANTS:
        vox_df    = pd.read_csv(DATA_PATH / f"sub-{p}_VoxelMetadata.csv")
        available = set(vox_df.columns).intersection(ROIS)
        all_participant_rois.append(available)
    return set.intersection(*all_participant_rois)


def merge_lr_rois(common_rois):
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


def get_ordered_rois(roi_names):
    ordered = []
    for r in ROIS:
        if r.startswith('l') or r.startswith('r'):
            base = r[1:]
            if base in roi_names and base not in ordered: ordered.append(base)
        else:
            if r in roi_names and r not in ordered: ordered.append(r)
    return ordered


def sign_flip_pval(d):
    n     = len(d)
    bits  = np.arange(2**n)
    signs = 2 * ((bits[:, None] >> np.arange(n)) & 1) - 1
    null  = (signs * d).mean(1)
    return (np.abs(null) >= np.abs(d.mean())).mean()


def fisher_pval(pvals):
    pvals = np.clip(pvals, 1e-16, 1.0)
    stat  = -2 * np.sum(np.log(pvals))
    return chi2.sf(stat, df=2 * len(pvals))


def fdr_bh(pvals_dict):
    keys      = list(pvals_dict.keys())
    p         = np.array([pvals_dict[k] for k in keys])
    n         = len(p)
    order     = np.argsort(p)
    corrected = np.minimum(1, p[order] * n / np.arange(1, n + 1))
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    result    = np.empty(n)
    result[order] = corrected
    return dict(zip(keys, result))


def load_roi_data_comparison(bb, common_rois, roi_map, roi_names):
    """Load ROI R2 for main vs static_linear from static_linear experiment npy files.

    The static_linear npy files contain r2_meta (SL model) and r2_main (main transformer),
    so a single load per participant suffices.
    """
    rows       = []
    fold_diffs = {roi_name: [] for roi_name in roi_names}

    for p_idx, p in enumerate(PARTICIPANTS):
        npy = BRAIN_PATH / f"{p_idx + 1}_static_linear_{bb}.npy"
        if not npy.exists(): continue
        res         = np.load(npy, allow_pickle=True).item()
        roi_results = res['roi_results']

        for roi_name in roi_names:
            lr_rois  = [c for c in roi_map[roi_name] if c in common_rois]
            matching = [r for r in lr_rois if r in roi_results]
            if not matching: continue
            r2_sl   = np.mean([roi_results[r]['r2_meta'] for r in matching])
            r2_main = np.mean([roi_results[r]['r2_main'] for r in matching])
            for model, r2 in [('multitask-learned', r2_sl), ('meta-learned', r2_main)]:
                rows.append({'ROI': roi_name, 'Model': model, 'R2': r2, 'Participant': p})
            d = np.mean([roi_results[r]['r2_main_folds'] - roi_results[r]['r2_meta_folds']
                         for r in matching], axis=0)
            fold_diffs[roi_name].append(d)

    raw_pvals = {}
    for roi_name in roi_names:
        if not fold_diffs[roi_name]: continue
        raw_pvals[roi_name] = fisher_pval([sign_flip_pval(d) for d in fold_diffs[roi_name]])
    roi_pvals = fdr_bh(raw_pvals) if raw_pvals else {}

    return pd.DataFrame(rows), roi_pvals


def main():
    noise_ceilings = get_noise_ceilings()
    trial_summary  = get_summary_data()
    all_stats      = get_all_stats()
    Path("figures").mkdir(exist_ok=True)

    all_panels = [
        ("things",           "all",           "THINGS",                noise_ceilings.get(("things", "all")),           0.1),
        ("levels",           "within_class",  "Levels\nWithin Class",  noise_ceilings.get(("levels", "within_class")),  0.1),
        ("levels",           "between_class", "Levels\nBetween Class", noise_ceilings.get(("levels", "between_class")), 0.1),
        ("levels",           "class_border",  "Levels\nClass Border",  noise_ceilings.get(("levels", "class_border")),  0.1),
        ("categorylearning", "all",           "Category Learning",     None,                                            0.05),
        ("rewardlearning",   "all",           "Reward Learning",       None,                                            0.05),
    ]

    import string as _string

    # --- Behavioral figure (meta) ---
    fig = plt.figure(figsize=(20, 5))
    gs  = fig.add_gridspec(1, 6, wspace=0.2)

    _first_ax = None
    for col, (ds, sub, title, nc, pad) in enumerate(all_panels):
        ax = fig.add_subplot(gs[0, col])
        plot_r2_bar(trial_summary, all_stats, ds, sub, ax, title,
                    noise_ceiling=nc, show_yticks=(col == 0), xlim_pad=pad)
        ax.text(-0.12 if col == 0 else -0.05, 1.08, _string.ascii_uppercase[col],
                transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
        if col == 0: _first_ax = ax

    handles, labels = _first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=2, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"figures/meta.{ext}", bbox_inches='tight',
                    **({"dpi": 300} if ext == "png" else {}))
    plt.close(fig)

    # --- Brain figure (meta_sup) ---
    common_rois        = find_common_rois()
    roi_map, roi_names = merge_lr_rois(common_rois)
    ordered_rois       = get_ordered_rois(roi_names)

    _colors   = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    hue_order = ['multitask-learned', 'meta-learned']
    brain_palette = {'multitask-learned': _colors[-3], 'meta-learned': _colors[1]}
    dot_color = sns.color_palette()[-1]
    dodge     = 0.4

    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(4, 1, hspace=0.45)
    _first_brain_ax = None

    for bb_row, bb in enumerate(BACKBONES):
        ax = fig.add_subplot(gs[bb_row, 0],
                             **({"sharex": _first_brain_ax} if _first_brain_ax else {}))
        if _first_brain_ax is None: _first_brain_ax = ax

        plot_df, roi_pvals = load_roi_data_comparison(bb, common_rois, roi_map, roi_names)
        if plot_df.empty:
            ax.set_visible(False)
            continue

        sns.barplot(data=plot_df, x='ROI', y='R2', hue='Model',
                    ax=ax, order=ordered_rois, hue_order=hue_order, palette=brain_palette,
                    alpha=0.8, edgecolor="black", linewidth=1, saturation=1, errorbar=None)
        ax.set_ylabel(r"$R^2$")
        ax.set_title(BACKBONE_LABELS[bb])

        ax.get_legend().remove()

        is_last = bb_row == len(BACKBONES) - 1
        if not is_last:
            ax.tick_params(axis='x', labelbottom=False)
            ax.set_xlabel("")

        for p in PARTICIPANTS:
            for i, roi in enumerate(ordered_rois):
                pts = plot_df[(plot_df.ROI == roi) & (plot_df.Participant == p)]
                if len(pts) < 2: continue
                xs, ys = [], []
                for j, model in enumerate(hue_order):
                    pt_row = pts[pts.Model == model]
                    if pt_row.empty: continue
                    xs.append(i + (j - 0.5) * dodge)
                    ys.append(pt_row['R2'].values[0])
                ax.plot(xs, ys, color='black',    linewidth=1.4, zorder=4, alpha=0.8)
                ax.plot(xs, ys, color=dot_color,  linewidth=0.8, zorder=5, alpha=0.8)
                ax.scatter(xs, ys, color=dot_color, s=10, zorder=6, alpha=0.8,
                           edgecolor='black', linewidth=0.5)

        for i, roi in enumerate(ordered_rois):
            pval = roi_pvals.get(roi)
            if pval is None or pval >= 0.05: continue
            roi_means = plot_df[plot_df.ROI == roi].groupby('Model')['R2'].mean()
            if (roi_means < 0).all(): continue
            star     = '***' if pval < 0.001 else ('**' if pval < 0.01 else '*')
            roi_vals = plot_df[plot_df.ROI == roi]['R2']
            ax.text(i, roi_vals.max() + 0.005, star, ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    handles, labels = _first_brain_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=2, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"figures/meta_sup.{ext}", bbox_inches='tight',
                    **({"dpi": 300} if ext == "png" else {}))


if __name__ == "__main__":
    main()
