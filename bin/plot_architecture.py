import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


def get_noise_ceilings():
    _things = pd.read_csv("data/external/THINGS_triplets.csv", sep="\t",
                          usecols=["subject_id", "image1", "image2", "image3", "choice"])
    _things["tid"] = _things.groupby(["subject_id", "image1", "image2", "image3"]).ngroup()
    _tcounts = _things.groupby("tid").size()
    _rep = _things[_things.tid.isin(_tcounts[_tcounts > 1].index)]
    _cc = _rep.groupby(["tid", "choice"]).size().unstack(fill_value=0)
    _probs = _cc.div(_cc.sum(axis=1), axis=0)
    _ent = -(_probs * np.log(_probs.clip(1e-10))).sum(axis=1)
    things_nc = 1 - _ent.mean() / np.log(3)

    with open("data/external/levels.pkl", "rb") as f:
        _levels = pickle.load(f)
    _lrows = []
    for _pid, _pdata in _levels.items():
        for _t in _pdata:
            if _t.get("exp_trial_type") != "exp_trial": continue
            _imgs = tuple(sorted([_t["image1Path"], _t["image2Path"], _t["image3Path"]]))
            _lrows.append((_pid, _imgs, _t["selected_image"], _t.get("triplet_type", "unknown")))
    _ldf = pd.DataFrame(_lrows, columns=["pid", "triplet", "selected", "triplet_type"])

    levels_nc = {}
    for _tt in ["within_class", "between_class", "class_border", "all"]:
        _sub = _ldf if _tt == "all" else _ldf[_ldf.triplet_type == _tt]
        _sub = _sub.copy()
        _sub["tid"] = _sub.groupby(["pid", "triplet"]).ngroup()
        _tc = _sub.groupby("tid").size()
        _rep = _sub[_sub.tid.isin(_tc[_tc > 1].index)]
        if len(_rep) == 0: levels_nc[_tt] = np.nan; continue
        _cc = _rep.groupby(["tid", "selected"]).size().unstack(fill_value=0)
        _p = _cc.div(_cc.sum(axis=1), axis=0)
        _e = -(_p * np.log(_p.clip(1e-10))).sum(axis=1)
        levels_nc[_tt] = 1 - _e.mean() / np.log(3)

    return {
        ("things", "all"): things_nc,
        ("levels", "within_class"): levels_nc["within_class"],
        ("levels", "between_class"): levels_nc["between_class"],
        ("levels", "class_border"): levels_nc["class_border"],
        ("levels", "all"): levels_nc["all"],
    }


def get_summary_data(experiments, backbones):
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
            for bb in backbones:
                path = Path(f"{dir_path}/{exp}_{bb}_trials.h5")
                if not path.exists(): continue
                df = pd.read_hdf(path, "trials")
                base_r2 = 1 - df.base_nll.values / null_nll
                meta_r2 = 1 - df.meta_nll.values / null_nll

                def _summarise(b, m):
                    n = len(b)
                    return dict(n=n, base_r2_mean=b.mean(), base_r2_sem=b.std()/np.sqrt(n),
                                meta_r2_mean=m.mean(), meta_r2_sem=m.std()/np.sqrt(n))

                if "triplet_type" in df.columns:
                    for tt, idx in df.groupby("triplet_type").groups.items():
                        idx = idx.values if hasattr(idx, 'values') else np.array(idx)
                        rows.append(dict(dataset=ds, experiment=exp, backbone=backbone_map[bb], subset=tt,
                                         **_summarise(base_r2[idx], meta_r2[idx])))
                rows.append(dict(dataset=ds, experiment=exp, backbone=backbone_map[bb], subset="all",
                                 **_summarise(base_r2, meta_r2)))
    return pd.DataFrame(rows)


def main():
    experiments = ["main", "8heads", "32heads", "mlp", "wd", "2layers", "lstm"]
    exp_labels = {"main": "Main", "8heads": "8 Attention Heads", "32heads": "32 Attention Heads",
                  "mlp": "Transformer With MLP", "wd": "Training With Weight Decay", "2layers": "2 Layer Transformer", "lstm" : "LSTM"}
    backbones = ["siglip2", "mae", "dinov3", "clip"]
    backbone_labels = {'clip': 'CLIP', 'dinov3': 'DINOv3', 'mae': 'MAE', 'siglip2': 'SigLIP2'}

    colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    exp_colors = [colors[i] for i in [1, 4, 5, 6, 7, -1, -2]]

    noise_ceilings = get_noise_ceilings()
    summary = get_summary_data(experiments, backbones)

    evals = [
        ("things",           "all",            "THINGS"),
        ("levels",           "within_class",   "Levels Within Class"),
        ("levels",           "between_class",  "Levels Between Class"),
        ("levels",           "class_border",   "Levels Class Border"),
        ("categorylearning", "all",            "Category Learning"),
        ("rewardlearning",   "all",            "Reward Learning"),
    ]

    n_exp = len(experiments)
    n_bb = len(backbones)
    gap = 1.0
    group_width = n_exp
    positions = []
    for i in range(n_bb):
        start = i * (group_width + gap)
        positions.append(np.arange(n_exp) + start)
    w = 0.8

    fig, axes = plt.subplots(len(evals), 1, figsize=(10, 14), sharex=True)

    for row, (ds, sub, title) in enumerate(evals):
        ax = axes[row]
        nc = noise_ceilings.get((ds, sub))
        nc = nc if (nc is not None and not np.isnan(nc)) else None

        for i, bb in enumerate(backbones):
            bb_label = backbone_labels[bb]
            base_m = None
            for j, exp in enumerate(experiments):
                mask = (summary.dataset == ds) & (summary.subset == sub) & \
                       (summary.experiment == exp) & (summary.backbone == bb_label)
                match = summary[mask]
                if len(match) == 0: m, s = np.nan, 0
                else:
                    r = match.iloc[0]
                    m, s = r.meta_r2_mean, r.meta_r2_sem
                    if exp == "main": base_m = r.base_r2_mean / nc if nc else r.base_r2_mean
                    if nc: m, s = m/nc, s/nc

                ax.bar(positions[i][j], m, w, yerr=s, capsize=3,
                       color=exp_colors[j], edgecolor="black", linewidth=1,
                       label=exp_labels[exp] if row == 0 and i == 0 else None, alpha=0.8)

            if base_m is not None:
                x0, x1 = positions[i][0] - w/2, positions[i][-1] + w/2
                ax.hlines(base_m, x0, x1, colors="black", linestyles="dashed", lw=1.5,
                          label="Base" if row == 0 and i == 0 else None)

        ax.set_ylabel("McFadden's $R^2$")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # x-axis: backbone labels at group centers
    group_centers = [pos.mean() for pos in positions]
    axes[-1].set_xticks(group_centers)
    axes[-1].set_xticklabels([backbone_labels[bb] for bb in backbones])
    leg = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                     bbox_to_anchor=(0.5, .97), ncol=4, frameon=False)
    for t in leg.get_texts(): t.set_ha("center"); t.set_ma("center")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/architecture.png", dpi=300)
    fig.savefig("figures/architecture.pdf")
    fig.savefig("figures/architecture.svg")
    print("Saved to figures/architecture.{png,pdf,svg}")


if __name__ == "__main__":
    main()
