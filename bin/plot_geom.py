"""Generate Figure 5: t-SNE visualizations and class separation analysis."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Patch
from scipy.spatial import procrustes
from sklearn.manifold import TSNE
from tqdm import tqdm

from metalign.data import Things, load_backbone, prepare_things_spose
from metalign.model import Transformer
from metalign.utils import calc_cka, calc_class_separation, fix_state_dict

MODEL_MAP = {"clip": "CLIP", "dinov3": "DINOv3", "mae": "MAE", "siglip2": "SigLIP2"}
ORDER = ["SigLIP2", "MAE", "DINOv3", "CLIP"]

BACKBONES = {
    "SigLIP2": "things_vit_base_patch16_siglip_256.v2_webli.h5",
    "DINOv3": "things_dinov3-vitb16-pretrain-lvd1689m.h5",
    "MAE": "things_webssl_mae300m_full2b_224.h5",
    "CLIP": "things_CLIP-ViT-B-32-DataComp.XL-s13B-b90K_sae-top_k-64-cls_only-layer_11-hook_resid_post_raw.h5"
}

CATEGORY_COLUMNS = ["animal", "clothing", "food", "furniture", "musical instrument",
                    "sports equipment", "vehicle", "body part", "plant"]

N_BOOTSTRAP = 100

_ = torch.set_grad_enabled(False)

def bootstrap_class_sep(X, y, n=N_BOOTSTRAP, desc="bootstrap class_sep"):
    "Bootstrap class separation by resampling observations"
    N = X.shape[0]
    results = []
    for _ in tqdm(range(n), desc=desc):
        idx = torch.randint(0, N, (N,))
        results.append(calc_class_separation(X[idx], y[idx]))
    return results

def bootstrap_cka(X, spose, n=N_BOOTSTRAP):
    "Bootstrap CKA by resampling observations — batched"
    N = X.shape[1]
    idx = torch.randint(0, N, (n, N))
    X_boot = X[:, idx].squeeze(0)       # (n, N, features)
    spose_boot = spose[:, idx].squeeze(0)
    return calc_cka(X_boot, spose_boot).tolist()

def load_metalign_reps(backbone_path, checkpoint_path):
    "Load backbone reps and transform through MetAlign model"
    reps = load_backbone(backbone_path)
    category_reps, spose = prepare_things_spose(reps)
    reps = torch.from_numpy(reps)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    m = Transformer(c=checkpoint['config'])
    m.load_state_dict(fix_state_dict(checkpoint['state_dict']))
    m.eval()

    return reps, m.encode(reps), category_reps, m.encode(category_reps), spose

def get_category_colors(md):
    "Assign colors based on category membership"
    colors = []
    for _, row in md.iterrows():
        cats = [col for col in CATEGORY_COLUMNS if row[col] == 1]
        colors.append(cats[0] if cats else 'uncategorised')
    return colors

def compute_metrics():
    "Compute class separation and CKA metrics across backbones with bootstrapped error bars"
    md = pd.read_table("data/external/THINGS_metadata.tsv")
    md_array = torch.from_numpy(md.to_numpy())

    imgs = [str(x).split("/")[-2] for x in Things().images]
    label_map = {label: idx for idx, label in enumerate(sorted(set(imgs)))}
    labels = torch.tensor([label_map[label] for label in imgs])

    class_sep_fine = {"model": [], "type": [], "r2": []}
    class_sep_coarse = {"model": [], "type": [], "r2": []}
    cka_rows = {"model": [], "type": [], "cka": []}

    for backbone_name, backbone_file in BACKBONES.items():
        backbone_path = f"data/backbone_reps/{backbone_file}"
        ckpt_path = f"data/checkpoints/[main]_{backbone_name.lower()}/model.pt"

        reps, metalign_reps, category_reps, metalign_category_reps, spose = load_metalign_reps(backbone_path, ckpt_path)

        # Bootstrap fine-grained class separation
        fine_base = bootstrap_class_sep(reps, labels, desc=f"{backbone_name} fine base")
        fine_meta = bootstrap_class_sep(metalign_reps, labels, desc=f"{backbone_name} fine meta")
        for b, m in [(fine_base, "base"), (fine_meta, "meta-learned")]:
            class_sep_fine["model"].extend([backbone_name] * N_BOOTSTRAP)
            class_sep_fine["type"].extend([m] * N_BOOTSTRAP)
            class_sep_fine["r2"].extend(b)

        # Bootstrap coarse-grained class separation
        coarse_base = bootstrap_class_sep(category_reps, md_array, desc=f"{backbone_name} coarse base")
        coarse_meta = bootstrap_class_sep(metalign_category_reps, md_array, desc=f"{backbone_name} coarse meta")
        for b, m in [(coarse_base, "base"), (coarse_meta, "meta-learned")]:
            class_sep_coarse["model"].extend([backbone_name] * N_BOOTSTRAP)
            class_sep_coarse["type"].extend([m] * N_BOOTSTRAP)
            class_sep_coarse["r2"].extend(b)

        # Bootstrap CKA between category reps and Hebart et al.
        spose_b = spose.unsqueeze(0)
        cka_base = bootstrap_cka(category_reps.unsqueeze(0), spose_b)
        cka_meta = bootstrap_cka(metalign_category_reps.unsqueeze(0), spose_b)
        for b, m in [(cka_base, "base"), (cka_meta, "meta-learned")]:
            cka_rows["model"].extend([backbone_name] * N_BOOTSTRAP)
            cka_rows["type"].extend([m] * N_BOOTSTRAP)
            cka_rows["cka"].extend(b)

    return pd.DataFrame(class_sep_fine), pd.DataFrame(class_sep_coarse), pd.DataFrame(cka_rows)

def main():
    # Load SigLIP2 for t-SNE visualization
    backbone_path = f"data/backbone_reps/{BACKBONES['SigLIP2']}"
    ckpt_path = "data/checkpoints/[main]_siglip2/model.pt"

    reps = load_backbone(backbone_path)
    category_reps, spose = prepare_things_spose(reps)

    checkpoint = torch.load(ckpt_path, weights_only=False)
    m = Transformer(c=checkpoint['config'])
    m.load_state_dict(fix_state_dict(checkpoint['state_dict']))
    m.eval()

    metalign_reps = m.encode(category_reps).detach().cpu().numpy()
    category_reps = category_reps.numpy()
    spose = spose.numpy()

    md = pd.read_table("data/external/THINGS_metadata.tsv")
    colors = get_category_colors(md)

    # Filter to categorised items
    categorised_mask = np.array([c != 'uncategorised' for c in colors])
    cat_reps = category_reps[categorised_mask]
    cat_metalign = metalign_reps[categorised_mask]
    cat_spose = spose[categorised_mask]
    cat_colors = [c for c, m in zip(colors, categorised_mask) if m]

    unique_colors = sorted(set(cat_colors))
    tab20 = plt.colormaps['tab20'].colors
    color_map = {cat: tab20[i % 20] for i, cat in enumerate(unique_colors)}
    point_colors = [color_map[c] for c in cat_colors]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=1234)
    tsne_reps = tsne.fit_transform(cat_reps)
    tsne_metalign = tsne.fit_transform(cat_metalign)
    tsne_spose = tsne.fit_transform(cat_spose)
    _, aligned_tsne_metalign, _ = procrustes(tsne_reps, tsne_metalign)
    _, aligned_tsne_spose, _ = procrustes(tsne_reps, tsne_spose)

    # Metrics (cached)
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / "geom.npz"
    if cache_file.exists():
        d = np.load(cache_file, allow_pickle=True)
        df_fine = pd.DataFrame({'model': d['fine_model'], 'type': d['fine_type'], 'r2': d['fine_r2']})
        df_coarse = pd.DataFrame({'model': d['coarse_model'], 'type': d['coarse_type'], 'r2': d['coarse_r2']})
        df_cka = pd.DataFrame({'model': d['cka_model'], 'type': d['cka_type'], 'cka': d['cka_cka']})
        for df in [df_fine, df_coarse, df_cka]:
            df["type"] = df["type"].str.lower()
    else:
        df_fine, df_coarse, df_cka = compute_metrics()
        np.savez_compressed(cache_file,
                           fine_model=df_fine['model'].values, fine_type=df_fine['type'].values, fine_r2=df_fine['r2'].values,
                           coarse_model=df_coarse['model'].values, coarse_type=df_coarse['type'].values, coarse_r2=df_coarse['r2'].values,
                           cka_model=df_cka['model'].values, cka_type=df_cka['type'].values, cka_cka=df_cka['cka'].values)

    # Plot: 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw={"hspace": 0.15})

    bar_kw = dict(alpha=0.8, edgecolor="black", linewidth=1, saturation=1, errorbar="se")

    # Row 1: t-SNE plots
    for ax, data, title in zip(axes[0], [tsne_reps, aligned_tsne_metalign, aligned_tsne_spose],
                                ["base", "meta-learned", "human similarity embedding (Hebart et al.)"]):
        ax.scatter(data[:, 0], data[:, 1], c=point_colors, alpha=0.8, edgecolor="black", linewidth=0.1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Row 2: bar plots
    sns.barplot(data=df_cka, x="model", y="cka", hue="type", ax=axes[1, 0], order=ORDER, **bar_kw)
    axes[1, 0].set_ylabel("CKA with human similarity embedding")
    axes[1, 0].set_xlabel("")
    axes[1, 0].get_legend().remove()

    sns.barplot(data=df_coarse, x="model", y="r2", hue="type", ax=axes[1, 1], order=ORDER, **bar_kw)
    axes[1, 1].set_ylabel("Coarse-Grained Class Separation")
    axes[1, 1].set_xlabel("")
    axes[1, 1].get_legend().remove()

    sns.barplot(data=df_fine, x="model", y="r2", hue="type", ax=axes[1, 2], order=ORDER, **bar_kw)
    axes[1, 2].set_ylabel("Fine-Grained Class Separation")
    axes[1, 2].set_xlabel("")

    # Bar plot legend (from last barplot)
    handles, labels = axes[1, 2].get_legend_handles_labels()
    axes[1, 2].get_legend().remove()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.53, 0.52), ncol=2, frameon=False)

    # Panel labels
    panel_labels = [['A', 'B', 'C'], ['D', 'E', 'F']]
    for i in range(2):
        for j in range(3):
            axes[i, j].text(-0.1, 1.1, panel_labels[i][j], transform=axes[i, j].transAxes,
                            fontsize=18, fontweight='bold', va='top', ha='right')

    plt.tight_layout(h_pad=0.2)

    # t-SNE legend above first row, after tight_layout so positions are final
    legend_elements = [
        Patch(facecolor=color_map[cat], edgecolor="black", linewidth=0.1, alpha=0.8, label=cat)
        for cat in unique_colors
    ]
    x_center = (axes[0, 0].get_position().x0 + axes[0, 2].get_position().x1) / 2
    tsne_top = max(axes[0, j].get_position().y1 for j in range(3)) + 0.04
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(x_center, tsne_top),
               ncol=len(unique_colors), title="", frameon=False)

    Path("figures").mkdir(exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(f"figures/geom.{ext}", bbox_inches='tight')

if __name__ == "__main__":
    main()
