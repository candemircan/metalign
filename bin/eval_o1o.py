import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastcore.script import call_parse
from tqdm import tqdm

from metalign.cognitive_model import softmax_cv_trial_metrics
from metalign.data import load_backbone, prepare_levels, prepare_things_spose
from metalign.model import get_model
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)
warnings.filterwarnings('ignore')

def _load_model_reps(ckpt_path, backbone_reps):
    "Load model and extract representations via model.encode()"
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = get_model(ckpt['config'])
    model.load_state_dict(fix_state_dict(ckpt['state_dict']))
    model.eval()
    x = backbone_reps if isinstance(backbone_reps, torch.Tensor) else torch.from_numpy(backbone_reps)
    return model.encode(x)

def _get_logits_things(reps, X, batch_size=2048):
    "Calculate triplet cosine similarities from representations (THINGS format with indices array)"
    all_sims = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        batch_reps = reps[X_batch]
        i_reps, j_reps, k_reps = batch_reps[:, 0], batch_reps[:, 1], batch_reps[:, 2]
        sim_ij = torch.nn.functional.cosine_similarity(i_reps, j_reps, dim=1)
        sim_ik = torch.nn.functional.cosine_similarity(i_reps, k_reps, dim=1)
        sim_jk = torch.nn.functional.cosine_similarity(j_reps, k_reps, dim=1)
        sims = torch.stack([sim_jk, sim_ik, sim_ij], dim=1)
        all_sims.append(sims.cpu())
    return torch.cat(all_sims, dim=0)

def _get_logits_levels(reps, trials, batch_size=2048):
    "Calculate triplet cosine similarities from representations (LEVELS format with trial dicts)"
    all_sims = []
    for i in range(0, len(trials), batch_size):
        batch_trials = trials[i:i+batch_size]
        for trial in batch_trials:
            img_indices = trial['images']
            trial_reps = reps[img_indices]
            i_reps, j_reps, k_reps = trial_reps[0:1], trial_reps[1:2], trial_reps[2:3]
            sim_ij = torch.nn.functional.cosine_similarity(i_reps, j_reps, dim=1)
            sim_ik = torch.nn.functional.cosine_similarity(i_reps, k_reps, dim=1)
            sim_jk = torch.nn.functional.cosine_similarity(j_reps, k_reps, dim=1)
            sims = torch.stack([sim_jk, sim_ik, sim_ij], dim=0)
            all_sims.append(sims.cpu())
    return torch.stack(all_sims, dim=0).squeeze(-1)

@call_parse
def main(
    dataset: str, # 'things' or 'levels'
    experiment_name: str, # One of main, raw, midsae, static_linear, 8heads, 32heads, mlp, wd, lstm, 2layers
    backbone_name: str, # clip, siglip2, dinov3, or mae
    batch_size: int = 2048,
    force: bool = False
):
    """Evaluate odd-one-out on THINGS or LEVELS using per-participant softmax temperature model with CV"""

    eval_path = Path(f"data/evals/{dataset}o1o")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{experiment_name}_{backbone_name}"
    output_file = eval_path / f"{file_name}_trials.h5"

    if output_file.exists() and not force:
        print(f"Output for {file_name} already exists, use --force to overwrite")
        return

    best_models = json.load(open(Path("data/checkpoints") / "best_models.json"))
    backbone_dict = json.load(open(Path("data/backbone_reps") / "backbones.json"))

    # Load data based on dataset
    if dataset == 'things':
        reps_file = f"data/backbone_reps/things_{backbone_dict[backbone_name]}.h5"
        backbone_reps, _ = prepare_things_spose(load_backbone(reps_file))
        df = pd.read_table("data/external/THINGS_triplets.csv")
        X_idx = df[["image1", "image2", "image3"]].values - 1
        y = df["choice"].values - 1
        participant_ids = df['subject_id'].values
        triplet_types = None
        _get_logits = _get_logits_things
    elif dataset == 'levels':
        reps_file = f"data/backbone_reps/levels_{backbone_dict[backbone_name]}.h5"
        backbone_reps, trials = prepare_levels(load_backbone(reps_file))
        y = np.array([trial['selected'] for trial in trials])
        participant_ids = np.array([trial['participant_id'] for trial in trials])
        triplet_types = np.array([trial.get('triplet_type', 'unknown') for trial in trials])
        X_idx = trials
        _get_logits = _get_logits_levels
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load experiment model reps
    exp_reps = _load_model_reps(best_models[f"[{experiment_name}]"][backbone_name], backbone_reps)

    # Baseline: backbone always, plus main-model for ablations
    base_reps = backbone_reps
    main_reps = None if experiment_name == "main" else _load_model_reps(best_models["[main]"][backbone_name], backbone_reps)

    # Compute similarities
    print("Computing similarities...")
    base_logits = _get_logits(base_reps, X_idx, batch_size=batch_size).numpy()
    main_logits = None if main_reps is None else _get_logits(main_reps, X_idx, batch_size=batch_size).numpy()
    exp_logits = _get_logits(exp_reps, X_idx, batch_size=batch_size).numpy()

    # Per-participant CV
    print("Running per-participant CV...")
    participants = np.unique(participant_ids)
    trial_rows = []

    for pid in tqdm(participants):
        pmask = participant_ids == pid
        y_p, X_base, X_meta = y[pmask], base_logits[pmask], exp_logits[pmask]

        base_nlls, base_correct = softmax_cv_trial_metrics(X_base, y_p)
        meta_nlls, meta_correct = softmax_cv_trial_metrics(X_meta, y_p)

        # Main model metrics (if ablation)
        main_nlls_p, main_correct_p = (None, None) if main_logits is None else softmax_cv_trial_metrics(main_logits[pmask], y_p)

        if base_nlls is None or meta_nlls is None: continue

        for i in range(len(y_p)):
            if np.isnan(base_nlls[i]) or np.isnan(meta_nlls[i]): continue
            row = {'participant_id': pid, 'base_nll': base_nlls[i], 'base_correct': base_correct[i],
                   'meta_nll': meta_nlls[i], 'meta_correct': meta_correct[i]}
            if main_nlls_p is not None and not np.isnan(main_nlls_p[i]):
                row['main_nll'] = main_nlls_p[i]
                row['main_correct'] = main_correct_p[i]
            if triplet_types is not None:
                row['triplet_type'] = triplet_types[pmask][i]
            trial_rows.append(row)

    # Save trial-level results
    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_hdf(output_file, key='trials', mode='w')

    # Save zeroshot accuracy
    zeroshot = {'experiment_name': experiment_name, 'backbone_name': backbone_name,
                'base_acc': float((base_logits.argmax(axis=1) == y).mean()),
                'metalign_acc': float((exp_logits.argmax(axis=1) == y).mean())}
    if main_logits is not None:
        zeroshot['main_acc'] = float((main_logits.argmax(axis=1) == y).mean())
    if triplet_types is not None:
        for tt in np.unique(triplet_types):
            tt_mask = triplet_types == tt
            zeroshot[tt] = {'base_acc': float((base_logits[tt_mask].argmax(axis=1) == y[tt_mask]).mean()),
                           'metalign_acc': float((exp_logits[tt_mask].argmax(axis=1) == y[tt_mask]).mean())}
            if main_logits is not None:
                zeroshot[tt]['main_acc'] = float((main_logits[tt_mask].argmax(axis=1) == y[tt_mask]).mean())
    with open(eval_path / f"{file_name}_zeroshot.json", "w") as f:
        json.dump(zeroshot, f, indent=2)
    print(f"Saved {len(trial_df)} trials to {output_file}")
