import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastcore.script import call_parse
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut

from metalign.data import Things, load_backbone
from metalign.model import get_model
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)

def _load_model_reps(ckpt_path, backbone_reps):
    "Load model and extract representations via model.encode()"
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = get_model(ckpt['config'])
    model.load_state_dict(fix_state_dict(ckpt['state_dict']))
    model.eval()
    x = backbone_reps if isinstance(backbone_reps, torch.Tensor) else torch.from_numpy(backbone_reps)
    return model.encode(x).numpy()

def compute_metrics(Y, P, NC, rois):
    "Compute voxel-wise and ROI-level R2 and correlation"
    r2 = r2_score(Y, P, multioutput="raw_values") / NC
    roi_res = {n: {'r2': r2[m].mean()} for n, m in rois.items()}
    return r2, roi_res

def _run_fold(tr_idx, te_idx, X, X_m, X_ma, Y, NC, rois, alphas, apt=True):
    "Run a single LOGO fold: fit Ridge on train, predict on test, return metrics"
    Y_tr, Y_te = Y[tr_idx], Y[te_idx]

    P_b = RidgeCV(alphas=alphas, alpha_per_target=apt).fit(X[tr_idx], Y_tr).predict(X[te_idx])
    r2_b, rm_b = compute_metrics(Y_te, P_b, NC, rois)

    P_m = RidgeCV(alphas=alphas, alpha_per_target=apt).fit(X_m[tr_idx], Y_tr).predict(X_m[te_idx])
    r2_m, rm_m = compute_metrics(Y_te, P_m, NC, rois)

    rm_ma = None
    if X_ma is not None:
        P_ma = RidgeCV(alphas=alphas, alpha_per_target=apt).fit(X_ma[tr_idx], Y_tr).predict(X_ma[te_idx])
        _, rm_ma = compute_metrics(Y_te, P_ma, NC, rois)

    return r2_b, r2_m, rm_b, rm_m, rm_ma

@call_parse
def main(
    experiment_name: str, # One of main, raw, midsae, static_linear
    backbone_name: str, # One of clip, siglip2, dinov3, mae
    par_no: int, # Which participant to evaluate
    threshold: float = 5., # Min ceiling for a voxel to be considered
    force: bool = False, # If True, will overwrite existing eval files
    n_jobs: int = 1, # Number of parallel jobs for LOGO folds
):
    "run brain ~ features encoding analysis with leave-one-session-out CV."
    eval_path = Path("data/evals/brain")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{par_no}_{experiment_name}_{backbone_name}"
    eval_file = eval_path / f"{file_name}.npy"

    if eval_file.exists() and not force:
        print(f"Eval file {eval_file} already exists. Use --force to overwrite.")
        return

    p_chck, p_reps = Path("data/checkpoints"), Path("data/backbone_reps")
    best_models = json.load((p_chck / "best_models.json").open())
    backbone_dict = json.load((p_reps / "backbones.json").open())

    things_reps_path = p_reps / f"things_{backbone_dict[backbone_name]}.h5"
    reps = load_backbone(things_reps_path)

    meta_reps = _load_model_reps(best_models[f"[{experiment_name}]"][backbone_name], reps)
    main_reps = None if experiment_name == "main" else _load_model_reps(best_models["[main]"][backbone_name], reps)

    p_brain = Path("data/external/brain_data")
    brain = pd.read_hdf(p_brain / f"sub-0{par_no}_ResponseData.h5")
    brain = brain.drop("voxel_id", axis=1).T.to_numpy()

    vox_df = pd.read_csv(p_brain / f"sub-0{par_no}_VoxelMetadata.csv")
    good_vox_df = vox_df[(vox_df.nc_singletrial > threshold)]
    voxels, NC = good_vox_df.voxel_id.to_list(), good_vox_df.nc_singletrial.to_numpy() / 100
    Y = brain[:, voxels]

    roi_cols = [c for c in vox_df.columns if c not in ['voxel_id', 'subject_id', 'voxel_x', 'voxel_y', 'voxel_z', 'nc_singletrial', 'nc_testset', 'splithalf_uncorrected', 'splithalf_corrected', 'prf-eccentricity', 'prf-polarangle', 'prf-rsquared', 'prf-size']]
    rois = {c: good_vox_df[c].to_numpy().astype(bool) for c in roi_cols}
    rois = {k: v for k, v in rois.items() if v.any()}

    stim_df = pd.read_csv(p_brain / f"sub-0{par_no}_StimulusMetadata.csv")
    groups = stim_df.session.to_list()
    img2idx = {str(x).split("THINGS/")[-1]: i for i, x in enumerate(Things().images)}
    stim_idxs = [img2idx[f"{c}/{s}"] for c,s in zip(stim_df.concept, stim_df.stimulus)]

    X, X_m = reps[stim_idxs], meta_reps[stim_idxs]
    X_ma = None if main_reps is None else main_reps[stim_idxs]

    logo = LeaveOneGroupOut()
    alphas = np.logspace(-6, 3, 100)
    splits = list(logo.split(X, Y, groups))

    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_fold)(tr_idx, te_idx, X, X_m, X_ma, Y, NC, rois, alphas)
        for tr_idx, te_idx in splits
    )
    print(f"All {len(splits)} folds complete.")
    roi_metrics = {n: {'r2_b': [], 'r2_m': [], 'r2_ma': []} for n in rois}
    vox_r2_base, vox_r2_meta = [], []

    for r2_b, r2_m, rm_b, rm_m, rm_ma in fold_results:
        vox_r2_base.append(r2_b)
        vox_r2_meta.append(r2_m)
        for n in rois:
            roi_metrics[n]['r2_b'].append(rm_b[n]['r2'])
            roi_metrics[n]['r2_m'].append(rm_m[n]['r2'])
            if rm_ma: roi_metrics[n]['r2_ma'].append(rm_ma[n]['r2'])

    roi_final = {}
    for n in rois:
        r2_b = np.array(roi_metrics[n]['r2_b'])
        r2_m = np.array(roi_metrics[n]['r2_m'])
        d = {
            'r2_base': r2_b.mean(),
            'r2_meta': r2_m.mean(),
            'r2_base_folds': r2_b,
            'r2_meta_folds': r2_m,
            'diff_mean': (r2_m - r2_b).mean(),
        }
        if roi_metrics[n]['r2_ma']:
            r2_ma = np.array(roi_metrics[n]['r2_ma'])
            d['r2_main'] = r2_ma.mean()
            d['r2_main_folds'] = r2_ma
        roi_final[n] = d

    res = {
        'roi_results': roi_final, 'voxel_idxs': voxels, 'noise_ceilings': NC,
        'r2_base': np.stack(vox_r2_base), 'r2_meta': np.stack(vox_r2_meta),
    }
    np.save(eval_file, res)
