import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastcore.script import call_parse
from nnsight import NNsight
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from tqdm.auto import tqdm

from metalign.data import Things, load_backbone
from metalign.model import Transformer, TwoLinear
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)

@call_parse
def main(
    experiment_name: str, # One of main, raw, midsae
    backbone_name: str, # One of clip, siglip2, dinov3
    par_no: int, # Which participant to evaluate,
    threshold: float = 5., # Min ceiling for a voxel to be considered
    force: bool = False # If True, will overwrite existing eval files
):
    "run brain ~ features  for `experiment_name` and `backbone_name` on participant `par_no`."

    eval_path = Path("data/evals/brain")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{par_no}_{experiment_name}_{backbone_name}"
    eval_file = eval_path / f"{file_name}.npy"

    if eval_file.exists() and not force:
        print(f"Eval file {eval_file} already exists. Use --force to overwrite.")
        return
    
    p_chck = Path("data/checkpoints")
    p_reps = Path("data/backbone_reps")
    
    best_models = json.load((p_chck / "best_models.json").open())
    backbone_dict = json.load((p_reps / "backbones.json").open())
    
    ckpt_path = best_models[f"[{experiment_name.upper()}]"][backbone_name]
    things_reps_path = p_reps / f"things_{backbone_dict[backbone_name]}.h5"

    ckpt = torch.load(ckpt_path,  weights_only=False)
    config, state_dict = ckpt['config'], fix_state_dict(ckpt['state_dict'])

    reps = load_backbone(things_reps_path)
    
    if experiment_name.lower() == 'notmeta':
        model = TwoLinear(c=config)
        model.load_state_dict(state_dict)
        model.eval()
        meta_reps = model.embed(torch.from_numpy(reps))
    else:
        model = Transformer(c=config)
        model.load_state_dict(state_dict)
        model.eval()
        model = NNsight(model)

        with model.trace(torch.from_numpy(reps).unsqueeze(1)): 
            meta_reps = model.embed.output.squeeze().save()

    
    meta_reps = meta_reps.numpy()

    p_brain = Path("data/external/brain_data")
    brain = pd.read_hdf(p_brain / f"sub-0{par_no}_ResponseData.h5")
    brain = brain.drop("voxel_id", axis=1).T.to_numpy()
    
    vox_df = pd.read_csv(p_brain / f"sub-0{par_no}_VoxelMetadata.csv")
    good_vox_df = vox_df[(vox_df.nc_singletrial > threshold)]
    
    voxels = good_vox_df.voxel_id.to_list() 
    noise_ceilings = good_vox_df.nc_singletrial.to_numpy() / 100
    Y = brain[:, voxels]

    stim_df = pd.read_csv(p_brain / f"sub-0{par_no}_StimulusMetadata.csv")
    groups = stim_df.session.to_list()
    
    all_images = [str(x).split("THINGS/")[-1] for x in Things().images]
    img2idx = {img: i for i, img in enumerate(all_images)} 
    
    shown_stim_fnames = [f"{c}/{s}" for c,s in zip(stim_df.concept, stim_df.stimulus)]
    stim_idxs = [img2idx[s] for s in shown_stim_fnames]
    
    X = reps[stim_idxs]
    X_meta = meta_reps[stim_idxs]

    logo = LeaveOneGroupOut()
    r2_base, r2_meta = [], []
    alphas = np.logspace(-6, 3, 100)
    
    for _, (train_idx, test_idx) in tqdm(enumerate(logo.split(X, Y, groups)), total=len(set(groups))):
        
        X_train, X_test = X[train_idx], X[test_idx]
        X_meta_train, X_meta_test = X_meta[train_idx], X_meta[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
        ridge.fit(X_train, Y_train)
        Y_pred = ridge.predict(X_test)
        r2_scores = r2_score(Y_test, Y_pred, multioutput="raw_values") / noise_ceilings
        r2_base.append(r2_scores)
        
        ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
        ridge.fit(X_meta_train, Y_train)
        Y_pred = ridge.predict(X_meta_test)
        r2_scores = r2_score(Y_test, Y_pred, multioutput="raw_values") / noise_ceilings
        r2_meta.append(r2_scores)


    results = {
        'r2_base': np.stack(r2_base),
        'r2_meta': np.stack(r2_meta),
        'noise_ceilings': noise_ceilings,
        'voxel_idxs': voxels
    }
    np.save(eval_file, results)