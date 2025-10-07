import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastcore.script import call_parse
from nnsight import NNsight
from tqdm import tqdm

from metalign.data import Things, load_backbone
from metalign.model import Transformer, TwoLinear
from metalign.utils import calculate_cka, fix_state_dict

_ = torch.set_grad_enabled(False)

@call_parse
def main(
    experiment_name: str, # has to be one of main, raw, midsae
    backbone_name: str, # has to be one of mae, clip, siglip2, dinov3
    force: bool = False # if True, will overwrite existing eval files
    ):

    eval_path = Path("data/evals/brain")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{experiment_name}_{backbone_name}"
    eval_file = eval_path / f"{file_name}.csv"

    if eval_file.exists() and not force:
        print(f"Eval file {eval_file} already exists. Use --force to overwrite.")
        return
    
    best_models = json.load(open(Path("data/checkpoints") / "best_models.json"))
    backbone_dict = json.load(open(Path("data/backbone_reps") / "backbones.json"))
    ckpt = best_models[f"[{experiment_name.upper()}]"][backbone_name]
    things_reps = f"data/backbone_reps/things_{backbone_dict[backbone_name]}.h5"

    ckpt = torch.load(ckpt,  weights_only=False)
    config, state_dict = ckpt['config'], fix_state_dict(ckpt['state_dict'])
    
    human_data = pd.read_csv("data/external/category_learning.csv")
    backbone_reps = load_backbone(things_reps)
    
    if experiment_name.lower() == 'notmeta':
        model = TwoLinear(c=config)
        model.load_state_dict(state_dict)
        model.eval()
        # for TwoLinear, we extract representations from the embed layer
        metalign_reps = model.embed(torch.from_numpy(backbone_reps))
    else:
        model = Transformer(c=config)
        model.load_state_dict(state_dict)
        model.eval()
        model = NNsight(model)
        # for Transformer, we extract representations from the embed layer using nnsight bc there is special processing of inputs
        with model.trace(torch.from_numpy(backbone_reps).unsqueeze(1)): 
            metalign_reps = model.embed.output.squeeze().save()

    # use things dataset ordering to match backbone feature extraction
    things_dataset = Things()
    imgs = [str(img) for img in things_dataset.images]
    
    brain_root = Path("data/external/brain_data")
    subs = brain_root.glob("sub-*ResponseData.h5")
    subs = [str(x).split("-")[-1].split("_")[0] for x in subs]

    results = []

    for sub in tqdm(subs):
        stim_df = pd.read_csv(brain_root/f"sub-{sub}_StimulusMetadata.csv")
        response_df = pd.read_hdf(brain_root/f"sub-{sub}_ResponseData.h5")
        voxel_meta_df = pd.read_csv(brain_root/f"sub-{sub}_VoxelMetadata.csv")


        it_voxels = voxel_meta_df[voxel_meta_df.IT == 1]["voxel_id"].tolist()
        response_df = response_df[response_df.voxel_id.isin(it_voxels)].reset_index(drop=True)
        assert len(it_voxels) == len(response_df)
        del response_df["voxel_id"]
        
        all_responses = response_df.to_numpy().T # obs by voxels now
        
        img_to_idx = {Path(img).name: i for i, img in enumerate(imgs)}
        
        stim_names = stim_df['stimulus'].tolist()
        trial_types = stim_df['trial_type'].tolist()
        trial_ids = stim_df['trial_id'].tolist()
        
        # Debug info
        print(f"Subject {sub}: {len(stim_names)} stimuli, response shape: {all_responses.shape}")
        
        valid_trials = []
        model_indices = []
        for i, stim_name in enumerate(stim_names):
            if stim_name in img_to_idx:
                valid_trials.append(i)
                model_indices.append(img_to_idx[stim_name])
        
        print(f"  Valid trials: {len(valid_trials)}/{len(stim_names)}")
        print(f"  Model indices range: {min(model_indices) if model_indices else 'N/A'} to {max(model_indices) if model_indices else 'N/A'}")
        print(f"  Backbone reps shape: {backbone_reps.shape}")
        
        # Check for duplicates in model_indices (this would be bad)
        unique_indices = len(set(model_indices))
        print(f"  Unique model indices: {unique_indices}/{len(model_indices)}")
        
        if unique_indices != len(model_indices):
            print("  WARNING: Duplicate model indices found! This suggests repeated stimuli.")
        
        # filter brain responses to valid trials only (which should be all)
        valid_responses = all_responses[valid_trials]  # n_valid_trials, n_voxels
        valid_stim_names = [stim_names[i] for i in valid_trials]
        valid_trial_types = [trial_types[i] for i in valid_trials]
        
        backbone_reps_aligned = backbone_reps[model_indices]  # n_valid_trials, n_features
        metalign_reps_aligned = metalign_reps[model_indices].numpy()  # same
        
        brain_df = pd.DataFrame({
            'stimulus': valid_stim_names,
            'trial_type': valid_trial_types,
            'trial_idx': range(len(valid_stim_names))
        })
        
        train_mask = brain_df['trial_type'] == 'train'
        test_mask = brain_df['trial_type'] == 'test'
        
        # for train data, use as is bc there are no repeats
        train_stimuli = brain_df[train_mask]['stimulus'].tolist()
        train_brain = valid_responses[train_mask]
        train_backbone = backbone_reps_aligned[train_mask]
        train_metalign = metalign_reps_aligned[train_mask]
        
        # for test data, average across repetitions
        test_df = brain_df[test_mask].copy()
        test_stimuli = test_df['stimulus'].unique()
        
        test_brain_avg = []
        test_backbone_avg = []
        test_metalign_avg = []
        test_brain_individual = []
        
        for stim in test_stimuli:
            stim_mask = test_df['stimulus'] == stim
            stim_indices = test_df[stim_mask]['trial_idx'].tolist()
            
            # get all reps
            stim_brain_reps = valid_responses[stim_indices]
            stim_backbone_reps = backbone_reps_aligned[stim_indices]
            stim_metalign_reps = metalign_reps_aligned[stim_indices]
            
            # average across reps
            test_brain_avg.append(stim_brain_reps.mean(axis=0))
            test_backbone_avg.append(stim_backbone_reps.mean(axis=0))
            test_metalign_avg.append(stim_metalign_reps.mean(axis=0))
            
            # keep individuals reps for noise ceiling
            test_brain_individual.append(stim_brain_reps)
        
        test_brain_avg = np.array(test_brain_avg)
        test_backbone_avg = np.array(test_backbone_avg)
        test_metalign_avg = np.array(test_metalign_avg)
        
        all_stimuli = train_stimuli + test_stimuli.tolist()
        combined_brain = np.vstack([train_brain, test_brain_avg])
        combined_backbone = np.vstack([train_backbone, test_backbone_avg])
        combined_metalign = np.vstack([train_metalign, test_metalign_avg])
        
        # calculate noise ceiling using split-half reliability across all test stimuli
        test_half1 = []
        test_half2 = []
        for stim_reps in test_brain_individual:
            n_reps = len(stim_reps)
            half1 = stim_reps[:n_reps//2].mean(axis=0)
            half2 = stim_reps[n_reps//2:].mean(axis=0)
            test_half1.append(half1)
            test_half2.append(half2)
        
        test_half1 = np.array(test_half1)  # test stimuli by voxels
        test_half2 = np.array(test_half2)  # test stimuli by voxels
        # our cka implementation is batched
        half1_tensor = torch.from_numpy(test_half1).unsqueeze(0) 
        half2_tensor = torch.from_numpy(test_half2).unsqueeze(0) 
        noise_ceiling = calculate_cka(half1_tensor, half2_tensor)

        brain_tensor = torch.from_numpy(combined_brain).unsqueeze(0) 
        backbone_tensor = torch.from_numpy(combined_backbone).unsqueeze(0) 
        metalign_tensor = torch.from_numpy(combined_metalign).unsqueeze(0)  
        
        backbone_brain_cka = calculate_cka(backbone_tensor, brain_tensor)
        metalign_brain_cka = calculate_cka(metalign_tensor, brain_tensor)

        print(f"subject {sub}: backbone CKA {backbone_brain_cka.item():.4f}, metalign CKA {metalign_brain_cka.item():.4f}, noise ceiling {noise_ceiling.item():.4f}")
        
        result = {
            'subject': sub,
            'backbone_brain_cka': backbone_brain_cka,
            'metalign_brain_cka': metalign_brain_cka,
            'noise_ceiling': noise_ceiling,
            'n_voxels': combined_brain.shape[1],
            'n_stimuli': combined_brain.shape[0],
            'n_train_stimuli': len(train_stimuli),
            'n_test_stimuli': len(test_stimuli)
        }
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(eval_file, index=False)
        
