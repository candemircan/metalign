import json
from pathlib import Path

import pandas as pd
import torch
from fastcore.script import call_parse
from nnsight import NNsight
from torch.nn import functional as F
from tqdm import tqdm

from metalign.data import load_backbone, prepare_levels
from metalign.model import Transformer, TwoLinear
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)

def _get_logits(reps, trials, batch_size=2048):
    "Calculate triplet similarities (logits) from representations and trials"
    all_sims = []
    
    for i in tqdm(range(0, len(trials), batch_size)):
        batch_trials = trials[i:i+batch_size]
        
        for trial in batch_trials:
            img_indices = trial['images']
            trial_reps = reps[img_indices]
            
            i_reps, j_reps, k_reps = trial_reps[0:1], trial_reps[1:2], trial_reps[2:3]
            
            sim_ij = F.cosine_similarity(i_reps, j_reps)
            sim_ik = F.cosine_similarity(i_reps, k_reps)
            sim_jk = F.cosine_similarity(j_reps, k_reps)
            
            # Stack so that index 0 -> 0 is odd (j,k similar), 1 -> 1 odd (i,k similar), 2 -> 2 odd (i,j similar)
            sims = torch.stack([sim_jk, sim_ik, sim_ij], dim=0)
            all_sims.append(sims.cpu())
    
    return torch.stack(all_sims, dim=0).squeeze()

@call_parse
def main(
    experiment_name: str, # has to be one of main, raw, midsae
    backbone_name: str, # has to be one of  clip, siglip2, dinov3
    batch_size: int = 2048, # batch size for evaluation
    force: bool = False # if True, will overwrite existing eval file
):
    """
    Evaluate 0-shot accuracy on LEVELS odd-one-out task of the base model and the given checkpoint
    """
    eval_path = Path("data/evals/levelso1o")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{experiment_name}_{backbone_name}"
    stats_file = eval_path / f"{file_name}_stats.csv"

    if stats_file.exists() and not force:
        print(f"Stats file for {file_name} already exists, use --force to overwrite")
        return

    best_models = json.load(open(Path("data/checkpoints") / "best_models.json"))
    backbone_dict = json.load(open(Path("data/backbone_reps") / "backbones.json"))
    ckpt = best_models[f"[{experiment_name.upper()}]"][backbone_name]
    levels_reps = f"data/backbone_reps/levels_{backbone_dict[backbone_name]}.h5"
    
    backbone_reps, trials = prepare_levels(load_backbone(levels_reps))

    ckpt = torch.load(ckpt, weights_only=False)
    config, state_dict = ckpt['config'], fix_state_dict(ckpt['state_dict'])
    
    if experiment_name.lower() == 'notmeta':
        model = TwoLinear(c=config)
        model.load_state_dict(state_dict)
        model.eval()
        metalign_reps = model.embed(backbone_reps)
    else:
        model = Transformer(c=config)
        model.load_state_dict(state_dict)
        model.eval()
        model = NNsight(model)
        with model.trace(backbone_reps.unsqueeze(1)): 
            metalign_reps = model.embed.output.squeeze().save()

    # Calculate logits for base and metalign models
    og_logits = _get_logits(backbone_reps, trials, batch_size=batch_size)
    metalign_logits = _get_logits(metalign_reps, trials, batch_size=batch_size)
    
    # Extract ground truth choices
    y = torch.tensor([trial['selected'] for trial in trials], dtype=torch.long)

    # Save stats for mixed effects modeling
    stats_df = pd.DataFrame({
        'participant_id': [trial['participant_id'] for trial in trials],
        'triplet_type': [trial.get('triplet_type', 'unknown') for trial in trials],
        'y': y.numpy(),
        'base_sim_0': og_logits[:, 0].numpy(),
        'base_sim_1': og_logits[:, 1].numpy(),
        'base_sim_2': og_logits[:, 2].numpy(),
        'metalign_sim_0': metalign_logits[:, 0].numpy(),
        'metalign_sim_1': metalign_logits[:, 1].numpy(),
        'metalign_sim_2': metalign_logits[:, 2].numpy()
    })
    
    stats_df.to_csv(stats_file, index=False)
