import json
from pathlib import Path

import torch
from fastcore.script import call_parse
from nnsight import NNsight
from torch.nn import functional as F
from tqdm import tqdm

from metalign.data import load_backbone, prepare_levels
from metalign.model import Transformer
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)

def _calculate_accuracy(reps, trials, batch_size=2048):
    "Calculate odd-one-out accuracy from representations and trials, both overall and by trial type"
    num_correct, num_total = 0, 0
    by_type = {}
            
    for i in tqdm(range(0, len(trials), batch_size)):
        batch_trials = trials[i:i+batch_size]
        
        for trial in batch_trials:
            img_indices = trial['images']
            selected_idx = trial['selected']
            trial_type = trial['triplet_type']
            
            trial_reps = reps[img_indices]
            
            i_reps, j_reps, k_reps = trial_reps[0:1], trial_reps[1:2], trial_reps[2:3]
            
            sim_ij = F.cosine_similarity(i_reps, j_reps)
            sim_ik = F.cosine_similarity(i_reps, k_reps)
            sim_jk = F.cosine_similarity(j_reps, k_reps)
            

            sims = torch.stack([sim_jk, sim_ik, sim_ij])
            
            predicted_odd = torch.argmax(sims).item()
            
            is_correct = predicted_odd == selected_idx
            
            if is_correct: num_correct += 1
            num_total += 1
            
            if trial_type not in by_type: by_type[trial_type] = {'correct': 0, 'total': 0}
            if is_correct: by_type[trial_type]['correct'] += 1
            by_type[trial_type]['total'] += 1
    
    overall_acc = num_correct / num_total
    type_accs = {t: counts['correct'] / counts['total'] for t, counts in by_type.items()}
    
    return overall_acc, type_accs

@call_parse
def main(
    experiment_name: str, # has to be one of main, raw, midsae
    backbone_name: str, # has to be one of mae, clip, siglip2, dinov3
    batch_size: int = 2048, # batch size for evaluation
    force: bool = False # if True, will overwrite existing eval file
):
    """
    Evaluate 0-shot accuracy on LEVELS odd-one-out task of the base model and the given checkpoint
    """
    eval_path = Path("data/evals/levelso1o")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{experiment_name}_{backbone_name}"
    eval_file = eval_path / f"{file_name}.json"
    if eval_file.exists() and not force:
        print(f"Eval file {eval_file} already exists. Use --force to overwrite.")
        return

    best_models = json.load(open(Path("data/checkpoints") / "best_models.json"))
    backbone_dict = json.load(open(Path("data/backbone_reps") / "backbones.json"))
    ckpt = best_models[f"[{experiment_name.upper()}]"][backbone_name]
    levels_reps = f"data/backbone_reps/levels_{backbone_dict[backbone_name]}.h5"
    
    backbone_reps, trials = prepare_levels(load_backbone(levels_reps))

    ckpt = torch.load(ckpt, weights_only=False)
    config, state_dict = ckpt['config'], fix_state_dict(ckpt['state_dict'])
    model = Transformer(c=config)
    model.load_state_dict(state_dict)
    model.eval()
    model = NNsight(model)

    with model.trace(backbone_reps.unsqueeze(1)): 
        metalign_reps = model.embed.output.squeeze().save()

    
    og_acc, og_type_accs = _calculate_accuracy(backbone_reps, trials, batch_size=batch_size)
    metalign_acc, metalign_type_accs = _calculate_accuracy(metalign_reps, trials, batch_size=batch_size)


    eval_data = {
        "model_name": backbone_name,
        "checkpoint_name": file_name,
        "base_model_accuracy": og_acc,
        "metalign_accuracy": metalign_acc,
        "base_model_accuracy_by_type": og_type_accs,
        "metalign_accuracy_by_type": metalign_type_accs,
        "num_trials": len(trials)
    }
    with open(eval_file, "w") as f: json.dump(eval_data, f, indent=4)
    print(f"Base model accuracy: {og_acc:.4f}")
    print(f"Metalign accuracy: {metalign_acc:.4f}")
    print(f"Base model by type: {og_type_accs}")
    print(f"Metalign by type: {metalign_type_accs}")
