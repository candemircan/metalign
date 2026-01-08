import json
from pathlib import Path

import pandas as pd
import torch
from fastcore.script import call_parse
from nnsight import NNsight
from torch.nn import functional as F
from tqdm import tqdm

from metalign.data import load_backbone, prepare_things_spose
from metalign.model import Transformer, TwoLinear
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)

def _get_logits(reps, X, batch_size=2048):
    "Calculate triplet similarities (logits) from representations"
    all_sims = []
            
    for i in tqdm(range(0, len(X), batch_size)):
        X_batch = X[i:i+batch_size]
        
        batch_reps = reps[X_batch]
        
        i_reps, j_reps, k_reps = batch_reps[:, 0], batch_reps[:, 1], batch_reps[:, 2]
        
        sim_ij = F.cosine_similarity(i_reps, j_reps)
        sim_ik = F.cosine_similarity(i_reps, k_reps)
        sim_jk = F.cosine_similarity(j_reps, k_reps)
        
        # Stack so that index 0 -> i is odd (j,k similar), 1 -> j odd (i,k similar), 2 -> k odd (i,j similar)
        sims = torch.stack([sim_jk, sim_ik, sim_ij], dim=1)
        all_sims.append(sims.cpu())
        
    return torch.cat(all_sims, dim=0)

def _accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == torch.tensor(y)).float().mean().item()

@call_parse
def main(
    experiment_name: str, # has to be one of main, raw, midsae
    backbone_name: str, # has to be one of  clip, siglip2, dinov3
    batch_size: int = 2048, # batch size for evaluation
    force:bool = False
):
    """
    Evaluate 0-shot accuracy on THINGS odd-one-out task of the base model and the given checkpoint
    """

    eval_path = Path("data/evals/thingso1o")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{experiment_name}_{backbone_name}"
    eval_file = eval_path / f"{file_name}.json"
    stats_file = eval_path / f"{file_name}_stats.csv"

    if eval_file.exists() and stats_file.exists() and not force:
        print(f"Eval files for {file_name} already exist, use --force to overwrite")
        return
    
    best_models = json.load(open(Path("data/checkpoints") / "best_models.json"))
    backbone_dict = json.load(open(Path("data/backbone_reps") / "backbones.json"))
    ckpt = best_models[f"[{experiment_name.upper()}]"][backbone_name]
    things_reps = f"data/backbone_reps/things_{backbone_dict[backbone_name]}.h5"
    

    df = pd.read_table("data/external/THINGS_triplets.csv")
    backbone_reps, ceiling_model = prepare_things_spose(load_backbone(things_reps))

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

    X = df[["image1", "image2", "image3"]].values - 1 # 0 index
    y = df["choice"].values -1 # 0 index
    
    # calculate logits for base and metaligned models
    og_logits = _get_logits(backbone_reps, X, batch_size=batch_size)
    metalign_logits = _get_logits(metalign_reps, X, batch_size=batch_size)
    
    # Calculate accuracy for reporting
    og_acc = _accuracy_from_logits(og_logits, y)
    metalign_acc = _accuracy_from_logits(metalign_logits, y)
    

    ceiling_acc    = _accuracy_from_logits(_get_logits(ceiling_model, X, batch_size=batch_size), y)

    eval_data = {
        "model_name": backbone_name,
        "checkpoint_name": file_name,
        "base_model_accuracy": og_acc,
        "metalign_accuracy": metalign_acc,
        "ceiling_accuracy": ceiling_acc
    }
    with open(eval_file, "w") as f: json.dump(eval_data, f, indent=4)
    print(f"Base model accuracy: {og_acc:.4f}")
    print(f"Metalign accuracy: {metalign_acc:.4f}")

    
    stats_df = pd.DataFrame({
        'subject_id': df['subject_id'],
        'y': y, # 0, 1, 2
        'm1_sim_0': og_logits[:, 0].numpy(),
        'm1_sim_1': og_logits[:, 1].numpy(),
        'm1_sim_2': og_logits[:, 2].numpy(),
        'm2_sim_0': metalign_logits[:, 0].numpy(),
        'm2_sim_1': metalign_logits[:, 1].numpy(),
        'm2_sim_2': metalign_logits[:, 2].numpy()
    })
    
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved statistical analysis data to {stats_file}")