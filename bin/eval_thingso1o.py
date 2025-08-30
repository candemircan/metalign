import json
from pathlib import Path

import pandas as pd
import torch
from fastcore.script import call_parse
from torch.nn import functional as F
from tqdm import tqdm

from metalign.data import load_backbone_representations, prepare_things_spose
from metalign.model import Transformer

_ = torch.set_grad_enabled(False)

def _calculate_accuracy(reps, X, y, batch_size=2048):
    "Calculate triplet accuracy from representations"
    num_correct, num_total = 0, 0
            
    for i in tqdm(range(0, len(X), batch_size)):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        
        batch_reps = reps[X_batch]
        
        i_reps, j_reps, k_reps = batch_reps[:, 0], batch_reps[:, 1], batch_reps[:, 2]
        
        sim_ij = F.cosine_similarity(i_reps, j_reps)
        sim_ik = F.cosine_similarity(i_reps, k_reps)
        sim_jk = F.cosine_similarity(j_reps, k_reps)
        
        sims = torch.stack([sim_jk, sim_ik, sim_ij], dim=1)
        
        preds = torch.argmax(sims, dim=1)
        
        num_correct += (preds == y_batch).sum().item()
        num_total += len(X_batch)
        
    return num_correct / num_total

@call_parse
def main(
    experiment_name: str, # has to be one of main, raw, midsae
    backbone_name: str, # has to be one of vit, clip, siglip2, dinov2
    batch_size: int = 2048 # batch size for evaluation
):
    """
    Evaluate 0-shot accuracy on THINGS odd-one-out task of the base model and the given checkpoint
    """
    
    best_models = json.load(open(Path("data/checkpoints") / "best_models.json"))
    backbone_dict = json.load(open(Path("data/backbone_reps") / "backbones.json"))
    ckpt = best_models[f"[{experiment_name.upper()}]"][backbone_name]
    things_reps = f"data/backbone_reps/things_{backbone_dict[backbone_name]}.h5"
    

    df = pd.read_table("data/external/THINGS_triplets.csv")
    backbone_reps, _ = prepare_things_spose(load_backbone_representations(things_reps))

    ckpt = torch.load(ckpt, weights_only=False)
    config, state_dict = ckpt['config'], ckpt['state_dict']
    model = Transformer(config=config)
    model.load_state_dict(state_dict)
    model.eval()

    metaligned_reps = torch.cat([torch.zeros(backbone_reps.shape[0], 2), backbone_reps], dim=1) @ model.embedding.weight.T + model.embedding.bias

    X = df[["image1", "image2", "image3"]].values - 1 # 0 index
    y = df["choice"].values -1 # 0 index
    
    og_acc = _calculate_accuracy(backbone_reps, X, y, batch_size=batch_size)
    metalign_acc = _calculate_accuracy(metaligned_reps, X, y, batch_size=batch_size)

    eval_path = Path("data/evals/thingso1o")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{experiment_name}_{backbone_name}"
    eval_file = eval_path / f"{file_name}.json"
    eval_data = {
        "model_name": backbone_name,
        "checkpoint_name": file_name,
        "base_model_accuracy": og_acc,
        "metalign_accuracy": metalign_acc
    }
    with open(eval_file, "w") as f: json.dump(eval_data, f, indent=4)
    print(f"Base model accuracy: {og_acc:.4f}")
    print(f"Metalign accuracy: {metalign_acc:.4f}")