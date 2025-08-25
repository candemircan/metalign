import json
from glob import glob
from pathlib import Path

import h5py
import pandas as pd
import torch
from fastcore.script import call_parse
from torch.nn import functional as F
from tqdm import tqdm

from metalign.data import prepare_things_spose
from metalign.model import Transformer

_ = torch.set_grad_enabled(False)

def calculate_accuracy(reps, X, y, batch_size=2048):
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
    checkpoint_name: str, # path to model checkpoint file
    batch_size: int = 2048 # batch size for evaluation
):
    """
    Evaluate 0-shot accuracy on THINGS odd-one-out task of the base model and the given checkpoint
    """
    
    checkpoint_path = Path("data/checkpoints") / checkpoint_name
    model_name = checkpoint_name.split("_")[0]
    things_reps = glob(f"data/backbone_reps/things*{model_name}*.h5")[0]

    df = pd.read_table("data/external/THINGS_odd_one_out/triplets_large_final_correctednc_correctedorder.csv")
    og_reps, _ = prepare_things_spose(h5py.File(things_reps, 'r')["representations"][:])

    ckpt = torch.load(checkpoint_path / "best.pt", weights_only=False)
    config, state_dict = ckpt['config'], ckpt['state_dict']
    model = Transformer(config=config)
    model.load_state_dict(state_dict)
    model.eval()

    metaligned_reps = torch.cat([torch.zeros(og_reps.shape[0], 2), og_reps], dim=1) @ model.embedding.weight.T + model.embedding.bias

    X = df[["image1", "image2", "image3"]].values - 1 # 0 index
    y = df["choice"].values -1 # 0 index
    
    og_acc = calculate_accuracy(og_reps, X, y, batch_size=batch_size)
    metalign_acc = calculate_accuracy(metaligned_reps, X, y, batch_size=batch_size)

    eval_path = Path("data/evals/things010")
    eval_path.mkdir(parents=True, exist_ok=True)
    eval_file = eval_path / f"{checkpoint_name}.json"
    eval_data = {
        "model_name": model_name,
        "checkpoint_name": checkpoint_name,
        "base_model_accuracy": og_acc,
        "metalign_accuracy": metalign_acc
    }
    with open(eval_file, "w") as f: json.dump(eval_data, f, indent=4)
    print(f"Base model accuracy: {og_acc:.4f}")
    print(f"Metalign accuracy: {metalign_acc:.4f}")