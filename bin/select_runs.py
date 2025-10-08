import json
from glob import glob
from pathlib import Path

import torch
from fastcore.script import call_parse
from tqdm import tqdm


@call_parse
def main(
    checkpoints_dir: Path = Path("data/checkpoints"),
):
    experiment_patterns = set(p.name.split("_")[0] for p in checkpoints_dir.iterdir() if p.is_dir())
    print(f"Found experiments: {experiment_patterns}")

    backbone_patterns = set(p.name.split("_")[1] for p in checkpoints_dir.iterdir() if p.is_dir())
    print(f"Found backbones: {backbone_patterns}")

    best_models = {exp: {bb: None for bb in backbone_patterns} for exp in experiment_patterns}
    for exp in tqdm(experiment_patterns):
        for bb in tqdm(backbone_patterns, leave=False):
            best_eval = -float("inf")
            all_cpts = glob(str(checkpoints_dir / f"*{exp.strip("[]")}*{bb}*" / "model.pt"))
            for checkpoint in all_cpts:
                cur_eval = torch.load(checkpoint, weights_only=False)
                metric = "eval_accuracy" if "eval_accuracy" in cur_eval.keys() else "eval_mAP"
                cur_eval = cur_eval[metric]
                if cur_eval > best_eval:
                    best_eval = cur_eval
                    best_models[exp][bb] = checkpoint

    with open(checkpoints_dir / "best_models.json", "w") as f:
        json.dump(best_models, f, indent=4)
