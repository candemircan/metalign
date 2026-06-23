import json
from pathlib import Path

from fastcore.script import call_parse


@call_parse
def main(checkpoints_dir: Path = Path("data/checkpoints")):
    "Simple mapping of experiment and backbone to their single model.pt file."
    best_models = {}
    
    for p in checkpoints_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("[") or "[noemb" in p.name: continue
        exp_part, bb = p.name.rsplit("_", 1)
            
        ckpt = p / "model.pt"
        if ckpt.exists():
            if exp_part not in best_models: best_models[exp_part] = {}
            best_models[exp_part][bb] = str(ckpt)

    out_file = checkpoints_dir / "best_models.json"
    with open(out_file, "w") as f:
        json.dump(best_models, f, indent=4)