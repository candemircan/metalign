import itertools
from pathlib import Path

import tomli_w
from fastcore.script import call_parse


@call_parse
def main(output_dir: Path = Path("data/configs")):
    "Generates TOML config files for a grid search."
    
    base_config = {
        "embedding": True,
        "hidden_act": "gelu",
        "bias": True,
        "sequence_length": 120,
        "logit_bias": True,
        "scale": False,
        "compile": True,
        "warmup_steps": 1000,
        "training_steps": 1000000,
        "attention_dropout": 0.1,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "weight_decay": 1e-4,
        "num_layers": 3,
        "batch_size": 4096
    }

    param_grid = {
        "lr": [0.0001 ,0.00025, 0.0005, 0.001], 
        "model_name": ["dinov3-vitb16-pretrain-lvd1689m", "CLIP-ViT-B-32-DataComp.XL-s13B-b90K_sae-top_k-64-cls_only-layer_11-hook_resid_post_raw",  "siglip2-base-patch16-224", "vit-base-patch16-224"]
    }

    keys, values = zip(*param_grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, params in enumerate(permutations):
        config = base_config.copy()
        config.update(params) 
        short_model_name = config["model_name"].split("-")[0]               
        lr_str = f"{config['lr']:.0e}".replace("-0", "-")
        config["name"] = f"[MAIN]_{short_model_name}_lr{lr_str}"
        config["train_backbone"] = f"coco_train_{config['model_name']}"
        config["eval_backbone"] = f"coco_eval_{config['model_name']}"
        file_path = output_dir / f"{i}.toml"
        with open(file_path, "wb") as f:
            tomli_w.dump(config, f)