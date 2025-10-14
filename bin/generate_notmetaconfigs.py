import json
import tomllib
from pathlib import Path

import tomli_w
from fastcore.script import call_parse


@call_parse
def main(output_dir: Path = Path("data/configs"), base_config_path: Path = Path("data/base_notmeta_config.toml")):
    "Generates TOML config files for non-meta learning baseline experiments."

    with open(base_config_path, "rb") as f:
        base_config = tomllib.load(f)

    # Four backbone models to test
    model_names = [
        "CLIP-ViT-B-32-DataComp.XL-s13B-b90K_sae-top_k-64-cls_only-layer_11-hook_resid_post_raw",  
        "vit_base_patch16_dinov3.lvd1689m", 
        "vit_base_patch16_siglip_256.v2_webli"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    backbone_json = {}
    backbone_dir = Path("data/backbone_reps")
    backbone_dir.mkdir(parents=True, exist_ok=True)

    # Single experiment setup for non-meta learning with focal loss
    experiment_setup = {
        "prefix": "notmeta_",
        "tags": ["notmeta", "focal"],
        "train_features": "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
        "eval_features": "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",
        "use_focal_loss": True,
        "focal_loss_alpha": 0.25,
        "focal_loss_gamma": 2.0,
    }

    for i, model_name in enumerate(model_names):
        config = base_config.copy()
        config["model_name"] = model_name
        
        # Generate short model name for file naming
        if "siglip" in model_name.lower():
            short_model_name = "siglip2"
        elif "dinov3" in model_name.lower():
            short_model_name = "dinov3"
        elif "clip" in model_name.lower():
            short_model_name = "clip"
        else:
            raise ValueError(f"Unknown model name {model_name}")         
        
        config["name"] = f"[NOTMETA]_{short_model_name}"
        config["train_backbone"] = f"coco_train_{model_name}"
        config["eval_backbone"] = f"coco_eval_{model_name}"
        config["train_features"] = experiment_setup["train_features"]
        config["eval_features"] = experiment_setup["eval_features"]
        config["tags"] = experiment_setup["tags"]
        config["use_focal_loss"] = experiment_setup["use_focal_loss"]
        config["focal_loss_alpha"] = experiment_setup["focal_loss_alpha"]
        config["focal_loss_gamma"] = experiment_setup["focal_loss_gamma"]
        
        file_path = output_dir / f"{experiment_setup['prefix']}{short_model_name}_{i}.toml"
        with open(file_path, "wb") as f:
            tomli_w.dump(config, f)

        backbone_json[short_model_name] = model_name

    # Update backbones.json (don't overwrite, just add to it)
    backbone_json_path = backbone_dir / "backbones.json"
    if backbone_json_path.exists():
        with open(backbone_json_path, "r") as f:
            existing_backbones = json.load(f)
        existing_backbones.update(backbone_json)
        backbone_json = existing_backbones
    
    with open(backbone_json_path, "w") as f:
        json.dump(backbone_json, f, indent=4)

