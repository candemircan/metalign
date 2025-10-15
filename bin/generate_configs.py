import itertools
import json
import tomllib
from pathlib import Path

import tomli_w
from fastcore.script import call_parse


@call_parse
def main(output_dir: Path = Path("data/configs"), base_config_path: Path = Path("data/base_config.toml")):
    "Generates TOML config files for a grid search."

    with open(base_config_path, "rb") as f:
        base_config = tomllib.load(f)

    param_grid = {
        "model_name": ["CLIP-ViT-B-32-DataComp.XL-s13B-b90K_sae-top_k-64-cls_only-layer_11-hook_resid_post_raw",  "vit_base_patch16_dinov3.lvd1689m", "vit_base_patch16_siglip_256.v2_webli"]
    }

    keys, values = zip(*param_grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    output_dir.mkdir(parents=True, exist_ok=True)
    backbone_json = {}
    backbone_dir = Path("data/backbone_reps")
    backbone_dir.mkdir(parents=True, exist_ok=True)

    experiment_setups = {
        "MAIN" : {
            "prefix" : "",
            "tags" : ["main"],
            "train_features" : "openimages_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "eval_features" : "openimages_test_sae-top_k-64-cls_only-layer_11-hook_resid_post",

        },
        "RAW" : {
            "prefix" : "raw_",
            "tags" : ["raw"],
            "train_features" : "openimages_train_CLIP-ViT-B-32-DataComp.XL-s13B-b90K_sae-top_k-64-cls_only-layer_11-hook_resid_post_raw",
            "eval_features" : "openimages_test_CLIP-ViT-B-32-DataComp.XL-s13B-b90K_sae-top_k-64-cls_only-layer_11-hook_resid_post_raw",
        },
        "MIDSAE": {
            "prefix" : "midsae_",
            "tags" : ["midsae"],
            "train_features" : "openimages_train_sae-top_k-64-cls_only-layer_6-hook_resid_post",
            "eval_features" : "openimages_test_sae-top_k-64-cls_only-layer_6-hook_resid_post",
        },
    }

    for setup_name, setup_params in experiment_setups.items():

        for i, params in enumerate(permutations):
            config = base_config.copy()
            config.update(params) 
            if "siglip" in config["model_name"].lower():
                short_model_name = "siglip2"
            elif "dinov3" in config["model_name"].lower():
                short_model_name = "dinov3"
            elif "clip" in config["model_name"].lower():
                short_model_name = "clip"
            else:
                raise ValueError(f"Unknown model name {config['model_name']}")         
            config["name"] = f"[{setup_name}]_{short_model_name}"
            config["train_backbone"] = f"openimages_train_{config['model_name']}"
            config["eval_backbone"] = f"openimages_test_{config['model_name']}"
            config["train_features"] = setup_params["train_features"]
            config["eval_features"] = setup_params["eval_features"]
            config["tags"] = setup_params["tags"]
            

            file_path = output_dir / f"{setup_params['prefix']}{short_model_name}_{i}.toml"
            with open(file_path, "wb") as f:
                tomli_w.dump(config, f)

            backbone_json[short_model_name] = config["model_name"]

    with open(backbone_dir / "backbones.json", "w") as f:
        json.dump(backbone_json, f, indent=4)