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

    model_names = ["CLIP-ViT-B-32-DataComp.XL-s13B-b90K_sae-top_k-64-cls_only-layer_11-hook_resid_post_raw",  "vit_base_patch16_dinov3.lvd1689m", "vit_base_patch16_siglip_256.v2_webli", "webssl_mae300m_full2b_224"]

    output_dir.mkdir(parents=True, exist_ok=True)
    backbone_json = {}
    backbone_dir = Path("data/backbone_reps")
    backbone_dir.mkdir(parents=True, exist_ok=True)

    experiment_setups = {
        "main" : {
            "prefix" : "main_",
            "train_features" : "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "eval_features" : "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",

        },
        "raw" : {
            "prefix" : "raw_",
            "train_features" : "coco_train_CLIP-ViT-B-32-DataComp.XL-s13B-b90K_sae-top_k-64-cls_only-layer_11-hook_resid_post_raw",
            "eval_features" : "coco_eval_CLIP-ViT-B-32-DataComp.XL-s13B-b90K_sae-top_k-64-cls_only-layer_11-hook_resid_post_raw",
        },
        "midsae": {
            "prefix" : "midsae_",
            "train_features" : "coco_train_sae-top_k-64-cls_only-layer_6-hook_resid_post",
            "eval_features" : "coco_eval_sae-top_k-64-cls_only-layer_6-hook_resid_post",
        },
        "2layers": {
            "prefix" : "2layers_",
            "train_features" : "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "eval_features" : "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "overrides" : {"n_layers": 2},
        },
        "mlp": {
            "prefix" : "mlp_",
            "train_features" : "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "eval_features" : "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "overrides" : {"use_mlp": True},
        },
        "wd": {
            "prefix" : "wd_",
            "train_features" : "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "eval_features" : "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "overrides" : {"weight_decay": 0.0001},
        },
        "8heads": {
            "prefix" : "8heads_",
            "train_features" : "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "eval_features" : "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "overrides" : {"n_heads": 8},
        },
        "32heads": {
            "prefix" : "32heads_",
            "train_features" : "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "eval_features" : "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "overrides" : {"n_heads": 32},
        },
        "noemb": {
            "prefix" : "noemb_",
            "train_features" : "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "eval_features" : "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "overrides" : {"no_embed": True},
        },
        "lstm": {
            "prefix" : "lstm_",
            "train_features" : "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "eval_features" : "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "overrides" : {"model_type": "lstm", "compile": False},
        },
        "static_linear": {
            "prefix" : "static_linear_",
            "train_features" : "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "eval_features" : "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",
            "overrides" : {"model_type": "static_linear"},
        },
    }

    for setup_name, setup_params in experiment_setups.items():

        for model_name in model_names:
            config = base_config.copy()
            config["model_name"] = model_name

            if "siglip" in model_name.lower():
                short_model_name = "siglip2"
            elif "dinov3" in model_name.lower():
                short_model_name = "dinov3"
            elif "clip" in model_name.lower():
                short_model_name = "clip"
            elif "webssl" in model_name.lower() or "mae" in model_name.lower():
                short_model_name = "mae"
            else:
                raise ValueError(f"Unknown model name {model_name}")

            config["name"] = f"[{setup_name}]_{short_model_name}"
            config["train_backbone"] = f"coco_train_{model_name}"
            config["eval_backbone"] = f"coco_eval_{model_name}"
            config["train_features"] = setup_params["train_features"]
            config["eval_features"] = setup_params["eval_features"]
            config["tags"] = [setup_params["prefix"].rstrip("_")]
            config.update(setup_params.get("overrides", {}))

            file_path = output_dir / f"{setup_params['prefix']}{short_model_name}.toml"
            with open(file_path, "wb") as f:
                tomli_w.dump(config, f)

            backbone_json[short_model_name] = model_name

    with open(backbone_dir / "backbones.json", "w") as f:
        json.dump(backbone_json, f, indent=4)
