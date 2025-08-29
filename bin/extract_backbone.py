import os
from pathlib import Path

import h5py
import open_clip
import timm
import torch
from fastcore.script import call_parse
from tqdm import tqdm

from metalign.data import Coco, Things

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # needed for aten::_upsample_bicubic2d_aa.out with the mps backend

_ = torch.set_grad_enabled(False)


def _get_model_and_transform(repo_id: str, device: str):
    """Load model and get appropriate transform based on repo_id format."""
    
    if "siglip" in repo_id.lower() and repo_id.startswith("timm/"):
        # SigLIP models via open_clip (even though they have timm/ prefix)
        model_name = f"hf-hub:{repo_id}"
        model, preprocess = open_clip.create_model_from_pretrained(model_name)
        model = model.to(device).eval()
        return model, preprocess, "open_clip"
        
    elif repo_id.startswith("timm/"):
        # Regular timm model
        model_name = f"hf_hub:{repo_id}"
        model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        return model, transform, "timm"
    
    else:
        # Assume open_clip format for non-timm models
        model, preprocess = open_clip.create_model_from_pretrained(repo_id)
        model = model.to(device).eval()
        return model, preprocess, "open_clip"


def _extract_features(model, inputs, model_type: str):
    """Extract features based on model type."""
    
    if model_type == "timm": return model(inputs)  # CLS token features
    elif model_type == "open_clip": return model.encode_image(inputs)
    else: raise ValueError(f"Unknown model type: {model_type}")


def _extract_and_save(model, transform, dataset, save_path, device, batch_size, model_type):
    """Helper function to extract features and save them."""
    all_features = []
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Extracting to {save_path}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        
        # Collect batch of tensors - dataset returns tensors
        batch_tensors = []
        for i in range(start_idx, end_idx):
            batch_tensors.append(dataset[i])
        inputs = torch.stack(batch_tensors).to(device)
        
        features = _extract_features(model, inputs, model_type)
        all_features.append(features.cpu())
    
    feature_reps = torch.cat(all_features, dim=0).numpy()
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('representations', data=feature_reps, compression='gzip')


@call_parse
def main(
    dataset: str, # one of things or coco
    repo_id: str,
    batch_size: int = 512, # batch size for the backbone model, for feature extraction
    force: bool = False # if True, will extract features even if the file already exists. Otherwise, will skip if the file  exists.
):
    """
    extract features from a model (timm or open_clip).
    The representations are saved in `data/backbone_reps/{dataset}_{model_name}.h5` file.
    For the COCO dataset, it creates separate files for train and evaluation sets.
    The representations are saved as a numpy array in an h5 file with compression.
    """
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, transform, model_type = _get_model_and_transform(repo_id, device)
    
    # Extract model name from repo for file naming
    model_name = repo_id.split('/')[-1]

    if dataset == "things":
        save_path = Path("data/backbone_reps") / f"things_{model_name}.h5"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists() and not force:
            print(f"File {save_path} already exists. Use --force to overwrite.")
            return
        
        ds = Things(transform=transform)
        _extract_and_save(model, transform, ds, save_path, device, batch_size, model_type)

    elif dataset == "coco":
        # Handle train set
        save_path_train = Path("data/backbone_reps") / f"coco_train_{model_name}.h5"
        save_path_train.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_train.exists() or force:
            ds_train = Coco(train=True, transform=transform)
            _extract_and_save(model, transform, ds_train, save_path_train, device, batch_size, model_type)
        else:
            print(f"File {save_path_train} already exists. Use --force to overwrite.")

        # Handle eval set
        save_path_eval = Path("data/backbone_reps") / f"coco_eval_{model_name}.h5"
        save_path_eval.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_eval.exists() or force:
            ds_eval = Coco(train=False, transform=transform)
            _extract_and_save(model, transform, ds_eval, save_path_eval, device, batch_size, model_type)
        else:
            print(f"File {save_path_eval} already exists. Use --force to overwrite.")
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'things', or 'coco'.")