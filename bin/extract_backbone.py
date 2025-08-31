import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # needed for aten::_upsample_bicubic2d_aa.out with the mps backend

from pathlib import Path

import h5py
import open_clip
import timm
import torch
from fastcore.script import call_parse
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from metalign.data import Coco, Levels, Things

_ = torch.set_grad_enabled(False)


def _get_model_and_transform(repo_id: str, device: str):
    """Load model and get appropriate transform based on repo_id format."""
    
    if repo_id.startswith("timm/") and "siglip" in repo_id.lower():
        # SigLIP models via open_clip (even though they have timm/ prefix)
        model_name = f"hf-hub:{repo_id}"
        model, preprocess = open_clip.create_model_from_pretrained(model_name)
        model = model.to(device).eval()
        return model, preprocess, "open_clip", False
        
    elif repo_id.startswith("timm/"):
        # regular timm model
        model_name = f"hf_hub:{repo_id}"
        model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        return model, transform, "timm", False
    
    elif "/" in repo_id:
        model = AutoModel.from_pretrained(repo_id).to(device).eval()
        processor = AutoProcessor.from_pretrained(repo_id)
        is_mae = "mae" in repo_id.lower()
        return model, processor, "transformers", is_mae
    
    else:
        # assume open_clip format for non-timm models
        model, preprocess = open_clip.create_model_from_pretrained(repo_id)
        model = model.to(device).eval()
        return model, preprocess, "open_clip", False


def _extract_features(model, inputs, model_type: str, is_mae: bool = False):
    """Extract features based on model type."""
    
    if model_type == "timm": return model(inputs)  # CLS token features
    elif model_type == "open_clip": return model.encode_image(inputs)
    elif model_type == "transformers": 
        outputs = model(pixel_values=inputs)
        if is_mae:
            # for MAE models, average pool the patch tokens (skip CLS token)
            return outputs.last_hidden_state[:, 1:].mean(dim=1)
        else:
            return outputs.last_hidden_state[:, 0]  # CLS token from last hidden state
    else: raise ValueError(f"Unknown model type: {model_type}")


def _identity_collate(batch):
    """Custom collate function that returns batch as-is (for PIL images)."""
    return batch


def _extract_and_save(model, dataset, save_path, device, batch_size, model_type, processor=None, is_mae=False):
    """Helper function to extract features and save them."""
    from torch.utils.data import DataLoader
    
    if model_type == "transformers":
        dataset.transform = None
        collate_fn = _identity_collate
    else:
        collate_fn = None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    with h5py.File(save_path, 'w') as f:
        total_samples = len(dataset)
        feature_dim = None
        h5_dataset = None
        current_idx = 0
        
        for batch in tqdm(dataloader, desc=f"Extracting to {save_path}"):
            if model_type == "transformers" and processor is not None:
                processed = processor(images=batch, return_tensors="pt")
                inputs = processed["pixel_values"].to(device, non_blocking=True)
            else:
                inputs = batch.to(device, non_blocking=True)
            
            features = _extract_features(model, inputs, model_type, is_mae)
            features = features.cpu().numpy()
            
            if h5_dataset is None:
                feature_dim = features.shape[1]
                h5_dataset = f.create_dataset('representations', 
                                            shape=(total_samples, feature_dim), 
                                            dtype=features.dtype, 
                                            compression='gzip')
            
            batch_size_actual = features.shape[0]
            h5_dataset[current_idx:current_idx + batch_size_actual] = features
            current_idx += batch_size_actual


@call_parse
def main(
    dataset: str, # one of things, coco, levels
    repo_id: str,
    batch_size: int = 512, # batch size for the backbone model, for feature extraction
    force: bool = False # if True, will extract features even if the file already exists. Otherwise, will skip if the file  exists.
):
    """
    extract features from a model (timm, open_clip, or transformers).
    The representations are saved in `data/backbone_reps/{dataset}_{model_name}.h5` file.
    For the COCO dataset, it creates separate files for train and evaluation sets.
    The representations are saved as a numpy array in an h5 file with compression.
    """
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, transform_or_processor, model_type, is_mae = _get_model_and_transform(repo_id, device)
    
    # Extract model name from repo for file naming
    model_name = repo_id.split('/')[-1]

    if dataset == "things":
        save_path = Path("data/backbone_reps") / f"things_{model_name}.h5"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists() and not force:
            print(f"File {save_path} already exists. Use --force to overwrite.")
            return
        
        if model_type == "transformers":
            ds = Things(transform=None)  # No preprocessing for transformers
            _extract_and_save(model, ds, save_path, device, batch_size, model_type, processor=transform_or_processor, is_mae=is_mae)
        else:
            ds = Things(transform=transform_or_processor)
            _extract_and_save(model, ds, save_path, device, batch_size, model_type, is_mae=is_mae)

    elif dataset == "coco":
        # Handle train set
        save_path_train = Path("data/backbone_reps") / f"coco_train_{model_name}.h5"
        save_path_train.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_train.exists() or force:
            if model_type == "transformers":
                ds_train = Coco(train=True, transform=None)
                _extract_and_save(model, ds_train, save_path_train, device, batch_size, model_type, processor=transform_or_processor, is_mae=is_mae)
            else:
                ds_train = Coco(train=True, transform=transform_or_processor)
                _extract_and_save(model, ds_train, save_path_train, device, batch_size, model_type, is_mae=is_mae)
        else:
            print(f"File {save_path_train} already exists. Use --force to overwrite.")

        # Handle eval set
        save_path_eval = Path("data/backbone_reps") / f"coco_eval_{model_name}.h5"
        save_path_eval.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_eval.exists() or force:
            if model_type == "transformers":
                ds_eval = Coco(train=False, transform=None)
                _extract_and_save(model, ds_eval, save_path_eval, device, batch_size, model_type, processor=transform_or_processor, is_mae=is_mae)
            else:
                ds_eval = Coco(train=False, transform=transform_or_processor)
                _extract_and_save(model, ds_eval, save_path_eval, device, batch_size, model_type, is_mae=is_mae)
        else:
            print(f"File {save_path_eval} already exists. Use --force to overwrite.")
    elif dataset == "levels":
        save_path = Path("data/backbone_reps") / f"levels_{model_name}.h5"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists() and not force:
            print(f"File {save_path} already exists. Use --force to overwrite.")
            return
        
        if model_type == "transformers":
            ds = Levels(transform=None)  # No preprocessing for transformers
            _extract_and_save(model, ds, save_path, device, batch_size, model_type, processor=transform_or_processor, is_mae=is_mae)
        else:
            ds = Levels(transform=transform_or_processor)
            _extract_and_save(model, ds, save_path, device, batch_size, model_type, is_mae=is_mae)

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'things', 'coco', or 'levels'.")