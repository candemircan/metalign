import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # needed for aten::_upsample_bicubic2d_aa.out with the mps backend

from pathlib import Path

import h5py
import timm
import torch
from fastcore.script import call_parse
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from metalign.data import Coco, Levels, Things

_ = torch.set_grad_enabled(False)


def _extract_and_save(model, dataset, save_path, device, batch_size):
    """Helper function to extract features and save them."""
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    with h5py.File(save_path, 'w') as f:
        total_samples = len(dataset)
        feature_dim = None
        h5_dataset = None
        current_idx = 0
        
        for batch in tqdm(dataloader, desc=f"Extracting to {save_path}"):
            inputs = batch.to(device, non_blocking=True)
            features = model(inputs)
            features = features.cpu().numpy()
            
            if h5_dataset is None:
                feature_dim = features.shape[1]
                h5_dataset = f.create_dataset('representations', shape=(total_samples, feature_dim), dtype=features.dtype, compression='gzip')
            
            batch_size_actual = features.shape[0]
            h5_dataset[current_idx:current_idx + batch_size_actual] = features
            current_idx += batch_size_actual


@call_parse
def main(
    dataset: str, # one of things, coco, levels, cifar100
    repo_id: str,
    batch_size: int = 512, # batch size for the backbone model, for feature extraction
    force: bool = False # if True, will extract features even if the file already exists. Otherwise, will skip if the file  exists.
):
    """
    extract features from a timm model.
    The representations are saved in `data/backbone_reps/{dataset}_{model_name}.h5` file.
    For the COCO and CIFAR-100 datasets, it creates separate files for train and evaluation sets.
    The representations are saved as a numpy array in an h5 file with compression.
    """
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = f"hf_hub:{repo_id}" if repo_id.startswith("timm/") else repo_id
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    # Extract model name from repo for file naming
    model_name = repo_id.split('/')[-1]

    if dataset == "things":
        save_path = Path("data/backbone_reps") / f"things_{model_name}.h5"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists() and not force:
            print(f"File {save_path} already exists. Use --force to overwrite.")
            return
        
        ds = Things(transform=transform)
        _extract_and_save(model, ds, save_path, device, batch_size)

    elif dataset == "coco":
        # Handle train set
        save_path_train = Path("data/backbone_reps") / f"coco_train_{model_name}.h5"
        save_path_train.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_train.exists() or force:
            ds_train = Coco(train=True, transform=transform)
            _extract_and_save(model, ds_train, save_path_train, device, batch_size)
        else:
            print(f"File {save_path_train} already exists. Use --force to overwrite.")

        # Handle eval set
        save_path_eval = Path("data/backbone_reps") / f"coco_eval_{model_name}.h5"
        save_path_eval.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_eval.exists() or force:
            ds_eval = Coco(train=False, transform=transform)
            _extract_and_save(model, ds_eval, save_path_eval, device, batch_size)
        else:
            print(f"File {save_path_eval} already exists. Use --force to overwrite.")
    elif dataset == "levels":
        save_path = Path("data/backbone_reps") / f"levels_{model_name}.h5"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists() and not force:
            print(f"File {save_path} already exists. Use --force to overwrite.")
            return
        
        ds = Levels(transform=transform)
        _extract_and_save(model, ds, save_path, device, batch_size)

    elif dataset == "cifar100":
        # Handle train set
        save_path_train = Path("data/backbone_reps") / f"cifar100_train_{model_name}.h5"
        save_path_train.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_train.exists() or force:
            ds_train = CIFAR100(root="data/external", train=True, download=True, transform=transform)
            _extract_and_save(model, ds_train, save_path_train, device, batch_size)
        else:
            print(f"File {save_path_train} already exists. Use --force to overwrite.")

        # Handle test set
        save_path_test = Path("data/backbone_reps") / f"cifar100_test_{model_name}.h5"
        save_path_test.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_test.exists() or force:
            ds_test = CIFAR100(root="data/external", train=False, download=True, transform=transform)
            _extract_and_save(model, ds_test, save_path_test, device, batch_size)
        else:
            print(f"File {save_path_test} already exists. Use --force to overwrite.")

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'things', 'coco', 'levels', or 'cifar100'.")