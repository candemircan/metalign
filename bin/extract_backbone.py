import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

import h5py
import timm
import torch
from fastcore.script import Param, call_parse
from torch.utils.data import DataLoader
from tqdm import tqdm

from metalign.data import Coco, Levels, Things

_ = torch.set_grad_enabled(False)

DATASET_MAKERS = {
    'things':   lambda **kwargs: Things(transform=kwargs['transform']),
    'coco':     lambda **kwargs: Coco(train=kwargs['split']=='train', transform=kwargs['transform']),
    'levels':   lambda **kwargs: Levels(transform=kwargs['transform']),
}

def _extract_and_save(model, dl, save_path, device):
    "helper function to extract features and save them to an HDF5 file."
    with h5py.File(save_path, 'w') as f:
        total_samples, feature_dim = len(dl.dataset), None
        h5_dset = None
        
        for i, batch in enumerate(tqdm(dl, desc=f"Extracting to {save_path.name}")):
            inputs = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device, non_blocking=True)
            features = model(inputs).cpu().numpy()
            
            if h5_dset is None:
                feature_dim = features.shape[1]
                h5_dset = f.create_dataset('representations', shape=(total_samples, feature_dim), dtype=features.dtype, compression='gzip')
            
            start_idx = i * dl.batch_size
            end_idx = start_idx + features.shape[0]
            h5_dset[start_idx:end_idx] = features

@call_parse
def main(
    dataset: Param(help="Dataset to use", choices=['things', 'coco', 'levels']),
    repo_id: str, # The repository ID of the model on Hugging Face Hub
    batch_size: int = 512, # Batch size for feature extraction
    force: bool = False # Overwrite existing files if True
):
    "extracts features from a timm model and saves them to `data/backbone_reps/.`"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = f"hf_hub:{repo_id}" if repo_id.startswith("timm/") else repo_id
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    file_model_name = repo_id.split('/')[-1]
    save_dir = Path("data/backbone_reps")
    save_dir.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'eval'] if dataset == "coco" else ['']
    
    for split in splits:
        fname = f"{dataset}_{split}_{file_model_name}.h5" if split else f"{dataset}_{file_model_name}.h5"
        save_path = save_dir / fname
        if save_path.exists() and not force:
            print(f"{save_path.name} already exists. Use --force to overwrite.")
            continue

        ds = DATASET_MAKERS[dataset](transform=transform, split=split)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        _extract_and_save(model, dl, save_path, device)