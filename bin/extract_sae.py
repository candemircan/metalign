import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

import h5py
import torch
from fastcore.script import Param, call_parse
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_prisma.models.model_loader import load_hooked_model
from vit_prisma.sae import SparseAutoencoder
from vit_prisma.transforms import get_clip_val_transforms

from metalign.data import DATASET_MAKERS

_ = torch.set_grad_enabled(False)


def _extract_and_save(model, sae, dl, sae_path, raw_path, device, force):
    "Extracts SAE and raw activations and saves them to HDF5 files."
    if sae_path.exists() and not force:
        print(f"{sae_path.name} already exists. Use --force to overwrite.")
        return
    
    if sae_path.exists() and force: sae_path.unlink()

    image_offset = 0
    with h5py.File(sae_path, 'w') as sae_f, h5py.File(raw_path, 'w') as raw_f:
        for batch in tqdm(dl, desc=f"Processing for {sae_path.stem}"):
            images = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            _, cache = model.run_with_cache(images, names_filter=sae.cfg.hook_point)
            hook_acts = cache[sae.cfg.hook_point].to(device)

            raw_acts = hook_acts[:, 0, :]
            feature_acts = sae.encode(raw_acts)[1]

            for i in range(feature_acts.shape[0]):
                global_idx = image_offset + i
                sae_group = sae_f.create_group(str(global_idx))
                
                nonzero_indices = feature_acts[i].nonzero().squeeze().cpu().numpy()
                nonzero_values = feature_acts[i][feature_acts[i] != 0].cpu().numpy()
                
                sae_group.create_dataset('indices', data=nonzero_indices)
                sae_group.create_dataset('activations', data=nonzero_values)
                raw_f.create_dataset(str(global_idx), data=raw_acts[i].cpu().numpy())
            
            image_offset += feature_acts.shape[0]

@call_parse
def main(
    dataset: Param(help="Dataset to use", choices=['things', 'coco', 'levels', 'openimages_train', 'openimages_test']),
    repo_id: str, # The repository ID of the SAE model on Hugging Face Hub
    batch_size: int = 256, # Batch size for processing images
    force: bool = False, # Overwrite existing files if True
):
    "Extracts SAE features and raw activations, saving them to data/sae/ and data/backbone_reps/."
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    sae_dir = Path("data/sae"); sae_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/backbone_reps"); raw_dir.mkdir(parents=True, exist_ok=True)
    
    sae_path = hf_hub_download(repo_id, "weights.pt")
    sae = SparseAutoencoder.load_from_pretrained(sae_path).to(device).eval()
    
    model = load_hooked_model(sae.cfg.model_name, device=device).to(device).eval()
    transform = get_clip_val_transforms()
    
    repo_id_suffix = repo_id.split('/')[-1]
    save_model_name = f"{sae.cfg.model_name.split('/')[-1]}_{repo_id_suffix}"
    
    splits = ['train', 'eval'] if dataset == "coco" else ['']
    
    for split in splits:
        split_prefix = f"{dataset}_{split}_" if split else f"{dataset}_"
        
        sae_file_path = sae_dir / f"{split_prefix}{repo_id_suffix}.h5"
        raw_file_path = raw_dir / f"{split_prefix}{save_model_name}_raw.h5"

        ds = DATASET_MAKERS[dataset](transform=transform, split=split)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True)
        _extract_and_save(model, sae, dl, sae_file_path, raw_file_path, device, force)