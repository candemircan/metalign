import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # needed for aten::_upsample_bicubic2d_aa.out with the mps backend

from pathlib import Path

import h5py
import torch
from fastcore.script import call_parse
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_prisma.models.model_loader import load_hooked_model
from vit_prisma.sae import SparseAutoencoder
from vit_prisma.transforms import get_clip_val_transforms

from metalign.data import Coco, Levels, Things

_ = torch.set_grad_enabled(False) 

def _extract_and_save(
    model,
    sae,
    dataloader,
    sae_file_path,
    raw_file_path,
    device,
    force
):
    if sae_file_path.exists() and force: os.remove(sae_file_path)
    elif sae_file_path.exists() and not force:
        print(f"SAE file {sae_file_path} already exists. Use --force to overwrite.")
        return

    image_offset = 0
    with h5py.File(sae_file_path, 'w') as sae_f, h5py.File(raw_file_path, 'w') as raw_f:
        for batch in tqdm(dataloader, desc=f"Processing images for {sae_file_path.stem}"):
            # some return (image, labels) while others return images only
            if isinstance(batch, (list, tuple)): images = batch[0].to(device)
            else: images = batch.to(device)
            _, cache = model.run_with_cache(images, names_filter=sae.cfg.hook_point)
            hook_point_activation = cache[sae.cfg.hook_point].to(device)

            raw_acts = hook_point_activation[:, 0, :]
            feature_acts = sae.encode(raw_acts)[1] 

            nonzero_mask = feature_acts != 0
            nonzero_coords = nonzero_mask.nonzero(as_tuple=False)
            nonzero_values = feature_acts[nonzero_mask]

            batch_img_indices = nonzero_coords[:, 0]
            counts = torch.bincount(batch_img_indices, minlength=feature_acts.shape[0])

            split_indices = torch.split(nonzero_coords[:, 1], counts.tolist())
            split_activations = torch.split(nonzero_values, counts.tolist())

            for i in range(feature_acts.shape[0]):
                global_image_index = image_offset + i
                
                indices_data = split_indices[i].cpu().numpy()
                activations_data = split_activations[i].cpu().numpy()
                raw_data = raw_acts[i].cpu().numpy()

                sae_f.create_dataset(f'{global_image_index}/indices', data=indices_data)
                sae_f.create_dataset(f'{global_image_index}/activations', data=activations_data)
                raw_f.create_dataset(f'{global_image_index}', data=raw_data)
            image_offset += feature_acts.shape[0]

@call_parse
def main(
    dataset: str, # dataset to use, "things", "coco", "levels"
    repo_id: str, # the repo ID of the SAE model on Hugging Face Hub
    batch_size: int = 256, # the batch size to use for processing images
    force: bool = False, # if True, remove the existing h5 file and remake one
):
    
    if dataset not in ["things", "coco", "levels"]:
        raise ValueError("Dataset must be either 'things', 'coco', 'levels'")
    
    sae_dir_path = Path("data/sae")
    sae_dir_path.mkdir(parents=True, exist_ok=True)
    raw_dir_path = Path("data/backbone_reps")
    raw_dir_path.mkdir(parents=True, exist_ok=True)
    
    repo_id_suffix = repo_id.split('/')[-1]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    sae_path = hf_hub_download(repo_id, "weights.pt") # we already have it downloaded, but need the path for it
    sae = SparseAutoencoder.load_from_pretrained(sae_path).to(device)
    model_name = sae.cfg.model_name
    model = load_hooked_model(model_name, device=device).to(device)
    model.eval()
    sae.eval()
    save_model_name = f"{model_name.split("/")[-1]}_{repo_id_suffix}"

    if dataset == "things":
        clip_transforms = get_clip_val_transforms()
        ds = Things(transform=clip_transforms)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        sae_file_path = sae_dir_path / f"things_{repo_id_suffix}.h5"
        raw_file_path = raw_dir_path / f"things_{save_model_name}_raw.h5"
        _extract_and_save(model, sae, dataloader, sae_file_path, raw_file_path, device, force)

    elif dataset == "levels":
        clip_transforms = get_clip_val_transforms()
        ds = Levels(transform=clip_transforms)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        sae_file_path = sae_dir_path / f"levels_{repo_id_suffix}.h5"
        raw_file_path = raw_dir_path / f"levels_{save_model_name}_raw.h5"
        _extract_and_save(model, sae, dataloader, sae_file_path, raw_file_path, device, force)

    
    elif dataset == "coco":
        clip_transforms = get_clip_val_transforms()
        # Process training data
        ds_train = Coco(train=True, transform=clip_transforms)
        dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=4)
        sae_file_path_train = sae_dir_path / f"coco_train_{repo_id_suffix}.h5"
        raw_file_path_train = raw_dir_path / f"coco_train_{save_model_name}_raw.h5"
        _extract_and_save(model, sae, dataloader_train, sae_file_path_train, raw_file_path_train, device, force)

        # Process eval data
        ds_eval = Coco(train=False, transform=clip_transforms)
        dataloader_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=4)
        sae_file_path_eval = sae_dir_path / f"coco_eval_{repo_id_suffix}.h5"
        raw_file_path_eval = raw_dir_path / f"coco_eval_{save_model_name}_raw.h5"
        _extract_and_save(model, sae, dataloader_eval, sae_file_path_eval, raw_file_path_eval, device, force)
