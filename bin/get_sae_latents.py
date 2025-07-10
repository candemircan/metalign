import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # needed for aten::_upsample_bicubic2d_aa.out with the mps backend

import warnings
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

from metarep.data import Things

warnings.filterwarnings("ignore", module="kaleido") # idk what this is, but it is annoying
_ = torch.set_grad_enabled(False) 

@call_parse
def main(
    repo_id: str = "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05", # the repo ID of the SAE model on Hugging Face Hub
    batch_size: int = 64, # the batch size to use for processing images
    force: bool = False, # if True, remove the existing h5 file and remake one
):
    
    dir_path = Path("data/sae")
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{repo_id.split('/')[-1]}.h5"

    if file_path.exists() and force: os.remove(file_path)
    elif file_path.exists() and not force:
        print(f"File {file_path} already exists. Use --force to overwrite.")
        return

    # model loading
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    sae_path = hf_hub_download(repo_id, "weights.pt")
    hf_hub_download(repo_id, "config.json")
    sae = SparseAutoencoder.load_from_pretrained(sae_path).to(device)
    model_name = sae.cfg.model_name
    model = load_hooked_model(model_name, device=device).to(device)
    model.eval()
    sae.eval()

    # data
    things = Things()
    things.transform = get_clip_val_transforms()
    things_loader = DataLoader(things, batch_size=batch_size, shuffle=False, num_workers=4)
    
    image_offset = 0
    with h5py.File(file_path, 'w') as f:
        for batch in tqdm(things_loader, desc="Processing images"):
            images = batch.to(device)
            _, cache = model.run_with_cache(images, names_filter=sae.cfg.hook_point)
            hook_point_activation = cache[sae.cfg.hook_point].to(device)

            # 0 is the input of the SAE of the returned tuple, 1 is the feature activations
            feature_acts = sae.encode(hook_point_activation[:,0,:])[1] 

            # Find non-zero activations for the entire batch
            nonzero_mask = feature_acts != 0
            # nonzero_coords is (n_nonzero, 2), where cols are (img_idx_in_batch, feature_idx)
            nonzero_coords = nonzero_mask.nonzero(as_tuple=False)
            nonzero_values = feature_acts[nonzero_mask]


            # image index for each non-zero value (0th column of nonzero_coords)
            batch_img_indices = nonzero_coords[:, 0]
            
            #  how many non-zero elements belong to each image in the batch
            counts = torch.bincount(batch_img_indices, minlength=feature_acts.shape[0])

            # split the results into a list of tensors, one for each image
            split_indices = torch.split(nonzero_coords[:, 1], counts.tolist())
            split_activations = torch.split(nonzero_values, counts.tolist())


            for i in range(feature_acts.shape[0]):
                global_image_index = image_offset + i
                
                indices_data = split_indices[i].cpu().numpy() if torch.is_tensor(split_indices[i]) else split_indices[i]
                activations_data = split_activations[i].cpu().numpy() if torch.is_tensor(split_activations[i]) else split_activations[i]

                f.create_dataset(f'{global_image_index}/indices', data=indices_data)
                f.create_dataset(f'{global_image_index}/activations', data=activations_data)

            image_offset += feature_acts.shape[0]