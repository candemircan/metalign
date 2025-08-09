import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # needed for aten::_upsample_bicubic2d_aa.out with the mps backend

from pathlib import Path

import numpy as np
import torch
from fastcore.script import call_parse
from torch.utils.data import DataLoader
from tqdm import tqdm

from metalign.data import Coco, Things

_ = torch.set_grad_enabled(False)


def _extract_and_save(model, dataloader, save_path, device):
    """Helper function to extract CLS tokens and save them."""
    all_cls_tokens = []
    for images in tqdm(dataloader, desc=f"Extracting to {save_path}"):
        images = images.to(device)
        output = model.forward_features(images)
        all_cls_tokens.append(output["x_norm_clstoken"].cpu())
    
    cls_reps = torch.cat(all_cls_tokens, dim=0).numpy()
    np.savez_compressed(save_path, data=cls_reps)
    print(f"Saved representations to {save_path}")


@call_parse
def main(
    dataset: str, # one of things or coco
    repo: str = "facebookresearch/dinov2", # repo given to `torch.hub.load`
    model_name: str = "dinov2_vitb14_reg", # model name in the repo, also given to `torch.hub.load`
    batch_size: int = 64, # batch size for the backbone model, for feature extraction
    hf_repo: str = None, # HuggingFace repo to the backbone model. If provded, `repo` and `model` are ignored.
    force: bool = False # if True, will extract features even if the file already exists. Otherwise, will skip if the file  exists.
):
    """
    extract the CLS token representations from a model in a repo.
    The representations are saved in `data/backbone_reps/{dataset}_{model_name}.npz` file.
    For the COCO dataset, it creates separate files for train and evaluation sets.
    The representations are saved as a numpy array in a compressed npz file with the key 'data'.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if hf_repo is None: model = torch.hub.load(repo, model_name).to(device)
    else: raise NotImplementedError("HuggingFace repo support is not implemented yet.")
    
    model.eval()

    if dataset == "things":
        save_path = Path("data/backbone_reps") / f"things_{model_name}.npz"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists() and not force:
            print(f"File {save_path} already exists. Use --force to overwrite.")
            return
        
        ds = Things()
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        _extract_and_save(model, dataloader, save_path, device)

    elif dataset == "coco":
        # Handle train set
        save_path_train = Path("data/backbone_reps") / f"coco_train_{model_name}.npz"
        save_path_train.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_train.exists() or force:
            ds_train = Coco(train=True)
            dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=4)
            _extract_and_save(model, dataloader_train, save_path_train, device)
        else:
            print(f"File {save_path_train} already exists. Use --force to overwrite.")

        # Handle eval set
        save_path_eval = Path("data/backbone_reps") / f"coco_eval_{model_name}.npz"
        save_path_eval.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_eval.exists() or force:
            ds_eval = Coco(train=False)
            dataloader_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=4)
            _extract_and_save(model, dataloader_eval, save_path_eval, device)
        else:
            print(f"File {save_path_eval} already exists. Use --force to overwrite.")
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'things' or 'coco'.")