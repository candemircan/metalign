import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # needed for aten::_upsample_bicubic2d_aa.out with the mps backend

from pathlib import Path

import numpy as np
import torch
from fastcore.script import call_parse
from torch.utils.data import DataLoader
from tqdm import tqdm

from metarep.data import Things

_ = torch.set_grad_enabled(False)


@call_parse
def main(
    repo: str = "facebookresearch/dinov2", # repo given to `torch.hub.load`
    model_name: str = "dinov2_vitb14_reg", # model name in the repo, also given to `torch.hub.load`
    batch_size: int = 64, # batch size for the backbone model, for feature extraction
    hf_repo: str = None, # HuggingFace repo to the backbone model. If provded, `repo` and `model` are ignored.
    force: bool = False # if True, will extract features even if the file already exists. Otherwise, will skip if the file  exists.
):
    """
    extract the backbone representations from a model in a repo.
    The representations are saved in `data/backbone_reps/{model}.npz` file.
    The representations are saved as a dictionary with keys:
    - "cls": the class token representation
    - "register": the register representations averaged over the registers (if available)
    - "patch": the patch representations averaged over the patches

    Currently, the extraction keys are formatted after the Dinov2 model only.
    """
    save_path = Path("data/backbone_reps") / f"{model_name}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and not force:
        print(f"File {save_path} already exists. Use --force to overwrite.")
        return
    

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if hf_repo is None: model = torch.hub.load(repo, model_name).to(device)
    else: raise NotImplementedError("HuggingFace repo support is not implemented yet.")
    
    model.eval()

    things_dataloader = DataLoader(Things(), batch_size=batch_size, shuffle=False, num_workers=4)
    representations=dict(cls=[], register=[], patch=[])

    for images in tqdm(things_dataloader, desc="extracting representations"):
        images = images.to(device)
        output = model.forward_features(images)

        representations["cls"].append(output["x_norm_clstoken"].cpu())
        if "x_norm_regtokens" in output.keys(): representations["register"].append(output["x_norm_regtokens"].mean(dim=1).cpu())
        representations["patch"].append(output["x_norm_patchtokens"].mean(dim=1).cpu())

    if len(representations["register"]) == 0: del representations["register"]
    for key in representations.keys():
        representations[key] = torch.cat(representations[key], dim=0).numpy()

    
    np.savez_compressed(save_path, **representations)