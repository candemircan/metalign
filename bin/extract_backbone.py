import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # needed for aten::_upsample_bicubic2d_aa.out with the mps backend

from pathlib import Path

import h5py
import torch
from fastcore.script import call_parse
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from metalign.data import Coco, Things

_ = torch.set_grad_enabled(False)


def _extract_and_save(model, processor, dataset, save_path, device, batch_size):
    """Helper function to extract CLS tokens and save them."""
    all_cls_tokens = []
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Extracting to {save_path}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        
        # Collect batch of PIL images
        batch_images = []
        for i in range(start_idx, end_idx):
            batch_images.append(dataset[i])
        
        # Process the batch
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0]  # CLS token is first token
        all_cls_tokens.append(cls_tokens.cpu())
    
    cls_reps = torch.cat(all_cls_tokens, dim=0).numpy()
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('representations', data=cls_reps, compression='gzip')
    print(f"Saved representations to {save_path}")


@call_parse
def main(
    dataset: str, # one of things or coco
    repo_id: str = "facebook/dinov2-with-registers-base", # HuggingFace repo for the backbone model
    batch_size: int = 64, # batch size for the backbone model, for feature extraction
    force: bool = False # if True, will extract features even if the file already exists. Otherwise, will skip if the file  exists.
):
    """
    extract the CLS token representations from a HuggingFace model.
    The representations are saved in `data/backbone_reps/{dataset}_{model_name}.h5` file.
    For the COCO dataset, it creates separate files for train and evaluation sets.
    The representations are saved as a numpy array in an h5 file with compression.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModel.from_pretrained(repo_id).to(device)
    processor = AutoImageProcessor.from_pretrained(repo_id)
    model.eval()
    
    # Extract model name from repo for file naming
    model_name = repo_id.split('/')[-1]

    if dataset == "things":
        save_path = Path("data/backbone_reps") / f"things_{model_name}.h5"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists() and not force:
            print(f"File {save_path} already exists. Use --force to overwrite.")
            return
        
        ds = Things(processor=processor)
        _extract_and_save(model, processor, ds, save_path, device, batch_size)

    elif dataset == "coco":
        # Handle train set
        save_path_train = Path("data/backbone_reps") / f"coco_train_{model_name}.h5"
        save_path_train.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_train.exists() or force:
            ds_train = Coco(train=True, processor=processor)
            _extract_and_save(model, processor, ds_train, save_path_train, device, batch_size)
        else:
            print(f"File {save_path_train} already exists. Use --force to overwrite.")

        # Handle eval set
        save_path_eval = Path("data/backbone_reps") / f"coco_eval_{model_name}.h5"
        save_path_eval.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_eval.exists() or force:
            ds_eval = Coco(train=False, processor=processor)
            _extract_and_save(model, processor, ds_eval, save_path_eval, device, batch_size)
        else:
            print(f"File {save_path_eval} already exists. Use --force to overwrite.")
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'things', or 'coco'.")