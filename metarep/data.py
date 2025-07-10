"""
common datasets and processing utils  used throughout the project
"""

__all__ = ["prepare_things_spose", "Things", "ThingsFunctionLearning", "h5_to_numpy"]

from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import NUM_THINGS_CATEGORIES, NUM_THINGS_IMAGES
from .transforms import image_transform


def h5_to_numpy(model_name: str = "sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05", # model name, meant to match the name of the HF repo (without the org)
                data_root: Path = Path("data/sae"), # where you keep your SAE activations
                min_nonzero: int = 0 # minimum number of non-zero activations per column to keep it in the final array
                ) -> np.ndarray:
    """
    Convert SAE activations from h5 file to numpy array.

    Activations are sparse over observations, but the space is very high dimensional.
    To save disk space, only non-zero activations and their indices are stored in the h5 file.
    Here, we convert the h5 file to a numpy array, filling in the zeros for the missing indices.
    If a column is all zeros, it is removed from the final array. Also, if a column has less than `min_nonzero` non-zero activations, it is removed from the final array.

    The original h5 file is expected to be in the format:
    ```
    {
        "0": {
            "activations": [...], # the activations for the first image
            "indices": [...] # the indices of the non-zero activations
        },
        "1": {
            "activations": [...], # the activations for the second image
            "indices": [...] # the indices of the non-zero activations
        },
        ...
    }
    ```

    The ordering is based on `sorted("data/external/THINGS/*/*jpg)`
    """

    file_path = data_root / f"{model_name}.h5"

    sae_h5 = h5py.File(file_path, 'r')
    indices = []
    for k in sae_h5.keys():
        indices.extend(sae_h5[k]["indices"][:].tolist())

    num_cols = max(indices) + 1
    activations = np.zeros((len(sae_h5), num_cols), dtype=np.float32)

    for img in range(len(sae_h5)):
        img_id = str(img)
        h5_activations = sae_h5[img_id]["activations"][:]
        indices = sae_h5[img_id]["indices"][:]
        filler_activations = np.zeros(num_cols, dtype=np.float32)
        filler_activations[indices] = h5_activations
        activations[img, :] = filler_activations

    sae_h5.close()
    activations = activations[:, (activations.mean(0) != 0)]

    # remove columns with less than min_nonzero non-zero activations
    non_zero_counts = np.count_nonzero(activations, axis=0)
    activations = activations[:, non_zero_counts >= min_nonzero]
    
    return activations



def prepare_things_spose(
    representations: np.lib.npyio.NpzFile, # directly from np.load("data/backbone_reps/{backbone}.npz"), keys are different token types (e.g. cls), values are numpy arrays of shape (N, D) where N is the number of images and D is the dimension of the representation 
    data_root: Path = Path("data/external"), # the root directory of the THINGS dataset, which contains the images and the unique_id.txt file
    return_tensors: str = "pt" # the type of tensors to return, can be "pt" for PyTorch tensors or "np" for NumPy arrays
) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
    """
    Match the representations from the backbone with the SPoSE embeddings for the THINGS dataset.

    Only use the first image of each category. The ordering is done based on the `unique_id.txt` file, which matches `spose_embedding_66d_sorted.txt`
    """
    img_names = sorted((data_root / "THINGS").glob("*/*.jpg"))
    unique_ids = [line.strip() for line in open(data_root / "unique_id.txt", "r")]

    assert all(img_name.parent.name in unique_ids for img_name in img_names), "Not all parent folders in img_names are in unique_ids"

    img_idx = []
    parent_name_list = [img_name.parent.name for img_name in img_names]
    for unique_id in unique_ids:
        img_idx.append(parent_name_list.index(unique_id)) # always gets the first item

    assert 0 in img_idx, "The first image must be in there"
    assert len(img_idx) == NUM_THINGS_CATEGORIES, f"Expected {NUM_THINGS_CATEGORIES} categories, found {len(img_idx)}"

    X = np.hstack([representations[key] for key in representations.keys()])[img_idx].astype(np.float32)
    Y = np.loadtxt(data_root / "spose_embedding_66d_sorted.txt").astype(np.float32)

    if return_tensors == "pt": X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    elif return_tensors == "np": pass
    else: raise ValueError(f"Unknown return_tensors type: {return_tensors}. Must be 'pt' or 'np'.")
    
    return X, Y


class Things(Dataset):
    """
    A dataset for the THINGS dataset, which contains images of objects.

    The images are expected to be in the format `data/external/THINGS/{category}/{image}.jpg`, which matches the original structure of the THINGS dataset.

    By default, ImageNet style transformations are applied to the images, including resizing, cropping, normalization, and conversion to tensor.
    """

    def __init__(self, root: Path = Path("data/external/THINGS"), total_images: int = NUM_THINGS_IMAGES):
        self.images = sorted(root.glob("*/*.jpg"))
        self.transform = image_transform()

        assert len(self.images) == total_images, f"Expected {total_images} images, found {len(self.images)} in {root}"

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path) as image:
            image = self.transform(image)
        return image


class ThingsFunctionLearning(Dataset):
    """
    This dataset contains the representations corresponding to the images, as well as the corresponding SPoSE embeddings.
    """

    def __init__(self, representations: torch.Tensor, root: Path = Path("data/external")):
        X, Y = prepare_things_spose(representations, data_root=root)
        
        self.X, self.Y = X, Y
            
        self.feature_dim = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idxs, dim):
        return self.X[idxs], self.Y[idxs, dim]