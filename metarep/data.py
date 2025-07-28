"""
common datasets and processing utils  used throughout the project
"""

__all__ = ["prepare_things_spose", "ImageDataset", "Things", "Coco", "ThingsFunctionLearning", "SimpleFunctionLearning", "h5_to_numpy", "image_transform"]

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import make_classification
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, NUM_COCO_TRAIN_IMAGES, NUM_THINGS_CATEGORIES, NUM_THINGS_IMAGES


def _convert_to_rgb(img):
    if img.mode != 'RGB': return img.convert('RGB')
    return img

def image_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    """
    transform pipeline for images, including resizing, cropping, normalization, and conversion to tensor.
    standard imagenet stuff, adapted from [here](https://github.com/facebookresearch/dinov2/blob/592541c8d842042bb5ab29a49433f73b544522d5/dinov2/data/transforms.py)
    """
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)



def h5_to_numpy(model_name: str = "things_sae-top_k-64-cls_only-layer_11-hook_resid_post", # model name, meant to match the name of the HF repo (without the org)
                data_root: Path = Path("data/sae"), # where you keep your SAE activations
                min_nonzero: int = 1, # minimum number of non-zero activations per column to keep it in the final array. The default is such that any column that is all 0s is removed.
                ) -> np.ndarray:
    """
    Convert SAE activations from h5 file to numpy array.

    Activations are sparse over observations, but the space is very high dimensional.
    To save disk space, only non-zero activations and their indices are stored in the h5 file.
    Here, we convert the h5 file to a numpy array, filling in the zeros for the missing indices.
   s If a column has less than `min_nonzero` non-zero activations, it is removed from the final array.

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

    The ordering is based on `sorted("data/external/THINGS/*/*jpg)` or `sorted("data/external/coco/train2017/*.jpg")`, depending on the dataset.
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

    # remove columns with less than min_nonzero non-zero activations
    non_zero_counts = np.count_nonzero(activations, axis=0)
    activations = activations[:, non_zero_counts >= min_nonzero]
    
    return activations



def prepare_things_spose(
    representations: np.lib.npyio.NpzFile, # directly from np.load("data/backbone_reps/{backbone}.npz"), keys are different token types (e.g. cls), values are numpy arrays of shape (N, D) where N is the number of images and D is the dimension of the representation 
    data_root: Path = Path("data/external"), # the root directory of the THINGS dataset, which contains the images and the unique_id.txt file
    return_tensors: str = "pt", # the type of tensors to return, can be "pt" for PyTorch tensors or "np" for NumPy arrays
    tokens: str = "all" # which tokens to use. if all, all tokens will be concatenated. otherwise needs to be one of cls, patch, or register (if avail)
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

    if tokens == "all":
        X = np.hstack([representations[key] for key in representations.keys()])[img_idx].astype(np.float32)
    elif tokens in representations:
        X = representations[tokens][img_idx].astype(np.float32)
    else:
        raise ValueError(f"Unknown token type: {tokens}. Must be 'all', 'cls', 'patch', or 'register'.")
    Y = np.loadtxt(data_root / "spose_embedding_66d_sorted.txt").astype(np.float32)

    if return_tensors == "pt": X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    elif return_tensors == "np": pass
    else: raise ValueError(f"Unknown return_tensors type: {return_tensors}. Must be 'pt' or 'np'.")
    
    return X, Y


class ImageDataset(Dataset):
    """
    A generic image dataset that can handle different directory structures.
    Images are sorted by name.

    By default, ImageNet style transformations are applied to the images, including resizing, cropping, normalization, and conversion to tensor.
    """

    def __init__(self, root: Path, glob_pattern: str = "*.jpg", total_images: int | None = None):
        self.images = sorted(root.glob(glob_pattern))
        self.transform = image_transform()

        if total_images is not None:
            assert len(self.images) == total_images, f"Expected {total_images} images, found {len(self.images)} in {root} with pattern {glob_pattern}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path) as image:
            image = self.transform(image)
        return image


class Things(ImageDataset):
    """
    A dataset for the THINGS dataset, which contains images of objects.

    The images are expected to be in the format `data/external/THINGS/{category}/{image}.jpg`, which matches the original structure of the THINGS dataset.
    """

    def __init__(self, root: Path = Path("data/external/THINGS"), total_images: int = NUM_THINGS_IMAGES):
        super().__init__(root=root, glob_pattern="*/*.jpg", total_images=total_images)


class Coco(ImageDataset):
    """
    A dataset for the COCO dataset, which contains images of objects.

    The images are expected to be in the format `data/external/coco/train2017/{image}.jpg`, which matches the original structure of the COCO dataset.
    """

    def __init__(self, root: Path = Path("data/external/coco/train2017"), total_images: int = NUM_COCO_TRAIN_IMAGES):
        super().__init__(root=root, glob_pattern="*.jpg", total_images=total_images)


class FunctionLearning(Dataset):
    "Base class for function learning datasets."
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, scale: bool = True):
        self.X, self.Y = X, Y
        self.feature_dim = self.X.shape[1]
        self.num_functions = self.Y.shape[1]

        if scale:
            self.mean = self.X.mean(dim=0, keepdim=True)
            self.std = self.X.std(dim=0, keepdim=True)
            self.X = (self.X - self.mean) / (self.std + 1e-8)
        else: self.mean, self.std = None, None

    def sample_episode(self, dim: int, seq_len: int, fixed_label: bool = False, weighted: bool = False):
        raise NotImplementedError

    def inverse_transform(self, X):
        "Inverse transform the data, i.e. scale it back to the original space."
        if self.mean is None or self.std is None:
            print("Warning: No scaling applied, returning original data.")
            return X
        return X * self.std + self.mean

    def __len__(self): return len(self.X)


class SAEFunctionLearning(FunctionLearning):
    "A dataset for learning functions from SAE embeddings."
    def __init__(self, inputs: np.ndarray, sae_features: str, data_root: Path = Path("data/sae"), scale: bool = True, min_nonzero: int = 100):
        "Initializes the dataset by preparing data and pre-calculating medians."
        X = torch.tensor(inputs, dtype=torch.float32)
        Y = torch.from_numpy(h5_to_numpy(sae_features, data_root=data_root, min_nonzero=min_nonzero))
        super().__init__(X, Y, scale)

    def sample_episode(self, dim: int, seq_len: int, fixed_label: bool = False, weighted: bool = False):
        "Sample an episode. Class 0: zero value. Class 1: non-zero value. If weighted, sample non-zero values based on magnitude."
        y_dim = self.Y[:, dim]
        
        std_dev = seq_len / 20.0
        n_pos = int(torch.normal(mean=torch.tensor(seq_len / 2), std=torch.tensor(std_dev)).round().clamp(0, seq_len).item())
        n_neg = seq_len - n_pos

        pos_mask = y_dim != 0
        neg_mask = ~pos_mask

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        if weighted:
            pos_weights = y_dim[pos_mask]
            pos_sample_indices = pos_indices[torch.multinomial(pos_weights, n_pos, replacement=False)]
        else:
            pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
        
        neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]

        indices = torch.cat([pos_sample_indices, neg_sample_indices])
        indices = indices[torch.randperm(len(indices))]

        X_episode = self.X[indices]
        Y_episode = (self.Y[indices, dim] != 0).float()

        if not fixed_label and torch.rand(1).item() < 0.5: Y_episode = 1 - Y_episode
        return X_episode, Y_episode


class ThingsFunctionLearning(FunctionLearning):
    "A dataset for classification on the THINGS dataset, using SPoSE embeddings."
    def __init__(self, representations: dict, data_root: Path = Path("data/external"), scale: bool = True):
        "Initializes the dataset by preparing data and pre-calculating medians."
        X, Y = prepare_things_spose(representations, data_root=data_root)
        super().__init__(X, Y, scale)
        self.medians = torch.median(self.Y, dim=0).values
    
    def sample_episode(self, dim: int, seq_len: int, fixed_label: bool = False, weighted: bool = False):
        "Sample an episode. Class 0: lower median split. Class 1: upper median split. If weighted, sample based on magnitude from median."
        std_dev = seq_len / 20.0
        n_pos = int(torch.normal(mean=torch.tensor(seq_len / 2), std=torch.tensor(std_dev)).round().clamp(0, seq_len).item())
        n_neg = seq_len - n_pos

        median = self.medians[dim]
        y_dim = self.Y[:, dim]
        
        pos_mask = y_dim >= median
        neg_mask = ~pos_mask

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        if weighted:
            pos_weights = y_dim[pos_mask] - median
            neg_weights = median - y_dim[neg_mask]
            pos_sample_indices = pos_indices[torch.multinomial(pos_weights, n_pos, replacement=False)]
            neg_sample_indices = neg_indices[torch.multinomial(neg_weights, n_neg, replacement=False)]
        else:
            pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
            neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]

        indices = torch.cat([pos_sample_indices, neg_sample_indices])
        indices = indices[torch.randperm(len(indices))]

        X_episode = self.X[indices]
        Y_episode = (self.Y[indices, dim] >= self.medians[dim]).float()

        if not fixed_label and torch.rand(1).item() < 0.5: Y_episode = 1 - Y_episode
        return X_episode, Y_episode


class SimpleFunctionLearning(FunctionLearning):
    "A dataset for learning simple binary classification functions using sklearn, generating them on the fly."
    def __init__(self, n_samples: int = 10000, n_features: int = 5, scale: bool = True, random_state: int = 42):
        "Generate a base dataset for sampling episodes from dynamically created functions."
        self.n_samples = n_samples
        self.n_features = n_features
        self.base_random_state = random_state
        
        # Create a dummy dataset just to initialize the base class
        X_dummy, y_dummy = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=random_state
        )
        
        X = torch.tensor(X_dummy, dtype=torch.float32)
        Y = torch.tensor(y_dummy.reshape(-1, 1), dtype=torch.float32)  # Single dummy function
        
        super().__init__(X, Y, scale)
        
        # Override num_functions to be effectively infinite
        self.num_functions = float('inf')
        
    def _generate_function(self, dim: int):
        "Generate a classification function on the fly using the dim as seed."
        X_func, y_func = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_features,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=self.base_random_state + dim
        )
        return torch.from_numpy(X_func).to(torch.float32), torch.from_numpy(y_func).to(torch.float32)
        
    def sample_episode(self, dim: int, seq_len: int, fixed_label: bool = False, weighted: bool = False):
        "Sample an episode by generating a function on the fly."
        # Generate the function for this specific dim
        X_func, y_func = self._generate_function(dim.item())
        
        # Apply scaling if it was used during initialization
        if self.mean is not None and self.std is not None:
            X_func = (X_func - self.mean) / (self.std + 1e-8)
        
        std_dev = seq_len / 20.0
        n_pos = int(torch.normal(mean=torch.tensor(seq_len / 2), std=torch.tensor(std_dev)).round().clamp(0, seq_len).item())
        n_neg = seq_len - n_pos

        pos_mask = y_func == 1
        neg_mask = y_func == 0

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
        neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]

        indices = torch.cat([pos_sample_indices, neg_sample_indices])
        indices = indices[torch.randperm(len(indices))]

        X_episode = X_func[indices]
        Y_episode = y_func[indices]

        if not fixed_label and torch.rand(1).item() < 0.5: Y_episode = 1 - Y_episode
        return X_episode, Y_episode