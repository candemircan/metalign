"""
common datasets and processing utils  used throughout the project
"""

__all__ = ["ImageDataset", "Things", "Coco", "h5_to_numpy", "image_transform","FunctionDataset", "prepare_things_spose"]

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from PIL import Image
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



def h5_to_numpy(features_path: Path, # path to the h5 file with the features
                min_nonzero: int = 1, # minimum number of non-zero activations per column to keep it in the final array. The default is such that any column that is all 0s is removed.
                ) -> np.ndarray:
    """
    Convert features from h5 file to numpy array.

    Features are sparse over observations, but the space is very high dimensional.
    To save disk space, only non-zero features and their indices are stored in the h5 file.
    Here, we convert the h5 file to a numpy array, filling in the zeros for the missing indices.
    If a column has less than `min_nonzero` non-zero features, it is removed from the final array.

    The original h5 file is expected to be in the format:
    ```
    {
        "0": {
            "activations": [...], # the features for the first image
            "indices": [...] # the indices of the non-zero features
        },
        "1": {
            "activations": [...], # the features for the second image
            "indices": [...] # the indices of the non-zero features
        },
        ...
    }
    ```

    The ordering is based on `sorted("data/external/THINGS/*/*jpg)` or `sorted("data/external/coco/train2017/*.jpg")`, depending on the dataset.
    """

    with h5py.File(features_path, 'r') as features_h5:
        indices = []
        for k in features_h5.keys():
            indices.extend(features_h5[k]["indices"][:].tolist())

        num_cols = max(indices) + 1 if indices else 0
        activations = np.zeros((len(features_h5), num_cols), dtype=np.float32)

        for img in range(len(features_h5)):
            img_id = str(img)
            h5_activations = features_h5[img_id]["activations"][:]
            indices = features_h5[img_id]["indices"][:]
            filler_activations = np.zeros(num_cols, dtype=np.float32)
            if indices.size > 0:
                filler_activations[indices.astype(np.int64)] = h5_activations
            activations[img, :] = filler_activations

    # remove columns with less than min_nonzero non-zero activations
    if activations.shape[1] > 0:
        non_zero_counts = np.count_nonzero(activations, axis=0)
        activations = activations[:, non_zero_counts >= min_nonzero]
    
    return activations


class ImageDataset(Dataset):
    """
    A generic image dataset that can handle different directory structures.
    Images are sorted by name.

    By default, ImageNet style transformations are applied to the images, including resizing, cropping, normalization, and conversion to tensor.
    If a processor is provided, it will be used instead of transforms. If a custom transform is provided, it will override the default.
    """

    def __init__(self, root: Path, glob_pattern: str = "*.jpg", total_images: int | None = None, processor=None, transform=None):
        self.images = sorted(root.glob(glob_pattern))
        self.processor = processor
        self.transform = transform if transform is not None else image_transform()

        if total_images is not None:
            assert len(self.images) == total_images, f"Expected {total_images} images, found {len(self.images)} in {root} with pattern {glob_pattern}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path) as image:
            if self.processor is not None:
                # Return PIL image for processor to handle
                return image.convert('RGB')
            else:
                # Apply transforms and return tensor
                image = self.transform(image)
        return image


class Things(ImageDataset):
    """
    A dataset for the THINGS dataset, which contains images of objects.

    The images are expected to be in the format `data/external/THINGS/{category}/{image}.jpg`, which matches the original structure of the THINGS dataset.
    """

    def __init__(self, root: Path = Path("data/external/THINGS"), total_images: int = NUM_THINGS_IMAGES, processor=None, transform=None):
        super().__init__(root=root, glob_pattern="*/*.jpg", total_images=total_images, processor=processor, transform=transform)

class Coco(ImageDataset):
    """
    A dataset for the COCO dataset, which contains images of objects.

    The images are expected to be in the format `data/external/coco/{split}/{image}.jpg`, which matches the original structure of the COCO dataset.
    """

    def __init__(self, root: Path = Path("data/external/coco"), train: bool = True, processor=None, transform=None):
        if train:
            super().__init__(root=root / "train2017", glob_pattern="*.jpg", total_images=NUM_COCO_TRAIN_IMAGES, processor=processor, transform=transform)
        else:
            # For eval, manually set images from val and test directories
            val_images = sorted((root / "val2017").glob("*.jpg"))
            test_images = sorted((root / "test2017").glob("*.jpg"))
            self.images = val_images + test_images
            self.processor = processor
            self.transform = transform if transform is not None else image_transform()


class FunctionDataset(Dataset):
    "Episode-based dataset for given features, optimized for DataLoader usage."
    def __init__(self, inputs: np.ndarray, features_path: Path,
                 seq_len: int = 120, scale: bool = False, min_nonzero: int = 120, train_dims: list = None, 
                 epoch_size: int = 100000):
        X = torch.tensor(inputs, dtype=torch.float32)
        
        Y = torch.from_numpy(h5_to_numpy(features_path, min_nonzero=min_nonzero))
        
        self.X, self.Y = X, Y
        self.feature_dim = self.X.shape[1]
        self.num_functions = self.Y.shape[1]
        self.seq_len = seq_len
        self.epoch_size = epoch_size
        
        if scale:
            self.mean = self.X.mean(dim=0, keepdim=True)
            self.std = self.X.std(dim=0, keepdim=True)
            self.X = (self.X - self.mean) / (self.std + 1e-8)
        else: self.mean, self.std = None, None
            
        self.train_dims = train_dims if train_dims is not None else list(range(self.num_functions))
        
    def __len__(self): return self.epoch_size
    
    def __getitem__(self, idx):
        dim = self.train_dims[idx % len(self.train_dims)]
        return self._sample_episode(dim)
        
    def _sample_episode(self, dim: int):
        y_dim = self.Y[:, dim]
        
        std_dev = self.seq_len / 20.0
        n_pos = int(torch.normal(mean=torch.tensor(self.seq_len / 2), std=torch.tensor(std_dev)).round().clamp(0, self.seq_len).item())
        n_neg = self.seq_len - n_pos

        pos_mask = y_dim != 0
        neg_mask = ~pos_mask

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
        neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]

        indices = torch.cat([pos_sample_indices, neg_sample_indices])
        indices = indices[torch.randperm(len(indices)).long()]

        X_episode = self.X[indices]
        Y_episode = (self.Y[indices, dim] != 0).float()

        if torch.rand(1).item() < 0.5: Y_episode = 1 - Y_episode
        return X_episode, Y_episode
    


def prepare_things_spose(
    representations: np.ndarray, # directly loaded from h5py.File(path, 'r')['representations'][:]
    data_root: Path = Path("data/external"), # the root directory of the THINGS dataset, which contains the images and the unique_id.txt file
    return_tensors: str = "pt", # the type of tensors to return, can be "pt" for PyTorch tensors or "np" for NumPy arrays
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

    X = representations[img_idx]
    Y = np.loadtxt(data_root / "spose_embedding_66d_sorted.txt").astype(np.float32)

    if return_tensors == "pt": X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    elif return_tensors == "np": pass
    else: raise ValueError(f"Unknown return_tensors type: {return_tensors}. Must be 'pt' or 'np'.")
    
    return X, Y