"""
common datasets and processing utils  used throughout the project
"""

__all__ = ["ImageDataset", "Things", "Coco", "h5_to_numpy", "FunctionDataset", "prepare_things_spose", "load_backbone_representations", "Levels"]

import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import NUM_COCO_TRAIN_IMAGES, NUM_THINGS_CATEGORIES, NUM_THINGS_IMAGES


def h5_to_numpy(features_path: Path, # path to the h5 file with the features
                min_nonzero: int = 1, # minimum number of non-zero activations per column to keep it in the final array. The default is such that any column that is all 0s is removed.
                ) -> np.ndarray:
    """
    Convert features from h5 file to numpy array.

    Handles two formats:
    1. Sparse SAE format with activations and indices (original behavior)
    2. Dense raw activations format where each image is stored directly under numeric keys

    For sparse format, features are sparse over observations but high dimensional.
    Only non-zero features and their indices are stored in the h5 file.
    We convert to numpy array, filling in zeros for missing indices.
    If a column has less than `min_nonzero` non-zero features, it is removed.

    Sparse format:
    ```
    {
        "0": {
            "activations": [...], # the features for the first image
            "indices": [...] # the indices of the non-zero features
        },
        ...
    }
    ```

    Dense format:
    ```
    {
        "0": [...], # dense feature vector for first image
        "1": [...], # dense feature vector for second image
        ...
    }
    ```

    The ordering is based on e.g. `sorted("data/external/THINGS/*/*jpg)` or `sorted("data/external/coco/train2017/*.jpg")`, depending on the dataset.
    """

    with h5py.File(features_path, 'r') as features_h5:
        # check if this is sparse format (has activations/indices) or dense format
        first_key = str(0)
        if first_key in features_h5:
            first_item = features_h5[first_key]
            if isinstance(first_item, h5py.Group) and "activations" in first_item and "indices" in first_item:
                # sparse format - original behavior
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
            else:
                # dense format - raw activations stored directly
                keys = sorted([int(k) for k in features_h5.keys() if k.isdigit()])
                activations = np.array([features_h5[str(key)][:] for key in keys], dtype=np.float32)
        else:
            raise ValueError(f"No data found in h5 file {features_path}")

    # remove columns with less than min_nonzero non-zero activations
    if activations.shape[1] > 0:
        non_zero_counts = np.count_nonzero(activations, axis=0)
        activations = activations[:, non_zero_counts >= min_nonzero]
    
    return activations


class ImageDataset(Dataset):
    """
    A generic image dataset that can handle different directory structures.
    Images are sorted by name.

    You must provide either a processor (AutoImageProcessor from transformers) or a transform (torchvision transforms).
    If processor is provided, it will be used to process PIL images and return tensors.
    If transform is provided, it will be applied to PIL images.
    """

    def __init__(self, root: Path, glob_pattern: str = "*.jpg", total_images: int | None = None, processor=None, transform=None):
        assert (processor is None) != (transform is None), "Must provide exactly one of processor or transform"
        
        self.images = sorted(root.glob(glob_pattern))
        self.processor = processor
        self.transform = transform

        if total_images is not None:
            assert len(self.images) == total_images, f"Expected {total_images} images, found {len(self.images)} in {root} with pattern {glob_pattern}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path) as image:
            image = image.convert('RGB')
            if self.processor is not None:
                # Use processor (returns dict with tensors)
                return self.processor(images=image, return_tensors="pt")
            else:
                # Apply transforms and return tensor
                return self.transform(image)


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
            assert (processor is None) != (transform is None), "Must provide exactly one of processor or transform"
            val_images = sorted((root / "val2017").glob("*.jpg"))
            test_images = sorted((root / "test2017").glob("*.jpg"))
            self.images = val_images + test_images
            self.processor = processor
            self.transform = transform

        
class Levels(ImageDataset):
    """
    This is an odd-one-out similarity dataset that uses a subset of the ImageNet dataset.
    The class will identify the relevant images from data/external/levels.pkl and load them from the local imagenet directory.
    """

    def __init__(self, root: Path = Path("data/external"), processor=None, transform=None):
        with open(root / "levels.pkl", "rb") as f:
            levels = pickle.load(f)
        
        image_names = []
        for values in levels.values():
            for trial in values:
                image_names.append(trial["image1Path"])
                image_names.append(trial["image2Path"])
                image_names.append(trial["image3Path"])

        unique_image_names = {img.split(".")[0] for img in image_names}
        
        # find matching images in local imagenet directory
        imagenet_root = root / "imagenet"
        train_root = imagenet_root / "train"
        val_root = imagenet_root / "val"
        
        # collect all imagenet images from train and val directories
        all_image_paths = []
        if train_root.exists():
            all_image_paths.extend(train_root.glob("*/*.JPEG"))
        if val_root.exists():
            all_image_paths.extend(val_root.glob("*.JPEG"))
        
        # filter to only include images that match our required image names
        self.images = []
        self.image_keys = []
        for img_path in all_image_paths:
            img_key = img_path.stem.split("_")[0]  # extract the key part (e.g., "n01440764" from "n01440764_18.JPEG")
            if img_key in unique_image_names:
                self.images.append(img_path)
                self.image_keys.append(img_key)
        
        # sort images and keys together based on keys to ensure consistent ordering
        sorted_pairs = sorted(zip(self.image_keys, self.images))
        self.image_keys, self.images = [list(t) for t in zip(*sorted_pairs)]
        
        self.processor = processor
        self.transform = transform
        assert (processor is None) != (transform is None), "Must provide exactly one of processor or transform"

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path) as image:
            image = image.convert('RGB')
            if self.processor is not None:
                # Use processor (returns dict with tensors)
                return self.processor(images=image, return_tensors="pt")
            else:
                # Apply transforms and return tensor
                return self.transform(image)



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
        
        # detect if this is dense (raw) or sparse (SAE) features
        # for sparse features, most values are 0; for dense features, all values are non-zero
        sparsity = (self.Y == 0).float().mean()
        self.is_sparse = sparsity > 0.5  # if more than 50% are zeros, treat as sparse
        
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

        if self.is_sparse:
            # sparse features: positive = non-zero, negative = zero
            pos_mask = y_dim != 0
            neg_mask = ~pos_mask
        else:
            # dense features: positive = above median, negative = below median
            median_val = torch.median(y_dim)
            pos_mask = y_dim > median_val
            neg_mask = y_dim <= median_val

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]


        # normal sampling
        n_pos = min(n_pos, len(pos_indices))
        n_neg = min(n_neg, len(neg_indices))
        
        pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
        neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]

        indices = torch.cat([pos_sample_indices, neg_sample_indices])
        indices = indices[torch.randperm(len(indices)).long()]

        X_episode = self.X[indices]
        if self.is_sparse:
            Y_episode = (self.Y[indices, dim] != 0).float()
        else:
            Y_episode = (self.Y[indices, dim] > median_val).float()

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


def load_backbone_representations(file_path: str) -> np.ndarray:
    """
    some backbone reps are saved under the key 'representations', while others (e.g. SAE raw format) have individual datasets with numeric keys. this function handles both cases.
    """
    with h5py.File(file_path, 'r') as f:
        if 'representations' in f: 
            return f['representations'][:]
        else:
            # handle SAE raw format: data stored as individual datasets with numeric keys
            keys = sorted([int(k) for k in f.keys() if k.isdigit()])
            reps = []
            for key in keys:
                reps.append(f[str(key)][:])
            return np.array(reps)