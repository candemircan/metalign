"""
common datasets and processing utils  used throughout the project
"""

__all__ = ["prepare_things_spose", "ImageDataset", "Things", "Coco", "h5_to_numpy", "image_transform", "ThingsEpisodeDataset", "SAEEpisodeDataset", "SimpleEpisodeDataset", "SAEActivationsCache"]

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


class SAEActivationsCache:
    "Efficient caching for SAE h5 file access with lazy loading."
    def __init__(self, model_name: str, data_root: Path = Path("data/sae"), min_nonzero: int = 1):
        self.file_path = data_root / f"{model_name}.h5"
        self.min_nonzero = min_nonzero
        self._activations = None
        self._valid_columns = None
        
    def _load_data(self):
        if self._activations is not None: return
        
        with h5py.File(self.file_path, 'r') as sae_h5:
            all_indices = []
            for k in sae_h5.keys():
                all_indices.extend(sae_h5[k]["indices"][:].tolist())
            
            num_cols = max(all_indices) + 1
            self._activations = np.zeros((len(sae_h5), num_cols), dtype=np.float32)
            
            for img in range(len(sae_h5)):
                img_id = str(img)
                h5_activations = sae_h5[img_id]["activations"][:]
                indices = sae_h5[img_id]["indices"][:]
                self._activations[img, indices] = h5_activations
        
        non_zero_counts = np.count_nonzero(self._activations, axis=0)
        self._valid_columns = non_zero_counts >= self.min_nonzero
        self._activations = self._activations[:, self._valid_columns]
    
    @property
    def activations(self):
        self._load_data()
        return self._activations



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


class ThingsEpisodeDataset(Dataset):
    "Episode-based dataset for THINGS, optimized for DataLoader usage."
    def __init__(self, representations: dict, data_root: Path = Path("data/external"), 
                 seq_len: int = 120, scale: bool = True, fixed_label: bool = False, 
                 weighted: bool = False, train_dims: list = None, epoch_size: int = 100000):
        X, Y = prepare_things_spose(representations, data_root=data_root)
        
        self.X, self.Y = X, Y
        self.feature_dim = self.X.shape[1]
        self.num_functions = self.Y.shape[1]
        self.seq_len = seq_len
        self.fixed_label = fixed_label
        self.weighted = weighted
        self.epoch_size = epoch_size
        
        if scale:
            self.mean = self.X.mean(dim=0, keepdim=True)
            self.std = self.X.std(dim=0, keepdim=True)
            self.X = (self.X - self.mean) / (self.std + 1e-8)
        else: self.mean, self.std = None, None
            
        self.medians = torch.median(self.Y, dim=0).values
        self.train_dims = train_dims if train_dims is not None else list(range(self.num_functions))
        
    def __len__(self): return self.epoch_size
    
    def __getitem__(self, idx):
        dim = self.train_dims[idx % len(self.train_dims)]
        return self._sample_episode(dim)
        
    def _sample_episode(self, dim: int):
        std_dev = self.seq_len / 20.0
        n_pos = int(torch.normal(mean=torch.tensor(self.seq_len / 2), std=torch.tensor(std_dev)).round().clamp(0, self.seq_len).item())
        n_neg = self.seq_len - n_pos

        median = self.medians[dim]
        y_dim = self.Y[:, dim]
        
        pos_mask = y_dim >= median
        neg_mask = ~pos_mask

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        if self.weighted:
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

        if not self.fixed_label and torch.rand(1).item() < 0.5: Y_episode = 1 - Y_episode
        return X_episode, Y_episode


class SAEEpisodeDataset(Dataset):
    "Episode-based dataset for SAE features, optimized for DataLoader usage."
    def __init__(self, inputs: np.ndarray, sae_features: str, data_root: Path = Path("data/sae"),
                 seq_len: int = 120, scale: bool = True, min_nonzero: int = 100,
                 fixed_label: bool = False, weighted: bool = False, train_dims: list = None, 
                 epoch_size: int = 100000):
        X = torch.tensor(inputs, dtype=torch.float32)
        
        # Use cached loading for SAE features
        self.sae_cache = SAEActivationsCache(sae_features, data_root=data_root, min_nonzero=min_nonzero)
        Y = torch.from_numpy(self.sae_cache.activations)
        
        self.X, self.Y = X, Y
        self.feature_dim = self.X.shape[1]
        self.num_functions = self.Y.shape[1]
        self.seq_len = seq_len
        self.fixed_label = fixed_label
        self.weighted = weighted
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

        if self.weighted:
            pos_weights = y_dim[pos_mask]
            pos_sample_indices = pos_indices[torch.multinomial(pos_weights, n_pos, replacement=False)]
        else:
            pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
        
        neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]

        indices = torch.cat([pos_sample_indices, neg_sample_indices])
        indices = indices[torch.randperm(len(indices))]

        X_episode = self.X[indices]
        Y_episode = (self.Y[indices, dim] != 0).float()

        if not self.fixed_label and torch.rand(1).item() < 0.5: Y_episode = 1 - Y_episode
        return X_episode, Y_episode


class SimpleEpisodeDataset(Dataset):
    "Episode-based dataset for simple functions, optimized for DataLoader usage."
    def __init__(self, n_samples: int = 10000, n_features: int = 5, scale: bool = True, 
                 random_state: int = 42, seq_len: int = 120, fixed_label: bool = False,
                 epoch_size: int = 100000):
        self.n_samples = n_samples
        self.n_features = n_features
        self.base_random_state = random_state
        self.seq_len = seq_len
        self.fixed_label = fixed_label
        self.epoch_size = epoch_size
        
        # Generate base data for scaling parameters
        X_dummy, _ = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=random_state
        )
        
        X_dummy = torch.tensor(X_dummy, dtype=torch.float32)
        
        if scale:
            self.mean = X_dummy.mean(dim=0, keepdim=True)
            self.std = X_dummy.std(dim=0, keepdim=True)
        else: self.mean, self.std = None, None
            
        self.feature_dim = n_features
        
    def __len__(self): return self.epoch_size
    
    def __getitem__(self, idx):
        # Use idx as the function dimension/seed
        return self._sample_episode(idx)
        
    def _sample_episode(self, dim: int):
        # Generate function on the fly
        X_func, y_func = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_features,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=self.base_random_state + dim
        )
        
        X_func = torch.tensor(X_func, dtype=torch.float32)
        y_func = torch.tensor(y_func, dtype=torch.float32)
        
        # Apply scaling
        if self.mean is not None and self.std is not None:
            X_func = (X_func - self.mean) / (self.std + 1e-8)
        
        std_dev = self.seq_len / 20.0
        n_pos = int(torch.normal(mean=torch.tensor(self.seq_len / 2), std=torch.tensor(std_dev)).round().clamp(0, self.seq_len).item())
        n_neg = self.seq_len - n_pos

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

        if not self.fixed_label and torch.rand(1).item() < 0.5: Y_episode = 1 - Y_episode
        return X_episode, Y_episode