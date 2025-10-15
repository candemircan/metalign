"""
common datasets and processing utils  used throughout the project
"""

__all__ = ["ImageDataset", "Things", "Coco", "h5_to_np", "FunctionStaticDataset", "FunctionDataset", "prepare_things_spose", "load_backbone", "Levels", "prepare_levels", "OpenImagesTrain", "OpenImagesTest", "DATASET_MAKERS"]

import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import NUM_COCO_TRAIN_IMAGES, NUM_THINGS_CATEGORIES, NUM_THINGS_IMAGES


def h5_to_np(features_path: Path, # path to the h5 file with the features
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

    Uses torchvision transforms to process PIL images.
    """

    def __init__(self, root: Path, glob_pattern: str = "*.jpg", total_images: int | None = None, transform=None):
        self.images = sorted(root.glob(glob_pattern))
        self.transform = transform

        if total_images is not None:
            assert len(self.images) == total_images, f"Expected {total_images} images, found {len(self.images)} in {root} with pattern {glob_pattern}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path) as image:
            image = image.convert('RGB')
            if self.transform is not None: return self.transform(image)
            return image

class Things(ImageDataset):
    """
    The THINGS dataset, which contains images of objects.

    The images are expected to be in the format `data/external/THINGS/{category}/{image}.jpg`, which matches the original structure of the THINGS dataset.
    """

    def __init__(self, root: Path = Path("data/external/THINGS"), total_images: int = NUM_THINGS_IMAGES, transform=None):
        super().__init__(root=root, glob_pattern="*/*.jpg", total_images=total_images, transform=transform)

class OpenImagesTrain(ImageDataset):
    """
    The OpenImages training dataset, which contains a large number of images.

    The images are expected to be in the format `data/external/openimages/train/{image}.jpg`, which matches the original structure of the OpenImages dataset.
    """

    def __init__(self, root: Path = Path("data/external/openimages/train"), transform=None):
        super().__init__(root=root, glob_pattern="*.jpg", transform=transform)

class OpenImagesTest(ImageDataset):
    """
    The OpenImages test dataset, which contains a large number of images.

    The images are expected to be in the format `data/external/openimages/test/{image}.jpg`, which matches the original structure of the OpenImages dataset.
    """

    def __init__(self, root: Path = Path("data/external/openimages/test"), transform=None):
        super().__init__(root=root, glob_pattern="*.jpg", transform=transform)

class Coco(ImageDataset):
    """
    The COCO dataset, which contains images of objects.

    The images are expected to be in the format `data/external/coco/{split}/{image}.jpg`, which matches the original structure of the COCO dataset.
    """

    def __init__(self, root: Path = Path("data/external/coco"), train: bool = True, transform=None):
        if train:
            super().__init__(root=root / "train2017", glob_pattern="*.jpg", total_images=NUM_COCO_TRAIN_IMAGES, transform=transform)
        else:
            # for eval, manually set images from val and test directories
            self.transform = transform
            val_images = sorted((root / "val2017").glob("*.jpg"))
            test_images = sorted((root / "test2017").glob("*.jpg"))
            self.images = val_images + test_images

class BOLD5000(ImageDataset):
    """
    Images used in BOLD5000.

    They are under `data/external/BOLD5000/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli`.

    Under that root folder, the iamges are at `COCO/*jpg`, `ImageNet/*JPEG`, and `Scene/*jpg`.
    """
    def __init__(self, root: Path =  Path("data/external/BOLD5000/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli"), transform=None):

        self.transform = transform
        coco = sorted((root / "COCO").glob("*jpg"))
        imagenet = sorted((root / "ImageNet").glob("*JPEG"))
        scenes = sorted((root / "scenes").glob("jpg"))
        self.images = sorted(coco + imagenet + scenes)

class Levels(ImageDataset):
    """
    This is an odd-one-out similarity dataset that uses a subset of the ImageNet dataset.
    The class will identify the relevant images from data/external/levels.pkl and load them from the local imagenet directory.
    """

    def __init__(self, root: Path = Path("data/external"), transform=None):
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
        all_image_paths = train_root.glob("*/*.JPEG")
        
        # filter to only include images that match our required image names
        self.images = []
        self.image_keys = []
        for img_path in all_image_paths:
            if img_path.stem in unique_image_names:
                self.images.append(img_path)
                self.image_keys.append(str(img_path).split("/")[-1])
        
        # sort images and keys together based on keys to ensure consistent ordering
        sorted_pairs = sorted(zip(self.image_keys, self.images))
        self.image_keys, self.images = [list(t) for t in zip(*sorted_pairs)]
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path) as image:
            image = image.convert('RGB')
            if self.transform is not None: return self.transform(image)
            return image


class FunctionStaticDataset(Dataset):
    "static SAE functions for given features with lazy loading, safe for multiprocessing."
    def __init__(self, inputs_path: Path, features_path: Path, min_nonzero: int = 120, valid_columns: np.ndarray = None):
        self.inputs_path = inputs_path
        self.features_path = features_path
        self.min_nonzero = min_nonzero
        
        # load metadata
        with h5py.File(inputs_path, 'r') as f:
            if 'representations' in f:
                self.n_samples = f['representations'].shape[0]
                self.feature_dim = f['representations'].shape[1]
            else:
                self.n_samples = len([k for k in f.keys() if k.isdigit()])
                self.feature_dim = f['0'].shape[0]
        
        # determine valid columns
        if valid_columns is not None:
            self.valid_columns = valid_columns
        else:
            self.valid_columns = self._get_valid_columns()
        
        self.num_functions = len(self.valid_columns)
        
        # file handles - will be initialized per worker
        self._inputs_h5 = None
        self._features_h5 = None
        
        # cache for lazy loading (optional, can help with repeated access)
        self._X_cache = None
        self._Y_cache = None
    
    def _get_valid_columns(self):
        "determine which columns have at least min_nonzero non-zero activations"
        with h5py.File(self.features_path, 'r') as f:
            first_key = str(0)
            if first_key in f and isinstance(f[first_key], h5py.Group):
                # sparse format
                index_counts = {}
                for i in range(self.n_samples):
                    if str(i) in f:
                        indices = f[str(i)]['indices'][:]
                        for idx in indices:
                            index_counts[idx] = index_counts.get(idx, 0) + 1
                return np.array([idx for idx, cnt in sorted(index_counts.items()) if cnt >= self.min_nonzero])
            else:
                # dense format
                if 'representations' in f:
                    data = f['representations'][:]
                else:
                    data = np.array([f[str(i)][:] for i in range(self.n_samples)])
                non_zero_counts = np.count_nonzero(data, axis=0)
                return np.where(non_zero_counts >= self.min_nonzero)[0]
    
    def _ensure_h5_handles(self):
        "open h5 files if not already open"
        if self._inputs_h5 is None:
            self._inputs_h5 = h5py.File(self.inputs_path, 'r')
        if self._features_h5 is None:
            self._features_h5 = h5py.File(self.features_path, 'r')
    
    def _load_all_data(self):
        "load all data into memory (called on first access in worker)"
        if self._X_cache is not None:
            return
        
        self._ensure_h5_handles()
        
        # load inputs
        if 'representations' in self._inputs_h5:
            X = torch.from_numpy(self._inputs_h5['representations'][:])
        else:
            X = torch.stack([torch.from_numpy(self._inputs_h5[str(i)][:]) for i in range(self.n_samples)])
        
        # load features
        first_key = str(0)
        if first_key in self._features_h5 and isinstance(self._features_h5[first_key], h5py.Group):
            # sparse format - convert to dense
            max_idx = self.valid_columns.max() if len(self.valid_columns) > 0 else 0
            Y_full = np.zeros((self.n_samples, max_idx + 1), dtype=np.float32)
            for i in range(self.n_samples):
                if str(i) in self._features_h5:
                    indices = self._features_h5[str(i)]['indices'][:].astype(np.int64)
                    activations = self._features_h5[str(i)]['activations'][:]
                    Y_full[i, indices] = activations
            Y = torch.from_numpy(Y_full[:, self.valid_columns])
        else:
            # dense format
            if 'representations' in self._features_h5:
                Y = torch.from_numpy(self._features_h5['representations'][:, self.valid_columns])
            else:
                Y_full = np.array([self._features_h5[str(i)][:] for i in range(self.n_samples)])
                Y = torch.from_numpy(Y_full[:, self.valid_columns])
        
        # binarize outputs: positive = non-zero, negative = zero
        Y = (Y != 0).float()
        
        self._X_cache = X
        self._Y_cache = Y
    
    def __del__(self):
        "ensure h5 files are closed"
        if self._inputs_h5 is not None:
            self._inputs_h5.close()
        if self._features_h5 is not None:
            self._features_h5.close()

    def __len__(self): 
        return self.n_samples

    def __getitem__(self, idx): 
        self._load_all_data()  # load once per worker
        return self._X_cache[idx], self._Y_cache[idx]
    
    @property
    def Y(self):
        "property for backward compatibility - loads all data"
        self._load_all_data()
        return self._Y_cache


class FunctionDataset(Dataset):
    "episode-based dataset for given features with lazy loading from h5, safe for DDP/multiprocessing."
    def __init__(self, inputs_path: Path, features_path: Path,
                 seq_len: int = 120, min_nonzero: int = 120, train_dims: list = None, epoch_size: int = None):
        self.inputs_path = inputs_path
        self.features_path = features_path
        self.seq_len = seq_len
        self.min_nonzero = min_nonzero
        self.epoch_size = epoch_size
        
        # only load metadata, not actual data
        with h5py.File(inputs_path, 'r') as f:
            if 'representations' in f:
                self.n_samples = f['representations'].shape[0]
                self.feature_dim = f['representations'].shape[1]
            else:
                self.n_samples = len([k for k in f.keys() if k.isdigit()])
                self.feature_dim = f['0'].shape[0]
        
        with h5py.File(features_path, 'r') as f:
            first_key = str(0)
            if first_key in f and isinstance(f[first_key], h5py.Group):
                # sparse format - need to check max index to determine dimension
                max_idx = 0
                for i in range(min(100, self.n_samples)):
                    if str(i) in f:
                        indices = f[str(i)]['indices'][:]
                        max_idx = max(max_idx, indices.max() if len(indices) > 0 else 0)
                self.num_functions = max_idx + 1
                self.is_sparse = True
            else:
                # dense format
                if 'representations' in f:
                    self.num_functions = f['representations'].shape[1]
                else:
                    self.num_functions = f['0'].shape[0]
                self.is_sparse = False
        
        # for filtering columns by min_nonzero, we need to scan once
        self.valid_columns = self._get_valid_columns()
        self.num_functions = len(self.valid_columns)
        
        # pre-compute medians for dense features
        if not self.is_sparse:
            self.medians = self._compute_medians()
            
        self.train_dims = train_dims if train_dims is not None else list(range(self.num_functions))
        
        # file handles - will be initialized per worker
        self._inputs_h5 = None
        self._features_h5 = None
    
    def _get_valid_columns(self):
        "determine which columns have at least min_nonzero non-zero activations"
        with h5py.File(self.features_path, 'r') as f:
            if self.is_sparse:
                # for sparse format, count occurrences of each index
                index_counts = {}
                for i in range(self.n_samples):
                    if str(i) in f:
                        indices = f[str(i)]['indices'][:]
                        for idx in indices:
                            index_counts[idx] = index_counts.get(idx, 0) + 1
                return np.array([idx for idx, cnt in index_counts.items() if cnt >= self.min_nonzero])
            else:
                # for dense format, load all and count non-zeros
                if 'representations' in f:
                    data = f['representations'][:]
                else:
                    data = np.array([f[str(i)][:] for i in range(self.n_samples)])
                non_zero_counts = np.count_nonzero(data, axis=0)
                return np.where(non_zero_counts >= self.min_nonzero)[0]
    
    def _compute_medians(self):
        "compute medians for dense features for threshold-based sampling"
        with h5py.File(self.features_path, 'r') as f:
            if 'representations' in f:
                data = f['representations'][:, self.valid_columns]
            else:
                data = np.array([f[str(i)][:] for i in range(self.n_samples)])[:, self.valid_columns]
        return torch.from_numpy(np.median(data, axis=0))
    
    def _ensure_h5_handles(self):
        "open h5 files if not already open (called in worker processes)"
        if self._inputs_h5 is None:
            self._inputs_h5 = h5py.File(self.inputs_path, 'r')
        if self._features_h5 is None:
            self._features_h5 = h5py.File(self.features_path, 'r')
    
    def __del__(self):
        "ensure h5 files are closed when dataset is deleted"
        if self._inputs_h5 is not None:
            self._inputs_h5.close()
        if self._features_h5 is not None:
            self._features_h5.close()
    
    def __len__(self): 
        return self.epoch_size if self.epoch_size is not None else 2**31 - 1

    
    def __getitem__(self, idx):
        self._ensure_h5_handles()
        # randomly sample a function dimension
        dim = self.train_dims[torch.randint(0, len(self.train_dims), (1,)).item()]
        return self._sample_episode(dim)
        
    def _sample_episode(self, dim: int):
        "sample an episode for a given function dimension with lazy loading"
        self._ensure_h5_handles()
        
        actual_dim = self.valid_columns[dim]
        
        # load features for this dimension lazily
        if self.is_sparse:
            y_dim = np.zeros(self.n_samples, dtype=np.float32)
            for i in range(self.n_samples):
                if str(i) in self._features_h5:
                    indices = self._features_h5[str(i)]['indices'][:]
                    if actual_dim in indices:
                        idx_pos = np.where(indices == actual_dim)[0][0]
                        y_dim[i] = self._features_h5[str(i)]['activations'][idx_pos]
            y_dim = torch.from_numpy(y_dim)
        else:
            # load dense features for this dimension
            if 'representations' in self._features_h5:
                y_dim = torch.from_numpy(self._features_h5['representations'][:, actual_dim])
            else:
                y_dim = torch.from_numpy(np.array([self._features_h5[str(i)][actual_dim] for i in range(self.n_samples)]))
        
        std_dev = self.seq_len / 20.0
        n_pos = int(torch.normal(mean=torch.tensor(self.seq_len / 2), std=torch.tensor(std_dev)).round().clamp(0, self.seq_len).item())
        n_neg = self.seq_len - n_pos

        if self.is_sparse:
            pos_mask = y_dim != 0
            neg_mask = y_dim == 0
        else:
            median = self.medians[dim]
            pos_mask = y_dim > median
            neg_mask = y_dim <= median

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        n_pos = min(n_pos, len(pos_indices))
        n_neg = min(n_neg, len(neg_indices))
        
        pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
        neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]

        indices = torch.cat([pos_sample_indices, neg_sample_indices])
        indices = indices[torch.randperm(len(indices)).long()]

        # h5py requires sorted indices for fancy indexing
        sorted_indices, sort_order = torch.sort(indices)

        # load inputs for sampled indices lazily
        if 'representations' in self._inputs_h5:
            X_episode = torch.from_numpy(self._inputs_h5['representations'][sorted_indices.numpy()])
        else:
            X_episode = torch.stack([torch.from_numpy(self._inputs_h5[str(i.item())][:]) for i in sorted_indices])
        
        # restore original random order
        X_episode = X_episode[torch.argsort(sort_order)]
        
        if self.is_sparse: 
            Y_episode = (y_dim[indices] != 0).float()
        else: 
            Y_episode = (y_dim[indices] > median).float()

        if torch.rand(1).item() < 0.5: 
            Y_episode = 1 - Y_episode
        
        return X_episode, Y_episode
    


def prepare_things_spose(
    representations: np.ndarray, # directly loaded from `load_backbone(h5_path)`
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


def load_backbone(file_path: str) -> np.ndarray:
    """
    some backbone reps are saved under the key 'representations', while others (e.g. SAE raw format) have individual datasets with numeric keys. this function handles both cases.
    """
    with h5py.File(file_path, 'r') as f:
        if 'representations' in f: return f['representations'][:]
        else:
            # handle SAE raw format: data stored as individual datasets with numeric keys
            keys = sorted([int(k) for k in f.keys() if k.isdigit()])
            reps = []
            for key in keys:
                reps.append(f[str(key)][:])
            return np.array(reps)


def prepare_levels(
    representations: np.ndarray, # directly loaded backbone representations
    data_root: Path = Path("data/external"), # the root directory containing levels.pkl
    return_tensors: str = "pt", # the type of tensors to return, can be "pt" for PyTorch tensors or "np" for NumPy arrays
) -> tuple[torch.Tensor, list] | tuple[np.ndarray, list]: # the backbone representations for all unique images, and the list of trials with image indices and ground truth
    """
    Prepare the levels odd-one-out dataset by matching backbone representations to trials.
    
    Assumes representations are extracted in the same order as the Levels dataset class would provide them.
    The Levels class sorts images by their keys, so we need to map trial images to the correct indices.
    
    Returns:
        representations: The backbone representations for all unique images  
        trials: List of trial dictionaries with image indices and ground truth
    """
    with open(data_root / "levels.pkl", "rb") as f:
        levels = pickle.load(f)
    
    # collect all unique image names and create the same ordering as Levels class
    image_names = []
    for participant_data in levels.values():
        for trial in participant_data:
            image_names.extend([trial["image1Path"], trial["image2Path"], trial["image3Path"]])
    
    # get unique image keys (without .JPEG extension) and sort them
    unique_image_keys = sorted(set(img.split(".")[0] for img in image_names))
    
    # create mapping from image key to representation index
    image_key_to_idx = {key: idx for idx, key in enumerate(unique_image_keys)}
    
    # the representations should be in the same order as unique_image_keys
    X = representations[:len(unique_image_keys)]
    
    # prepare trials with indices and ground truth
    trials = []
    for participant_data in levels.values():
        for trial in participant_data:
            if trial.get('exp_trial_type') == 'exp_trial':  # only experimental trials
                img1_key = trial["image1Path"].split(".")[0]
                img2_key = trial["image2Path"].split(".")[0]
                img3_key = trial["image3Path"].split(".")[0]
                selected_key = trial["selected_image"].split(".")[0]
                
                if all(key in image_key_to_idx for key in [img1_key, img2_key, img3_key]):
                    img_indices = [
                        image_key_to_idx[img1_key],
                        image_key_to_idx[img2_key], 
                        image_key_to_idx[img3_key]
                    ]
                    
                    # find which position the selected image is in
                    if selected_key == img1_key: selected_idx = 0
                    elif selected_key == img2_key: selected_idx = 1
                    elif selected_key == img3_key: selected_idx = 2
                    else: continue  # skip if selected image not in triplet
                    
                    trials.append({
                        'images': img_indices,
                        'selected': selected_idx,
                        'triplet_type': trial.get('triplet_type')
                    })
    
    if return_tensors == "pt": X = torch.from_numpy(X)
    elif return_tensors == "np": pass
    else: raise ValueError(f"Unknown return_tensors type: {return_tensors}. Must be 'pt' or 'np'.")
    
    return X, trials


DATASET_MAKERS = {
    'things':   lambda **kwargs: Things(transform=kwargs['transform']),
    'coco':     lambda **kwargs: Coco(train=kwargs['split']=='train', transform=kwargs['transform']),
    'levels':   lambda **kwargs: Levels(transform=kwargs['transform']),
    'openimages_train': lambda **kwargs: OpenImagesTrain(transform=kwargs['transform']),
    'openimages_test':  lambda **kwargs: OpenImagesTest(transform=kwargs['transform']),
    "bold5000": lambda **kwargs: BOLD5000(transform=kwargs['transform'])
}


if __name__ == "__main__":
    import tempfile

    from torchvision import transforms

    def _test_transform():
        """Simple transform for testing"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda img: img.convert('RGB') if img.mode != 'RGB' else img,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup_things_root(tmp_path: Path) -> Path:
        """Create a dummy THINGS dataset directory."""
        root = tmp_path / "THINGS"
        category1 = root / "category1"
        category2 = root / "category2"
        category1.mkdir(parents=True)
        category2.mkdir()

        # Create dummy images
        Image.new("RGB", (100, 100), color="red").save(category1 / "image1.jpg")
        Image.new("RGB", (100, 100), color="green").save(category1 / "image2.jpg")
        Image.new("RGB", (100, 100), color="blue").save(category2 / "image3.jpg")
        
        return root

    def test_things_dataset(things_root: Path):
        dataset = Things(root=things_root, total_images=3, transform=_test_transform())
        
        assert len(dataset) == 3
        
        image = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)

        image_2 = dataset[2]
        assert isinstance(image_2, torch.Tensor)
        assert image_2.shape == (3, 224, 224)

    def test_things_dataset_with_custom_transform(things_root: Path):
        
        # Create a custom transform that resizes to a different size
        custom_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        dataset = Things(root=things_root, total_images=3, transform=custom_transform)
        
        assert len(dataset) == 3
        
        image = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 128, 128)  # Should be the custom size

    def test_things_dataset_no_transform(things_root: Path):
        """Test Things dataset with no transform - should return PIL images."""
        dataset = Things(root=things_root, total_images=3, transform=None)
        
        assert len(dataset) == 3
        
        result = dataset[0]  # Should return PIL image when no transform is provided
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)  # Original image size

    def test_coco_dataset(tmp_path: Path):
        
        root = tmp_path / "coco"
        train_root = root / "train2017"
        train_root.mkdir(parents=True)
        
        Image.new("RGB", (100, 100), color="red").save(train_root / "image1.jpg")
        Image.new("RGB", (100, 100), color="green").save(train_root / "image2.jpg")
        
        dataset = Coco(root=root, train=True, transform=_test_transform())
        assert len(dataset) == 2
        
        image = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)

    def test_image_dataset(tmp_path: Path):
        root = tmp_path / "images"
        root.mkdir()
        
        Image.new("RGB", (50, 50), color="blue").save(root / "test.jpg")
        Image.new("RGB", (50, 50), color="yellow").save(root / "test2.jpg")
        
        dataset = ImageDataset(root=root, glob_pattern="*.jpg", total_images=2, transform=_test_transform())
        assert len(dataset) == 2
        
        image = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)

    def test_test_transform():
        transform = _test_transform()
        
        rgb_img = Image.new("RGB", (300, 300), color="red")
        tensor = transform(rgb_img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        
        rgba_img = Image.new("RGBA", (300, 300), color="blue")
        tensor = transform(rgba_img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_h5_to_np_sparse_format(tmp_path: Path):
        """Test h5_to_np with sparse SAE format (original behavior)."""
        data_root = tmp_path / "sae"
        data_root.mkdir(exist_ok=True)
        
        test_file = data_root / "test_model.h5"
        
        with h5py.File(test_file, 'w') as f:
            f.create_group("0")
            f["0"].create_dataset("activations", data=[1.0, 2.0, 3.0])
            f["0"].create_dataset("indices", data=[0, 2, 4])
            
            f.create_group("1")
            f["1"].create_dataset("activations", data=[4.0, 5.0])
            f["1"].create_dataset("indices", data=[1, 3])
        
        result = h5_to_np(test_file, min_nonzero=1)
        
        assert result.shape[0] == 2
        assert result[0, 0] == 1.0
        assert result[0, 2] == 2.0
        assert result[0, 4] == 3.0
        assert result[1, 1] == 4.0
        assert result[1, 3] == 5.0

    def test_h5_to_np_dense_format(tmp_path: Path):
        """Test h5_to_np with dense raw activations format."""
        data_root = tmp_path / "backbone_reps"
        data_root.mkdir(exist_ok=True)
        
        test_file = data_root / "test_raw_model.h5"
        
        # Create dense activations
        dense_data = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ], dtype=np.float32)
        
        with h5py.File(test_file, 'w') as f:
            for i in range(3):
                f.create_dataset(str(i), data=dense_data[i])
        
        result = h5_to_np(test_file, min_nonzero=1)
        
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, dense_data)

    def test_h5_to_np_dense_format_with_filtering(tmp_path: Path):
        """Test h5_to_np dense format with min_nonzero filtering."""
        data_root = tmp_path / "backbone_reps"
        data_root.mkdir(exist_ok=True)
        
        test_file = data_root / "test_raw_filtered.h5"
        
        # Create dense data where some columns have zeros
        dense_data = np.array([
            [1.0, 0.0, 3.0, 4.0],
            [5.0, 0.0, 7.0, 8.0],
            [9.0, 0.0, 11.0, 0.0]  # last column has one zero
        ], dtype=np.float32)
        
        with h5py.File(test_file, 'w') as f:
            for i in range(3):
                f.create_dataset(str(i), data=dense_data[i])
        
        # Filter out columns with less than 3 non-zero values
        result = h5_to_np(test_file, min_nonzero=3)
        
        # Should keep columns 0 and 2 (both have 3 non-zero values)
        # Column 1 has 0 non-zero values, column 3 has 2 non-zero values
        expected = dense_data[:, [0, 2]]
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, expected)

    def test_function_static_dataset(tmp_path: Path):
        """Test FunctionStaticDataset with lazy loading."""
        # Create dummy inputs h5 file
        inputs_path = tmp_path / "inputs_static.h5"
        inputs = np.random.rand(50, 10).astype(np.float32)
        with h5py.File(inputs_path, 'w') as f:
            f.create_dataset('representations', data=inputs)
        
        # Create dummy h5 file with sparse activations
        features_path = tmp_path / "features_static.h5"
        with h5py.File(features_path, 'w') as f:
            for i in range(50):
                g = f.create_group(str(i))
                if i % 3 == 0:
                    g.create_dataset("activations", data=[1.5, 2.5])
                    g.create_dataset("indices", data=[0, 2])
                else:
                    g.create_dataset("activations", data=[])
                    g.create_dataset("indices", data=[])
        
        dataset = FunctionStaticDataset(
            inputs_path=inputs_path,
            features_path=features_path,
            min_nonzero=5
        )
        
        assert len(dataset) == 50
        assert dataset.feature_dim == 10
        assert dataset.num_functions > 0
        
        X, Y = dataset[0]
        assert X.shape[0] == 10
        assert Y.shape[0] == dataset.num_functions
        assert isinstance(X, torch.Tensor)
        assert isinstance(Y, torch.Tensor)
        
        # test Y property for backward compatibility
        Y_all = dataset.Y
        assert Y_all.shape == (50, dataset.num_functions)

    def test_function_dataset_sparse(tmp_path: Path):
        """Test FunctionDataset with sparse features (SAE format)."""
        # Create dummy inputs h5 file
        inputs_path = tmp_path / "inputs.h5"
        inputs = np.random.rand(100, 10).astype(np.float32)
        with h5py.File(inputs_path, 'w') as f:
            f.create_dataset('representations', data=inputs)
        
        # Create dummy h5 file with sparse activations
        features_path = tmp_path / "features_sparse.h5"
        with h5py.File(features_path, 'w') as f:
            for i in range(100):
                g = f.create_group(str(i))
                # Simulate sparse activations
                if i % 2 == 0:
                    g.create_dataset("activations", data=[1.0, 2.0])
                    g.create_dataset("indices", data=[i % 5, (i+1) % 5]) # Use modulo 5 to ensure columns get reused
                else:
                    g.create_dataset("activations", data=[])
                    g.create_dataset("indices", data=[])

        dataset = FunctionDataset(
            inputs_path=inputs_path,
            features_path=features_path,
            seq_len=20,
            epoch_size=50,
            min_nonzero=1
        )

        assert len(dataset) == 50
        assert dataset.feature_dim == 10
        assert dataset.num_functions > 0
        assert dataset.is_sparse  # should detect as sparse

        X_ep, Y_ep = dataset[0]
        assert X_ep.shape[0] <= 20
        assert X_ep.shape[1] == 10
        assert Y_ep.shape[0] == X_ep.shape[0]
        assert isinstance(X_ep, torch.Tensor)
        assert isinstance(Y_ep, torch.Tensor)

    def test_function_dataset_dense(tmp_path: Path):
        """Test FunctionDataset with dense features (raw format)."""
        # Create dummy inputs h5 file
        inputs_path = tmp_path / "inputs_dense.h5"
        inputs = np.random.rand(100, 10).astype(np.float32)
        with h5py.File(inputs_path, 'w') as f:
            f.create_dataset('representations', data=inputs)
        
        # Create dummy h5 file with dense activations
        features_path = tmp_path / "features_dense.h5"
        dense_data = np.random.rand(100, 8).astype(np.float32)  # all non-zero
        
        with h5py.File(features_path, 'w') as f:
            for i in range(100):
                f.create_dataset(str(i), data=dense_data[i])

        dataset = FunctionDataset(
            inputs_path=inputs_path,
            features_path=features_path,
            seq_len=20,
            epoch_size=50,
            min_nonzero=1
        )

        assert len(dataset) == 50
        assert dataset.feature_dim == 10
        assert dataset.num_functions == 8
        assert not dataset.is_sparse  # should detect as dense

        X_ep, Y_ep = dataset[0]
        assert X_ep.shape[0] <= 20
        assert X_ep.shape[1] == 10
        assert Y_ep.shape[0] == X_ep.shape[0]
        assert isinstance(X_ep, torch.Tensor)
        assert isinstance(Y_ep, torch.Tensor)

    def test_load_backbone_standard_format(tmp_path: Path):
        """Test loading backbone representations with standard 'representations' key."""
        backbone_path = tmp_path / "backbone_reps.h5"
        dummy_reps = np.random.rand(50, 768).astype(np.float32)
        
        with h5py.File(backbone_path, 'w') as f:
            f.create_dataset('representations', data=dummy_reps, compression='gzip')
        
        loaded_reps = load_backbone(str(backbone_path))
        
        assert np.array_equal(loaded_reps, dummy_reps)
        assert loaded_reps.shape == (50, 768)
        assert loaded_reps.dtype == np.float32

    def test_load_backbone_sae_raw_format(tmp_path: Path):
        """Test loading backbone representations with SAE raw format (numeric keys)."""
        backbone_path = tmp_path / "backbone_reps_raw.h5"
        dummy_reps = np.random.rand(3, 768).astype(np.float32)
        
        with h5py.File(backbone_path, 'w') as f:
            for i in range(3):
                f.create_dataset(str(i), data=dummy_reps[i])
        
        loaded_reps = load_backbone(str(backbone_path))
        
        assert np.array_equal(loaded_reps, dummy_reps)
        assert loaded_reps.shape == (3, 768)
        assert loaded_reps.dtype == np.float32

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        things_root = setup_things_root(tmp_path)
        test_things_dataset(things_root)
        test_things_dataset_with_custom_transform(things_root)
        test_things_dataset_no_transform(things_root)
        
        original_coco_train_images = NUM_COCO_TRAIN_IMAGES
        NUM_COCO_TRAIN_IMAGES = 2
        test_coco_dataset(tmp_path)
        NUM_COCO_TRAIN_IMAGES = original_coco_train_images
        
        test_image_dataset(tmp_path)
        test_test_transform()
        test_h5_to_np_sparse_format(tmp_path)
        test_h5_to_np_dense_format(tmp_path)
        test_h5_to_np_dense_format_with_filtering(tmp_path)
        test_function_static_dataset(tmp_path)
        test_function_dataset_sparse(tmp_path)
        test_function_dataset_dense(tmp_path)
        test_load_backbone_standard_format(tmp_path)
        test_load_backbone_sae_raw_format(tmp_path)
