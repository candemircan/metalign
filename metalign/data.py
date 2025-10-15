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
    "static SAE functions for given features. not used for meta-learning but for a supervised baseline."
    def __init__(self, inputs: np.ndarray, features_path: Path, min_nonzero: int = 120, valid_columns: np.ndarray = None):
        X = torch.tensor(inputs, dtype=torch.float32)
        
        # if valid_columns is provided, use it to filter columns consistently
        if valid_columns is not None:
            Y_full = torch.from_numpy(h5_to_np(features_path, min_nonzero=0))  # don't filter here
            Y = Y_full[:, valid_columns]
            self.valid_columns = valid_columns
        else:
            # original behavior: filter based on min_nonzero
            Y = torch.from_numpy(h5_to_np(features_path, min_nonzero=min_nonzero))
            # store which columns were kept for potential reuse
            Y_full = torch.from_numpy(h5_to_np(features_path, min_nonzero=0))
            non_zero_counts = (Y_full != 0).sum(dim=0)
            self.valid_columns = (non_zero_counts >= min_nonzero).numpy()
        
        self.X, self.Y = X, Y

        # for Y, we want to binarise the outputs
        # in this class, we assume all functions are sparse
        # so a positive is a non-zero value, and a negative is a zero value
        self.Y = (self.Y != 0).float()

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

class FunctionDataset(Dataset):
    "episode-based dataset for given features, optimized for DataLoader usage."
    def __init__(self, inputs: np.ndarray, features_path: Path,
                 seq_len: int = 120, min_nonzero: int = 120, train_dims: list = None, epoch_size: int = None):
        X = torch.tensor(inputs, dtype=torch.float32)
        
        Y = torch.from_numpy(h5_to_np(features_path, min_nonzero=min_nonzero))
        
        self.X, self.Y = X, Y
        self.feature_dim = self.X.shape[1]
        self.num_functions = self.Y.shape[1]
        self.seq_len = seq_len
        self.epoch_size = epoch_size
        
        # detect if this is dense (raw) or sparse (SAE) features
        # for sparse features, most values are 0
        # for dense features, all values are non-zero
        sparsity = (self.Y == 0).float().mean()

        # heuristic: if more than 50% are zeros, treat as sparse
        # with SAE feautres sparsity is way higher than 50%
        # and with raw features sparsity is ~ 0
        self.is_sparse = sparsity > 0.5  
        
        # pre-compute medians for dense features to avoid repeated computation
        if not self.is_sparse: self.medians = torch.median(self.Y, dim=0).values
            
        self.train_dims = train_dims if train_dims is not None else list(range(self.num_functions))
            
    def __len__(self): return self.epoch_size if self.epoch_size is not None else 2**31 - 1 # effectively infinite for training (when epoch is not set)

    
    def __getitem__(self, idx):
        # randomly sample a function dimension instead of cycling through them
        dim = self.train_dims[torch.randint(0, len(self.train_dims), (1,)).item()]
        return self._sample_episode(dim)
        
    def _sample_episode(self, dim: int):
        y_dim = self.Y[:, dim]
        
        std_dev = self.seq_len / 20.0 # with a sequence length of 120, this gives a decent but not too exeggerated of a spread
        n_pos = int(torch.normal(mean=torch.tensor(self.seq_len / 2), std=torch.tensor(std_dev)).round().clamp(0, self.seq_len).item())
        n_neg = self.seq_len - n_pos

        if self.is_sparse:
            # sparse features: positive = non-zero, negative = zero
            pos_mask = y_dim != 0
            neg_mask = ~pos_mask
        else:
            # dense features: positive = above median, negative = below median
            # use pre-computed median instead of computing each time
            median_val = self.medians[dim]
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
        if self.is_sparse: Y_episode = (self.Y[indices, dim] != 0).float()
        else: Y_episode = (self.Y[indices, dim] > median_val).float()

        if torch.rand(1).item() < 0.5: Y_episode = 1 - Y_episode # 0 and 1 are arbitrary, so flip with 50% chance at the episode level
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

    def test_function_dataset_sparse(tmp_path: Path):
        """Test FunctionDataset with sparse features (SAE format)."""
        # Create dummy inputs
        inputs = np.random.rand(100, 10).astype(np.float32)
        
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
            inputs=inputs,
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
        # Create dummy inputs
        inputs = np.random.rand(100, 10).astype(np.float32)
        
        # Create dummy h5 file with dense activations
        features_path = tmp_path / "features_dense.h5"
        dense_data = np.random.rand(100, 8).astype(np.float32)  # all non-zero
        
        with h5py.File(features_path, 'w') as f:
            for i in range(100):
                f.create_dataset(str(i), data=dense_data[i])

        dataset = FunctionDataset(
            inputs=inputs,
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
        test_function_dataset_sparse(tmp_path)
        test_function_dataset_dense(tmp_path)
        test_load_backbone_standard_format(tmp_path)
        test_load_backbone_sae_raw_format(tmp_path)
