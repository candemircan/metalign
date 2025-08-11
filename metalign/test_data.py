from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from metalign.data import Coco, FunctionDataset, ImageDataset, Things, h5_to_numpy, image_transform


@pytest.fixture
def things_root(tmp_path: Path) -> Path:
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
    dataset = Things(root=things_root, total_images=3)
    
    assert len(dataset) == 3
    
    image = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)

    image_2 = dataset[2]
    assert isinstance(image_2, torch.Tensor)
    assert image_2.shape == (3, 224, 224)


def test_things_dataset_with_processor(things_root: Path):
    # Mock processor that just returns PIL images
    class MockProcessor:
        def __call__(self, images, return_tensors=None):
            return {"pixel_values": torch.stack([torch.randn(3, 224, 224) for _ in images])}
    
    mock_processor = MockProcessor()
    dataset = Things(root=things_root, total_images=3, processor=mock_processor)
    
    assert len(dataset) == 3
    
    image = dataset[0]  # Should return PIL Image when processor is used
    assert isinstance(image, Image.Image)


def test_things_dataset_with_custom_transform(things_root: Path):
    from torchvision import transforms
    
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


def test_coco_dataset(tmp_path: Path, monkeypatch):
    from metalign import data
    monkeypatch.setattr(data, "NUM_COCO_TRAIN_IMAGES", 2)
    root = tmp_path / "coco"
    train_root = root / "train2017"
    train_root.mkdir(parents=True)
    
    Image.new("RGB", (100, 100), color="red").save(train_root / "image1.jpg")
    Image.new("RGB", (100, 100), color="green").save(train_root / "image2.jpg")
    
    dataset = Coco(root=root, train=True)
    assert len(dataset) == 2
    
    image = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)


def test_image_dataset(tmp_path: Path):
    root = tmp_path / "images"
    root.mkdir()
    
    Image.new("RGB", (50, 50), color="blue").save(root / "test.jpg")
    Image.new("RGB", (50, 50), color="yellow").save(root / "test2.jpg")
    
    dataset = ImageDataset(root=root, glob_pattern="*.jpg", total_images=2)
    assert len(dataset) == 2
    
    image = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)


def test_image_transform():
    transform = image_transform()
    
    rgb_img = Image.new("RGB", (300, 300), color="red")
    tensor = transform(rgb_img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    
    rgba_img = Image.new("RGBA", (300, 300), color="blue")
    tensor = transform(rgba_img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_h5_to_numpy(tmp_path: Path):
    data_root = tmp_path / "sae"
    data_root.mkdir()
    
    test_file = data_root / "test_model.h5"
    
    with h5py.File(test_file, 'w') as f:
        f.create_group("0")
        f["0"].create_dataset("activations", data=[1.0, 2.0, 3.0])
        f["0"].create_dataset("indices", data=[0, 2, 4])
        
        f.create_group("1")
        f["1"].create_dataset("activations", data=[4.0, 5.0])
        f["1"].create_dataset("indices", data=[1, 3])
    
    result = h5_to_numpy(test_file, min_nonzero=1)
    
    assert result.shape[0] == 2
    assert result[0, 0] == 1.0
    assert result[0, 2] == 2.0
    assert result[0, 4] == 3.0
    assert result[1, 1] == 4.0
    assert result[1, 3] == 5.0


def test_function_dataset(tmp_path: Path):
    # Create dummy inputs
    inputs = np.random.rand(100, 10).astype(np.float32)
    
    # Create dummy h5 file
    features_path = tmp_path / "features.h5"
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

    X_ep, Y_ep = dataset[0]
    assert X_ep.shape[0] <= 20
    assert X_ep.shape[1] == 10
    assert Y_ep.shape[0] == X_ep.shape[0]
    assert isinstance(X_ep, torch.Tensor)
    assert isinstance(Y_ep, torch.Tensor)


def test_backbone_representations_loading(tmp_path: Path):
    # Create dummy backbone representations file
    backbone_path = tmp_path / "backbone_reps.h5"
    dummy_reps = np.random.rand(50, 768).astype(np.float32)
    
    with h5py.File(backbone_path, 'w') as f:
        f.create_dataset('representations', data=dummy_reps, compression='gzip')
    
    # Test loading
    with h5py.File(backbone_path, 'r') as f:
        loaded_reps = f['representations'][:]
    
    assert np.array_equal(loaded_reps, dummy_reps)
    assert loaded_reps.shape == (50, 768)
    assert loaded_reps.dtype == np.float32
