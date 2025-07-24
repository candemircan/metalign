from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from metarep.data import Coco, ImageDataset, Things, ThingsFunctionLearning, h5_to_numpy, image_transform, prepare_things_spose


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


def test_coco_dataset(tmp_path: Path):
    root = tmp_path / "coco"
    root.mkdir()
    
    Image.new("RGB", (100, 100), color="red").save(root / "image1.jpg")
    Image.new("RGB", (100, 100), color="green").save(root / "image2.jpg")
    
    dataset = Coco(root=root, total_images=2)
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
    
    result = h5_to_numpy("test_model", data_root=data_root, min_nonzero=1)
    
    assert result.shape[0] == 2
    assert result[0, 0] == 1.0
    assert result[0, 2] == 2.0
    assert result[0, 4] == 3.0
    assert result[1, 1] == 4.0
    assert result[1, 3] == 5.0


def test_prepare_things_spose(tmp_path: Path, monkeypatch):
    from metarep import data
    monkeypatch.setattr(data, "NUM_THINGS_CATEGORIES", 2)
    
    data_root = tmp_path / "external"
    data_root.mkdir()
    
    things_dir = data_root / "THINGS"
    things_dir.mkdir()
    
    cat1 = things_dir / "category1"
    cat2 = things_dir / "category2"
    cat1.mkdir()
    cat2.mkdir()
    
    Image.new("RGB", (100, 100)).save(cat1 / "img1.jpg")
    Image.new("RGB", (100, 100)).save(cat1 / "img2.jpg")  
    Image.new("RGB", (100, 100)).save(cat2 / "img1.jpg")
    
    with open(data_root / "unique_id.txt", "w") as f:
        f.write("category1\ncategory2\n")
    
    np.savetxt(data_root / "spose_embedding_66d_sorted.txt", 
               np.random.rand(2, 66))
    
    representations = {
        "cls": np.random.rand(3, 10).astype(np.float32),
        "patch": np.random.rand(3, 20).astype(np.float32)
    }
    
    X, Y = prepare_things_spose(representations, data_root=data_root, tokens="cls")
    assert X.shape == (2, 10)
    assert Y.shape == (2, 66)
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    
    X_all, Y_all = prepare_things_spose(representations, data_root=data_root, tokens="all")
    assert X_all.shape == (2, 30)


def test_things_function_learning(tmp_path: Path, monkeypatch):
    from metarep import data
    monkeypatch.setattr(data, "NUM_THINGS_CATEGORIES", 2)
    
    data_root = tmp_path / "external" 
    data_root.mkdir()
    
    things_dir = data_root / "THINGS"
    things_dir.mkdir()
    
    cat1 = things_dir / "category1"
    cat2 = things_dir / "category2" 
    cat1.mkdir()
    cat2.mkdir()
    Image.new("RGB", (100, 100)).save(cat1 / "img1.jpg")
    Image.new("RGB", (100, 100)).save(cat2 / "img1.jpg")
    
    with open(data_root / "unique_id.txt", "w") as f:
        f.write("category1\ncategory2\n")
    
    np.savetxt(data_root / "spose_embedding_66d_sorted.txt", 
               np.random.rand(2, 66))
    
    representations = {"cls": np.random.rand(2, 10).astype(np.float32)}
    
    dataset = ThingsFunctionLearning(representations, data_root=data_root)
    assert len(dataset) == 2
    assert dataset.feature_dim == 10
    assert dataset.num_functions == 66
    
    X_ep, Y_ep = dataset.sample_episode(dim=0, seq_len=2)
    assert X_ep.shape[1] == 10
    assert Y_ep.shape[0] == X_ep.shape[0]
    assert 0 <= Y_ep.shape[0] <= 2
