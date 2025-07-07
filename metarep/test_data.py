from pathlib import Path

import pytest
import torch
from PIL import Image

from metarep.data import Things


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
    """Test the Things dataset."""
    dataset = Things(root=things_root, total_images=3)


    # Test __getitem__
    image = dataset[0]
    assert isinstance(image, torch.Tensor)
    
    # The default transform resizes to 224x224
    assert image.shape == (3, 224, 224)

    # Test another item
    image_2 = dataset[2]
    assert isinstance(image_2, torch.Tensor)
    assert image_2.shape == (3, 224, 224)
