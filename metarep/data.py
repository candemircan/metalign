"""
common datasets used throughout the project
"""

__all__ = ["Things"]

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from metarep.transforms import image_transform


class Things(Dataset):
    """
    A dataset for the THINGS dataset, which contains images of objects.
    
    The images are expected to be in the format `data/THINGS/{category}/{image}.jpg`, which matches the original structure of the THINGS dataset.
    
    By default, ImageNet style transformations are applied to the images, including resizing, cropping, normalization, and conversion to tensor.
    """

    def __init__(self, root: Path = Path("data/THINGS"), total_images: int = 26107):
        self.root = root
        self.images = sorted(root.glob("*/*.jpg"))
        self.transform = image_transform()

        assert len(self.images) == total_images, f"Expected {total_images} images, found {len(self.images)} in {self.root}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path) as image: image = self.transform(image)
        return image