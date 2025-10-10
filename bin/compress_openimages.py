from pathlib import Path

from fastcore.script import call_parse

from metalign.data import images_to_h5


@call_parse
def main(
    image_dir: Path, # directory containing JPG images
):
    h5_path = image_dir / "images.h5"
    images_to_h5(image_dir, h5_path)
    for img_path in image_dir.glob("*.jpg"):
        img_path.unlink()