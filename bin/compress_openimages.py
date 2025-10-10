from pathlib import Path

import h5py
import numpy as np
from fastcore.script import call_parse
from PIL import Image
from tqdm import tqdm


@call_parse
def main(
    image_dir: Path, # directory containing JPG images
):
    h5_path = image_dir / "images.h5"
    imgs = sorted(image_dir.glob("*.jpg"))
    print(f"Creating HDF5 file at {h5_path} with {len(imgs)} images.")
    with h5py.File(h5_path, 'w') as f:
        for img_path in tqdm(imgs):
            img = Image.open(img_path)
            img_array = np.array(img)
            dataset_name = img_path.stem
            f.create_dataset(
                dataset_name, 
                data=img_array,
                compression="gzip",
                compression_opts=9 
            )
            
            f[dataset_name].attrs['original_filename'] = img_path.name

    for img_path in image_dir.glob("*.jpg"):
        img_path.unlink()