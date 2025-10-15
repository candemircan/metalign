"""convert h5 files to memory-mapped npy format for faster loading"""
from pathlib import Path

import h5py
import numpy as np
from fastcore.script import call_parse


@call_parse
def main(
    h5_dir: str = "data/backbone_reps",  # directory containing h5 files
    output_dir: str = None,  # output directory for npy files. if None, saves next to h5 files
    pattern: str = "*.h5",  # glob pattern for h5 files to convert
    overwrite: bool = False,  # overwrite existing npy files
):
    """
    convert h5 files to npy format for faster memory-mapped loading.
    
    npy files are much faster for random access than h5 files, especially with multiprocessing.
    the npy files will be saved with the same name but .npy extension.
    """
    h5_path = Path(h5_dir)
    out_path = Path(output_dir) if output_dir else h5_path
    if output_dir: out_path.mkdir(exist_ok=True, parents=True)
    
    h5_files = sorted(h5_path.glob(pattern))
    print(f"found {len(h5_files)} h5 files in {h5_dir}")
    
    for h5_file in h5_files:
        npy_file = out_path / f"{h5_file.stem}.npy"
        
        if npy_file.exists() and not overwrite:
            print(f"skipping {h5_file.name} (npy already exists)")
            continue
            
        print(f"converting {h5_file.name}...", end=" ", flush=True)
        
        with h5py.File(h5_file, 'r') as f:
            if 'representations' in f:
                data = f['representations'][:]
            else:
                # numeric keys format (SAE raw or individual datasets)
                n_samples = len([k for k in f.keys() if k.isdigit()])
                data = np.array([f[str(i)][:] for i in range(n_samples)])
        
        np.save(npy_file, data)
        size_mb = npy_file.stat().st_size / (1024**2)
        print(f"✓ {data.shape} {data.dtype} ({size_mb:.1f}MB)")
    
    print(f"\nconverted {len(h5_files)} files to {out_path}")
