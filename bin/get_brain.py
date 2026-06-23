import shutil
import zipfile
from pathlib import Path

import requests


def download_and_extract(file_id, target_dir, final_name, sub_folder_to_keep=None):
    """
    Downloads from Figshare, unzips, moves the useful folder, and cleans up.
    """
    base_url = f"https://api.figshare.com/v2/file/download/{file_id}"
    zip_path = Path(f"data/external/{final_name}.zip")
    temp_dir = Path(f"data/external/{final_name}_temp")
    
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"--- Downloading {final_name} (ID: {file_id}) ---")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(base_url, headers=headers, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Extracting {final_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    macosx_dir = temp_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

    source_folder = temp_dir / sub_folder_to_keep if sub_folder_to_keep else temp_dir
    
    if sub_folder_to_keep:
        target_final = Path("data/external/brain_data") / final_name
        if final_name == "brain_data":
            for item in source_folder.iterdir():
                shutil.move(str(item), str(Path("data/external/brain_data")))
        else:
            shutil.move(str(source_folder), str(target_final))
    
    shutil.rmtree(temp_dir)
    zip_path.unlink()
    print(f"Done with {final_name}.\n")

if __name__ == "__main__":
    base_path = Path("data/external/brain_data")

    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)

        download_and_extract("43635873", "brain_data", "brain_data", sub_folder_to_keep="betas_csv")
        download_and_extract("36682242", "brain_data", "masks", sub_folder_to_keep="brainmasks")
        download_and_extract("36693528", "brain_data", "surface", sub_folder_to_keep="pycortex_filestore")

        print("Finished all downloads and organization!")
    else:
        print("Data already exists. Skipping.")