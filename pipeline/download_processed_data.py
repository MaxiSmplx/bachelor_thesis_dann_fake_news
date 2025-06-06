import gdown
import yaml
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import GOOGLE_DRIVE_FINAL_IDS

def download_drive_file(file_id: str, output_path: str) -> None:
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"Data {os.path.basename(output_path)} already exists in corresponding folder")


def get_files() -> None:
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    for folder_name, type_dict in GOOGLE_DRIVE_FINAL_IDS.items():
        print(f"\n📦 Downloading Folder {folder_name}")
        if not os.path.isdir(f"{config['output']}/{folder_name}"):
            os.mkdir(f"{config['output']}/{folder_name}")
        for type, id in type_dict.items():
            print(f"\n📁 Downloading Dataset -> {type}")
            download_drive_file(id, f"pipeline/{config['output']}/{folder_name}/{type}.parquet")


if __name__ == "__main__":
    """Download all pipeline output files from Google Drive."""
    get_files()