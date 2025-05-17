import gdown
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import GOOGLE_DRIVE_IDS


def download_drive_file(file_id: str, output_path: str) -> None:
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"Dataset {os.path.basename(output_path)} already exists in corresponding folder")


def get_all_files() -> None:
    for dataset_name, dataset in GOOGLE_DRIVE_IDS.items():
        print(f"\n📁 Downloading Dataset -> {dataset_name}")
        for type, id in dataset.items():
            print(f"📄 Retrieving {type}.parquet")
            download_drive_file(id, f"datasets/{dataset_name}/{type}.parquet")


if __name__ == "__main__":
    """Download and save all real and fake dataset files from Google Drive."""
    get_all_files()
