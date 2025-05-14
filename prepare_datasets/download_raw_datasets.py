import gdown
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import DATASETS, GOOGLE_DRIVE_IDS_RAW


def download_drive_file(file_id: str, output_path: str) -> None:
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"Dataset {os.path.basename(output_path)} already exists in corresponding folder")


def get_all_raw_files() -> None:
    for folder in DATASETS:
        print(f"\n📁Downloading Dataset -> {folder}")
        download_drive_file(GOOGLE_DRIVE_IDS_RAW[folder], f"prepare_datasets/{folder}/{folder}.parquet")


if __name__ == "__main__":
    get_all_raw_files()
