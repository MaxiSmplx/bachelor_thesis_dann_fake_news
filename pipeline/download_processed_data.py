import gdown
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import GOOGLE_DRIVE_FINAL_IDS

def download_drive_file(file_id: str, output_path: str) -> None:
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"Data {os.path.basename(output_path)} already exists in corresponding folder")


def get_files(only_augmented: bool = False) -> None:
    GOOGLE_DRIVE_FINAL_IDS.pop("preprocessed_data", None) if only_augmented else None

    for data_name, id in GOOGLE_DRIVE_FINAL_IDS.items():
        print(f"\n📁 Downloading Dataset -> {data_name}")
        download_drive_file(id, f"pipeline/output/{data_name}.npz")


if __name__ == "__main__":
    """Download all pipeline output files from Google Drive."""
    get_files(only_augmented=False)
