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

    for domain_type, domain_content in GOOGLE_DRIVE_FINAL_IDS.items():
        for preprocessed_type, content_ids in domain_content.items():
            print(f"\n📦 Downloading Folder {domain_type}/{preprocessed_type}")
            if not os.path.isdir(f"{config['output']}/{domain_type}/{preprocessed_type}"):
                os.mkdir(f"{config['output']}/{domain_type}/{preprocessed_type}")
            for dataset_type, id in content_ids.items():
                print(f"\n📁 Downloading Dataset -> {dataset_type}")
                download_drive_file(id, f"{config['output']}/{domain_type}/{preprocessed_type}/{dataset_type}.parquet")

if __name__ == "__main__":
    """Download all pipeline output files from Google Drive."""
    get_files()