import pandas as pd
from config import (
    NUM_CLASSES,
    NUM_DOMAINS,
    INPUT_DIM,
    FEATURE_DIM
)
from model import DANN
from test import test
import torch
import os
from time import perf_counter
import argparse
import yaml

def get_data(cross_domain: bool, balanced: bool, augmented: bool):
    with open("pipeline/config.yml", "r") as f:
        config = yaml.safe_load(f)
    data_folder_name = {
        (False, False): 'raw',
        (True,  False): 'balanced',
        (False, True):  'augmented',
        (True,  True):  'balanced_augmented'
    }[(balanced, augmented)]
    data_folder_attribute = "cross_domain" if cross_domain else "in_domain"
    data_folder_path = os.path.join(f"pipeline/{config['output']}", data_folder_attribute, data_folder_name)
    return pd.read_parquet(f"{data_folder_path}/preprocessed_data_test.parquet")

def get_param_count(model):
    return sum(p.numel() for p in model.parameters())

def get_model_size(model):
    file_size_bytes = os.path.getsize(model)
    return file_size_bytes / (1024 ** 2) #in MB

def get_model_size_in_ram(model, dtype=torch.float32):
    param_count = get_param_count(model)
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    total_size_bytes = param_count * bytes_per_param
    return total_size_bytes / (1024 ** 2) #in MB

def measure_gpu_memory_and_inference(model_checkpoint: str, device, cross_domain: bool, balanced: bool, augmented: bool, test_data: pd.DataFrame):
    torch.cuda.reset_peak_memory_stats(device)
    
    start_inference = perf_counter()
    test(
        model_checkpoint=model_checkpoint,
        logging=False,
        cross_domain=cross_domain,
        balanced=balanced,
        augmented=augmented
    )
    inference_total = perf_counter() - start_inference

    inference_per_sample = inference_total / len(test_data)

    return {
        "mem_allocation": torch.cuda.max_memory_allocated(device) / (1024 ** 2), # MB
        "total_inference_time": inference_total,
        "inference_per_sample": inference_per_sample
    }

def get_attributes(model_checkpoint: str, cross_domain: bool, augmented: bool, balanced: bool):
    """Load a trained model checkpoint and report its attributes and performance stats.

    Parameters
    ----------
    model_checkpoint : str
        Name of the saved model checkpoint (without `.pt` extension).
    cross_domain : bool
        If True, evaluate using cross-domain data; else in-domain.
    augmented : bool
        Use augmented data if available.
    balanced : bool
        Use balanced data if available.

    Notes
    -----
    - Loads the model onto GPU (CUDA/MPS) if available, else CPU.
    - Prints parameter count, checkpoint size, in-RAM model size.
    - Runs inference on test data to measure peak GPU memory usage and inference time.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device >> {device}\n")

    model_path = f"models/Domain Adversarial Neural Network/checkpoints/{model_checkpoint}.pt"


    model = DANN(
    input_dim=INPUT_DIM,
    feature_dim=FEATURE_DIM,
    num_classes=NUM_CLASSES,
    num_domains=NUM_DOMAINS
    )   

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)

    test_data = get_data(cross_domain, balanced, augmented)

    inference_dict = measure_gpu_memory_and_inference(model_checkpoint, device, cross_domain, balanced, augmented, test_data)

    print("\n=== Model Attributes ===")
    print(f"{'Parameters':<25}: {get_param_count(model):,}")
    print(f"{'Model size (disk)':<25}: {get_model_size(model_path)} MB")
    print(f"{'Model size (RAM)':<25}: {get_model_size_in_ram(model)} MB")
    print(f"{'Peak GPU Memory':<25}: {inference_dict['mem_allocation']} MB")
    print(f"{'Inference time (total)':<25}: {inference_dict['total_inference_time']}")
    print(f"{'Inference time (per sample)':<25}: {inference_dict['inference_per_sample']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best DANN model")

    parser.add_argument("--model_checkpoint", type=str, help="Model checkpoint")
    parser.add_argument("--cross_domain", action="store_true", help="Train in cross-domain setting")
    parser.add_argument("--augmented", action="store_true", help="Use augmented data")
    parser.add_argument("--balanced", action="store_true", help="Use balanced dataset")

    args = parser.parse_args()

    get_attributes(
        model_checkpoint=args.model_checkpoint,
        cross_domain=args.cross_domain,
        augmented=args.augmented,
        balanced=args.balanced,
    )