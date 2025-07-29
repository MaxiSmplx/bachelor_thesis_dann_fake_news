import pandas as pd
from LLM_finetuned_test import test
import torch
import os
from time import perf_counter
from transformers import BertForSequenceClassification
from transformers import RobertaForSequenceClassification
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

def get_model_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 ** 2)  # MB

def get_model_size_in_ram(model, dtype=torch.float32):
    param_count = get_param_count(model)
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    total_size_bytes = param_count * bytes_per_param
    return total_size_bytes / (1024 ** 2) #in MB

def measure_gpu_memory_and_inference(model_checkpoint: str, device, cross_domain: bool, balanced: bool, augmented: bool, test_data: pd.DataFrame):
    torch.cuda.reset_peak_memory_stats(device)
    
    start_inference = perf_counter()
    test(
        model_path=model_checkpoint,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device >> {device}\n")

    model_path = f"models/LLM/models/{model_checkpoint}"
    
    print(f"Detected model architecture: {model_checkpoint.split('_', 1)[0]}")
    if model_checkpoint.split('_', 1)[0] == "BERT":
        model = BertForSequenceClassification.from_pretrained(model_path)
    elif model_checkpoint.split('_', 1)[0] == "RoBERTa":
        model = RobertaForSequenceClassification.from_pretrained(model_path)
    else:
        print("ERROR! Model neither BERT nor RoBERTa")

    test_data = get_data(cross_domain, balanced, augmented)


    print(f"Parameters: {get_param_count(model)}")
    print(f"Model size: {get_model_size(model_path)} MB")
    print(f"Model size in RAM: {get_model_size_in_ram(model)} MB")
    inference_dict = measure_gpu_memory_and_inference(model_checkpoint, device, cross_domain, balanced, augmented, test_data)
    print(f"Peak Memory consumption: {inference_dict['mem_allocation']} MB")
    print(f"Inference time: total -> {inference_dict['total_inference_time']}, per sample -> {inference_dict['inference_per_sample']}")


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