# check_loader.py

from data_loader import get_dataloader
import torch

def main():
    # Create a loader (uses your FILE_PATH or augmented—pick one)
    loader = get_dataloader(augmented=False, batch_size=4, shuffle=False, num_workers=0)

    # Grab one batch
    batch = next(iter(loader))
    input_ids, attention_mask, labels, domains = batch

    # Print shapes and dtypes
    print("input_ids      :", input_ids.shape, input_ids.dtype)
    print("attention_mask :", attention_mask.shape, attention_mask.dtype)
    print("labels         :", labels.shape, labels.dtype, "unique:", torch.unique(labels))
    print("domains        :", domains.shape, domains.dtype, "unique:", torch.unique(domains))

if __name__ == "__main__":
    main()
