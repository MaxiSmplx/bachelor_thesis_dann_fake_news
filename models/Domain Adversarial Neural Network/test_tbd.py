# check_loader.py

from data_loader import get_dataloader
import torch

def test_data_loader():
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


import os
from datetime import datetime
from config import LOG_DIR

def test_logging():
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    
    num_epochs = 10
    avg_c_loss = 0.5
    avg_d_loss = 0.4
    acc = 70.56

    for i in range(10):
        with open(f"{LOG_DIR}/logs_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt", "a") as f:
            f.write(f"[Epoch {i:2d}/{num_epochs}]\n"
                    f"  →Class-Loss: {avg_c_loss:.4f}\n"
                    f"  →Domain-Loss: {avg_d_loss:.4f}\n"
                    f"  →Accuracy: {acc:.2f}% \n\n")

if __name__ == "__main__":
    test_logging()
