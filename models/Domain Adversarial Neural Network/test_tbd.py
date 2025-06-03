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
    epoch = 4
    avg_c_loss = 0.5
    avg_d_loss = 0.4
    acc = 70.56
    log_file = f"{LOG_DIR}/logs_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt" 

    for i in range(10):
        with open(f"{LOG_DIR}/logs_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt", "a") as f:
            with open(log_file, "a") as f: #Doesnt work
                f.write(f"[Epoch {epoch:2d}/{num_epochs}]\n"
                        f"  ‣ Class-Loss: {avg_c_loss:.4f}\n"
                        f"  ‣ Domain-Loss: {avg_d_loss:.4f}\n"
                        f"  ‣ Accuracy: {acc:.2f}% \n\n")

def test_progress_print():
    elapsed_time_batch = 8.0       
    batch_idx = 4                   
    batch_time_avg = 40.0           
    loader = [None] * 20 

    print(
            f"     Batch time: {elapsed_time_batch:.0f} sec >> "
            f"Time to finish epoch: {((len(loader) - (batch_idx+1)) * (batch_time_avg / (batch_idx+1) / 60)):.1f} min"
        )
    
def test_epoch_print():
    import numpy as np

    epoch        = 3
    num_epochs   = 20
    avg_c_loss   = 0.6173
    avg_d_loss   = 2.8568
    acc          = 54.29
    epoch_times  = [1.1, 1.4, 1.3]  # in minutes, one entry per completed epoch
    
    print(f"🎯Epoch {epoch}/{num_epochs} Summary: "
          f"Class-Loss: {avg_c_loss:.4f} | "
          f"Domain-Loss: {avg_d_loss:.4f} | "
          f"Accuracy: {acc:.2f}% \n"
          f"    • Epoch time: {epoch_times[-1]:.1f} min >> "
          f"Time to finish run: {((num_epochs - epoch) * (np.mean(epoch_times)) / 60):.2f} hrs")
    
def test_num_workers():
    from time import perf_counter
    for nw in [0, 2, 4, 8]:
        loader = get_dataloader(batch_size=64, num_workers=nw)
        t0 = perf_counter()
        for _ in loader:
            pass
        print(f"num_workers={nw:<2} → {perf_counter() - t0:.2f}s")

def test_data_loader():
    for _ in range(10):
        loader = get_dataloader(batch_size=64)
    augmented = False

    print(f"Loaded dataset with {len(loader.dataset)} datapoints... \n"
          f"    • Configured batch size: {loader.batch_size} \n"
          f"    • {len(loader)} batches per epoch \n"
          f"    • Detected {len(loader.dataset.df['domain'].unique())} domains \n"
          f"    • Data Augmentation is {'enabled' if augmented else 'disabled'} \n")
    print(len(loader.dataset.domain2idx))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title("Simple Plot")
    plt.show()  # Required in .py files
