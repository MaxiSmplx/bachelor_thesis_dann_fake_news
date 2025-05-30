import os
import numpy as np
import torch
from torch import nn
from adan_pytorch import Adan
from config import (
    INPUT_DIM, 
    NUM_CLASSES, 
    NUM_DOMAINS, 
    LEARNING_RATE, 
    NUM_EPOCHS, 
    CHECKPOINT_DIR, 
    FEATURE_DIM, 
    LOG_DIR, 
    BATCH_SIZE,
    TOKENIZER_NAME
)
from model import DANN
from data_loader import get_dataloader
from datetime import datetime
from time import perf_counter

def grl_lambda(iter_num: int, max_iter: int) -> float:
    p = float(iter_num) / max_iter
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

def train(
    batch_size: int = BATCH_SIZE,
    feature_dim: int = FEATURE_DIM,
    lr: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
    save_dir: str = CHECKPOINT_DIR,
    logging: bool = False,
    augmented: bool = False
):
    print("🚀 Starting training run...")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device >> {device}")

    # Data Loader
    num_classes = NUM_CLASSES
    # num_domains = NUM_DOMAINS

    loader = get_dataloader(augmented=augmented, batch_size=batch_size, shuffle=True, num_workers=4)

    num_domains = len(loader.dataset.domain2idx) #TODO Remove

    print(f"Loaded dataset with {len(loader.dataset)} datapoints... \n"
          f"    • Configured batch size: {loader.batch_size} \n"
          f"    • {len(loader)} batches per epoch \n"
          f"    • Detected {len(loader.dataset.df['domain'].unique())} domains \n"
          f"    • Data Augmentation is {'enabled' if augmented else 'disabled'} \n"
          f"    • Using Tokenizer {TOKENIZER_NAME}")

    # Model
    model = DANN(
        input_dim=INPUT_DIM,
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_domains=num_domains
    ).to(device)

    model_name_prefix = f"dann_{datetime.now().strftime('%Y-%m-%d-%H')}"
    log_file = f"{LOG_DIR}/logs_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"

    # Losses & optimizer
    class_criterion  = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = Adan(model.parameters(), lr=lr, weight_decay=5e-4)

    # Training loop
    total_iters = num_epochs * len(loader)
    iter_num = 0

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if logging:
        if not os.path.isdir(LOG_DIR):
            os.mkdir(LOG_DIR)

    best_acc = 0.0
    best_model_path = None

    epoch_times = []

    progress_treshold = max(1, len(loader) // 5)

    for epoch in range(1, num_epochs + 1):
        if device.type == "mps":
            torch.mps.empty_cache()

        start_epoch_time = perf_counter()
        print(f"\n── Epoch {epoch}/{num_epochs} ──")

        model.train()
        running_class_loss = 0.0
        running_domain_loss = 0.0
        correct = 0
        total = 0

        batch_time_avg = 0

        for batch_idx, batch in enumerate(loader):
            start_batch_time = perf_counter()

            input_ids, attention_mask, y_lab, y_dom = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y_lab = y_lab.to(device)
            y_dom = y_dom.to(device)

            x = (input_ids, attention_mask)

            # schedule GRL lambda
            lambda_p = grl_lambda(iter_num, total_iters)

            iter_num += 1

            # forward
            class_logits, domain_logits = model(x, lambda_p)

            # compute losses
            loss_class  = class_criterion(class_logits, y_lab)
            loss_domain = domain_criterion(domain_logits, y_dom)
            loss = loss_class + loss_domain

            # backward + step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track stats
            running_class_loss += loss_class.item()
            running_domain_loss += loss_domain.item()

            preds = class_logits.argmax(dim=1)
            correct += (preds == y_lab).sum().item()
            total += y_lab.size(0)

            batch_acc = (preds == y_lab).sum().item() / y_lab.size(0)

            elapsed_time_batch = (perf_counter() - start_batch_time)
            batch_time_avg += elapsed_time_batch

            # print 5 times (+ first and last)
            if (batch_idx+1) == 1 or (batch_idx+1) % progress_treshold == 0 or (batch_idx + 1) == len(loader):
                print(f"  → Processing batch {batch_idx+1}/{len(loader)}")
                print(f"     λ = {lambda_p:.4f}")
                print(f"     Class-Loss: {loss_class.item():.4f} | Domain-Loss: {loss_domain.item():.4f}")
                print(f"     Accuracy: {batch_acc*100:.2f}%")
                print(
                    f"     Batch time: {elapsed_time_batch:.0f} sec >> "
                    f"Time to finish epoch: {((len(loader) - (batch_idx+1)) * (batch_time_avg / (batch_idx+1) / 60)):.1f} min"
                )

        avg_c_loss = running_class_loss / len(loader)
        avg_d_loss = running_domain_loss / len(loader)
        acc = correct / total * 100

        epoch_times.append((perf_counter() - start_epoch_time) / 60)
            
        print(f"\n🎯Epoch {epoch}/{num_epochs} Summary: "
              f"Class-Loss: {avg_c_loss:.4f} | "
              f"Domain-Loss: {avg_d_loss:.4f} | "
              f"Accuracy: {acc:.2f}% \n"
              f"    • Epoch time: {epoch_times[-1]:.1f} min >> "
              f"Time to finish run: {((num_epochs - epoch) * (np.mean(epoch_times)) / 60):.2f} hrs")


        if logging:
            with open(log_file, "a") as f:
                f.write(f"[Epoch {epoch:2d}/{num_epochs}]\n"
                        f"  ‣ Class-Loss: {avg_c_loss:.4f}\n"
                        f"  ‣ Domain-Loss: {avg_d_loss:.4f}\n"
                        f"  ‣ Accuracy: {acc:.2f}% \n"
                        f"  ‣ Epoch time: {epoch_times[-1]:.1f} min \n\n")

        # save if its the best model
        if acc > best_acc:
            best_acc = acc
            if best_model_path and os.path.isfile(best_model_path):
                os.remove(best_model_path)

            model_name = f"{model_name_prefix}_acc-{acc:.2f}"
            best_model_path = os.path.join(save_dir, f"{model_name}.pt")
            torch.save(model.state_dict(), best_model_path)

            print(f"💾 Saved checkpoint: {best_model_path}")

    print("Training complete.")

if __name__ == "__main__":
    train(logging=True)
