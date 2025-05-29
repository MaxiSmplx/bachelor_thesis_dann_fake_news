import os
import numpy as np
import torch
from torch import nn, optim
from config import INPUT_DIM, NUM_CLASSES, NUM_DOMAINS, LEARNING_RATE, NUM_EPOCHS, CHECKPOINT_DIR, FEATURE_DIM
from model import DANN
from data_loader import get_dataloader
from datetime import datetime

def grl_lambda(iter_num: int, max_iter: int) -> float:
    """
    Compute the GRL coefficient lambda_p = 2/(1+exp(-10*p)) - 1
    where p = iter_num / max_iter.
    """
    p = float(iter_num) / max_iter
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

def train(
    batch_size: int = 64,
    feature_dim: int = FEATURE_DIM,
    lr: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
    save_dir: str = CHECKPOINT_DIR
):
    print("🚀 Starting training run...")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loader
    num_classes = NUM_CLASSES
    # num_domains = NUM_DOMAINS

    loader = get_dataloader(augmented=False, batch_size=batch_size, shuffle=True, num_workers=4)

    num_domains = len(loader.dataset.domain2idx) #TODO Remove

    print(f"Loaded dataset with {len(loader.dataset)} examples → {len(loader)} batches per epoch")

    # Model
    model = DANN(
        input_dim=INPUT_DIM,
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_domains=num_domains
    ).to(device)

    # Losses & optimizer
    class_criterion  = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    total_iters = num_epochs * len(loader)
    iter_num = 0

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n── Epoch {epoch}/{num_epochs} ──")

        model.train()
        running_class_loss = 0.0
        running_domain_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(loader):
            print(f"  → Processing batch {batch_idx+1}/{len(loader)}")

            input_ids, attention_mask, y_lab, y_dom = batch

            x = (input_ids, attention_mask)

            # schedule GRL lambda
            lambda_p = grl_lambda(iter_num, total_iters)
            print(f"     λ = {lambda_p:.4f}")

            iter_num += 1

            # forward
            class_logits, domain_logits = model(x, lambda_p)

            # compute losses
            loss_class  = class_criterion(class_logits, y_lab)
            loss_domain = domain_criterion(domain_logits, y_dom)
            loss = loss_class + loss_domain

            print(f"     Class-Loss: {loss_class.item():.4f} | Domain-Loss: {loss_domain.item():.4f}")


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

            print(f"     Accuracy: {batch_acc*100:.2f}%")

        avg_c_loss = running_class_loss / len(loader)
        avg_d_loss = running_domain_loss / len(loader)
        acc = correct / total * 100

        print(f"[Epoch {epoch:2d}/{num_epochs}] "
              f"ClassLoss: {avg_c_loss:.4f}  "
              f"DomLoss: {avg_d_loss:.4f}  "
              f"Acc: {acc:.2f}%")

        # save if its the best model
        if acc > best_acc:
            best_acc = acc
            model_name = f"dann_{datetime.now().strftime('%Y-%m-%d-%H')}_acc-{acc:.2f}"
            ckpt_path = os.path.join(save_dir, f"{model_name}.pt")
            torch.save(model.state_dict(), ckpt_path)

            print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")

if __name__ == "__main__":
    train()
