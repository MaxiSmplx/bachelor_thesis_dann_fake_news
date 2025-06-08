import os
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score
from config import (
    INPUT_DIM, 
    NUM_CLASSES, 
    NUM_DOMAINS, 
    LEARNING_RATE, 
    NUM_EPOCHS, 
    CHECKPOINT_DIR, 
    FEATURE_DIM, 
    LOG_DIR, 
    TENSORBOARD_DIR,
    BATCH_SIZE,
    TOKENIZER_NAME,
    GRL_LAMBDA_CEILING,
    GRL_WARMUP
)
from model import DANN
from data_loader import get_dataloader
from datetime import datetime, timedelta
from time import perf_counter
import argparse

def grl_lambda(iter_num: int, max_iter: int, ceiling: float = GRL_LAMBDA_CEILING, delay: float = GRL_WARMUP) -> float:
    p = iter_num / max_iter
    if p < delay:
        return 0.0
    p = (p - delay) / (1.0 - delay)
    raw = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    return ceiling * raw

def train(
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
    save_dir: str = CHECKPOINT_DIR,
    logging: bool = False,
    augmented: bool = False,
    balanced: bool = True,
):
    print("\n🚀 Starting training run...")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"Using device >> {device}")

    # Data Loader
    num_classes = NUM_CLASSES
    num_domains = NUM_DOMAINS
    input_dim = INPUT_DIM
    feature_dim = FEATURE_DIM

    train_loader, val_loader = get_dataloader(split="train", 
                            val_fraction=0.1, 
                            augmented=augmented,
                            balanced=balanced,
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4)

    print(f"Loaded Train dataset with {len(train_loader.dataset)} datapoints... \n"
          f"    • Configured batch size: {train_loader.batch_size} \n"
          f"    • {len(train_loader)} batches per epoch \n"
          f"    • Detected {(no_dom := len(train_loader.dataset.df['domain'].unique()))} domains \n"
          f"        • Ideal domain accuracy: {(1/no_dom)*100:.2f}% \n"
          f"    • Data Augmentation is {'enabled' if augmented else 'disabled'} \n"
          f"    • Domain and Class balancing is {'enabled' if balanced else 'disabled'} \n"
          f"    • Using Tokenizer {TOKENIZER_NAME} \n"
          f"Loaded Validation dataset with {len(val_loader.dataset)} datapoints... \n"
          f"    • {len(val_loader)} batches per epoch \n"
          f"Enabled Logging, to view Tensorboard logs call 'tensorboard --logdir={TENSORBOARD_DIR}'")

    # Model
    model = DANN(
        input_dim=input_dim,
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_domains=num_domains
    ).to(device)

    model_name_prefix = f"dann_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    log_file = f"{LOG_DIR}/logs_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"

    # Losses & optimizer
    class_criterion  = nn.BCEWithLogitsLoss()
    domain_criterion = nn.CrossEntropyLoss()

    transformer_params = list(model.feature_extractor.encoder.parameters())
    newly_init_params = (
        list(model.feature_extractor.feature.parameters())
        + list(model.label_predictor.parameters())
        + list(model.domain_classifier.parameters())
    )

    def decay_filter(p):
        return p.ndim >= 2 and "bias" not in p.name and "LayerNorm" not in p.name

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if decay_filter(p) else no_decay).append(p)

    optimizer = AdamW(
        [
            {"params": decay,     "lr": 2e-5, "weight_decay": 1e-2},
            {"params": no_decay,  "lr": 2e-5, "weight_decay": 0.0},
            {"params": newly_init_params, "lr": 1e-4, "weight_decay": 1e-2},
        ]
    )

    # Training loop
    total_iters = num_epochs * len(train_loader)
    iter_num = 0

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if logging:
        if not os.path.isdir(LOG_DIR):
            os.mkdir(LOG_DIR)
        if not os.path.isdir(TENSORBOARD_DIR):
            os.mkdir(TENSORBOARD_DIR)

        with open(log_file, "a") as f:
            f.write(f"=== Run Configuration === \n\n"
                    f"  Tokenizer: {TOKENIZER_NAME} \n"
                    f"  Data Augmentation is {'enabled' if augmented else 'disabled'} \n"
                    f"  Domain and Class balancing is {'enabled' if balanced else 'disabled'} \n"
                    f"  Batch Size: {batch_size} \n"
                    f"  Num Epochs: {num_epochs} \n"
                    f"  Optimizer: {optimizer.__class__.__name__} \n\n\n\n")
            
        writer = SummaryWriter(log_dir=f"{TENSORBOARD_DIR}/run_{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    best_val_acc = 0.0
    best_model_path = None

    es_patience = 3
    es_epochs_no_improve = 0

    epoch_times = []

    progress_treshold_train = max(1, len(train_loader) // 10)
    progress_treshold_val = max(1, len(val_loader) // 5)

    for epoch in range(1, num_epochs + 1):
        if device.type == "mps":
            torch.mps.empty_cache()

        start_epoch_time = perf_counter()
        print(f"\n── Epoch {epoch}/{num_epochs} ──")

        model.train()

        running_class_loss = np.empty(len(train_loader))
        running_domain_loss = np.empty(len(train_loader))
        correct = 0
        total = 0
        domain_correct = 0
        domain_total = 0

        avg_batch_times = 0.0 

        for batch_idx, batch in enumerate(train_loader):
            start_batch_time = perf_counter()

            input_ids, attention_mask, y_lab, y_dom = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y_lab = y_lab.float().unsqueeze(1).to(device)
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
            running_class_loss[batch_idx] = loss_class.item()
            running_domain_loss[batch_idx] = loss_domain.item()

            probs = torch.sigmoid(class_logits)
            preds = (probs > 0.5).long().squeeze(1)
            true_labels = y_lab.long().squeeze(1)
            correct += (preds == true_labels).sum().item()
            total += true_labels.size(0)

            domain_preds = domain_logits.argmax(dim=1)
            domain_correct += (domain_preds == y_dom).sum().item()
            domain_total += y_dom.size(0)

            batch_acc = (preds == true_labels).sum().item() / true_labels.size(0)

            batch_time = (perf_counter() - start_batch_time)
            avg_batch_times += batch_time
            
            # print 5 times (+ first and last)
            if (batch_idx+1) == 1 or (batch_idx+1) % progress_treshold_train == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  → [Training] Processing batch {batch_idx+1}/{len(train_loader)}")
                print(f"     λ = {lambda_p:.4f}")
                print(f"     Class-Loss: {loss_class.item():.4f} | Domain-Loss: {loss_domain.item():.4f}")
                print(f"     Accuracy: {batch_acc*100:.2f}%")
                print(
                    f"     Batch time: {batch_time:.1f} sec >> "
                    f"Time to finish epoch: {((len(train_loader) - (batch_idx+1)) * (avg_batch_times / (batch_idx+1) / 60)):.1f} min"
                )

        avg_c_loss = np.mean(running_class_loss)
        avg_d_loss = np.mean(running_domain_loss)
        acc = correct / total * 100
        domain_acc = domain_correct / domain_total * 100
            
        print(f"🎯Epoch {epoch}/{num_epochs} Training Summary: "
              f"Class-Loss: {avg_c_loss:.4f} | "
              f"Domain-Loss: {avg_d_loss:.4f} | "
              f"Accuracy: {acc:.2f}% | " 
              f"Domain Accuracy: {domain_acc:.2f}% \n")


        model.eval()
        val_class_losses = np.empty(len(val_loader))
        val_correct = 0
        val_total = 0
        val_domain_correct = 0
        val_domain_total = 0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_loader):
                input_ids, attention_mask, y_lab, y_dom = val_batch

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                y_lab = y_lab.float().unsqueeze(1).to(device)
                y_dom = y_dom.to(device)

                x = (input_ids, attention_mask)

                # For validation, freeze GRL
                class_logits, domain_logits = model(x, lambda_=0.0)

                loss_class = class_criterion(class_logits, y_lab)
                val_class_losses[val_batch_idx] = loss_class.item()

                probs = torch.sigmoid(class_logits)
                preds = (probs > 0.5).long().squeeze(1)
                true_labels = y_lab.long().squeeze(1)

                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(true_labels.cpu().numpy())

                val_correct += (preds == true_labels).sum().item()
                val_total += true_labels.size(0)

                domain_preds = domain_logits.argmax(dim=1)
                val_domain_correct += (domain_preds == y_dom.to(device)).sum().item()
                val_domain_total += y_dom.size(0)

                batch_acc = (preds == true_labels).sum().item() / y_lab.size(0)
                if (val_batch_idx+1) % progress_treshold_val == 0:
                    print(f"  → [Validation] Processing batch {val_batch_idx+1}/{len(val_loader)} ")
                    print(f"     Class-Loss: {loss_class.item():.4f}")
                    print(f"     Accuracy: {batch_acc*100:.2f}%")

        avg_val_c_loss = np.mean(val_class_losses)
        val_acc = val_correct / val_total * 100
        val_domain_acc = val_domain_correct / val_domain_total * 100

        val_all_preds = np.array(val_all_preds)
        val_all_labels = np.array(val_all_labels)
        f1 = f1_score(val_all_labels, val_all_preds, average='binary')
        precision = precision_score(val_all_labels, val_all_preds, average='binary')
        recall = recall_score(val_all_labels, val_all_preds, average='binary')

        print(f"🔍Epoch {epoch}/{num_epochs} Validation summary: "
              f"Class-Loss: {avg_val_c_loss:.4f} | "
              f"Accuracy: {val_acc:.2f}% | "
              f"Domain Accuracy: {val_domain_acc:.2f}%\n")


        epoch_times.append((perf_counter() - start_epoch_time) / 60)
        print(f"    • Epoch time: {epoch_times[-1]:.1f} min >> "
              f"Time to finish run: {(eta_run := ((num_epochs - epoch) * (np.mean(epoch_times)) / 60)):.2f} hrs "
              f"(ETA at {(datetime.now() + timedelta(hours=eta_run)).strftime('%d/%m/%Y %H:%M:%S')}) \n")


        if logging:
            with open(log_file, "a") as f:
                f.write(f"[Epoch {epoch:2d}/{num_epochs}]\n"
                        f"‣ Training\n"
                        f"  ‣ Class-Loss: {avg_c_loss:.4f}\n"
                        f"      ‣ Max. Class-Loss: {np.max(running_class_loss):.4f}\n"
                        f"      ‣ Min. Class-Loss: {np.min(running_class_loss):.4f}\n"
                        f"  ‣ Domain-Loss: {avg_d_loss:.4f}\n"
                        f"      ‣ Max. Domain-Loss: {np.max(running_domain_loss):.4f}\n"
                        f"      ‣ Min. Domain-Loss: {np.min(running_domain_loss):.4f}\n"
                        f"  ‣ Accuracy: {acc:.2f}%\n"
                        f"  ‣ Domain Accuracy: {domain_acc:.2f}% \n"
                        f"‣ Validation\n"
                        f"  ‣ Class-Loss: {avg_val_c_loss:.4f}\n"
                        f"      ‣ Max. Class-Loss: {np.max(val_class_losses):.4f}\n"
                        f"      ‣ Min. Class-Loss: {np.min(val_class_losses):.4f}\n"
                        f"  ‣ Classification Metrics \n"
                        f"      ‣ Accuracy: {val_acc:.2f}%\n"
                        f"      ‣ F1-Score: {f1 * 100:.2f}%\n"
                        f"      ‣ Precision: {precision * 100:.2f}%\n"
                        f"      ‣ Recall: {recall * 100:.2f}%\n"
                        f"‣ Epoch time: {epoch_times[-1]:.1f} min \n\n")
            
            writer.add_scalars("Loss", {
                "Train": avg_c_loss,
                "Validation": avg_val_c_loss
            }, epoch)
            writer.add_scalars("Accuracy", {
                "Train": acc,
                "Validation": val_acc
            }, epoch)
            writer.add_scalars("Domain", {
                "Loss": avg_d_loss,
                "Accuracy": domain_acc
            }, epoch)
            writer.add_scalars("Other Metrics", {
                "F1": f1,
                "Precision": precision,
                "Recall": recall
            }, epoch)

        # save if its the best model
        if val_acc > best_val_acc + 0.1:
            best_val_acc = val_acc
            es_epochs_no_improve = 0

            if best_model_path and os.path.isfile(best_model_path):
                os.remove(best_model_path)

            model_name = f"{model_name_prefix}_acc-{val_acc:.2f}"
            best_model_path = os.path.join(save_dir, f"{model_name}.pt")
            torch.save(model.state_dict(), best_model_path)

            print(f"💾 Saved checkpoint: {best_model_path}\n")
        else:
            es_epochs_no_improve += 1
            if es_epochs_no_improve >= es_patience:
                if logging:
                    with open(log_file, "a") as f:
                        f.write(f"Early stopping initiated")
                print(f"⏹ Early stopping after {epoch} epochs (no improvement).")
                break
    
    if logging:
        writer.close()
    print(f"Training complete. Best weights found achieved accuracy of {best_val_acc:.2f}%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DANN model")

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--augmented", action="store_true", help="Use augmented data")
    parser.add_argument("--balanced", action="store_true", help="Use balanced dataset")
    parser.add_argument("--log", action="store_true", help="Enable TensorBoard and logging")

    args = parser.parse_args()

    train(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        augmented=args.augmented,
        balanced=args.balanced,
        logging=args.log
    )