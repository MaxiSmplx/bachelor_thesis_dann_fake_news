from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from data_loader import get_dataloader
from model import DANN
from time import perf_counter
import numpy as np
import torch
import argparse
import os
from config import (
    NUM_CLASSES,
    NUM_DOMAINS,
    INPUT_DIM,
    FEATURE_DIM,
    LOG_DIR
)

def test(model_checkpoint: str , logging: bool = True, cross_domain: bool = True, balanced: bool = False, augmented: bool = True):
    print(f"\n🚀 Starting evaluation...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device >> {device}\n")

    test_loader = get_dataloader("test", cross_domain=cross_domain, balanced=balanced, augmented=augmented, batch_size=48)
    print(f"Loaded Test Data with {len(test_loader.dataset)} datapoints")
    print(f"Testing on domain(s): {test_loader.dataset.df['domain'].unique().tolist()}") if cross_domain else print("Testing on all {NUM_DOMAINS} domains")

    num_classes = NUM_CLASSES
    num_domains = NUM_DOMAINS
    input_dim = INPUT_DIM
    feature_dim = FEATURE_DIM

    print(f"Loading model '{model_checkpoint}' from checkpoint...\n")
    model = DANN(
        input_dim=input_dim,
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_domains=num_domains
    )
    model.load_state_dict(
        torch.load(f"models/Domain Adversarial Neural Network/checkpoints/{model_checkpoint}.pt", map_location=device)
    )
    model.to(device)

    model.eval()

    all_preds = []
    all_labels = []

    domain_correct = 0
    domain_total = 0

    batch_times = np.empty(len(test_loader))
    progress_treshold = max(1, len(test_loader) // 10)

    print(f"🔍 Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_start_time = perf_counter()
            input_ids, attention_mask, y_lab, y_dom = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y_lab = y_lab.float().unsqueeze(1).to(device)
            y_dom = y_dom.to(device)

            x = (input_ids, attention_mask)

            class_logits, domain_logits = model(x, lambda_=0.0)

            probs = torch.sigmoid(class_logits)
            preds = (probs > 0.5).long().squeeze(1)
            true_labels = y_lab.long().squeeze(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

            domain_preds = domain_logits.argmax(dim=1)
            domain_correct += (domain_preds == y_dom.to(device)).sum().item()
            domain_total += y_dom.size(0)

            batch_times[batch_idx] = (perf_counter() - batch_start_time)

            if (batch_idx + 1) == 1 or (batch_idx + 1) % progress_treshold == 0 or (batch_idx + 1) == len(test_loader):
                print(f"    → Processed {batch_idx + 1}/{len(test_loader)} batches | "
                      f"ETA: {((len(test_loader) - (batch_idx + 1)) * np.mean(batch_times[:batch_idx + 1])):.2f} sec")

    domain_acc = domain_correct / domain_total
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print("\n\n===== Evaluation Summary =====")
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"Domain Accuracy: {domain_acc * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")

    if logging:
        if not os.path.isdir(f"{LOG_DIR}/results"):
            os.makedirs(f"{LOG_DIR}/results")
        log_file = f"{LOG_DIR}/results/{model_checkpoint.split('_acc')[0]}"
        with open(log_file, "w") as f:
            f.write(f"Summary of Test Run for model \n"
                    f"  → {model_checkpoint} \n"
                    f" in a {'cross-domain' if cross_domain else 'in-domain'} setting \n"
                    f"  → {test_loader.dataset.df['domain'].unique().tolist() if cross_domain else NUM_DOMAINS} domains \n"
                    f"Classification Accuracy: {accuracy * 100:.2f}% \n"
                    f"Domain Accuracy: {domain_acc * 100:.2f}% \n"
                    f"F1 Score: {f1 * 100:.2f}% \n"
                    f"Precision: {precision * 100:.2f}% \n"
                    f"Recall: {recall * 100:.2f}%")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DANN model")

    parser.add_argument("--model", type=str, help="Model checkpoint")
    parser.add_argument("--augmented", action="store_true", help="Use augmented data")
    parser.add_argument("--cross_domain", action="store_true", help="Train in cross-domain setting")
    parser.add_argument("--balanced", action="store_true", help="Use balanced dataset")
    parser.add_argument("--log", action="store_true", help="Enable TensorBoard and logging")

    args = parser.parse_args()

    test(
        model_checkpoint=args.model,
        logging=args.log,
        cross_domain=args.cross_domain,
        balanced=args.balanced,
        augmented=args.augmented
    )
