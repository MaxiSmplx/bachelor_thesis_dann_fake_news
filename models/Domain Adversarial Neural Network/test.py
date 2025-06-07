from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from data_loader import get_dataloader
from model import DANN
from time import perf_counter
from datetime import datetime
import numpy as np
import torch
from config import (
    NUM_CLASSES,
    NUM_DOMAINS,
    INPUT_DIM,
    FEATURE_DIM
)

def test(model_checkpoint: str):
    print(f"\n🚀 Starting evaluation...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device >> {device}\n")

    test_loader = get_dataloader("test", balanced=True, batch_size=48)
    print(f"Loaded Test Data with {len(test_loader.dataset)} datapoints")

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

            if (batch_idx + 1) == 1 or (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
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


            

if __name__ == "__main__":
    test("dann_2025-06-06-17-28_acc-61.00")
