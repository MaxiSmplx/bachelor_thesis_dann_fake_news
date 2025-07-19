from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from data_loader import get_dataset, BERTDataset
import os

def get_data(cross_domain: bool = True, augmented: bool = False, balanced: bool = False) -> pd.DataFrame:
    test_data = get_dataset("test", cross_domain=cross_domain, augmented=augmented, balanced=balanced)
    return test_data

def tokenize_data(test_data: pd.DataFrame) -> TensorDataset:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    test_enc = tokenizer(test_data["text"].tolist(), padding=True, truncation=True, return_tensors="pt")

    test_labels = torch.tensor(test_data["label"].values)

    test_dataset = BERTDataset(test_enc, test_labels)

    return test_dataset


def test(model_path: str, cross_domain: bool = True, augmented: bool = False, balanced: bool = False):
    print(f"Loading model from: models/LLM/models/{model_path}...")
    model = BertForSequenceClassification.from_pretrained(f"models/LLM/models/{model_path}")

    trainer = Trainer(model=model)

    test_data = get_data(cross_domain=cross_domain, augmented=augmented, balanced=balanced)

    print("Tokenizing test data...")
    test_dataset = tokenize_data(test_data)

    print("Running evaluation...")
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(axis=1)
    y_true = predictions.label_ids

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="real")
    recall = recall_score(y_true, y_pred, pos_label="real")
    f1 = f1_score(y_true, y_pred, pos_label="real")

    print(
        f"\nEvaluation Metrics:\n"
        f"---------------------------\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"F1 Score : {f1:.4f}\n"
    )

    if not os.path.isdir("models/LLM/output/results"):
        os.mkdir("models/LLM/output/results")

    print(f"Saving results to: models/LLM/output/results/{os.path.basename(model_path)}")
    with open(f"models/LLM/output/results/{os.path.basename(model_path)}", "w") as f:
        f.write(f"Summary of Test Run for model \n"
                f"  → {model_path} \n"
                f" in a {'cross-domain' if cross_domain else 'in-domain'} setting \n"
                f"  → {test_data['domain'].unique().tolist()} domains \n"
                f"Classification Accuracy: {accuracy * 100:.2f}% \n"
                f"F1 Score: {f1 * 100:.2f}% \n"
                f"Precision: {precision * 100:.2f}% \n"
                f"Recall: {recall * 100:.2f}%")