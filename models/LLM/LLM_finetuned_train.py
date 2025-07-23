from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import TensorDataset
from data_loader import get_dataset, BERTDataset
from sklearn.metrics import accuracy_score
from datetime import datetime
import pandas as pd
import argparse
import os
import torch

def get_data(cross_domain: bool = True, augmented: bool = False, balanced: bool = False, val_fraction: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data, val_data = get_dataset("train", val_fraction=val_fraction, cross_domain=cross_domain, augmented=augmented, balanced=balanced)
    test_data = get_dataset("test", cross_domain=cross_domain, augmented=augmented, balanced=balanced)

    return train_data, val_data, test_data

def tokenize_data(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, model_arch: str = "BERT") -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    if model_arch == "BERT":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_arch == "RoBERTa":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_enc = tokenizer(train_data["text"].tolist(), padding=True, truncation=True, return_tensors="pt")
    val_enc = tokenizer(val_data["text"].tolist(), padding=True, truncation=True, return_tensors="pt")
    test_enc = tokenizer(test_data["text"].tolist(), padding=True, truncation=True, return_tensors="pt")

    train_labels = torch.tensor(train_data["label"].values)
    val_labels = torch.tensor(val_data["label"].values)
    test_labels = torch.tensor(test_data["label"].values)

    train_dataset = BERTDataset(train_enc, train_labels)
    val_dataset = BERTDataset(val_enc, val_labels)
    test_dataset = BERTDataset(test_enc, test_labels)

    return (train_dataset, val_dataset, test_dataset)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds)
    }

def prepare_trainer(train_dataset, val_dataset, epochs: int = 10, batch_size: int = 48, model_arch: str = "BERT") -> Trainer:
    if model_arch == "BERT":
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    elif model_arch == "RoBERTa":
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)


    training_args = TrainingArguments(
        output_dir="models/LLM/results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        metric_for_best_model="accuracy",
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    return trainer

def train(
    batch_size: int = 64,
    num_epochs: int = 10,
    cross_domain: bool = True,
    augmented: bool = False,
    balanced: bool = False,
    model_arch: str = "BERT"):
    if not os.path.isdir(f"models/LLM/output/{model_arch}"):
        os.mkdir(f"models/LLM/output/{model_arch}")
    if not os.path.isdir("models/LLM/models"):
        os.mkdir("models/LLM/models")
    print(f"Started training... \n" 
          f"Training with model: {model_arch} \n")

    output_folder_path = f"models/LLM/output/{model_arch}/training_summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    assert model_arch in ("BERT", "RoBERTa"), "model should be 'BERT' or 'RoBERTa'"

    print("Loading data...")
    train_data, val_data, test_data = get_data(cross_domain=cross_domain, augmented=augmented, balanced=balanced)

    with open(output_folder_path, "a") as f:
        f.write(
            f"Training Details \n"
            f"  Training Data: {len(train_data)} data points, {len(train_data['domain'].unique())} domains \n"
            f"  Evaluation Data: {len(val_data)}, {len(val_data['domain'].unique())} domain(s) \n"
            f"  Test Data: {len(test_data)}, {len(test_data['domain'].unique())} domain(s) \n"
            f"  Training in {'cross-domain' if cross_domain else 'in-domain'} setting \n"
            f"      {test_data['domain'].unique().tolist() if cross_domain else len(test_data['domain'].unique())} domain(s) \n"
            f"  Data Augmentation is {'enabled' if augmented else 'disabled'} \n"
            f"  Domain and Class balancing is {'enabled' if balanced else 'disabled'} \n\n\n"
        )

    print("Tokenizing data...")
    train_dataset, val_dataset, test_dataset = tokenize_data(train_data, val_data, test_data, model_arch)

    print("Preparing trainer...")
    trainer = prepare_trainer(train_dataset, val_dataset, epochs=num_epochs, batch_size=batch_size, model_arch=model_arch)

    print("Starting training...")
    trainer.train()

    print("Evaluating on test dataset...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)

    print(f"Saving results to {output_folder_path}...")
    with open(output_folder_path, "a") as f:
        f.write("===== Training Log Summary =====\n")
        for entry in trainer.state.log_history:
            for key, value in entry.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

        f.write("===== Final Test Evaluation =====\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"Final Test Accuracy: {test_metrics['eval_accuracy'] * 100}%")

    trainer.save_model(f"models/LLM/models/{model_arch}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{test_metrics['eval_accuracy'] * 100}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune LLM model")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--cross_domain", action="store_true", help="Train in cross-domain setting")
    parser.add_argument("--augmented", action="store_true", help="Use augmented data")
    parser.add_argument("--balanced", action="store_true", help="Use balanced dataset")
    parser.add_argument("--model", type=str, default="BERT", help="BERT variant - BERT or RoBERTa")

    args = parser.parse_args()

    train(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        cross_domain=args.cross_domain,
        augmented=args.augmented,
        balanced=args.balanced,
        model_arch=args.model
    )