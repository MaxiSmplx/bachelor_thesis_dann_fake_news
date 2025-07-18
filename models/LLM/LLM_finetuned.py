from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import TensorDataset
from data_loader import get_dataset, BERTDataset
from sklearn.metrics import accuracy_score
import pandas as pd
import torch

def get_data(cross_domain: bool = True, augmented: bool = False, balanced: bool = False, val_fraction: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data, val_data = get_dataset("train", val_fraction=val_fraction, cross_domain=cross_domain, augmented=augmented, balanced=balanced)
    test_data = get_dataset("test", cross_domain=cross_domain, augmented=augmented, balanced=balanced)

    return train_data, val_data, test_data

def tokenize(tokenizer, sample):
    return tokenizer(sample['text'], truncation=True, padding='max_length', max_length=256)

def tokenize_data(train_data, val_data, test_data) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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

def prepare_trainer(train_dataset, val_dataset, epochs: int = 10, batch_size: int = 48) -> Trainer:
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    return trainer


if __name__ == "__main__":
    train_data, val_data, test_data = get_data(cross_domain=True, augmented=False, balanced=False)

    train_dataset, val_dataset, test_dataset = tokenize_data(train_data, val_data, test_data)

    trainer = prepare_trainer(train_dataset, val_dataset, epochs=10, batch_size=8)

    trainer.train()

    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    print(eval_result)

    # trainer.save_model("bert-fake-news-detector")






