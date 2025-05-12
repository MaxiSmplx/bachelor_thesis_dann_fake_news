"""
%run ../common_functions.py
"""

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets, Dataset
from random import randint
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


print("load_dataset()")
def load_dataset(dataset_path: str) -> Dataset: 
    real_data = load_dataset("csv", data_files="../datasets/FakeNewsDetection/fakenewsdetection_real.csv")["train"]
    real_data = real_data.add_column("label", [0] * len(real_data))

    fake_data = load_dataset("csv", data_files="../datasets/FakeNewsDetection/fakenewsdetection_fake.csv")["train"]
    fake_data = fake_data.add_column("label", [1] * len(fake_data))

    return concatenate_datasets([real_data, fake_data])


print("combine_text_columns()")
def combine_text_columns(dataset:Dataset, cols:list[str]) -> Dataset:
    def combine(data):
        return {
            "combined_text": " ".join(str(data[col]) for col in cols if data.get(col) is not None)
        }
    dataset = dataset.map(combine)
    dataset = dataset.remove_columns(cols)
    return dataset


print("split_data()")
def split_data(dataset:Dataset, split_size:int = 0.2) -> (Dataset, Dataset):
    dataset = dataset.shuffle(seed=randint(0, 1_000_000))
    dataset = dataset.train_test_split(test_size=split_size)

    return dataset["train"], dataset["test"]


print("extract_data()")
def extract_data(train_data:Dataset, test_data:Dataset, text_col:str = "combined_text", label_col:str = "label"):
    return train_data[text_col], train_data[label_col], test_data[text_col], test_data[label_col]


print("evaluate_model()")
def evaluate_model(y_pred, y_test):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print()
    print("Overall Metrics:")
    print(f"Accuracy       : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (avg): {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall    (avg): {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score  (avg): {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


print("detect_missing_values()")
def detect_missing_values(df: pd.DataFrame):
    missing_per_column = df.isna().sum()
    total_missing = missing_per_column.sum()
    columns_with_missing = missing_per_column[missing_per_column > 0].index.tolist()
    
    print("Missing values per column:")
    print(missing_per_column)
    print("\nTotal missing values in DataFrame:", total_missing)
    print("\nColumns with missing values:", columns_with_missing)

