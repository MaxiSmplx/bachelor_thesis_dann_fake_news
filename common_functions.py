"""
%run ../common_functions.py
"""

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


print("read_dataset()")
def read_dataset(dataset_path: str) -> pd.DataFrame: 
    real_data = pd.read_csv(f"{dataset_path}/real.csv")
    fake_data = pd.read_csv(f"{dataset_path}/fake.csv")

    return pd.concat([real_data, fake_data], ignore_index=True)


print("combine_text_columns()")
def combine_text_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df["combined_text"] = df[cols].apply(
        lambda row: " ".join(str(val) for val in row if pd.notnull(val)),
        axis=1
    )
    df.drop(columns=cols, inplace=True)
    return df



print("split_data()")
def split_data(df: pd.DataFrame, split_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    random_seed = randint(0, 1_000_000)
    train_df, test_df = train_test_split(df, test_size=split_size, random_state=random_seed, shuffle=True)
    return train_df, test_df


print("extract_data()")
def extract_data(train_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str = "combined_text", label_col: str = "label"):
    return (
        train_df[text_col], train_df[label_col],
        test_df[text_col], test_df[label_col]
    )


print("evaluate_model()")
def evaluate_model(y_test, y_pred):
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()
    print("Overall Metrics:")
    print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (avg): {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall    (avg): {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score  (avg): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


print("detect_missing_values()")
def detect_missing_values(df: pd.DataFrame):
    missing_per_column = df.isna().sum()
    total_missing = missing_per_column.sum()
    columns_with_missing = missing_per_column[missing_per_column > 0].index.tolist()
    
    print("Missing values per column:")
    print(missing_per_column)
    print("\nTotal missing values in DataFrame:", total_missing)
    print("\nColumns with missing values:", columns_with_missing)

