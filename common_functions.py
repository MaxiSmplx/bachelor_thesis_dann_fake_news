"""
%run ../common_functions.py
"""

#Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from constants import DATASETS



print("read_dataset()")
def read_dataset(dataset_name: str) -> pd.DataFrame: 
    real_data = pd.read_parquet(f"../datasets/{dataset_name}/real.parquet")
    fake_data = pd.read_parquet(f"../datasets/{dataset_name}/fake.parquet")

    return pd.concat([real_data, fake_data], ignore_index=True)


print("read_all_datasets()")
def read_all_datasets() -> pd.DataFrame:
    path = "../datasets"

    all_dfs = [
        pd.read_parquet(f"{path}/{data}/{type}.parquet")
        for data in DATASETS
        for type in ("real", "fake")
    ]

    return pd.concat(all_dfs, ignore_index=True)


print("read_processed_data()")
def read_processed_data(augmented: bool = False):
    path = f"../pipeline/output/preprocessed_data{'_augmented' if augmented else ''}.npz"

    data = np.load(path)

    return data["X"], data["y"]
    


print("combine_text_columns()")
def combine_text_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df["text"] = df[cols].apply(
        lambda row: " ".join(str(val) for val in row if pd.notnull(val)),
        axis=1
    )
    df.drop(columns=filter(lambda c: c != "text", cols), inplace=True)
    return df


print("split_train_test()")
def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    stratify: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y if stratify else None
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