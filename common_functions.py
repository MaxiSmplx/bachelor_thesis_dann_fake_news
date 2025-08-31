"""
%run ../common_functions.py
"""

# Libraries
import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
from constants import DATASETS
import matplotlib.pyplot as plt



print("read_dataset()")
def read_dataset(dataset_name: str) -> pd.DataFrame: 
    """Load and concatenate the real and fake parquet files for a single dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the subdirectory under "../datasets/" containing
        "real.parquet" and "fake.parquet".

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with rows from both files, re-indexed consecutively.
    """

    real_data = pd.read_parquet(f"../datasets/{dataset_name}/real.parquet")
    fake_data = pd.read_parquet(f"../datasets/{dataset_name}/fake.parquet")

    return pd.concat([real_data, fake_data], ignore_index=True)


print("read_all_datasets()")
def read_all_datasets() -> pd.DataFrame:
    """
    Load and concatenate all real and fake parquet files from datasets in DATASETS.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing rows from `real.parquet` and `fake.parquet`
        across all datasets in DATASETS, with consecutive reindexing.
    """

    path = "../datasets"

    all_dfs = [
        pd.read_parquet(f"{path}/{data}/{type}.parquet")
        for data in DATASETS
        for type in ("real", "fake")
    ]

    return pd.concat(all_dfs, ignore_index=True)


print("read_processed_data()")
def read_processed_data(cross_domain: bool = True, augmented: bool = False, balanced: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train/validation and test datasets from the pipeline output.

    Parameters
    ----------
    cross_domain : bool, default True
        If True, load data from the cross-domain folder; else from in-domain.
    augmented : bool, default False
        If True, load augmented data.
    balanced : bool, default True
        If True, load balanced data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (data_train, data_test) loaded from parquet files.
    """

    with open("../pipeline/config.yml", "r") as f:
        config = yaml.safe_load(f)

    folder_name = {
        (False, False): 'raw',
        (True,  False): 'balanced',
        (False, True):  'augmented',
        (True,  True):  'balanced_augmented'
    }[(balanced, augmented)]

    folder_attribute = "cross_domain" if cross_domain else "in_domain"

    folder_path = os.path.join(f"../pipeline/{config['output']}", folder_attribute, folder_name)

    data_train = pd.read_parquet(f"{folder_path}/preprocessed_data_train_val.parquet")
    data_test = pd.read_parquet(f"{folder_path}/preprocessed_data_test.parquet")
    
    return data_train, data_test
    


print("combine_text_columns()")
def combine_text_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Combine multiple text columns into a single 'text' column and drop the originals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns to combine.
    cols : list[str]
        List of column names to concatenate.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'text' column and without the original specified columns.
    """

    df["text"] = df[cols].apply(
        lambda row: " ".join(str(val) for val in row if pd.notnull(val)),
        axis=1
    )
    df.drop(columns=filter(lambda c: c != "text", cols), inplace=True)
    return df


print("split_train_test()")
def split_train_test(
    data: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a DataFrame into training and testing subsets.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset to split.
    test_size : float, default 0.2
        Fraction of samples to allocate to the test set (between 0.0 and 1.0).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (train_set, test_set)
    """

    return train_test_split(
        data,
        test_size=test_size,
        random_state=42,
    )

print("extract_data()")
def extract_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract feature and label arrays from training and test DataFrames.

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame containing the training data.
    test_df : pd.DataFrame
        DataFrame containing the test data.
    text_col : str
        Name of the column to use as features.
    label_col : str
        Name of the column to use as labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, y_train, X_test, y_test)
    """

    X_train = train_df[text_col].to_numpy()
    y_train = train_df[label_col].to_numpy()
    X_test  = test_df[text_col].to_numpy()
    y_test  = test_df[label_col].to_numpy()

    return X_train, y_train, X_test, y_test


print("evaluate_model()")
def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Evaluate a classifier's predictions using key metrics and a confusion matrix.

    Parameters
    ----------
    y_test : np.ndarray
        True labels for the test set.
    y_pred : np.ndarray
        Predicted labels from the model.

    Returns
    -------
    None
        Prints:
        - Classification report
        - Accuracy, weighted precision, recall, and F1 score
        - Displays confusion matrix plot with "Real News" and "Fake News" labels
    """

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()
    print("Overall Metrics:")
    print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (avg): {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall    (avg): {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score  (avg): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print()
    ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["Real News", "Fake News"], cmap="Blues", colorbar=True
    )
    plt.title("Confusion Matrix", fontsize=10)
    plt.tight_layout()
    plt.show()


print("detect_missing_values()")
def detect_missing_values(df: pd.DataFrame) -> None:
    """Analyze and report missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect for missing values.

    Returns
    -------
    None
        Prints:
        - Count of missing values per column
        - Total number of missing values
        - List of columns containing missing values
    """

    missing_per_column = df.isna().sum()
    total_missing = missing_per_column.sum()
    columns_with_missing = missing_per_column[missing_per_column > 0].index.tolist()
    
    print("Missing values per column:")
    print(missing_per_column)
    print("\nTotal missing values in DataFrame:", total_missing)
    print("\nColumns with missing values:", columns_with_missing)