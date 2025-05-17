"""
%run ../common_functions.py
"""

# Libraries
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
    """Load and concatenate the real and fake portions of the specified dataset.

    Args:
        dataset_name (str): Name of the subdirectory under "../datasets/" containing
                            "real.parquet" and "fake.parquet".

    Returns:
        pd.DataFrame: DataFrame containing all rows from both real.parquet and fake.parquet,
                      re-indexed consecutively.
    """
    real_data = pd.read_parquet(f"../datasets/{dataset_name}/real.parquet")
    fake_data = pd.read_parquet(f"../datasets/{dataset_name}/fake.parquet")

    return pd.concat([real_data, fake_data], ignore_index=True)


print("read_all_datasets()")
def read_all_datasets() -> pd.DataFrame:
    """Load and concatenate the real and fake Parquet files from all datasets listed in DATASETS.

    Args:
        None

    Returns:
        pd.DataFrame: DataFrame containing all rows from real.parquet and fake.parquet
                      across every dataset in DATASETS, re-indexed consecutively.
    """
    path = "../datasets"

    all_dfs = [
        pd.read_parquet(f"{path}/{data}/{type}.parquet")
        for data in DATASETS
        for type in ("real", "fake")
    ]

    return pd.concat(all_dfs, ignore_index=True)


print("read_processed_data()")
def read_processed_data(augmented: bool = False):
    """Load preprocessed (and optionally augmented) data from the pipeline output.

    Args:
        augmented (bool): If True, load the augmented dataset (`preprocessed_data_augmented.npz`);
                          otherwise load the standard `preprocessed_data.npz`.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            X – feature array loaded from the NPZ file,
            y – label array loaded from the NPZ file.
    """
    path = f"../pipeline/output/preprocessed_data{'_augmented' if augmented else ''}.npz"

    data = np.load(path)

    return data["X"], data["y"]
    


print("combine_text_columns()")
def combine_text_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Combine multiple text columns into a single 'text' column and drop the originals.

    Args:
        df (pd.DataFrame): DataFrame containing the columns to combine.
        cols (list[str]): List of column names whose values will be concatenated.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'text' column and without the original specified columns.
    """
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
    """Split features and labels into train and test sets.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Label array.
        test_size (float): Proportion of the dataset to include in the test split.
        stratify (bool): Whether to stratify the split by labels.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            X_train, X_test, y_train, y_test arrays.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y if stratify else None
    )


print("evaluate_model()")
def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Evaluate a classifiers predictions by printing key metrics and the confusion matrix.

    Args:
        y_test (np.ndarray): True labels for the test set.
        y_pred (np.ndarray): Predicted labels from the model.

    Returns:
        None: Prints the classification report, overall accuracy, weighted precision, recall,
              F1 score, and the confusion matrix to stdout.
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
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


print("detect_missing_values()")
def detect_missing_values(df: pd.DataFrame) -> None:
    """Analyze and report missing values in the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to inspect for missing values.

    Returns:
        None: Prints the count of missing values per column, the total missing count,
              and a list of columns containing missing values.
    """
    missing_per_column = df.isna().sum()
    total_missing = missing_per_column.sum()
    columns_with_missing = missing_per_column[missing_per_column > 0].index.tolist()
    
    print("Missing values per column:")
    print(missing_per_column)
    print("\nTotal missing values in DataFrame:", total_missing)
    print("\nColumns with missing values:", columns_with_missing)