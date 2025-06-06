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
def read_processed_data(augmented: bool = False, balanced: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed (and optionally augmented) data from the pipeline output.

    Args:
        augmented (bool): If True, load the augmented dataset (`preprocessed_data_augmented.npz`);
                          otherwise load the standard `preprocessed_data.npz`.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            X – feature array loaded from the NPZ file,
            y – label array loaded from the NPZ file.
    """
    with open("../pipeline/config.yml", "r") as f:
        config = yaml.safe_load(f)

    folder_name = {
        (False, False): 'raw',
        (True,  False): 'balanced',
        (False, True):  'augmented',
        (True,  True):  'balanced_augmented'
    }[(balanced, augmented)]

    folder_path = os.path.join(f"../pipeline/{config['output']}", folder_name)

    data_train = pd.read_parquet(f"{folder_path}/preprocessed_data_train_val.parquet")
    data_test = pd.read_parquet(f"{folder_path}/preprocessed_data_test.parquet")
    
    return data_train, data_test
    


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
    data: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split DataFrame into training and testing subsets.

    Args:
        data (pd.DataFrame):  
            The full dataset to split, as a pandas DataFrame.
        test_size (float, optional):  
            Fraction of samples to allocate to the test set (between 0.0 and 1.0).  
            Defaults to 0.2 (20% test / 80% train).

    Returns:
        Tuple[np.ndarray, np.ndarray]:  
            - train_set: A NumPy array containing the training portion of `data`.  
            - test_set: A NumPy array containing the testing portion of `data`.
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
    """
    Extract feature and label arrays from train and test DataFrames.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training data.
        test_df (pd.DataFrame): DataFrame containing the test data.
        text_col (str):     Name of the column to use as the feature.
        label_col (str):    Name of the column to use as the label.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train: numpy array of features from train_df
            y_train: numpy array of labels from train_df
            X_test:  numpy array of features from test_df
            y_test:  numpy array of labels from test_df
    """
    X_train = train_df[text_col].to_numpy()
    y_train = train_df[label_col].to_numpy()
    X_test  = test_df[text_col].to_numpy()
    y_test  = test_df[label_col].to_numpy()

    return X_train, y_train, X_test, y_test


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
    ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["Real News", "Fake News"], cmap="Blues", colorbar=True
    )
    plt.title("Confusion Matrix", fontsize=10)
    plt.tight_layout()
    plt.show()


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