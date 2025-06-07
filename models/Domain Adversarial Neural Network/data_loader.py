import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import TOKENIZER, BATCH_SIZE, FOLDER_PATH_BALANCED, FOLDER_PATH_AUGMENTED, FOLDER_PATH_BALANCED_AUGMENTED, FOLDER_PATH_RAW

tokenizer = TOKENIZER


class ParquetDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

        domains = sorted(self.df['domain'].unique())
        self.domain2idx = {d:i for i,d in enumerate(domains)}

        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()
        self.domains = [self.domain2idx[d] for d in self.df['domain']]

        self.encodings = tokenizer(
            self.texts,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=256,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        label = self.labels[idx]
        domain = self.domains[idx]

        return input_ids, attention_mask, label, domain


def get_dataloader(
    split: str =  "train",
    val_fraction: float = 0.1,
    augmented: bool = False,
    balanced: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    folder_path = (
        FOLDER_PATH_BALANCED_AUGMENTED if balanced else FOLDER_PATH_AUGMENTED
    ) if augmented else (
        FOLDER_PATH_BALANCED if balanced else FOLDER_PATH_RAW
    )

    if split.lower() in ("train", "tr", "validation", "val"):
        data = pd.read_parquet(f"{folder_path}/preprocessed_data_train_val.parquet").sample(n=30_000)
        split_idx = int(len(data) * (1 - val_fraction))
        dataset_train, dataset_val = ParquetDataset(data.iloc[:split_idx]), ParquetDataset(data.iloc[split_idx:])

        loader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        loader_val = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        return loader_train, loader_val

    elif split.lower() in ("test", "tst"):
        data = pd.read_parquet(f"{folder_path}/preprocessed_data_test.parquet")
        dataset_test = ParquetDataset(data)

        loader_test = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        return loader_test
    
    else:
        raise ValueError("Unknown data split, either 'train', 'validation' or 'test'!")