import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import TOKENIZER, BATCH_SIZE, FILE_PATH, FILE_PATH_AUGMENTED, FILE_PATH_AUGMENTED_BALANCED, FILE_PATH_BALANCED

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
    file_path = (
        FILE_PATH_AUGMENTED_BALANCED if balanced else FILE_PATH_AUGMENTED
    ) if augmented else (
        FILE_PATH_BALANCED if balanced else FILE_PATH
    )

    data = pd.read_parquet(file_path).sample(1_000) #TODO remove

    if split.lower() == "train" or split.lower() == "tr":
        dataset = ParquetDataset(data.sample(int((len(data) * (1 - val_fraction)))))
    elif split.lower() == "validation" or split.lower() == "val":
        dataset = ParquetDataset(data.sample(int((len(data) * val_fraction))))
    else:
        raise ValueError("Unknown data split, either 'train' or 'validation'!")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return loader