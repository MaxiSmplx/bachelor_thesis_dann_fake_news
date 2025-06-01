import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import TOKENIZER, BATCH_SIZE, FILE_PATH, FILE_PATH_AUGMENTED

tokenizer = TOKENIZER


class ParquetDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_parquet(file_path).sample(500) # TODO Remove just for testing

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
    augmented: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    dataset = ParquetDataset(FILE_PATH_AUGMENTED) if augmented else ParquetDataset(FILE_PATH)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return loader