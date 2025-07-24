import pandas as pd
import yaml, os
from torch.utils.data import Dataset, DataLoader
from config import TOKENIZER, BATCH_SIZE

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
    cross_domain: bool = True,
    augmented: bool = False,
    balanced: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    with open("pipeline/config.yml", "r") as f:
        config = yaml.safe_load(f)

    folder_name = {
        (False, False): 'raw',
        (True,  False): 'balanced',
        (False, True):  'augmented',
        (True,  True):  'balanced_augmented'
    }[(balanced, augmented)]

    folder_attribute = "cross_domain" if cross_domain else "in_domain"

    folder_path = os.path.join(f"pipeline/{config['output']}", folder_attribute, folder_name)

    if split.lower() in ("train", "tr", "validation", "val"):     
        data = pd.read_parquet(f"{folder_path}/preprocessed_data_train_val.parquet").sample(n=200)

        if folder_attribute == "cross_domain":
            domains = data['domain'].unique()
            val_domain_count = max(1, int(len(domains) * val_fraction))

            val_domains = pd.Series(domains).sample(val_domain_count, random_state=42).tolist()

            train_data = data[~data['domain'].isin(val_domains)]
            val_data = data[data['domain'].isin(val_domains)]
        else:
            train_data = []
            val_data = []

            grouped = data.groupby("domain")
            for domain, group in grouped:
                val_split = group.sample(int(len(group) * val_fraction), random_state=42)
                train_split = group.drop(val_split.index)

                val_data.append(val_split)
                train_data.append(train_split)

            train_data = pd.concat(train_data).reset_index(drop=True)
            val_data = pd.concat(val_data).reset_index(drop=True)

        dataset_train, dataset_val = ParquetDataset(train_data), ParquetDataset(val_data)

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