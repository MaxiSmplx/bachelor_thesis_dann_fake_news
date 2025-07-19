import yaml, os
import pandas as pd
from torch.utils.data import Dataset

def get_dataset(split: str =  "train",
    val_fraction: float = 0.1,
    cross_domain: bool = True,
    augmented: bool = False,
    balanced: bool = False):
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
        data = pd.read_parquet(f"{folder_path}/preprocessed_data_train_val.parquet")
        
        domains = data['domain'].unique()
        val_domain_count = max(1, int(len(domains) * val_fraction))

        val_domains = pd.Series(domains).sample(val_domain_count, random_state=42).tolist()

        dataset_train = data[~data['domain'].isin(val_domains)]
        dataset_val = data[data['domain'].isin(val_domains)]

        return dataset_train, dataset_val
    
    elif split.lower() in ('test', 'tst'):
        dataset_test = pd.read_parquet(f"{folder_path}/preprocessed_data_test.parquet")
        return dataset_test
    
    else:
        raise ValueError("Unknown data split, either 'train', 'validation' or 'test'!")


class BERTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)