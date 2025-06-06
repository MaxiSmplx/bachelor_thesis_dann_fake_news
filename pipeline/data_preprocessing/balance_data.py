import pandas as pd
import numpy as np

def balance_data(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    cap = int(df["domain"].value_counts().min() * (1 + tolerance))
    print(f"Balancing every domain to approx. {cap} datapoints")

    balanced_chunks = []
    for domain, domain_count in df["domain"].value_counts().items():
        domain_df = df[df["domain"] == domain]
        
        if domain_count > cap:
            for label, label_count in domain_df["label"].value_counts().items():
                sampled = domain_df[domain_df["label"] == label].sample(n=min(label_count, cap//2))
                balanced_chunks.append(sampled)
        else:
            balanced_chunks.append(domain_df) 
    
    balanced_df = pd.concat(balanced_chunks, ignore_index=True).sample(frac=1).reset_index(drop=True)

    count_domain_old_df = df["domain"].value_counts()
    count_domain_new_df = balanced_df["domain"].value_counts()
    count_label_old_df = df["label"].value_counts()
    count_label_new_df = balanced_df["label"].value_counts()
    
    print(f"Max. difference in assigned rows to domain was {count_domain_old_df.max() - count_domain_old_df.min()} rows and is now {count_domain_new_df.max() - count_domain_new_df.min()} rows")
    print(f"Label imbalance (0 vs. 1) was {count_label_old_df[0] / count_label_old_df.sum():.2} vs. {count_label_old_df[1] / count_label_old_df.sum():.2%} "
          f"and is now {count_label_new_df[0] / count_label_new_df.sum():.2%} vs. {count_label_new_df[1] / count_label_new_df.sum():.2%}")

    return balanced_df


def augmentation_threshold(df: pd.DataFrame, augmentation_budget: int) -> dict:
    counts = df["domain"].value_counts().to_list()
    names = df["domain"].value_counts().index.tolist()
    
    rows_to_augment = [0] * len(counts)
    domain_diff = [(counts[i] - counts[i+1]) for i in range(len(counts) - 1)]

    for i in range(len(domain_diff)):
        diff_value = min(domain_diff[-(i+1)], augmentation_budget)
        indices_to_update = range(len(counts) - i - 1, len(counts))

        if len(indices_to_update) == 0 or diff_value == 0:
            continue

        # Distribute the diff_value equally or as much as possible
        per_index = diff_value // len(indices_to_update)
        remainder = diff_value % len(indices_to_update)

        for idx, j in enumerate(indices_to_update):
            rows_to_augment[j] += per_index + (1 if idx < remainder else 0)

        augmentation_budget -= diff_value
        if augmentation_budget == 0:
            break

    return dict(zip(names, rows_to_augment))