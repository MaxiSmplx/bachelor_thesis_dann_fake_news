import pandas as pd
import itertools

def print_stats(data: pd.DataFrame) -> None:
    total_rows      = len(data)
    unique_texts    = data['text'].nunique()
    duplicate_rows  = total_rows - unique_texts
    dup_pct         = (duplicate_rows / total_rows) * 100

    print(
        f"Total rows                 : {total_rows}\n"
        f"Unique 'text' entries      : {unique_texts}\n"
        f"Duplicate 'text' entries   : {duplicate_rows} "
        f"({dup_pct:.2f}% of total)"
    )


def duplicates_single(datasets: dict[str: pd.DataFrame]) -> dict[str: pd.DataFrame]:
    print("==DUPLICATES WITHIN DATASET==")
    total_duplicates_single = 0

    for name, df in datasets.items():
        # count
        duplicate_count = len(df) - df["text"].nunique()
        total_duplicates_single += duplicate_count
        print(f"Found {duplicate_count} duplicated rows for Dataset '{name}'")
        
        # remove
        old_length = len(df)
        df.drop_duplicates(subset=["text"], inplace=True)
        print(f"Row count: {old_length} → {len(df)} "
            f"(-{((old_length - len(df)) / old_length * 100):.2f}%)")
        print("=" * 50)

    print(f"=> Found {total_duplicates_single} duplicated rows in total")


def duplicates_across(datasets: dict[str: pd.DataFrame]) -> dict[str: pd.DataFrame]:
    print("==DUPLICATES ACROSS DATASETS==")
    text_sets = {name: set(df['text']) for name, df in datasets.items()}
    total_duplicates_across = 0

    for name1, name2 in itertools.combinations(text_sets, 2):
        common_texts = text_sets[name1].intersection(text_sets[name2])
        if len(common_texts) > 0:
            # count
            total_duplicates_across += len(common_texts)
            print(f"Found {len(common_texts)} duplicates between '{name1}' and '{name2}' → {(len(common_texts) / len(datasets[name1])) * 100:.2f}%")

            # remove
            df1 = datasets[name1]
            datasets[name1] = df1[~df1['text'].isin(common_texts)]
    print(f"=> Found {total_duplicates_across} duplicated rows in total")


def find_and_remove_dups(data: pd.DataFrame, datasets: dict[str: pd.DataFrame]) -> pd.DataFrame:
    print_stats(data)
    print()
    duplicates_single(datasets)
    print()
    duplicates_across(datasets)
    
    return pd.concat(datasets.values(), ignore_index=True)