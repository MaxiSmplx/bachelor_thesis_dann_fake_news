from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd

def tokenize_text(df,
                  tokenizer_name: str = "bert-base-uncased",
                  max_length: int = 256,
                  text_column: str = "text") -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    input_ids = []
    attention_masks = []

    for txt in tqdm(df[text_column], desc="Tokenizing"):
        enc = tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        input_ids.append(enc["input_ids"])
        attention_masks.append(enc["attention_mask"])

    df = df.copy()
    df["input_ids"] = input_ids
    df["attention_mask"] = attention_masks
    return df
