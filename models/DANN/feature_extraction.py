import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel
from tqdm import tqdm

def extract_features(df,
                     model_name: str = "bert-base-uncased",
                     batch_size: int = 32,
                     device: str = None,
                     input_id_col: str = "input_ids",
                     mask_col: str = "attention_mask") -> np.ndarray:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    input_ids = torch.tensor(df[input_id_col].tolist())
    attention_mask = torch.tensor(df[mask_col].tolist())
    dataset = TensorDataset(input_ids, attention_mask)
    loader  = DataLoader(dataset, batch_size=batch_size)

    all_embeddings = []
    with torch.no_grad():
        for batch_ids, batch_mask in tqdm(loader, desc="Extracting features"):
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            outputs = model(input_ids=batch_ids,
                               attention_mask=batch_mask)
            cls_emb = outputs.pooler_output.cpu().numpy()
            all_embeddings.append(cls_emb)

    return np.vstack(all_embeddings)
