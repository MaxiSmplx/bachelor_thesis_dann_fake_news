import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ast
from sentence_transformers import SentenceTransformer
from data_augmentation._openai_api import call_openai
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np

def sentence_encoding(df: pd.DataFrame, save_to_file: bool) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")   
    
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    if save_to_file:
        np.save("output/embeddings.npy", embeddings)

    return embeddings

def kMeans(k: int = 13): #TODO Tweak parameters and check if it works better
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=1024,
        random_state=42,
        reassignment_ratio=0.01
    )

    return kmeans

def ids_to_topic(samples: dict[int: list[str]]) -> str:
    context = (
        "You are an expert at mapping LDA topic keywords to concise, human-readable domain labels. "
        "Input is a Python dict where keys are topic IDs (ints) and values are lists of some examples for that topic. "
        "Output must be a Python dict with the same keys, mapping each topic ID to a single string label."
    )
    prompt = (
            f"Here is my dict with topic ids and example sentences for each topic:\n{samples}\n\n"
            "Produce only the literal Python dict mapping each topic ID to a distinct, non-overlapping domain label."
            "Also do not include '```python' in your response."
            "You are free to be assign labels such as 'Satire', but do not frame your labels as judgments on their truthfulness or morality: Avoid anything like 'False Statements', 'Lies', or similar."
        )
    response = call_openai(context=context, prompt=prompt, model="gpt-4.1-2025-04-14", temperature=0.2)
    
    return ast.literal_eval(response)



def add_domain(df: pd.DataFrame, save_embeddings: bool = False, use_saved_embeddings: bool = False) -> pd.DataFrame:
    if use_saved_embeddings and os.path.exists("output/embeddings.npy"):
        print("Reading saved sentence embeddings...")
        embeddings = np.load("output/embeddings.npy")
    else:
        print("Computing sentence embeddings...")
        embeddings = sentence_encoding(df, save_embeddings)

    k = 13
    print(f"Initializing KMeans with k={k} Clusters...")
    kmeans = kMeans(k=k)

    print("Predicting domains with KMeans...")
    df["domain"] = kmeans.fit_predict(embeddings)

    n_samples = 6
    samples = {
        cid: (
            df[df["domain"] == cid]
            ["text"]
            .sample(n_samples, random_state=cid)
            .str.slice(0, 1000)
            .tolist()
        )
        for cid in range(k)
    }

    print("Mapping domain IDs to topic names...")
    df["domain"] = df["domain"].map(ids_to_topic(samples))

    print(f"\n\nTopic Distribution: {df['domain'].value_counts()}")

    return df

