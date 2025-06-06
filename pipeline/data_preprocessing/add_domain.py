import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ast
import torch
from sentence_transformers import SentenceTransformer
from data_augmentation._openai_api import call_openai
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def sentence_encoding(df: pd.DataFrame, tokenizer_model, save_to_file: bool) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
    print(f"Using device >> {device}")

    model = SentenceTransformer(tokenizer_model, device=device) 
    
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if save_to_file:
        np.save("output/embeddings.npy", embeddings)

    return embeddings

def kMeans(k: int = 13):
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
            "You are free to be assign labels such as 'Satire', but do not frame your labels as judgments on their truthfulness or morality: Avoid anything like 'False Statements', 'Lies', 'Unusual or Contradictory Claims', 'Definitions & Factual Statements' or similar."
        )
    response = call_openai(context=context, prompt=prompt, model="gpt-4.1-2025-04-14", temperature=0.2)
    
    return ast.literal_eval(response)


def visualize_kmeans(cluster, embeddings, cluster_names, sample_size=5_000):
    # Randomly sample to increase readability
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[indices]
        cluster = cluster[indices]

    # Reduce Dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    for i in range(13):
        plt.scatter(
            embeddings_2d[cluster == i, 0],
            embeddings_2d[cluster == i, 1],
            label=f"{cluster_names.get(i)}",
            alpha=0.6,
            s=20
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('kMeans Clustering Visualization')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.grid(True)
    plt.show()


def add_domain(df: pd.DataFrame, 
               k: int, 
               tokenizer_model,
               save_embeddings: bool = False, 
               use_saved_embeddings: bool = False, 
               plot_kmeans: bool = False) -> pd.DataFrame:
    if use_saved_embeddings and os.path.exists("output/embeddings.npy"):
        print("Reading saved sentence embeddings...")
        embeddings = np.load("output/embeddings.npy")
    else:
        print("Computing sentence embeddings...")
        embeddings = sentence_encoding(df, tokenizer_model, save_embeddings)

    print(f"Initializing KMeans with k={k} Clusters...")
    kmeans = kMeans(k=k)

    print("Predicting domains with KMeans...")
    cluster = kmeans.fit_predict(embeddings)
    df["domain"] = cluster

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
    cluster_names = ids_to_topic(samples)
    df["domain"] = df["domain"].map(cluster_names)

    if plot_kmeans:
        print("Plotting...")
        visualize_kmeans(cluster, embeddings, cluster_names, sample_size=5_000)

    print(f"\n\nTopic Distribution: {df['domain'].value_counts()}")

    return df

