import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ast
from sentence_transformers import SentenceTransformer
from data_augmentation._openai_api import call_openai
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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


def visualize_kmeans(kmeans, embeddings):
    labels = kmeans.fit_predict(embeddings)

    # Convert to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    for i in range(13):
        plt.scatter(
            embeddings_2d[labels == i, 0],
            embeddings_2d[labels == i, 1],
            label=f'Cluster {i}',
            alpha=0.6
        )

    # Plot centroids (also reduced to 2D)
    centroids_2d = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', s=200, marker='X', label='Centroids')

    plt.legend()
    plt.title('kMeans Clustering Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()


def add_domain(df: pd.DataFrame, 
               k: int, save_embeddings: bool = False, 
               use_saved_embeddings: bool = False, 
               plot_kmeans: bool = False) -> pd.DataFrame:
    if use_saved_embeddings and os.path.exists("output/embeddings.npy"):
        print("Reading saved sentence embeddings...")
        embeddings = np.load("output/embeddings.npy")
    else:
        print("Computing sentence embeddings...")
        embeddings = sentence_encoding(df, save_embeddings)

    print(f"Initializing KMeans with k={k} Clusters...")
    kmeans = kMeans(k=k)

    print("Predicting domains with KMeans...")
    df["domain"] = kmeans.fit_predict(embeddings)

    if plot_kmeans:
        visualize_kmeans(kmeans, embeddings)

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

