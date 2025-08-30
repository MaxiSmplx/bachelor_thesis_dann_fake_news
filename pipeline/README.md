# 🧪 NLP Processing Pipeline

This folder contains the complete NLP data processing pipeline, including text preprocessing, and optional data augmentation. The entire flow is orchestrated by a central `pipeline.py` script and is fully configurable via a `config.yml` file.

---

## 📁 Contents

### 🧹 Duplicate Removal

To prevent data leakage and inflated metrics, the pipeline **always removes exact text duplicates** before any other processing. This step guarantees that each `text` entry appears **at most once** in the final training corpus—even if it exists multiple times **within** a dataset or **across** different datasets.

What it does:

- **Project-Wide De-duplication**
  - **Within a dataset**: drops repeated rows based on the `text` column.
  - **Across datasets**: detects overlaps and removes the overlapping texts from the first dataset in each pairwise comparison to ensure each unique text survives **once** globally.

- **Deterministic Keep-Rule**
  - Keeps the **first occurrence** encountered for a given `text` (subsequent duplicates are discarded).

How it works (order of operations):

1. **Summary Stats**  
   Prints the total rows, number of unique `text` entries, and duplicate count/percentage for the merged view.

2. **Intra-Dataset Cleanup** (`duplicates_single`)  
   For each dataset, removes duplicates by `text` and reports the number removed and the percentage shrinkage.

3. **Cross-Dataset Cleanup** (`duplicates_across`)  
   Compares all dataset pairs; overlapping texts are removed from the **left** dataset in each pairwise comparison (the iteration order is deterministic). Reports per-pair overlap counts and relative percentages.

4. **Concatenate Clean Data**  
   Returns a single, cleaned `DataFrame` created by concatenating all pruned datasets.

Outputs & logging:

- **Printed summary** (before and after cleanup):
  - Total rows  
  - Unique `text` entries  
  - Duplicate count & percentage  
  - Per-dataset removals (counts and % shrinkage)  
  - Per-pair overlaps across datasets

- **Final artifact**: a **deduplicated `DataFrame`** ready for downstream steps (domain tagging → balancing → augmentation).


---

### 🔧 Preprocessing

The `clean_text()` function applies a sequence of configurable text cleaning steps:

- **Strip HTML** tags
- **Remove URLs**
- **Remove digits**
- **Remove punctuation**
- **Remove special characters**
- **Convert to lowercase**
- **Normalize whitespace**

Each step can be toggled via the `config.yml` file.

---


### 🔁 Optional: Data Augmentation

Before preprocessing, the pipeline can optionally enhance the dataset through various **data augmentation techniques**. These are particularly useful for improving model generalization, especially in low-resource or imbalanced scenarios.

The following augmentation strategies are available:

- **Paraphrasing**  
  Generates alternate versions of input text that express the same meaning using different phrasing. This helps the model learn to generalize across varied sentence structures. Paraphrasing is powered by a large language model (LLM) to ensure fluency and semantic equivalence.

- **Style Transfer**  
  Rewrites text in different rhetorical or stylistic tones. For example, a neutral sentence could be transformed into a headline-style summary, a simplified general-audience version, or one with a skeptical or inquisitive tone. This encourages the model to be robust to stylistic variance and is especially valuable for tasks like sentiment analysis or content classification.

Each augmentation method:
- Is applied to a **sampled portion** of the dataset (e.g., 10–30% of samples)
- Produces **new augmented examples**, which are then **appended to the dataset**
- Can be selectively **enabled or disabled** through the `config.yml` file
- Supports **LLM-based** augmentation

📁 **Configurable via**:  
All augmentation logic — including methods used, percentage of data to augment, and the specific style/tone targets — can be defined in the `augmentation` section of your `config.yml`.

This design provides a flexible and modular way to enrich training data without modifying the original samples.


---

### ⚖️ Optional: Data Balancing  

Before training, the pipeline can optionally **rebalance the dataset** to reduce class or domain skew. This ensures fairer representation across data sources and labels, preventing the model from overfitting to dominant groups.  

The following balancing strategies are available:  

- **Domain-Level Balancing**  
  Caps the number of samples per domain based on the smallest domain size and a configurable **tolerance**. This ensures that no single domain overwhelms the dataset.  

- **Label-Aware Sampling**  
  Within each domain, samples are proportionally reduced per label (e.g., `0` vs. `1`) so that both classes remain present while respecting the domain cap.  

Each balancing operation:  
- Works by **downsampling** (i.e., reducing over-represented groups) rather than oversampling  
- Is applied **before preprocessing and augmentation**  
- Preserves natural variation while **reducing dataset volume**  
- Can be selectively **enabled or disabled** through the `config.yml` file  

📁 **Configurable via**:  
All balancing logic — including whether balancing is active and the acceptable **tolerance level** for imbalance — can be defined in the `balance_data` section of your `config.yml`.  


---


### 🏷️ Domain Tagging

Before preprocessing, the pipeline **infers and attaches domain labels** to each sample. These labels help with analysis, stratified splits, balancing, and targeted augmentation.

What it does:

- **Sentence Embeddings**  
  Encodes each `text` entry using a `SentenceTransformer` model.  
  Optionally saves/loads embeddings to speed up reruns.

- **Unsupervised Clustering (MiniBatchKMeans)**  
  Groups samples into `n_domains` clusters for scalability on large datasets.

- **LLM-Based Cluster Naming**  
  Converts opaque cluster IDs into concise, human-readable **domain labels** (e.g., “Health Policy”, “Sports”).  
  Labels avoid value judgments (no “Lies”, “False Statements”, etc.) to keep domains non-normative.

- **Optional 2D Visualization**  
  Uses t-SNE for dimensionality reduction and plots clusters for a quick qualitative check.

Each tagging run:
- Adds a **`domain` column** with human-readable labels
- Can **cache embeddings** to `output/embeddings.npy`
- Can **reuse cached embeddings** to avoid recompute
- Can **plot** a sampled t-SNE scatter of clusters with legend

📁 **Configurable via**:  
All domain tagging options — embedding reuse/saving, number of domains, and plotting — live in the `domain_tagging` section of your `config.yml`.


---

## ⚙️ Configuration

All preprocessing, augmentation, and feature extraction steps are controlled through the `config.yml` file for flexible and reproducible experimentation.

---

## 🚀 Centralized Pipeline

The entire process is coordinated in `pipeline.py`, which serves as the main entry point for running the complete pipeline.

---

## 📦 Output Dataset

At the end of preprocessing, the pipeline exports **train/val** and **test** splits as `.parquet` files.  
The output folder structure reflects both the **test set strategy** and whether **balancing/augmentation** were applied.

#### 🔀 Test Set Options
- **Cross-Domain Testing** (`test_cross_domains: true`)  
  - Hold out entire domains for testing.  
  - Domains can be:
    - Randomly sampled (`random_domains: true`)  
    - Manually specified (`use_manual_domains: true`)  
    - Or chosen as the *n* smallest domains.  
- **In-Domain Testing** (`test_cross_domains: false`)  
  - Standard random split using `test_size` (e.g., 10%).  

#### 📂 Folder Naming
- `raw/` – no balancing or augmentation  
- `balanced/` – balancing only  
- `augmented/` – augmentation only  
- `balanced_augmented/` – both balancing & augmentation  

Each run is further grouped by:
- **`cross_domain/`** or **`in_domain/`** (depending on test strategy)

#### 📁 **Configurable via**:  
All test split logic is defined in the `test_set` section of the `config.yml`.

---

## ⬇️ Download Preprocessed Data

You can skip processing and download already-prepared datasets using:

```bash
python download_processed_data.py
