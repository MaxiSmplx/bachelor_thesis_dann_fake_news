# 🧪 NLP Processing Pipeline

This folder contains the complete NLP data processing pipeline, including text preprocessing, and optional data augmentation. The entire flow is orchestrated by a central `pipeline.py` script and is fully configurable via a `config.yml` file.

---

## 📁 Contents

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

- **Synonym Replacement**  
  Randomly replaces selected words in a sentence with their synonyms. This preserves the original meaning while increasing linguistic variety. The method can use either a local synonym database or an LLM-based lookup for greater contextual accuracy.

- **Paraphrasing**  
  Generates alternate versions of input text that express the same meaning using different phrasing. This helps the model learn to generalize across varied sentence structures. Paraphrasing is powered by a large language model (LLM) to ensure fluency and semantic equivalence.

- **Style Transfer**  
  Rewrites text in different rhetorical or stylistic tones. For example, a neutral sentence could be transformed into a headline-style summary, a simplified general-audience version, or one with a skeptical or inquisitive tone. This encourages the model to be robust to stylistic variance and is especially valuable for tasks like sentiment analysis or content classification.

Each augmentation method:
- Is applied to a **sampled portion** of the dataset (e.g., 10–30% of samples)
- Produces **new augmented examples**, which are then **appended to the dataset**
- Can be selectively **enabled or disabled** through the `config.yml` file
- Supports **LLM-based** augmentation and/or **lightweight local alternatives**

📁 **Configurable via**:  
All augmentation logic — including methods used, percentage of data to augment, and the specific style/tone targets — can be defined in the `augmentation` section of your `config.yml`.

This design provides a flexible and modular way to enrich training data without modifying the original samples.


---

## ⚙️ Configuration

All preprocessing, augmentation, and feature extraction steps are controlled through the `config.yml` file for flexible and reproducible experimentation.

---

## 🚀 Centralized Pipeline

The entire process is coordinated in `pipeline.py`, which serves as the main entry point for running the complete pipeline.

---

## 💾 Output

All processed data is saved automatically in the `/output` directory.

---

## ⬇️ Download Preprocessed Data

You can skip processing and download already-prepared datasets using:

```bash
python download_processed_data.py
