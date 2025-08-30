# 🧠 Models

This folder contains all modeling approaches used in the project — from **traditional ML baselines** to **Domain-Adversarial Neural Networks (DANN)** and **Large Language Models (LLMs)**. All models consume the **preprocessed Parquet outputs** produced by the pipeline.

---

## 📊 Traditional Machine Learning Models

The following models are implemented as Jupyter Notebooks (`.ipynb`), serving as interpretable and lightweight baselines:

- Logistic Regression  
- Naive Bayes  
- Support Vector Machines (SVM)  
- k-Nearest Neighbors (kNN)  
- Random Forest  
- Gradient Boosting  

Each notebook loads the preprocessed dataset, trains the model, and evaluates performance with metrics such as accuracy, precision, recall and F1-score.

---

## 🌐 Domain-Adversarial Neural Network (DANN)

A deep neural architecture designed to improve **domain generalization** by learning domain-invariant representations.

**Files included:**
- `train.py` – Training loop for DANN  
- `test.py` – Evaluation script  
- `model.py` – Model definition (feature extractor, label predictor, domain classifier, GRL)  
- `data_loader.py` – Data loading and preprocessing utilities  
- `config.py` – Central configuration of hyperparameters and paths  
- `get_model_attributes.py` – Benchmarking tool for memory usage and inference speed  

---

## 🤖 Large Language Models (LLMs)

This folder contains both fine-tuned transformer models and API-prompted inference:

- **Fine-tuned Models (BERT, RoBERTa)**  
  - `LLM_finetuned_train.py` – Fine-tunes BERT/RoBERTa on the fake news dataset  
  - `LLM_finetuned_test.py` – Evaluates the fine-tuned models
  - `data_loader.py` – Prepares input data for both fine-tuned and prompted models  
  - `get_model_attributes.py` – Measures memory footprint and inference time  

- **Prompted Models (OpenAI API)**  
  - `LLM_prompted.py` – Zero/few-shot inference using the OpenAI API  

---

## ⚖️ Model Attributes

For **DANN** and **LLMs**, the script `get_model_attributes.py` can be run to quickly profile:
- **Parameters** – Total number of trainable parameters  
- **Model size (disk)** – File size of the stored model checkpoint  
- **Model size (RAM)** – Memory used when loaded into GPU/CPU RAM  
- **Peak memory consumption** – Maximum GPU memory allocation during inference  
- **Inference time**  
  - **Total** inference time on the test set  
  - **Per-sample** average inference time  

This allows for direct comparison between lightweight ML models, DANN, and LLMs.

---

