# Fake News Detection Using Domain-Adversarial Neural Networks

**Bachelor Thesis by Maximilian Schulz**  
University of Hamburg – Faculty of Mathematics, Informatics and Natural Sciences  
🗓️ *May 2025*

---

## 📘 Overview

This project investigates the development of a **Domain-Adversarial Neural Network (DANN)** for **robust fake news detection** across multiple domains. The central aim is to enhance the model's **semantic generalization** ability, enabling it to identify fake news not only in a specific dataset or domain but also across diverse platforms (e.g., news articles, social media posts, journalistic statements).

The architecture incorporates **Multi-Domain Learning** and **Adversarial Invariance Techniques**, with extensive comparisons to other machine learning and deep learning baselines—including **Support Vector Machines, Naive Bayes, Random Forests**, and **Large Language Models**.

---

## ❓ Research Question

> _"What training strategies and regularization methods enable the development of a Domain-Adversarial Neural Network for Fake News detection that generalizes across domains and prioritizes semantic robustness over stylistic correlations?"_

---

## 🧠 Core Contributions

- Implementation of a **Domain-Adversarial Neural Network (DANN)** with a Gradient Reversal Layer (GRL)
- Benchmarking against **SVM, Naive Bayes, Logistic Regression, CNN**, and **LLMs (Transformer-based models)**
- Integration of **domain-specific data** from various public datasets
- Usage of **data augmentation** and **robustness techniques** (dropout, adversarial training, noise injection)
- Performance evaluation across metrics: Accuracy, Precision, Recall, F1-score, False Positive Rate, and more

---

## 📊 Datasets Used

### 🚂 Training Datasets

| Dataset | Domain | Type | Size |
|--------|--------|------|------|

TODO

### 🧪 Test Datasets

| Dataset | Domain | Type | Size |
|--------|--------|------|------|

TODO

All datasets can be found at:  
🔗 **[Raw Datasets](https://drive.google.com/drive/folders/1d_2XZ3N9c1Nmncj1CSrnQAaUaoG_xkyv?usp=share_link)** \
🔗 **[Processed Datasets](https://drive.google.com/drive/folders/1-uSCjx6wC7Rh-6gJ1MC0rUFKoF1chw42?usp=share_link)**

---

## ⚙️ Key Methods

- **Domain-Adversarial Neural Network** with:
  - Feature Extractor
  - Label Predictor
  - Domain Classifier
  - Gradient Reversal Layer
- **Text Preprocessing**:
  - Cleaning, tokenization, normalization
  - Feature extraction via Transformers (e.g., BERT)
- **Data Augmentation**:
  - Synonym replacement
  - LLM-based paraphrasing and style transfer
- **Regularization Techniques**:
  - Dropout, Weight Decay, Adversarial Training, Noise Injection

---

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- False Positive / Negative Rate
- Specificity
- Statistical Significance (p-values, confidence intervals)

---

## 🔬 Baseline Models

- **Traditional ML**: SVM, Naive Bayes, Random Forest
- **Deep Learning**: CNN
- **Transformer-based LLMs**

---

## 📦 Environment Setup

All dependencies are listed in the `environment.yml` file.

Create a new conda environment using:

`conda env create -f environment.yml` \
`conda activate fake-news-detection`





