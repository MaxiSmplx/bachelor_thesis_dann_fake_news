# 📊 Dataset Processing Pipeline

This project provides a streamlined pipeline for preparing machine learning-ready datasets from various raw data sources.

## 📁 Data Sources

All raw datasets are available in a shared [Google Drive folder](https://drive.google.com/drive/folders/1d_2XZ3N9c1Nmncj1CSrnQAaUaoG_xkyv?usp=sharing). These datasets were either:

- Directly downloaded and exported as `.parquet`
- Programmatically generated using the **make_dataset.ipynb** notebooks

> 📥 **All raw datasets (before preprocessing) can be downloaded automatically**  
> using the `download_raw_datasets.py` script in the `prepare_datasets/` directory.


## 📃 Datasets

| Dataset Name       | Source / Download Link |
|--------------------|------------------------|
| WELFAKE    | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| Source based FN    | [Kaggle](https://www.kaggle.com/datasets/yash0956/fakenews) |
| llm-misinformation | [GitHub](https://github.com/llm-misinformation/llm-misinformation) |
| LIAR2 | [Hugging Facce](https://huggingface.co/datasets/chengxuphd/liar2)|
| MultiFC | [Hugging Face](https://huggingface.co/datasets/pszemraj/multi_fc)) |
| FineFake | [GitHub](https://github.com/Accuser907/FineFake?tab=readme-ov-file) |
| FEVER | [FEVER](https://fever.ai/dataset/fever.html) |
| Fakeddit | [Fakeddit](https://fakeddit.netlify.app) |
| FakeNewsNet | [Kaggle](https://www.kaggle.com/datasets/algord/fake-news) |
| Fake News Corpus | [GitHub](https://github.com/several27/FakeNewsCorpus?tab=readme-ov-file) |
| Climate-FEVER | [Hugging Face](https://huggingface.co/datasets/tdiggelm/climate_fever)|


## ⚙️ Processing Pipeline

Each raw dataset is processed individually using the **process_dataset.ipynb** notebooks. This step cleans and formats the data, producing two structured files per dataset:

- `fake.parquet`
- `real.parquet`

These files are ready to be used for machine learning training tasks.

## 📌 Notes

- Each dataset follows the same format and structure after processing for consistency.
- You can explore or modify the notebooks to customize the pipeline for your own needs.
