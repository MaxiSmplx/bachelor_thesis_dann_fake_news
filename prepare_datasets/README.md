# 📊 Dataset Processing Pipeline

This project provides a streamlined pipeline for preparing machine learning-ready datasets from various raw data sources.

## 📁 Data Sources

All raw datasets are available in a shared [Google Drive folder](https://drive.google.com/drive/folders/1d_2XZ3N9c1Nmncj1CSrnQAaUaoG_xkyv?usp=sharing). These datasets were either:

- Directly downloaded from online repositories
- Programmatically generated using the [`make_dataset.ipynb`](make_dataset.ipynb) notebooks

## ⚙️ Processing Pipeline

Each raw dataset is processed individually using the [`process_dataset.ipynb`](process_dataset.ipynb) notebooks. This step cleans and formats the data, producing two structured files per dataset:

- `fake.csv`
- `real.csv`

These files are ready to be used for machine learning training tasks.

## 📌 Notes

- Each dataset follows the same format and structure after processing for consistency.
- You can explore or modify the notebooks to customize the pipeline for your own needs.
