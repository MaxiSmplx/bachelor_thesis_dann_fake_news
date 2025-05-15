## 🗂️ Project Structure

This section outlines the organization of datasets and file access patterns used throughout the project.

---

### 📁 Folder Layout

All datasets are stored in the `datasets/` directory, with each dataset housed in its own subfolder containing `real.csv` and `fake.csv` files.

```plaintext
datasets/
├── dataset1/
│   ├── real.parquet
│   └── fake.parquet
├── dataset2/
│   ├── real.parquet
│   └── fake.parquet
├── dataset3/
│   ├── real.parquet
│   └── fake.parquet
...
```

---

### 📌 How to Access in Code Notebooks

To load the files in your Jupyter notebooks or Python scripts, use the following path pattern:

`real_path = "../datasets/<dataset_name>/real.parquet"`\
`fake_path = "../datasets/<dataset_name>/fake.parquet"`\
\
Or use the `load_dataset()` function in **common_functions**
