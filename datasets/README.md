## 🗂️ Project Structure

This section outlines the organization of datasets and file access patterns used throughout the project.

---

### 📁 Folder Layout

All datasets are stored in the `datasets/` directory, with each dataset housed in its own subfolder containing `real.csv` and `fake.csv` files.

```plaintext
datasets/
├── dataset1/
│   ├── real.csv
│   └── fake.csv
├── dataset2/
│   ├── real.csv
│   └── fake.csv
├── dataset3/
│   ├── real.csv
│   └── fake.csv
...
```

---

### 📌 How to Access in Code Notebooks

To load the files in your Jupyter notebooks or Python scripts, use the following path pattern:

`real_path = "../datasets/<dataset_name>/real.csv"`\
`fake_path = "../datasets/<dataset_name>/fake.csv"`\
\
Or use the `load_dataset()` function in **common_functions**
