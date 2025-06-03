import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Imports
import yaml
from transformers import AutoTokenizer, AutoConfig

# Read config
with open("pipeline/config.yml", "r") as f:
    config = yaml.safe_load(f)

TOKENIZERS = {
    "MiniL6": "sentence-transformers/all-MiniLM-L6-v2",
    "MiniL12": "sentence-transformers/all-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "roberta": "sentence-transformers/all-distilroberta-v1"
}


# Data Loading
TOKENIZER_NAME = TOKENIZERS["MiniL6"]
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
FILE_PATH = "pipeline/output/preprocessed_data.parquet"
FILE_PATH_AUGMENTED = "pipeline/output/preprocessed_data_augmented.parquet"
FILE_PATH_BALANCED = "pipeline/output/preprocessed_data_balanced.parquet"
FILE_PATH_AUGMENTED_BALANCED = "pipeline/output/preprocessed_data_augmented_balanced.parquet"

# Model architecture
INPUT_DIM = AutoConfig.from_pretrained(TOKENIZER_NAME).hidden_size
FEATURE_DIM = 128 
NUM_CLASSES = 2
NUM_DOMAINS = config["domain_tagging"]["n_domains"]

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 48
NUM_EPOCHS = 20

# Gradient Reversal Layer schedule
GRL_LAMBDA_MAX = 1.0
GRL_WARMUP_EPOCHS = 1.0

# Model
CHECKPOINT_DIR = "models/Domain Adversarial Neural Network/checkpoints"
LOG_DIR = "models/Domain Adversarial Neural Network/logs"
TENSORBOARD_DIR = "models/Domain Adversarial Neural Network/tensorboard"