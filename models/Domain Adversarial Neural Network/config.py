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
    "roberta": "sentence-transformers/all-distilroberta-v1",
    "bert": "bert-base-uncased"
}


# Data Loading
TOKENIZER_NAME = TOKENIZERS["MiniL6"]
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
FOLDER_PATH_RAW = "pipeline/output/raw"
FOLDER_PATH_AUGMENTED = "pipeline/output/augmented"
FOLDER_PATH_BALANCED = "pipeline/output/balanced"
FOLDER_PATH_BALANCED_AUGMENTED = "pipeline/output/balanced_augmented"

# Model architecture
INPUT_DIM = AutoConfig.from_pretrained(TOKENIZER_NAME).hidden_size
FEATURE_DIM = 128 
NUM_CLASSES = 1
NUM_DOMAINS = config["domain_tagging"]["n_domains"]

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 48
NUM_EPOCHS = 20

# Gradient Reversal Layer schedule
GRL_LAMBDA_CEILING = 1.2
GRL_WARMUP = 0.0

# Model
CHECKPOINT_DIR = "models/Domain Adversarial Neural Network/checkpoints"
LOG_DIR = "models/Domain Adversarial Neural Network/logs"
TENSORBOARD_DIR = "models/Domain Adversarial Neural Network/tensorboard"