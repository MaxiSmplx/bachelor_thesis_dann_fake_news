import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Imports
import yaml
from transformers import AutoTokenizer, AutoConfig

# Read config
with open("pipeline/config.yml", "r") as f:
    config = yaml.safe_load(f)

# Data Loading
TOKENIZER_NAME = "bert-base-uncased"
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
FILE_PATH = "pipeline/output/preprocessed_data.parquet"
FILE_PATH_AUGMENTED = "pipeline/output/preprocessed_data_augmented.parquet"

# Model architecture
INPUT_DIM = AutoConfig.from_pretrained(TOKENIZER_NAME).hidden_size
FEATURE_DIM = 128 
NUM_CLASSES = 2
NUM_DOMAINS = config["domain_tagging"]["n_domains"]

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 20

# Gradient Reversal Layer schedule
GRL_LAMBDA_MAX = 1.0
GRL_WARMUP_EPOCHS = 1.0

# Model
CHECKPOINT_DIR = "models/Domain Adversarial Neural Network/checkpoints"
LOG_DIR = "models/Domain Adversarial Neural Network/logs"