from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# Experiment Configuration
# =============================================================================

EXPERIMENT = "exp_e_window_sens" 
'''
baseline
exp_a_prefix
exp_b_entities
exp_c_tabular
exp_d_hybrid
exp_e_window_sens - sub folders : exp_e_window3 & exp_e_window10
'''

USE_HYBRID = EXPERIMENT == "exp_d_hybrid"

# =============================================================================
# Model Selection
# =============================================================================

MODEL_TYPE = "hybrid" if USE_HYBRID else "distilbert"  # "lstm" | "distilbert" for else

TOKENIZER_NAMES = {
    "lstm": "distilbert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "hybrid": "distilbert-base-uncased",
}

TOKENIZER_NAME = TOKENIZER_NAMES[MODEL_TYPE]

MODEL_FILES = {
    "lstm": "lstm_baseline.pth",
    "distilbert": "bert_momentum.pth",
    "hybrid": "bert_hybrid_momentum.pth",
}

DISPLAY_NAMES = {
    "lstm": "LSTM Baseline",
    "distilbert": "DistilBERT",
    "hybrid": "Hybrid DistilBERT"
}

# =============================================================================
# Dataset
# =============================================================================

MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_WORKERS = 0

CLASS_NAMES = [
    "Home Dominant",
    "Away Dominant",
    "Balanced"
]

# =============================================================================
# Model Hyperparameters
# =============================================================================

VOCAB_SIZE = 30522
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 3
DROPOUT = 0.3

# =============================================================================
# Training
# =============================================================================

EPOCHS = 10

LEARNING_RATE = 1e-3
BERT_LEARNING_RATE = 2e-5

EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 1e-4