# Data
DATA_ROOT = './patches'
STAIN_REFERENCE_PATH = './data/stain_reference.png'

CLASS_NAMES = {
    0: 'Background',
    1: 'Neutrophil',
    2: 'Epithelial',
    3: 'Lymphocyte',
    4: 'Plasma',
    5: 'Eosinophil',
    6: 'Connective Tissue'
}

# Class weights for imbalanced dataset
# Based on inverse frequency: weight = total_samples / (num_classes * class_count)
CLASS_WEIGHTS = [
    0.0,      # Background (not used in loss typically)
    104.96,   # Neutrophil
    2.05,     # Epithelial
    4.68,     # Lymphocyte
    17.37,    # Plasma
    144.98,   # Eosinophil
    4.44,     # Connective Tissue
]

# Model architecture
NUM_CLASSES = 7
IN_CHANNELS = 3
BASE_CHANNELS = 64
DEPTH = 4

# Training parameters
PATCH_SIZE = 256
STRIDE = 128  
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Validation
VAL_BATCH_SIZE = 8

# Checkpointing
CHECKPOINT_DIR = './checkpoints'
LOG_DIR = './logs'
EXPERIMENT_NAME = "baseline"

# Weights & Biases
USE_WANDB = True
WANDB_PROJECT = 'zneus2'
WANDB_ENTITY = 'eakosh-' 
WANDB_LOG_MODEL = True 
WANDB_WATCH_MODEL = True  