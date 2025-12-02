import torch

# Data
DATA_ROOT = '/kaggle/input/lizard-in-patches/patches'
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

CLASS_WEIGHTS = torch.tensor([
    1.0000,  
    491.3849,
    7.8228,  
    43.6471,  
    137.7204, 
    903.1321, 
    25.1461,  
], dtype=torch.float32)


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
EARLY_STOPPING_PATIENCE = 5

# Validation
VAL_BATCH_SIZE = 8

# Checkpointing
CHECKPOINT_DIR = './checkpoints'
LOG_DIR = './logs'
EXPERIMENT_NAME = "unet_class_weights"

# Weights & Biases
USE_WANDB = True
WANDB_PROJECT = 'zneus2'
WANDB_ENTITY = 'eakosh-' 
WANDB_LOG_MODEL = True 
WANDB_WATCH_MODEL = True  

# Visualization
VISUALIZE_NUM_SAMPLES = 3
VISUALIZE_EVERY_N_EPOCHS = 10
VAL_IMG_DIR = DATA_ROOT + '/val/img'
VAL_MASK_DIR = DATA_ROOT + '/val/mask'