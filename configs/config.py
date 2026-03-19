from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/'data'
RAW_DATA_DIR = DATA_DIR/'raw'
ANNOTATION_DIR = DATA_DIR/'annotations'
MASK_DIR = DATA_DIR/'masks'

BATCH_SIZE = 2
EPOCHS = 50
LR = 3e-4
VAL_SPLIT = 0.2
SEED = 42

# Training settings
REMOVE_EMPTY_SLICES = True
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5