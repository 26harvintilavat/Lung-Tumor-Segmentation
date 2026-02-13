"""
Configuration file for TCIA dataset download.

Modify these settings according to your needs.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/'data'
RAW_DATA_DIR = DATA_DIR/'raw'
ANNOTATION_DIR = DATA_DIR/'annotations'
MASK_DIR = DATA_DIR/'masks'

BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-4
VAL_SPLIT = 0.2
SEED = 42