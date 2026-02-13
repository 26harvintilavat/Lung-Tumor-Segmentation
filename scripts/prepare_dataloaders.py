from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.train_dataset import LungSegmentationDataset

from configs.config import RAW_DATA_DIR, MASK_DIR

BATCH_SIZE = 4
VAL_SPLIT = 0.2
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


def get_patient_ids(mask_dir):
    patient_ids = []
    for p in mask_dir.glob("*_mask.npy"):
        patient_ids.append(p.stem.replace("_mask", ""))
    return sorted(patient_ids)

def split_patients(patient_ids, val_split, seed):
    random.seed(seed)
    random.shuffle(patient_ids)

    n_val = max(1, int(len(patient_ids) * val_split))    
    val_ids = patient_ids[:n_val]
    train_ids = patient_ids[n_val:]

    return train_ids, val_ids

def main():
    patient_ids = get_patient_ids(MASK_DIR)
    print("Total patients:", len(patient_ids))

    train_ids, val_ids = split_patients(patient_ids, VAL_SPLIT, SEED)

    print("Train patients:", train_ids)
    print("Val patients:", val_ids)

    # Dataset
    train_dataset = LungSegmentationDataset(
        RAW_DATA_DIR,
        MASK_DIR, 
        train_ids
    )

    val_dataset = LungSegmentationDataset(
        RAW_DATA_DIR,
        MASK_DIR,
        val_ids
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    images, masks = next(iter(train_loader))

    images = images.to(device)
    masks = masks.to(device)

    print("Batch image shape:", images.shape)
    print("Batch mask shape:", masks.shape)
    print("Batch device:", images.device)

if __name__ == "__main__":
    main()
