import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader

from configs.config import RAW_DATA_DIR, MASK_DIR, BATCH_SIZE, VAL_SPLIT, SEED
from src.train_dataset import LungSegmentationDataset
from src.model import LungAttentionUNet
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_patient_ids(mask_dir):
    mask_dir = Path(mask_dir)
    return [
        f.stem.replace('_mask', '')
        for f in mask_dir.glob('*_mask.npy')
    ]

def split_patients(patient_ids, val_split, seed):
    return train_test_split(
        patient_ids,
        test_size=val_split,
        random_state=seed
    )

def dice_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

def main():
    patient_ids = get_patient_ids(MASK_DIR)
    _, val_ids = split_patients(patient_ids, VAL_SPLIT, SEED)

    val_dataset = LungSegmentationDataset(RAW_DATA_DIR, MASK_DIR, val_ids)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LungAttentionUNet(in_channels=3, out_channels=1).to(device)

    checkpoint_path = Path("checkpoints/best_model.pth")
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dices = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)

            for p, m in zip(preds, masks):
                dices.append(dice_score(p, m).item())

    print(f"\nFinal Validation Dice: {np.mean(dices):.4f}")

if __name__=="__main__":
    main()