import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs.config import RAW_DATA_DIR, MASK_DIR
from src.train_dataset import LungSegmentationDataset
from src.model import UNet
from scripts.prepare_dataloaders import get_patient_ids, split_patients
from src.losses import DiceLoss

BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 5
VAL_SPLIT = 0.2
SEED = 42

# Train one epoch
def train_one_epoch(model, loader, optimizer, bce_loss, dice_loss, device):
    model.train()
    total_loss = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(loader)

# Validation
def validate(model, loader, bce_loss, dice_loss, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

            total_loss += loss.item()
    
    return total_loss/len(loader)

# Main training function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Patients
    patient_ids = get_patient_ids(MASK_DIR)
    train_ids, val_ids = split_patients(patient_ids, VAL_SPLIT, SEED)

    print("Train patients:", train_ids)
    print("Val patients:", val_ids)

    # Datasets 
    train_dataset = LungSegmentationDataset(RAW_DATA_DIR, MASK_DIR, train_ids)
    val_dataset = LungSegmentationDataset(RAW_DATA_DIR, MASK_DIR, val_ids)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    # Model
    model = UNet(in_channels=1, out_channels=1).to(device)

    # Loss & optimizer
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    print("\nStarting training...\n")

    # Epoch loop
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, dice_loss, device)
        val_loss = validate(model, val_loader, bce_loss, dice_loss, device)

        print(f"Epoch [{epoch+1}/{EPOCHS}]"
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
if __name__ == "__main__":
    main()