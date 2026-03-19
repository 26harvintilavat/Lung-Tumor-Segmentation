import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from configs.config import RAW_DATA_DIR, MASK_DIR, BATCH_SIZE, LR, EPOCHS, VAL_SPLIT, SEED
from src.train_dataset import LungSegmentationDataset
from src.model import UNet
from scripts.prepare_dataloaders import get_patient_ids, split_patients
from src.losses import BCEDiceLoss, dice_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Train one epoch
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, masks in progress_bar:
        images = images.to(device, memory_format= torch.channels_last, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # outputs = model(images)
        # loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        with autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss/len(loader)

# Validation
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0

    progress_bar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device, memory_format= torch.channels_last, non_blocking=True)
            masks = masks.to(device)

            with autocast(device_type = "cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)

            batch_dice = dice_score(outputs, masks)
            total_dice += batch_dice

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
    
    return total_loss/len(loader), total_dice/len(loader)

# Main training function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    scaler = GradScaler()

    # Patients
    patient_ids = get_patient_ids(MASK_DIR)
    train_ids, val_ids = split_patients(patient_ids, VAL_SPLIT, SEED)

    print("Train patients:", train_ids)
    print("Val patients:", val_ids)

    # Datasets 
    print("Creating train dataset...")
    train_dataset = LungSegmentationDataset(RAW_DATA_DIR, MASK_DIR, train_ids)
    print("Train dataset created")

    print("Creating val dataset...")
    val_dataset = LungSegmentationDataset(RAW_DATA_DIR, MASK_DIR, val_ids)
    print("Val dataset created")

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers= 0,
        pin_memory=True,
        persistent_workers=False 
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
        )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    # Model
    model = UNet(in_channels=3, out_channels=1).to(device, memory_format=torch.channels_last)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Loss & optimizer
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        verbose=True
    )

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    checkpoint_path = save_dir/"best_model.pth"
    start_epoch = 0

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    print("\nStarting training...\n")

    if checkpoint_path.exists():
        print(f"Checkpoints found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get("best_val_loss", float('inf'))
 
            print(f"Resuming from epoch {start_epoch}")

        else:
            model.load_state_dict(checkpoint)
            start_epoch = 0
            print("Loaded weights only. Optimizer reinitialized.")

    else:
        print("No checkpoint found. Starting training from scratch.")

    # Epoch loop
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)

        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:6f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"Model saved at {checkpoint_path}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Curve")

    plot_path = save_dir/"training_curve.png"
    plt.savefig(plot_path)
    print(f"Training curve saved at {plot_path}")

if __name__ == "__main__":
    main()