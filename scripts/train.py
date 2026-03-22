import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from configs.config import (
    RAW_DATA_DIR, MASK_DIR,
    BATCH_SIZE, LR, EPOCHS,
    VAL_SPLIT, SEED, IMG_SIZE
)
from src.train_dataset import LungSegmentationDataset
from src.model import LungAttentionUNet
from src.losses import TverskyFocalLoss, dice_score

# Train one epoch
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    total_dice = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, masks in progress_bar:

        optimizer.zero_grad(set_to_none=True)

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

        scaler.scale(loss).backward()
        
        # gradient clipping 
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        batch_dice = dice_score(outputs, masks)
        total_loss += loss.item()
        total_dice += batch_dice

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{batch_dice:.4f}"
        )

    n = len(loader)
    return total_loss/n, total_dice/n

# Validation
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_samples = 0 # weighted average by batch size

    progress_bar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device, memory_format= torch.channels_last, non_blocking=True)

            masks = masks.to(device, non_blocking=True)

            with autocast(device_type = "cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)

            batch_size = images.size(0)
            batch_dice = dice_score(outputs, masks)

            total_loss += loss.item() * batch_size
            total_dice += batch_dice * batch_size
            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{batch_dice:.4f}"
            )

    return total_loss/total_samples, total_dice/total_samples

# Main training function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    scaler = GradScaler()

    # Patients
    mask_dir    = Path(MASK_DIR)
    patient_ids = [
    f.stem.replace('_mask', '')
    for f in mask_dir.glob('*_mask.npy')
]

    train_ids, val_ids = train_test_split(
        patient_ids,
        test_size=VAL_SPLIT,
        random_state=SEED
)

    print(f"Total patients : {len(patient_ids)}")
    print(f"Train patients : {len(train_ids)}")
    print(f"Val patients   : {len(val_ids)}")

    print("Train patients:", train_ids)
    print("Val patients:", val_ids)

    # Datasets 
    print("Creating train dataset...")
    train_dataset = LungSegmentationDataset(
        RAW_DATA_DIR, 
        MASK_DIR, 
        train_ids,
        img_size=IMG_SIZE,
        augment=True,
        min_tumor_pixels=10,
        bg_ratio=2
        )
    print("Train dataset created")

    print("Creating val dataset...")
    val_dataset = LungSegmentationDataset(
        RAW_DATA_DIR, 
        MASK_DIR, 
        val_ids,
        img_size=IMG_SIZE,
        augment=False,
        min_tumor_pixels=10,
        bg_ratio=2)
    print("Val dataset created")

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    num_workers = min(4, os.cpu_count())
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
    model = LungAttentionUNet(in_channels=3, out_channels=1).to(device, memory_format=torch.channels_last)

    # Loss & optimizer
    criterion = TverskyFocalLoss(
        tversky_alpha=0.3,
        tversky_beta=0.7,
        focal_gamma=2.0,
        tversky_weight=0.7
    )
    optimizer = Adam(model.parameters(), lr=LR)
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=5
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=45,
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[5]
    )

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    checkpoint_path = save_dir/"best_model.pth"
    start_epoch = 0
    best_val_dice = 0.0
    early_stop_counter = 0
    PATIENCE = 15

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    print("\nStarting training...\n")

    if checkpoint_path.exists():
        print(f"Checkpoints found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            best_val_dice = checkpoint.get('best_val_dice', 0.0)
            print(f"Resuming from epoch {start_epoch} | Best Val Dice so far: {best_val_dice:.4f}")

        else:
            model.load_state_dict(checkpoint)
            start_epoch = 0
            print("Loaded weights only. Optimizer reinitialized.")

    else:
        print("No checkpoint found. Starting training from scratch.")

    # Epoch loop
    for epoch in range(start_epoch, EPOCHS):
        train_loader.dataset.resample_per_epoch()
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)

        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_dice': best_val_dice,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Best model saved - Val Dice: {val_dice:.4f}")

        else:
            early_stop_counter += 1
            print(f"No improvement. Patience: {early_stop_counter}/{PATIENCE}")

            if early_stop_counter >= PATIENCE:
                print(f"\n Early stopping at epoch {epoch+1}.")
                print(f"Best Val Dice: {best_val_dice:.4f}")
                break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss", color="blue")
    ax1.plot(val_losses,   label="Val Loss",   color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_dices, label="Train Dice", color="green")
    ax2.plot(val_dices,   label="Val Dice",   color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Dice Score Curve")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = save_dir / "training_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nTraining curve saved at {plot_path}")


if __name__ == "__main__":
    main()