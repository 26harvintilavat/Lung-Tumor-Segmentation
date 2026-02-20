import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm

from configs.config import RAW_DATA_DIR, MASK_DIR, BATCH_SIZE, EPOCHS, LR, VAL_SPLIT, SEED, REMOVE_EMPTY_SLICES, BCE_WEIGHT, DICE_WEIGHT
from configs.config import RAW_DATA_DIR, MASK_DIR, BATCH_SIZE, EPOCHS, LR, VAL_SPLIT, SEED, REMOVE_EMPTY_SLICES, BCE_WEIGHT, DICE_WEIGHT
from src.dataset import LungCTDataset
from models.unet import UNet
from evaluation.metrics import CombinedLoss, compute_metrics

# Optional callback type for dynamic status updates
from typing import Callable, Dict, Any, Optional

class PyTorchLungDataset(Dataset):
    """
    Wraps the existing abstraction `LungCTDataset` into a PyTorch compatible loader.
    Iterates through patients, extracts slices and pairs them with 3D numpy arrays.
    """
    def __init__(self, raw_dir, mask_dir, remove_empty=True):
        self.raw_dir = Path(raw_dir)
        self.mask_dir = Path(mask_dir)
        self.remove_empty = remove_empty
        
        self.samples = [] # List of tuples: (patient_id, slice_idx)
        self.patient_datasets = {} # patient_id -> LungCTDataset
        self.patient_masks = {}    # patient_id -> numpy 3D array
        
        self._prepare_dataset()
        
    def _prepare_dataset(self):
        print("Preparing PyTorch Dataset. Scanning patients...")
        # Find all patients with valid mask files
        valid_patients = [p.stem for p in self.mask_dir.glob("*.npy")]
        
        for patient_id in valid_patients:
            patient_dir = self.raw_dir / patient_id
            if not patient_dir.exists():
                continue
                
            try:
                # Load CT Volumetric slice abstraction
                ct_data = LungCTDataset(patient_dir)
                # Load corresponding binary mask
                mask_vol = np.load(self.mask_dir / f"{patient_id}.npy")
                
                # Verify depth matches
                if len(ct_data) != mask_vol.shape[0]:
                    print(f"Warning: Depth mismatch for {patient_id}. CT: {len(ct_data)}, Mask: {mask_vol.shape[0]}")
                    continue
                    
                self.patient_datasets[patient_id] = ct_data
                self.patient_masks[patient_id] = mask_vol
                
                for i in range(len(ct_data)):
                    if self.remove_empty:
                        # Only add slices that have at least one True (tumor) pixel
                        if np.any(mask_vol[i]):
                            self.samples.append((patient_id, i))
                    else:
                        self.samples.append((patient_id, i))
                        
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                
        print(f"Dataset prepared. Total valid slices loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        patient_id, slice_idx = self.samples[idx]
        
        # Extract pre-processed HU normalized 2D image
        ct_data = self.patient_datasets[patient_id]
        img = ct_data.get_slice(slice_idx)
        
        # Normalization 0-1 for CNN compatibility
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        
        # Extract mask slice
        mask = self.patient_masks[patient_id][slice_idx]
        
        # PyTorch expects (C, H, W). Image is (H, W), add channel axis.
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return img_tensor, mask_tensor
        

def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    loss_function: str = "dice_bce",
    remove_empty_slices: bool = REMOVE_EMPTY_SLICES,
    update_status_callback: Optional[Callable[[Dict[str, Any]], None]] = None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Initiating Training Pipeline on: {device}")
    
    # Optional status updater
    def emit_status(updates):
        if update_status_callback:
            update_status_callback(updates)
            
    emit_status({"status": "running", "current_epoch": 0})
    
    # 1. Dataset & DataLoader configuration
    torch.manual_seed(SEED)
    dataset = PyTorchLungDataset(RAW_DATA_DIR, MASK_DIR, remove_empty=remove_empty_slices)
    
    if len(dataset) == 0:
        msg = "Error: Dataset is empty. Make sure masks are generated and available in data/masks/."
        print(msg)
        emit_status({"status": "failed", "error": msg})
        return
        
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 2. Model Initialization
    model = UNet(n_channels=1, n_classes=1).to(device)
    
    # 3. Optimizer & Loss Config
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    if loss_function == "focal":
        # Simplified focal loss implementation fallback or placeholder
        print("Focal loss selected. (Using BCE+Dice as robust alternative if pure Focal isn't defined)")
        criterion = CombinedLoss(bce_weight=0.8, dice_weight=0.2) 
    else:
        criterion = CombinedLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)
    
    best_val_dice = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    # 4. Training Loop
    for epoch in range(epochs):
        emit_status({"current_epoch": epoch + 1})
        
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            
            metrics = compute_metrics(logits, masks)
            
            train_loss += loss.item()
            train_dice += metrics['dice']
            train_iou += metrics['iou']
            
            progress_bar.set_postfix({"Loss": loss.item(), "Dice": metrics['dice']})
            
            # Emit intra-epoch progress occasionally to feel responsive
            if batch_idx % max(1, len(train_loader) // 5) == 0:
                emit_status({
                    "train_loss": train_loss / (batch_idx + 1)
                })
            
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                metrics = compute_metrics(logits, masks)
                
                val_loss += loss.item()
                val_dice += metrics['dice']
                val_iou += metrics['iou']
                
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        scheduler.step(val_dice)
        
        print(f"\n[Epoch {epoch+1}/{epochs}] Summary:")
        print(f"Train -> Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "checkpoints/best_unet.pth")
            print(f"🔥 New best model saved! (Validation Dice: {best_val_dice:.4f})")
            
        # Emit End of Epoch metrics
        emit_status({
            "val_dice": val_dice,
            "val_iou": val_iou,
            "train_loss": train_loss,
            "best_val_dice": best_val_dice
        })
            
    print(f"\nTraining Complete! Best Validation Dice: {best_val_dice:.4f}")
    emit_status({"status": "completed"})

if __name__ == "__main__":
    train()
