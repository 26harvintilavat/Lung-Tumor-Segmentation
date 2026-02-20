import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.unet import UNet
from src.dataset import LungCTDataset
from configs.config import RAW_DATA_DIR

def run_inference(patient_id, slice_idx=None, checkpoint_path="checkpoints/best_unet.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Model on {device}")
    
    # Initialize and load model
    model = UNet(n_channels=1, n_classes=1).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights not found at {checkpoint_path}. Train the model first.")
        return
        
    model.eval()
    
    print(f"Loading Patient Data: {patient_id}")
    patient_dir = Path(RAW_DATA_DIR) / patient_id
    
    if not patient_dir.exists():
        print(f"Patient directory {patient_dir} not found.")
        return
        
    dataset = LungCTDataset(patient_dir)
    print(f"Loaded {len(dataset)} slices for {patient_id}.")
    
    # Focus on a specific slice if requested, otherwise find middle slice
    if slice_idx is None:
        slice_idx = len(dataset) // 2
        print(f"No slice_idx provided, testing middle slice: {slice_idx}")
        
    img_np = dataset.get_slice(slice_idx)
    
    # Normalize
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-8)
    
    # Convert to Tensor (B, C, H, W)
    img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    
    print("Running Inference...")
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
    pred_np = preds.squeeze().cpu().numpy()
    
    # Visualization setup
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original CT Slice ({slice_idx})")
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Predicted Tumor Overlay")
    plt.imshow(img_np, cmap='gray')
    # Overlay predictions in red with alpha
    plt.imshow(pred_np, cmap='Reds', alpha=0.4 * pred_np) 
    plt.axis('off')
    
    plt.tight_layout()
    save_path = f"inference_{patient_id}_slice{slice_idx}.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Visualized result saved to {save_path}")

if __name__ == "__main__":
    # Test on the synthetic patient built previously
    run_inference(patient_id="LIDC-IDRI-0001", slice_idx=2)
