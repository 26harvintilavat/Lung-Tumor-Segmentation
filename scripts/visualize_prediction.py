import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt

from configs.config import RAW_DATA_DIR, MASK_DIR
from src.train_dataset import LungSegmentationDataset
from src.model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(device):
    model = UNet(in_channels=1, out_channels=1).to(device)

    checkpoint_path = Path("checkpoints/best_model.pth")

    if checkpoint_path.exists():
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found. Using random weights.")

    model.eval()
    return model

def visualize_patient(patient_id, max_slices=3):
    print(f"\nPatient: {patient_id}")
    dataset = LungSegmentationDataset(RAW_DATA_DIR, MASK_DIR, [patient_id])

    model = load_model(device)
    shown = 0

    for i in range(len(dataset)):
        image, mask = dataset[i]

        # Only show slices where tumor exists
        if mask.sum() == 0:
            continue

        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.sigmoid(output)

        image = image.squeeze().numpy()
        mask = mask.squeeze().numpy()
        pred = pred.squeeze().numpy()

        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.title("CT")
        plt.imshow(image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(image, cmap="gray")
        plt.imshow(mask, cmap="jet", alpha=0.5)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred, cmap="gray")
        plt.imshow(pred > 0.5, cmap="jet", alpha=0.5)
        plt.axis("off")

        plt.show()

        shown += 1
        if shown >= max_slices:
            break

def main():
    patients=[
        "LIDC-IDRI-0001",
        "LIDC-IDRI-0005"
    ]

    for pid in patients:
        visualize_patient(pid, max_slices=2)

if __name__=="__main__":
    main()