import numpy as np
import os
from pathlib import Path

patient_id = "LIDC-IDRI-0001"
dcm_dir = Path("data/masks")
dcm_dir.mkdir(parents=True, exist_ok=True)

# 5 slices, 512x512
mask = np.zeros((5, 512, 512), dtype=np.uint8)
# Add some dummy tumor
mask[2:4, 250:260, 250:260] = 1 
mask[2, 300:310, 300:310] = 1 

# Fix missing save format 
np.save(dcm_dir / f"{patient_id}.npy", mask)
print("created dummy mask")
