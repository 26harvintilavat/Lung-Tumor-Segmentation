from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.dataset import LungCTDataset

class LungSegmentationDataset(Dataset):
    "Returns (image, mask) slice pairs for training"

    def __init__(self, raw_dir, mask_dir, patient_ids):
        self.raw_dir = Path(raw_dir)
        self.mask_dir = Path(mask_dir)
        self.samples = []

        for pid in patient_ids:
            patient_raw_dir = self.raw_dir/pid

            # find series folder
            series_dirs = [d for d in patient_raw_dir.iterdir() if d.is_dir()]
            if len(series_dirs) == 0:
                continue

            series_dir = series_dirs[0]

            # load CT volume
            dataset = LungCTDataset(series_dir)
            volume = dataset.volume

            # load mask
            mask_path = self.mask_dir/f"{pid}_mask.npy"
            if not mask_path.exists():
                continue

            mask = np.load(mask_path)

            # store slice-level samples
            for z in range(volume.shape[0]):
                self.samples.append({
                    "series_dir": series_dir,
                    "mask_path": mask_path,
                    "slice_idx": z
                })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        dataset = LungCTDataset(sample['series_dir'])
        volume = dataset.volume

        mask = np.load(sample['mask_path'])

        image = volume[sample['slice_idx']]
        mask = mask[sample['slice_idx']]

        # normalize image
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # add channel dimension -> (1, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)