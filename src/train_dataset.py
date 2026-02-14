from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.dataset import LungCTDataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import random

class LungSegmentationDataset(Dataset):
    "Returns (image, mask) slice pairs for training"

    def __init__(self, raw_dir, mask_dir, patient_ids):
        self.raw_dir = Path(raw_dir)
        self.mask_dir = Path(mask_dir)
        self.samples = []

        self.img_resize = T.Resize((256, 256), interpolation=InterpolationMode.BILINEAR)
        self.mask_resize = T.Resize((256, 256), interpolation=InterpolationMode.NEAREST)

        for pid in patient_ids:
            patient_raw_dir = self.raw_dir/pid

            # find series folder
            series_dirs = [d for d in patient_raw_dir.iterdir() if d.is_dir()]
            if len(series_dirs) == 0:
                continue

            series_dir = series_dirs[0]

            # load mask
            mask_path = self.mask_dir/f"{pid}_mask.npy"
            if not mask_path.exists():
                continue

            mask = np.load(mask_path)

            # store slice-level samples
            tumor_slices = []
            non_tumor_slices = []

            for z in range(mask.shape[0]):
                if mask[z].sum() > 0:
                    tumor_slices.append(z)
                else:
                    non_tumor_slices.append(z)

            # Add all tumor slices
            for z in tumor_slices:
                self.samples.append({
                    "pid": pid,
                    "slice_idx": z,
                    "series_dir": series_dir,
                    "mask_path": mask_path
                })

            # Add 2x non-tumor slices
            num_bg = min(len(non_tumor_slices), 2 * len(tumor_slices))
            bg_selected = random.sample(non_tumor_slices, num_bg)

            for z in bg_selected:
                self.samples.append({
                    "pid": pid,
                    "slice_idx": z,
                    "series_dir": series_dir,
                    "mask_path": mask_path
                })

            print(f"Total training slices: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample["pid"]
        z = sample["slice_idx"]
        series_dir = sample['series_dir']
        mask_path = sample['mask_path']

        # Load CT volume only when needed
        dataset = LungCTDataset(series_dir)
        image = dataset.volume[z]

        # Load mask slice
        mask_volume = np.load(mask_path)
        mask = mask_volume[z]

        # normalize
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # add channel dimenstion
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # resize
        image = self.img_resize(image)
        mask = self.mask_resize(mask)

        return image, mask