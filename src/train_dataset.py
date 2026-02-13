from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.dataset import LungCTDataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

class LungSegmentationDataset(Dataset):
    "Returns (image, mask) slice pairs for training"

    def __init__(self, raw_dir, mask_dir, patient_ids):
        self.raw_dir = Path(raw_dir)
        self.mask_dir = Path(mask_dir)
        self.samples = []

        self.volume_cache = {}
        self.mask_cache = {}

        self.img_resize = T.Resize((256, 256), interpolation=InterpolationMode.BILINEAR)
        self.mask_resize = T.Resize((256, 256), interpolation=InterpolationMode.NEAREST)

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

            self.volume_cache[pid] = volume
            self.mask_cache[pid] = mask

            # store slice-level samples
            for z in range(volume.shape[0]):
                if mask[z].sum() > 0:
                    self.samples.append({
                        "pid": pid,
                        "slice_idx": z
                    })
            print(f"Total tumor slices: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample["pid"]
        z = sample["slice_idx"]

       # get from cache(no disk loading)
        volume = self.volume_cache[pid]
        mask = self.mask_cache[pid]

        image = volume[z]
        mask = mask[z]

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