from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import random
from src.preprocessing import get_lung_bbox
from src.dataset import LungCTDataset  
import torchvision.transforms.functional as TF

class LungSegmentationDataset(Dataset):
    """
    Returns (image, mask) slice pairs for training.
    Uses fully preprocessed volumes from LungCTDataset.
    """

    def __init__(self, raw_dir, mask_dir, patient_ids, img_size=384):

        self.samples = []
        self.patient_volumes = {}
        self.patient_masks = {}

        self.img_resize = T.Resize((img_size, img_size),
                                   interpolation=InterpolationMode.BILINEAR)

        self.mask_resize = T.Resize((img_size, img_size),
                                    interpolation=InterpolationMode.NEAREST)
        
        self.augment = True

        raw_dir = Path(raw_dir)
        mask_dir = Path(mask_dir)

        for pid in patient_ids:

            patient_raw_dir = raw_dir / pid

            series_dirs = [d for d in patient_raw_dir.iterdir() if d.is_dir()]
            if len(series_dirs) == 0:
                continue

            series_dir = series_dirs[0]

            mask_path = mask_dir / f"{pid}_mask.npy"
            if not mask_path.exists():
                continue

            # Load FULLY PREPROCESSED volume + mask from dataset.py
            ct_dataset = LungCTDataset(series_dir, np.load(mask_path))

            volume = ct_dataset.volume
            mask_volume = ct_dataset.mask_volume

            self.patient_volumes[pid] = volume
            self.patient_masks[pid] = mask_volume

            tumor_slices = []
            non_tumor_slices = []

            for z in range(mask_volume.shape[0]):
                if mask_volume[z].sum() > 0:
                    tumor_slices.append(z)
                else:
                    non_tumor_slices.append(z)

            # add all tumor slices
            for z in tumor_slices:
                self.samples.append((pid, z))

            # add 2x background slices
            num_bg = min(len(non_tumor_slices), 2 * len(tumor_slices))
            bg_selected = random.sample(non_tumor_slices, num_bg)

            for z in bg_selected:
                self.samples.append((pid, z))

        print("Total training samples:", len(self.samples))

        tumor = 0
        for pid, z in self.samples:
            if self.patient_masks[pid][z].sum() > 0:
                tumor += 1

        print(f"Tumor slices: {tumor}, Background slices: {len(self.samples) - tumor}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        pid, z = self.samples[idx]

        #  Load numpy slices
        volume = self.patient_volumes[pid]
        mask_volume = self.patient_masks[pid]

        prev_slice = volume[z-1] if z>0 else volume[z]
        curr_slice = volume[z]
        next_slice = volume[z+1] if z < len(volume)-1 else volume[z]

        image = np.stack([prev_slice, curr_slice, next_slice], axis=0)
        mask = mask_volume[z]

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # Lung crop FIRST (on numpy)
        ymin, ymax, xmin, xmax = get_lung_bbox(curr_slice)

        image = image[:, ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]

        if self.augment:

            # Horizontal flip
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            # Small rotation
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                image = TF.rotate(torch.from_numpy(image).unsqueeze(0), angle).squeeze(0).numpy()
                mask = TF.rotate(torch.from_numpy(mask).unsqueeze(0), angle, interpolation=InterpolationMode.NEAREST).squeeze(0).numpy()

        #  Then tensor conversion
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).unsqueeze(0)

        #  Then resize
        image = self.img_resize(image)
        mask = self.mask_resize(mask)

        return image, mask