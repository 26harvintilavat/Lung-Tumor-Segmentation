from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import random
import pydicom

from src.preprocessing import (
    convert_to_hu,
    resample_volume,
    resample_mask,
    window_and_normalize,
    get_lung_bbox,
    resize_image,
    resize_mask,
    get_tumor_slices,
    verify_image_mask_alignment,
)


class LungSegmentationDataset(Dataset):


    def __init__(
        self,
        raw_dir,
        mask_dir,
        patient_ids,
        img_size=256,
        augment=False,
        min_tumor_pixels=10,
        bg_ratio=2,
    ):
        self.img_size  = img_size
        self.augment   = augment
        self.bg_ratio  = bg_ratio
        self.raw_dir   = Path(raw_dir)
        self.mask_dir  = Path(mask_dir)

        self.tumor_samples = []
        self.bg_samples    = []

        
        self.patient_series_dirs = {}

        for pid in patient_ids:
            patient_raw_dir = self.raw_dir / pid
            series_dirs = [
                d for d in patient_raw_dir.iterdir()
                if d.is_dir()
            ]
            if not series_dirs:
                print(f"  [SKIP] No series for {pid}")
                continue

            mask_path = self.mask_dir / f"{pid}_mask.npy"
            if not mask_path.exists():
                print(f"  [SKIP] No mask for {pid}")
                continue

            self.patient_series_dirs[pid] = series_dirs[0]

            mask_volume = np.load(mask_path)

            tumor_indices = get_tumor_slices(
                mask_volume,
                min_tumor_pixels=min_tumor_pixels
            )
            non_tumor_indices = [
                z for z in range(mask_volume.shape[0])
                if z not in set(tumor_indices)
            ]

            for z in tumor_indices:
                self.tumor_samples.append((pid, z))
            for z in non_tumor_indices:
                self.bg_samples.append((pid, z))

            print(f"  {pid} — "
                f"tumor: {len(tumor_indices)} | "
                f"bg: {len(non_tumor_indices)}")

            del mask_volume

        self.samples = self._build_samples()

       
        print(f"\nTotal samples    : {len(self.samples)}")
        print(f"Tumor samples    : {len(self.tumor_samples)}")
        print(f"Background samples: {len(self.bg_samples)}")

    def _build_samples(self):

        num_bg = min(
            len(self.bg_samples),
            self.bg_ratio * len(self.tumor_samples)
        )
        bg_selected = random.sample(self.bg_samples, num_bg)
        samples = self.tumor_samples + bg_selected
        random.shuffle(samples)
        return samples

    def resample_per_epoch(self):
        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, z = self.samples[idx]

        cache_dir = Path("data/cache")
        pid_dir   = cache_dir / pid
        bbox_path = cache_dir / f"{pid}_bboxes.npy"
        mask_path = self.mask_dir / f"{pid}_mask.npy"

        bboxes  = np.load(bbox_path)
        total_z = len(bboxes)

        z        = min(z, total_z - 1)
        prev_idx = max(0, z - 1)
        next_idx = min(total_z - 1, z + 1)

        prev_slice = np.load(pid_dir / f"{prev_idx:04d}.npy")
        curr_slice = np.load(pid_dir / f"{z:04d}.npy")
        next_slice = np.load(pid_dir / f"{next_idx:04d}.npy")

        mask_volume = np.load(mask_path)
        mask        = mask_volume[z].copy().astype(np.float32)
        del mask_volume

        y_min, y_max, x_min, x_max = bboxes[z]
        del bboxes

        image = np.stack(
            [prev_slice, curr_slice, next_slice],
            axis=0
        ).astype(np.float32)

   
        image = image[:, y_min:y_max, x_min:x_max]
        mask  = mask[y_min:y_max, x_min:x_max]

        resized = []
        for c in range(3):
            resized.append(resize_image(image[c], self.img_size))
        image = np.stack(resized, axis=0)
        mask  = resize_mask(mask, self.img_size)

  
        mask = (mask > 0).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)

      
        image = torch.tensor(image, dtype=torch.float32)
        mask  = torch.tensor(
            mask, dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


    def _augment(self, image, mask):
        """
        All augmentations in numpy space.
        image: (3, H, W) float32
        mask:  (H, W)    float32
        """

        #  Horizontal flip — axis=2 is width for (3,H,W)
        if random.random() < 0.5:
            image = np.flip(image, axis=2).copy()
            mask  = np.flip(mask,  axis=1).copy()

        #  Vertical flip
        if random.random() < 0.2:
            image = np.flip(image, axis=1).copy()
            mask  = np.flip(mask,  axis=0).copy()

        #  Rotation — consistent for image and mask
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img_t  = torch.from_numpy(image)           # (3,H,W)
            msk_t  = torch.from_numpy(mask).unsqueeze(0)  # (1,H,W)

            img_t  = TF.rotate(img_t,  angle,
                        interpolation=InterpolationMode.BILINEAR)
            msk_t  = TF.rotate(msk_t, angle,
                        interpolation=InterpolationMode.NEAREST)

            image = img_t.numpy()
            mask  = msk_t.squeeze(0).numpy()

        #  Brightness/contrast — image ONLY, never mask
        if random.random() < 0.3:
            factor = random.uniform(0.9, 1.1)
            image  = np.clip(image * factor, 0.0, 1.0)

        #  Gaussian noise — image ONLY
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        if random.random() < 0.2:
            h, w    = mask.shape
            n_holes = random.randint(1, 4)
            for _ in range(n_holes):
                y1 = random.randint(0, h - 16)
                x1 = random.randint(0, w - 16)
                y2 = min(h, y1 + random.randint(8, 24))
                x2 = min(w, x1 + random.randint(8, 24))
                image[:, y1:y2, x1:x2] = 0.0

        return image, mask