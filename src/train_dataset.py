from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import random

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
import pydicom


class LungSegmentationDataset(Dataset):
    """
    Patient-level dataset for lung tumor segmentation.
    - Loads + preprocesses entire volume per patient at init
    - Precomputes lung bboxes once
    - Builds balanced tumor/background sample list
    - Applies consistent augmentation to image + mask
    """

    def __init__(
        self,
        raw_dir,
        mask_dir,
        patient_ids,
        img_size=384,
        augment=False,              #  controllable — True for train, False for val
        min_tumor_pixels=10,        #  filter noise annotations
        bg_ratio=2,                  
    ):
        self.img_size  = img_size
        self.augment   = augment
        self.bg_ratio  = bg_ratio

        self.patient_volumes = {}
        self.patient_masks   = {}
        self.patient_bboxes  = {}   # precomputed per slice per patient

        self.tumor_samples  = []    # separated for dynamic sampling
        self.bg_samples     = []

        raw_dir  = Path(raw_dir)
        mask_dir = Path(mask_dir)

        for pid in patient_ids:
            patient_raw_dir = raw_dir / pid
            series_dirs = [d for d in patient_raw_dir.iterdir() if d.is_dir()]

            if not series_dirs:
                print(f"  [SKIP] No series found for patient {pid}")
                continue

            mask_path = mask_dir / f"{pid}_mask.npy"
            if not mask_path.exists():
                print(f"  [SKIP] No mask found for patient {pid}")
                continue

            # Load DICOM
            series_dir  = series_dirs[0]
            dicom_files = sorted(
                series_dir.rglob("*.dcm"),
                key=lambda f: float(pydicom.dcmread(f).ImagePositionPatient[2])
            )
            slices = [pydicom.dcmread(f) for f in dicom_files]

            # Spacing 
            pixel_spacing   = tuple(map(float, slices[0].PixelSpacing))
            slice_thickness = float(slices[0].SliceThickness)
            spacing = np.array([
                slice_thickness,
                pixel_spacing[0],
                pixel_spacing[1]
            ])

            # Preprocess 
            volume      = convert_to_hu(slices)
            mask_volume = np.load(mask_path)

            #  resample_volume for CT, resample_mask for mask
            volume      = resample_volume(volume, spacing)
            mask_volume = resample_mask(mask_volume,spacing)

            #  Alignment check
            verify_image_mask_alignment(volume, mask_volume)

            #  Window + normalize entire volume ONCE
            volume = window_and_normalize(volume)

            self.patient_volumes[pid] = volume
            self.patient_masks[pid]   = mask_volume.astype(np.uint8)

            # Precompute bboxes ONCE per slice 
            # Not recomputed every __getitem__ call
            self.patient_bboxes[pid] = {}
            for z in range(volume.shape[0]):
                self.patient_bboxes[pid][z] = get_lung_bbox(volume[z])

            # ── Build sample lists ────────────────────────────
            tumor_indices = get_tumor_slices(
                mask_volume,
                min_tumor_pixels=min_tumor_pixels   # filter noise
            )
            non_tumor_indices = [
                z for z in range(mask_volume.shape[0])
                if z not in set(tumor_indices)
            ]

            for z in tumor_indices:
                self.tumor_samples.append((pid, z))

            for z in non_tumor_indices:
                self.bg_samples.append((pid, z))

        # Build balanced sample list 
        self.samples = self._build_samples()

        # Stats
        tumor_count = sum(
            1 for pid, z in self.samples
            if self.patient_masks[pid][z].sum() > 0
        )
        print(f"Total samples   : {len(self.samples)}")
        print(f"Tumor slices    : {tumor_count}")
        print(f"Background slices: {len(self.samples) - tumor_count}")

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

        volume      = self.patient_volumes[pid]
        mask_volume = self.patient_masks[pid]
        total_z     = volume.shape[0]

        # 3-slice context 
        prev_idx = max(0, z - 1)
        next_idx = min(total_z - 1, z + 1)

        prev_slice = volume[prev_idx]
        curr_slice = volume[z]
        next_slice = volume[next_idx]

        # Shape: (3, H, W)
        image = np.stack(
            [prev_slice, curr_slice, next_slice],
            axis=0
        ).astype(np.float32)

        mask = mask_volume[z].astype(np.float32)  # (H, W)

        # Lung crop
        # Uses precomputed bbox — not recomputed here
        y_min, y_max, x_min, x_max = self.patient_bboxes[pid][z]
        image = image[:, y_min:y_max, x_min:x_max]  # (3, H, W)
        mask  = mask[y_min:y_max, x_min:x_max]       # (H, W)

        # Resize
        resized = []
        for c in range(3):
            resized.append(resize_image(image[c], self.img_size))
        image = np.stack(resized, axis=0)             # (3, H, W)
        mask  = resize_mask(mask, self.img_size)       # (H, W)

        # Binary mask 
        mask = (mask > 0).astype(np.float32)

        # Augmentation 
        #  All ops on numpy before tensor conversion
        #  Same transform applied to image AND mask
        if self.augment:
            image, mask = self._augment(image, mask)

        image = torch.tensor(image, dtype=torch.float32)       # (3,H,W)
        mask  = torch.tensor(mask,  dtype=torch.float32).unsqueeze(0)  # (1,H,W)

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