from pathlib import Path
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

from src.preprocessing import (
    convert_to_hu,
    resample_volume,
    resample_mask,
    window_and_normalize,
    get_lung_bbox,
    resize_image,
    resize_mask,
    verify_image_mask_alignment,
)


class LungCTDataset(Dataset):

    def __init__(
        self,
        series_dir: Path,
        mask_volume=None,
        img_size=256
    ):
        self.series_dir = Path(series_dir)
        self.img_size   = img_size

        self.slices = self._load_slices()

        pixel_spacing = tuple(map(float,
                                self.slices[0].PixelSpacing))
        slice_thickness = float(self.slices[0].SliceThickness)
        self.spacing = np.array([
            slice_thickness,
            pixel_spacing[0],
            pixel_spacing[1]
        ])

        raw_volume = convert_to_hu(self.slices)

        self.volume = resample_volume(raw_volume, self.spacing)

        if mask_volume is not None:
            self.mask_volume = resample_mask(
                mask_volume, self.spacing
            )
            verify_image_mask_alignment(
                self.volume, self.mask_volume
            )
            self.mask_volume = (
                self.mask_volume > 0
            ).astype(np.uint8)
        else:
            self.mask_volume = np.zeros(
                self.volume.shape, dtype=np.uint8
            )

        self.volume = window_and_normalize(self.volume)

    def _load_slices(self):
        dicom_files = sorted(
            self.series_dir.rglob("*.dcm"),
            key=lambda f: float(
                pydicom.dcmread(f).ImagePositionPatient[2]
            )
        )
        if not dicom_files:
            raise RuntimeError(
                f"No DICOM files found in {self.series_dir}"
            )
        return [pydicom.dcmread(f) for f in dicom_files]

    def __len__(self):
        return self.volume.shape[0]
    
    def __getitem__(self, idx):
        img  = self.volume[idx]           # (H, W) float32
        mask = self.mask_volume[idx]      # (H, W) uint8

        # Resize 
        img  = resize_image(img,  self.img_size)
        mask = resize_mask(mask,  self.img_size)

        #  Binary mask
        mask = (mask > 0).astype(np.float32)

        # To Tensor
        img  = torch.tensor(img,  dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask