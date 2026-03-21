from pathlib import Path
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

from src.preprocessing import (
    convert_to_hu,
    resample_volume,
    window_image,
    zscore_normalize,
    get_lung_bbox,
    resize_image,
    resize_mask
)

class LungCTDataset(Dataset):
    def __init__(self, series_dir: Path, mask_volume: None, img_size=256):
        self.series_dir = series_dir
        self.img_size = img_size
        self.slices = self._load_slices()

        self.pixel_spacing = tuple(map(float, self.slices[0].PixelSpacing))
        self.slice_thickness = float(self.slices[0].SliceThickness)
        
        self.spacing = np.array([
            self.slice_thickness,
            self.pixel_spacing[0],
            self.pixel_spacing[1]
        ])

        volume = convert_to_hu(self.slices)

        self.volume = resample_volume(volume, self.spacing)

        if mask_volume is not None:
            self.mask_volume = resample_volume(mask_volume, self.spacing)

            z = min(self.volume.shape[0], self.mask_volume.shape[0])
            y = min(self.volume.shape[1], self.mask_volume.shape[1])
            x = min(self.volume.shape[2], self.mask_volume.shape[2])

            self.volume = self.volume[:z, :y, :x]
            self.mask_volume = self.mask_volume[:z, :y, :x]
        else:
            self.mask_volume = None

        

    def _load_slices(self):
        dicom_files = list(self.series_dir.rglob("*.dcm"))

        if not dicom_files:
            raise RuntimeError(f"No DICOM files found in {self.series_dir}")
        
        slices = [pydicom.dcmread(f) for f in dicom_files]
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
        return slices

    def __len__(self):
        return self.volume.shape[0]
    
    def __getitem__(self, idx):

        img = self.volume[idx]
        mask = self.mask_volume[idx]

        # Window
        img = window_image(img)

        # Normalize 
        img = zscore_normalize(img)

        # Lung Crop
        bbox = get_lung_bbox(img)

        if bbox is not None:
            y_min, y_max, x_min, x_max = bbox
            img = img[y_min:y_max, x_min:x_max]
            mask = mask[y_min:y_max, x_min:x_max]

        # Resize
        img = resize_image(img, self.img_size)
        mask = resize_mask(mask, self.img_size)

        # Binary mask
        mask = (mask > 0).astype(np.float32)

        # To Tensor
        img = torch.tensor(img).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask