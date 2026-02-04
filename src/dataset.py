from pathlib import Path
import numpy as np
import pydicom
from src.preprocessing import convert_to_hu, window_image

class LungCTDataset:
    def __init__(self, series_dir: Path):
        self.series_dir = series_dir
        self.slices = self._load_slices()

        self.pixel_scaling = self._get_pixel_spacing()
        self.slice_thickness = self._get_slice_thickness()
        self.voxel_spacing = self._get_voxel_spacing()

        self.volume = self._build_volume()

    def _load_slices(self):
        dicom_files = list(self.series_dir.rglob("*.dcm"))

        if not dicom_files:
            raise RuntimeError(f"No DICOM files found in {self.series_dir}")
        
        slices = [pydicom.dcmread(f) for f in dicom_files]
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
        return slices
    
    def _get_pixel_spacing(self):
        return tuple(map(float, self.slices[0].PixelSpacing))
    
    def _get_slice_thickness(self):
        return float(self.slices[0].SliceThickness)
    
    def _get_voxel_spacing(self):
        row, col = self.pixel_scaling
        z = self.slice_thickness
        return (z, row, col)
    
    def _build_volume(self):
        return convert_to_hu(self.slices)
    
    def get_slice(self, idx):
        img = self.volume[idx]
        img = window_image(img)
        return img
    
    def __len__(self):
        return self.volume.shape[0]