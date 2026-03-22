import sys
from pathlib import Path
import numpy as np
import pydicom

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import RAW_DATA_DIR, MASK_DIR
from src.preprocessing import (
    convert_to_hu,
    resample_volume,
    resample_mask,
    window_and_normalize,
    get_lung_bbox,
)


def main():
    raw_dir   = Path(RAW_DATA_DIR)
    mask_dir  = Path(MASK_DIR)
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = [
        f.stem.replace('_mask', '')
        for f in mask_dir.glob('*_mask.npy')
    ]

    print(f"Total patients: {len(patient_ids)}")

    for i, pid in enumerate(patient_ids):
        pid_cache_dir = cache_dir / pid
        mask_cache_dir = cache_dir / f"{pid}_masks"
        bbox_path     = cache_dir / f"{pid}_bboxes.npy"

        print(f"Checking {pid}: cache {pid_cache_dir.exists()}, masks {mask_cache_dir.exists()}, bbox {bbox_path.exists()}")

        # ✅ Skip if already done
        if pid_cache_dir.exists() and mask_cache_dir.exists() and bbox_path.exists():
            print(f"  [{i+1}/{len(patient_ids)}] "
                  f"{pid} — skipping")
            continue

        pid_cache_dir.mkdir(parents=True, exist_ok=True)
        mask_cache_dir.mkdir(parents=True, exist_ok=True)

        patient_dir = raw_dir / pid
        series_dirs = [
            d for d in patient_dir.iterdir()
            if d.is_dir()
        ]
        if not series_dirs:
            continue

        try:
            series_dir  = series_dirs[0]
            
            def get_slice_pos(f):
                ds = pydicom.dcmread(f)
                if hasattr(ds, 'ImagePositionPatient'):
                    return float(ds.ImagePositionPatient[2])
                return float(getattr(ds, 'InstanceNumber', 0))

            dicom_files = sorted(
                series_dir.rglob("*.dcm"),
                key=get_slice_pos
            )
            slices = [pydicom.dcmread(f) for f in dicom_files]

       
            pixel_spacing   = tuple(map(float,
                                slices[0].PixelSpacing))
            slice_thickness = float(slices[0].SliceThickness)
            spacing = np.array([
                slice_thickness,
                pixel_spacing[0],
                pixel_spacing[1]
            ])

            
            volume = convert_to_hu(slices)
            volume = resample_volume(volume, spacing)
            volume = window_and_normalize(volume)

            # Resample mask to match volume
            mask_path = mask_dir / f"{pid}_mask.npy"
            raw_mask = np.load(mask_path)
            resampled_mask = resample_mask(raw_mask, spacing)

            if volume.shape[0] != resampled_mask.shape[0]:
                print(f"  [WARNING] Shape mismatch for {pid}: "
                      f"Img {volume.shape} vs Mask {resampled_mask.shape}")

            print(f"  [{i+1}/{len(patient_ids)}] {pid} — "
                  f"saving {volume.shape[0]} slices...")

            bboxes = np.zeros(
                (volume.shape[0], 4), dtype=np.int32
            )

            for z in range(volume.shape[0]):
                # Save Image slice
                slice_path = pid_cache_dir / f"{z:04d}.npy"
                np.save(slice_path, volume[z])

                # Save Mask slice
                mask_slice_path = mask_cache_dir / f"{z:04d}.npy"
                np.save(mask_slice_path, resampled_mask[z])

                y_min, y_max, x_min, x_max = get_lung_bbox(
                    volume[z]
                )
                bboxes[z] = [y_min, y_max, x_min, x_max]

        
            np.save(bbox_path, bboxes)

            print(f"  [{i+1}/{len(patient_ids)}] {pid} — "
                  f"done!")

            del volume, resampled_mask, slices, bboxes

        except Exception as e:
            print(f"  [{i+1}/{len(patient_ids)}] "
                  f"{pid} — ERROR: {e}")
            continue

    print("\n✅ All patients cached by slice!")


if __name__ == "__main__":
    main()