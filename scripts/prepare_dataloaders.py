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
        bbox_path     = cache_dir / f"{pid}_bboxes.npy"

        # ✅ Skip if already done
        if pid_cache_dir.exists() and bbox_path.exists():
            print(f"  [{i+1}/{len(patient_ids)}] "
                  f"{pid} — skipping")
            continue

        pid_cache_dir.mkdir(parents=True, exist_ok=True)

        patient_dir = raw_dir / pid
        series_dirs = [
            d for d in patient_dir.iterdir()
            if d.is_dir()
        ]
        if not series_dirs:
            continue

        try:
            series_dir  = series_dirs[0]
            dicom_files = sorted(
                series_dir.rglob("*.dcm"),
                key=lambda f: float(
                    pydicom.dcmread(f).ImagePositionPatient[2]
                )
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

            print(f"  [{i+1}/{len(patient_ids)}] {pid} — "
                  f"saving {volume.shape[0]} slices...")

            bboxes = np.zeros(
                (volume.shape[0], 4), dtype=np.int32
            )

            for z in range(volume.shape[0]):
            
                slice_path = pid_cache_dir / f"{z:04d}.npy"
                np.save(slice_path, volume[z])

                y_min, y_max, x_min, x_max = get_lung_bbox(
                    volume[z]
                )
                bboxes[z] = [y_min, y_max, x_min, x_max]

        
            np.save(bbox_path, bboxes)

            print(f"  [{i+1}/{len(patient_ids)}] {pid} — "
                  f"done!")

            del volume, slices, bboxes

        except Exception as e:
            print(f"  [{i+1}/{len(patient_ids)}] "
                  f"{pid} — ERROR: {e}")
            continue

    print("\n✅ All patients cached by slice!")


if __name__ == "__main__":
    main()