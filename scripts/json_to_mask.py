from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import json
import numpy as np
from skimage.draw import polygon
from src.dataset import LungCTDataset


def build_z_index_map(slices):
    """
    Map physical Z position -> slice index
    """
    z_map = {}
    for idx, s in enumerate(slices):
        z = float(s.ImagePositionPatient[2])
        z_map[z] = idx
    return z_map

def contour_to_mask(contour, shape):
    """
    contour: list of [x, y]
    shape: (H, W)
    """
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]

    rr, cc = polygon(ys, xs, shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rr, cc] = 1

    return mask

def build_mask_from_json(json_path, series_dir):
    # Load annotation
    with open(json_path, 'r') as f:
        annotation = json.load(f)

    # Load CT
    dataset = LungCTDataset(series_dir)
    volume_shape = dataset.volume.shape
    mask_volume = np.zeros(volume_shape, dtype=np.uint8)

    # Build Z-position lookup
    z_map = build_z_index_map(dataset.slices)

    for nodule in annotation['nodules']:
        for sl in nodule['slices']:
            z = sl['z_position']

            if z not in z_map:
                continue

            z_idx = z_map[z]
            contour = sl['contour']

            slice_mask = contour_to_mask(
                contour,
                volume_shape[1:]
            )

            mask_volume[z_idx] |= slice_mask

    return mask_volume

def save_mask(mask, patient_id):
    out_dir = Path("data/masks")
    out_dir.mkdir(exist_ok=True)

    path = out_dir/f"{patient_id}_mask.npy"
    np.save(path, mask)
    print(f"Saved mask: {path}")

def main():
    annotation_dir = Path("data/annotations")
    raw_dir = Path("data/raw")

    for json_file in annotation_dir.glob("*.json"):
        patient_id = json_file.stem
        series_dir = raw_dir/patient_id

        if not series_dir.exists():
            continue

        mask = build_mask_from_json(json_file, series_dir)
        save_mask(mask, patient_id)

if __name__=="__main__":
    main()