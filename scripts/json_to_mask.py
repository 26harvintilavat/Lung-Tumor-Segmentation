import json
import sys
from pathlib import Path
import numpy as np
from skimage.draw import polygon
from tcia_utils import nbia

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import RAW_DATA_DIR, ANNOTATION_DIR, MASK_DIR
import pydicom

def load_dicom_slices(series_dir):
    """Load and sort DICOM slices by z position"""
    dicom_files = sorted(
        Path(series_dir).rglob("*.dcm"),
        key=lambda f: float(pydicom.dcmread(f).ImagePositionPatient[2])
    )
    if not dicom_files:
        raise RuntimeError(f"No DICOM files in {series_dir}")
    return [pydicom.dcmread(f) for f in dicom_files]

def build_sop_uid_map(slices):
    """Map SOPInstanceUID → slice index"""
    return {s.SOPInstanceUID: idx for idx, s in enumerate(slices)}


def contour_to_mask(contour, shape):
    """
    Convert contour points to binary mask.
    contour: list of [x, y] in DICOM image coordinates
    shape:   (H, W) of original DICOM slice
    """
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    rr, cc = polygon(ys, xs, shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


def build_mask_from_json(json_path, series_dir):
    """
    ✅ Build mask in ORIGINAL DICOM space first,
    then resample mask separately.
    Never build mask on resampled volume coordinates.
    """
    with open(json_path, 'r') as f:
        annotation = json.load(f)
    slices = load_dicom_slices(series_dir)


    n_slices   = len(slices)
    orig_H     = int(slices[0].Rows)
    orig_W     = int(slices[0].Columns)
    orig_shape = (n_slices, orig_H, orig_W)

    print(f"  Original DICOM shape: {orig_shape}")


    mask_volume = np.zeros(orig_shape, dtype=np.uint8)
    sop_map     = build_sop_uid_map(slices)

    for nodule in annotation["nodules"]:
        for sl in nodule["slices"]:
            sop_uid = sl["sop_uid"]

            if sop_uid not in sop_map:
                print(f"  [WARN] SOPInstanceUID not found: {sop_uid}")
                continue

            z_idx   = sop_map[sop_uid]
            contour = sl["contour"]

            slice_mask = contour_to_mask(
                contour,
                (orig_H, orig_W)    
            )
            mask_volume[z_idx] |= slice_mask

    tumor_slices = np.sum(mask_volume.sum(axis=(1,2)) > 0)
    print(f"  Tumor slices found: {tumor_slices}/{n_slices}")

    pixel_spacing   = tuple(map(float, slices[0].PixelSpacing))
    slice_thickness = float(slices[0].SliceThickness)
    spacing = np.array([
        slice_thickness,
        pixel_spacing[0],
        pixel_spacing[1]
    ])
    from src.preprocessing import resample_mask
    mask_resampled = resample_mask(mask_volume, spacing)

    print(f"  Resampled mask shape: {mask_resampled.shape}")
    print(f"  Tumor slices after resample: "
          f"{np.sum(mask_resampled.sum(axis=(1,2)) > 0)}")

    return mask_resampled


def save_mask(mask, patient_id):
    out_dir = Path(MASK_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{patient_id}_mask.npy"
    np.save(path, mask)
    print(f"  Saved mask: {path}")


def find_series_dir(raw_patient_dir, target_series_uid):
    if not Path(raw_patient_dir).exists():
        return None
    for d in Path(raw_patient_dir).iterdir():
        if d.is_dir() and d.name == target_series_uid:
            return d
    return None

def download_series_if_missing(patient_id, series_uid, raw_dir):
    patient_dir = Path(raw_dir) / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    series_dir = patient_dir / series_uid

    if series_dir.exists() and any(series_dir.rglob("*.dcm")):
        print(f"  CT already downloaded for {patient_id}")
        return series_dir

    print(f"  Downloading CT for {patient_id}...")
    nbia.downloadSeries(
        series_data=[series_uid],
        input_type="list",
        path=str(patient_dir)
    )
    return series_dir if series_dir.exists() else None


def main():
    annotation_dir = Path(ANNOTATION_DIR)
    raw_dir        = Path(RAW_DATA_DIR)

    json_files = list(annotation_dir.glob("*.json"))
    print(f"Found {len(json_files)} annotation files\n")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            annotation = json.load(f)

        patient_id = annotation['patient_id']
        series_uid = annotation['series_instance_uid']

        print(f"Processing: {patient_id}")

        
        mask_path = Path(MASK_DIR) / f"{patient_id}_mask.npy"
        if mask_path.exists():
            print(f"  Mask already exists — skipping\n")
            continue


        series_dir = find_series_dir(
            raw_dir / patient_id, series_uid
        )
        if series_dir is None:
            series_dir = download_series_if_missing(
                patient_id, series_uid, raw_dir
            )
        if series_dir is None:
            print(f"  [ERROR] Could not get series — skipping\n")
            continue

        
        try:
            mask = build_mask_from_json(json_file, series_dir)
            save_mask(mask, patient_id)
            print(f"  Done ✅\n")
        except Exception as e:
            print(f"  [ERROR] {patient_id}: {e}\n")
            continue


if __name__ == "__main__":
    main()