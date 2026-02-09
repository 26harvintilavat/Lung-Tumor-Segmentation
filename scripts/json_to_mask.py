import json
import sys
from pathlib import Path
from tcia_utils import nbia
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import numpy as np
from skimage.draw import polygon
from src.dataset import LungCTDataset

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

def build_sop_uid_map(slices):
    sop_map = {}
    for idx, s in enumerate(slices):
        sop_map[s.SOPInstanceUID] = idx
    return sop_map

def build_mask_from_json(json_path, series_dir):
    # Load annotation
    with open(json_path, 'r') as f:
        annotation = json.load(f)

    # Load CT
    dataset = LungCTDataset(series_dir)
    volume_shape = dataset.volume.shape
    mask_volume = np.zeros(volume_shape, dtype=np.uint8)

    sop_map = build_sop_uid_map(dataset.slices)

    for nodule in annotation["nodules"]:
        for sl in nodule["slices"]:
            sop_uid = sl["sop_uid"]

            if sop_uid not in sop_map:
                continue

            z_idx = sop_map[sop_uid]
            contour = sl["contour"]

            slice_mask = contour_to_mask(
                contour,
                volume_shape[1:]
            )

            mask_volume[z_idx] |= slice_mask


    return mask_volume

def save_mask(mask, patient_id):
    out_dir = Path("data/masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir/f"{patient_id}_mask.npy"
    np.save(path, mask)
    print(f"Saved mask: {path}")

def find_series_dir(raw_patient_dir, target_series_uid):

    if not raw_patient_dir.exists():
        return None
    
    for d in raw_patient_dir.iterdir():
        if d.is_dir() and d.name == target_series_uid:
            return d
    return None

def download_series_if_missing(patient_id, series_uid, raw_dir):
    patient_dir = raw_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    series_dir = patient_dir / series_uid
    if series_dir.exists():
        return series_dir

    print(f"Downloading CT series for {patient_id}")
    nbia.downloadSeries(
        series_data=[series_uid],
        input_type="list",
        path=str(patient_dir)
    )

    return series_dir if series_dir.exists() else None


def main():
    annotation_dir = Path("data/annotations")
    raw_dir = Path("data/raw")

    for json_file in annotation_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            annotation = json.load(f)

        patient_id = annotation['patient_id']
        series_uid = annotation['series_instance_uid']

        series_dir = find_series_dir(raw_dir/patient_id, series_uid)

        if series_dir is None:
            series_dir = download_series_if_missing(patient_id, series_uid, raw_dir)

        if series_dir is None:
            print(f"Failed to obtain series for {patient_id}")
            continue

        mask = build_mask_from_json(json_file, series_dir)
        save_mask(mask, patient_id)

if __name__=="__main__":
    main()