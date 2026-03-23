import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import cv2
import pydicom
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import zipfile
import io
import base64

from src.model import LungAttentionUNet
from src.preprocessing import (
    convert_to_hu,
    resample_volume,
    window_and_normalize,
    get_lung_bbox,
    resize_image,
    resize_mask
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
MODEL_PATH = PROJECT_ROOT/"checkpoints"/"best_model.pth"
MODEL_INFO_CACHE = None        # cached at startup, never re-read from disk
MAX_UPLOAD_MB = 500            # reject ZIPs larger than this

def load_model():
    global model
    model = LungAttentionUNet(
        in_channels=3,
        out_channels=1
    ).to(device, memory_format=torch.channels_last)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Best Val Dice: {checkpoint.get('best_val_dice', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Model ready on {device}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_INFO_CACHE
    print("Loading model...")
    load_model()
    # Cache model info so /model-info never hits disk again
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        MODEL_INFO_CACHE = {
            'architecture': "Attention U-Net",
            'in_channels': 3,
            'out_channels': 1,
            'input_size': "256x256",
            'trained_epoch': checkpoint.get('epoch', 'N/A'),
            'best_val_dice': checkpoint.get('best_val_dice', 'N/A'),
            'device': str(device)
        }
        del checkpoint
    print("API ready!")
    yield

app = FastAPI(
    title="Lung Tumor Segmentation API",
    description="Attention U-Net based lung tumor segmentation",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_dicom_series(dicom_files):
    """Preprocess list of DICOM files into model input"""
    slices = []
    for f in dicom_files:
        slices.append(pydicom.dcmread(f))
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    pixel_spacing = tuple(map(float, slices[0].PixelSpacing))
    slice_thickness = float(slices[0].SliceThickness)
    spacing = np.array([
        slice_thickness,
        pixel_spacing[0],
        pixel_spacing[1]
    ])

    volume = convert_to_hu(slices)
    volume = resample_volume(volume, spacing)
    volume = window_and_normalize(volume)

    return volume

def predict_volume(volume):
    """Run inference on preprocessed volume"""
    total_z = volume.shape[0]
    predictions = np.zeros(volume.shape, dtype=np.float32)

    with torch.no_grad():
        for z in range(total_z):
            prev_idx = max(0, z-1)
            next_idx = min(total_z - 1, z+1)

            prev_slice = volume[prev_idx]
            curr_slice = volume[z]
            next_slice = volume[next_idx]

            y_min, y_max, x_min, x_max = get_lung_bbox(curr_slice)

            image = np.stack(
                [prev_slice, curr_slice, next_slice],
                axis=0
            ).astype(np.float32)

            image = image[:, y_min:y_max, x_min:x_max]

            resized = []
            for c in range(3):
                resized.append(resize_image(image[c], 256))
            image = np.stack(resized, axis=0)

            image_tensor = torch.tensor(
                image, dtype=torch.float32
            ).unsqueeze(0).to(device)

            if device.type == "cuda":
                from torch.amp import autocast
                with autocast(device_type="cuda"):
                    output = model(image_tensor)
            else:
                output = model(image_tensor)

            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred = (pred > 0.5).astype(np.float32)

            orig_h = y_max - y_min
            orig_w = x_max - x_min
            pred_resized = cv2.resize(
                pred,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

            full_pred = np.zeros(
                volume.shape[1:],dtype=np.float32
            )
            full_pred[y_min:y_max, x_min:x_max] = pred_resized
            predictions[z] = full_pred
    
    return predictions

def create_overlay(ct_slice, mask_slice):
    """Create colored overlay of prediction on CT"""

    ct_uint8 = (ct_slice * 255).astype(np.uint8)
    ct_rgb = cv2.cvtColor(ct_uint8, cv2.COLOR_GRAY2BGR)

    # Red overlay for tumor
    overlay = ct_rgb.copy()
    tumor_pixels = mask_slice > 0.5
    overlay[tumor_pixels] = [255, 0, 0]

    result = cv2.addWeighted(ct_rgb, 0.7, overlay, 0.3, 0)
    return result

def encode_image(image_array):
    """Encode numpy image to base64 string"""
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')

@app.get('/health')
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "model_path": str(MODEL_PATH)
    }

@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if MODEL_INFO_CACHE is None:
        raise HTTPException(status_code=503, detail="Model info not available")
    return MODEL_INFO_CACHE

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a ZIP file containing DICOM files.
    Returns predictions with tumor slices and overlays.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    if not file.filename.endswith('.zip'):
        raise HTTPException(
            status_code=400,
            detail="Please upload a ZIP file containing DICOM files"
        )

    # File size guard — read first chunk to check Content-Length or stream size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.0f} MB). Maximum allowed: {MAX_UPLOAD_MB} MB"
        )
    
    try:
        zip_buffer = io.BytesIO(contents)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            with zipfile.ZipFile(zip_buffer) as zf:
                zf.extractall(tmpdir)

            dicom_files = list(tmpdir.rglob("*dcm"))
            if not dicom_files:
                raise HTTPException(
                    status_code=400,
                    detail="No DICOM files found in ZIP"
                )
            
            print(f"Found {len(dicom_files)} DICOM files")

            volume = preprocess_dicom_series(dicom_files)
            print(f"Volume shape: {volume.shape}")

            predictions = predict_volume(volume)

            tumor_slices = []
            for z in range(len(predictions)):
                if predictions[z].sum() > 10:
                    tumor_slices.append(z)

            print(f"Tumor slices: {len(tumor_slices)}")

            overlays = []
            for z in tumor_slices:
                overlay = create_overlay(
                    volume[z],
                    predictions[z]
                )
                overlay_resized = cv2.resize(overlay, (256, 256))
                overlays.append({
                    "slice_index": z,
                    "image_base64": encode_image(overlay_resized),
                    "tumor_pixels": int(predictions[z].sum())
                })

            return JSONResponse({
                "status"          : "success",
                "total_slices"    : int(volume.shape[0]),
                "tumor_slices"    : len(tumor_slices),
                "tumor_slice_ids" : tumor_slices,
                "total_tumor_volume": float(predictions.sum()),
                "overlays"        : overlays
            })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
if __name__=="__main__":
    uvicorn.run(
        "api.main:app",
        host="localhost",
        port=8000,
        reload=False
    )