import sys
from pathlib import Path
import numpy as np
import torch
import cv2
import pydicom
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import zipfile
import io
import base64

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import LungAttentionUNet
from src.preprocessing import (
    convert_to_hu,
    resample_volume,
    window_and_normalize,
    get_lung_bbox,
    resize_image
)

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
MODEL_INFO_CACHE = None
MAX_UPLOAD_MB = 500

def load_model():
    global model
    print(f"--- Init: Loading weights from {MODEL_PATH} ---")
    try:
        model = LungAttentionUNet(
            in_channels=3,
            out_channels=1
        ).to(device, memory_format=torch.channels_last)

        # Load weights (weights_only=False required for complex model structures)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"SUCCESS: Model loaded from epoch {checkpoint['epoch']}")
        else:
            model.load_state_dict(checkpoint)
            print("SUCCESS: Model weights loaded directly.")

        model.eval()
        print(f"SUCCESS: Model ready on {device}")
        return checkpoint
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {str(e)}")
        model = None
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_INFO_CACHE
    print("\n--- API Lifecycle Startup ---")
    checkpoint = load_model()
    
    if checkpoint and MODEL_PATH.exists():
        MODEL_INFO_CACHE = {
            'architecture': "Attention U-Net",
            'input_size': "256x256",
            'trained_epoch': checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A',
            'device': str(device)
        }
    print("--- API Lifecycle Ready ---\n")
    yield

# Create App (Lifespan MUST be defined before this line)
app = FastAPI(
    title="Lung Tumor Segmentation API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
async def health_check():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
    
    contents = await file.read()
    zip_buffer = io.BytesIO(contents)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with zipfile.ZipFile(zip_buffer) as zf:
                zf.extractall(tmpdir)

            dicom_files = list(tmpdir.rglob("*dcm"))
            if not dicom_files:
                raise HTTPException(status_code=400, detail="No DICOM files in ZIP")
            
            # 1. Preprocess
            slices = [pydicom.dcmread(f) for f in dicom_files]
            slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            
            spacing = np.array([
                float(slices[0].SliceThickness),
                float(slices[0].PixelSpacing[0]),
                float(slices[0].PixelSpacing[1])
            ])

            volume = convert_to_hu(slices)
            volume = resample_volume(volume, spacing)
            volume = window_and_normalize(volume)

            # 2. Predict
            total_z = volume.shape[0]
            predictions = np.zeros(volume.shape, dtype=np.float32)

            with torch.no_grad():
                for z in range(total_z):
                    # Simple 2.5D context
                    prev_idx = max(0, z-1)
                    next_idx = min(total_z - 1, z+1)
                    
                    # Stack context
                    img_stack = np.stack([volume[prev_idx], volume[z], volume[next_idx]], axis=0)
                    
                    # Resize to model input
                    resized = np.stack([resize_image(img_stack[c], 256) for c in range(3)], axis=0)
                    
                    tensor = torch.from_numpy(resized).unsqueeze(0).to(device)
                    output = model(tensor)
                    pred = torch.sigmoid(output).squeeze().cpu().numpy()
                    
                    # Resize back (simplified for this update)
                    pred_back = cv2.resize(pred, (volume.shape[2], volume.shape[1]), interpolation=cv2.INTER_NEAREST)
                    predictions[z] = (pred_back > 0.5).astype(np.float32)

            # 3. Format Overlays
            tumor_slices = [z for z in range(total_z) if predictions[z].sum() > 10]
            overlays = []
            for z in tumor_slices:
                ct_uint8 = (volume[z] * 255).astype(np.uint8)
                ct_rgb = cv2.cvtColor(ct_uint8, cv2.COLOR_GRAY2BGR)
                
                overlay = ct_rgb.copy()
                overlay[predictions[z] > 0.5] = [255, 0, 0] # Red
                res = cv2.addWeighted(ct_rgb, 0.7, overlay, 0.3, 0)
                
                _, buf = cv2.imencode('.png', cv2.resize(res, (256, 256)))
                overlays.append({
                    "slice_index": z,
                    "image_base64": base64.b64encode(buf).decode('utf-8'),
                    "tumor_pixels": int(predictions[z].sum())
                })

            return {
                "status": "success",
                "total_slices": total_z,
                "tumor_slices": len(tumor_slices),
                "tumor_slice_ids": tumor_slices,
                "total_tumor_volume": float(predictions.sum()),
                "overlays": overlays
            }

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
