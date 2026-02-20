import os
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian
import numpy as np

patient_id = "LIDC-IDRI-0001"
dcm_dir = f"data/raw/{patient_id}"
os.makedirs(dcm_dir, exist_ok=True)

for i in range(5):
    filename = f"{dcm_dir}/slice_{i}.dcm"
    
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = f"1.2.3.{i}"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.InstanceNumber = i
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    
    arr = np.zeros((512, 512), dtype=np.int16)
    arr[200:300, 200:300] = 500  # "Tumor"
    
    ds.PixelData = arr.tobytes()
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    ds.save_as(filename)

print("Created 5 dummy DICOM files.")
