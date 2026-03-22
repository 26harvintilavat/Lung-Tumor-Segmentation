import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
import cv2

def convert_to_hu(slices):
    """
    Convert raw DICOM pixel values to Hounsfield Units.
    Handles both integer and float slopes correctly.
    """
    images = np.stack([s.pixel_array for s in slices]).astype(np.float32)

    for i,s in enumerate(slices):
        intercept = float(s.RescaleIntercept)
        slope = float(s.RescaleSlope)

        images[i] = images[i] * slope + intercept

    return images.astype(np.float32)

# Resampling 
def resample_volume(volume, old_spacing, new_spacing=(1.0,1.0,1.0)):

    old_spacing = np.array(old_spacing, dtype=np.float64)   # convert to z,y,x
    new_spacing = np.array(new_spacing, dtype=np.float64)

    resize_factor = old_spacing / new_spacing
    new_shape = np.round(
        np.array(volume.shape) * resize_factor
    ).astype(int)

    image = sitk.GetImageFromArray(volume.astype(np.float32))
    # spacing in (x, y, z) for SimpleITK
    image.SetSpacing([
        float(old_spacing[2]),   # x
        float(old_spacing[1]),   # y
        float(old_spacing[0])    # z
    ])


    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing([
        float(new_spacing[2]),
        float(new_spacing[1]),
        float(new_spacing[0])
    ])

    resampler.SetSize([
        int(new_shape[2]),
        int(new_shape[1]),
        int(new_shape[0])
    ])

    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    resampled = resampler.Execute(image)
    result = sitk.GetArrayFromImage(resampled)
    print(f"  Volume resampled: {volume.shape} → {result.shape}")
    return result.astype(np.float32)


# RESAMPLE CT MASK
def resample_mask(mask, old_spacing, new_spacing=(1.0,1.0,1.0)):

    old_spacing = np.array(old_spacing, dtype=np.float64)
    new_spacing = np.array(new_spacing, dtype=np.float64)

    resize_factor = old_spacing / new_spacing
    new_shape = np.round(np.array(mask.shape) * resize_factor).astype(int)

    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetSpacing([
        float(old_spacing[2]),
        float(old_spacing[1]),
        float(old_spacing[0])
    ])

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing([
        float(new_spacing[2]),
        float(new_spacing[1]),
        float(new_spacing[0])
    ])

    resampler.SetSize([
        int(new_shape[2]),
        int(new_shape[1]),
        int(new_shape[0])
    ])

    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    resampled = resampler.Execute(image)
    result = sitk.GetArrayFromImage(resampled)

    print(f"  Mask resampled:   {mask.shape} → {result.shape}")
    return (result > 0).astype(np.uint8)

# CONSISTENCY CHECK
def verify_image_mask_alignment(image_volume, mask_volume):
    """
    If shapes still mismatch after resampling,
    crop both to the minimum overlapping shape
    instead of crashing
    """
    if image_volume.shape == mask_volume.shape:
        return image_volume, mask_volume

    print(f"  [WARN] Shape mismatch — "
          f"Image: {image_volume.shape} | "
          f"Mask: {mask_volume.shape}")
    print(f"  [WARN] Cropping both to minimum overlap")

    # Crop to minimum shape on each axis
    z = min(image_volume.shape[0], mask_volume.shape[0])
    y = min(image_volume.shape[1], mask_volume.shape[1])
    x = min(image_volume.shape[2], mask_volume.shape[2])

    image_volume = image_volume[:z, :y, :x]
    mask_volume  = mask_volume[:z,  :y, :x]

    print(f"  [INFO] Cropped to: {image_volume.shape}")

    return image_volume, mask_volume

# LUNG WINDOW + NORMALIZATION
def window_and_normalize(image, min_hu=-1000, max_hu=400):
    """
    1. Clip to lung window HU range
    2. Z-score normalize
    3. Rescale to [0, 1] for model input
    """

    image = np.clip(image, min_hu, max_hu)

    mean = np.mean(image)
    std = np.std(image) + 1e-8
    image = (image - mean) / std

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image.astype(np.float32)

#  LUNG BOUNDING BOX
def get_lung_bbox(slice_img, margin=10):
    H, W = slice_img.shape

    if slice_img.max() <= 1.0:
        body_threshold = 0.25
        lung_threshold = 0.50
    else:
        body_threshold = -600
        lung_threshold = -300

    body = slice_img > body_threshold
    body = ndi.binary_closing(body, iterations=5)
    body = ndi.binary_fill_holes(body)

    lung = (slice_img < lung_threshold) & body
    lung = ndi.binary_opening(lung, iterations=2)

    label, num = ndi.label(lung)

    if num == 0:
        return 0, H, 0, W

    sizes          = ndi.sum(lung, label, range(1, num + 1))
    top_n          = min(2, num)
    largest_labels = np.argsort(sizes)[-top_n:] + 1
    lung_mask      = np.isin(label, largest_labels)

    coords        = np.column_stack(np.where(lung_mask))
    y_min, x_min  = coords.min(axis=0)
    y_max, x_max  = coords.max(axis=0)

    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(H, y_max + margin)
    x_max = min(W, x_max + margin)


    if y_min >= y_max or x_min >= x_max:
        return 0, H, 0, W

    return y_min, y_max, x_min, x_max

# Resize
def resize_image(image, size=256):
   
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return np.zeros((size, size), dtype=np.float32)
    
    return cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

def resize_mask(mask, size=256):
  
    if mask.size == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
        return np.zeros((size, size), dtype=np.uint8)
    
    resized = cv2.resize(
        mask.astype(np.uint8),
        (size, size),
        interpolation=cv2.INTER_NEAREST
    )
    return (resized > 0).astype(np.uint8)

# tumor slice filter
def get_tumor_slices(mask_volume, min_tumor_pixels=10):
    tumor_indices = []
    for i in range(len(mask_volume)):
        pixel_count = np.sum(mask_volume[i] > 0)
        if pixel_count >= min_tumor_pixels:
            tumor_indices.append(i)
    return tumor_indices

def preprocess_patient(dicom_slices, mask_volume, old_spacing):
    """
    Full preprocessing pipeline for one patient.
    Call this in your dataset __init__ or prepare script.
    """
    # 1. HU conversion
    ct_volume = convert_to_hu(dicom_slices)

    # 2. Resample both to isotropic 1mm spacing
    ct_resampled   = resample_volume(ct_volume,   old_spacing)
    mask_resampled = resample_mask(mask_volume,   old_spacing)

    # 3. Verify alignment
    verify_image_mask_alignment(ct_resampled, mask_resampled)

    # 4. Window and normalize
    ct_normalized = window_and_normalize(ct_resampled)

    return ct_normalized, mask_resampled