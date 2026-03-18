import numpy as np
import SimpleITK as sitk
import cv2

def convert_to_hu(slices):
    images = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    for i,s in enumerate(slices):
        intercept = s.RescaleIntercept
        slope = s.RescaleSlope

        if slope != 1:
            images[i] = slope * images[i].astype(np.float64)
            images[i] = images[i].astype(np.int16)

        images[i] += np.int16(intercept)

    return images

# Resampling 
def resample_volume(volume, old_spacing, new_spacing=(1,1,1)):

    old_spacing = np.array(old_spacing[::-1])   # convert to z,y,x
    new_spacing = np.array(new_spacing)

    resize_factor = old_spacing / new_spacing
    new_shape = np.round(volume.shape * resize_factor).astype(int)

    image = sitk.GetImageFromArray(volume)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetSize([int(x) for x in new_shape[::-1]])
    resampler.SetOutputSpacing(list(new_spacing[::-1]))
    resampled = resampler.Execute(image)

    return sitk.GetArrayFromImage(resampled)

# lung window
def window_image(image, min_hu=-1000, max_hu=400):
    image = np.clip(image, min_hu, max_hu)
    return image

# Z-score normalization
def zscore_normalize(image):
    image = (image - np.mean(image)) / (np.std(image) + 1e-8)

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

# ---------- Lung BBOX ----------
import numpy as np
import SimpleITK as sitk


# ==============================
# HU CONVERSION
# ==============================
def convert_to_hu(slices):

    images = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    for i, s in enumerate(slices):

        intercept = s.RescaleIntercept
        slope = s.RescaleSlope

        if slope != 1:
            images[i] = slope * images[i].astype(np.float64)
            images[i] = images[i].astype(np.int16)

        images[i] += np.int16(intercept)

    return images


# ==============================
# RESAMPLE CT VOLUME
# ==============================
def resample_volume(volume, old_spacing, new_spacing=(1,1,1)):

    image = sitk.GetImageFromArray(volume)

    resize_factor = old_spacing / new_spacing
    new_shape = np.round(volume.shape * resize_factor)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)

    resampler.SetOutputSpacing(list(new_spacing))
    resampler.SetSize([int(x) for x in new_shape[::-1]])

    resampled = resampler.Execute(image)
    resampled = sitk.GetArrayFromImage(resampled)

    return resampled


# ==============================
# RESAMPLE MASK (VERY IMPORTANT)
# ==============================
def resample_mask(volume, old_spacing, new_spacing=(1,1,1)):

    image = sitk.GetImageFromArray(volume)

    resize_factor = old_spacing / new_spacing
    new_shape = np.round(volume.shape * resize_factor)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize([int(x) for x in new_shape[::-1]])

    resampled = resampler.Execute(image)
    resampled = sitk.GetArrayFromImage(resampled)

    return resampled


# ==============================
# LUNG WINDOW + NORMALIZATION
# ==============================
def window_and_normalize(image, min_hu=-1000, max_hu=400):

    image = np.clip(image, min_hu, max_hu)

    mean = np.mean(image)
    std = np.std(image) + 1e-8

    image = (image - mean) / std

    return image

#  LUNG BOUNDING BOX
import scipy.ndimage as ndi

def get_lung_bbox(slice_img):

    # Step 1 — body mask (remove outside air)
    body = slice_img > -600

    body = ndi.binary_closing(body, iterations=5)
    body = ndi.binary_fill_holes(body)

    # Step 2 — lung air inside body
    lung = (slice_img < -300) & body

    lung = ndi.binary_opening(lung, iterations=2)

    # Step 3 — keep largest components
    label, num = ndi.label(lung)

    if num == 0:
        return 0, slice_img.shape[0], 0, slice_img.shape[1]

    sizes = ndi.sum(lung, label, range(1, num+1))
    largest_labels = np.argsort(sizes)[-2:] + 1

    lung_mask = np.isin(label, largest_labels)

    coords = np.column_stack(np.where(lung_mask))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # add small margin
    margin = 10
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(slice_img.shape[0], y_max + margin)
    x_max = min(slice_img.shape[1], x_max + margin)

    return y_min, y_max, x_min, x_max

# Resize
def resize_image(image, size=256):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

def resize_mask(mask, size=256):
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

# tumor slice filter
def get_tumor_slices(mask_volume):
    tumor_indices = []
    for i in range(len(mask_volume)):
        if np.sum(mask_volume[i]) > 0:
            tumor_indices.append(i)
    return tumor_indices