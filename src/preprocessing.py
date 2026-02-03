import numpy as np

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

def window_image(image, min_hu=-1000, max_hu=400):
    image = np.clip(image, min_hu, max_hu)
    image = (image - min_hu) / (max_hu - min_hu)
    return image