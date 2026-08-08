import cv2
import numpy as np


def apply_mask_to_crop(mask: np.ndarray, crop_bgr: np.ndarray) -> np.ndarray:
    """
    Applies the binary mask to the BGR crop and returns an RGBA image
    with a transparent background.
    """
    b, g, r = cv2.split(crop_bgr)
    return cv2.merge((b, g, r, mask))
