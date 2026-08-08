from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils_app.image_utils import center_crop, remove_alpha


STRAWBERRY_COLOR_RANGES = [
    (np.array([0, 25, 18]), np.array([25, 255, 255])),
    (np.array([160, 25, 18]), np.array([180, 255, 255])),
    (np.array([5, 20, 15]), np.array([45, 255, 245])),
    (np.array([35, 25, 15]), np.array([100, 255, 245])),
]

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
SMALL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
GRABCUT_OUTER_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))


@dataclass(frozen=True)
class PreprocessResult:
    roi: np.ndarray
    mask: np.ndarray
    confidence: float
    bbox: tuple[int, int, int, int]


def apply_grayworld(image: np.ndarray) -> np.ndarray:
    try:
        b, g, r = cv2.split(image)
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        if min(b_mean, g_mean, r_mean) == 0:
            return image
        avg_mean = (b_mean + g_mean + r_mean) / 3.0
        b_scaled = b.astype(np.float32) * (avg_mean / b_mean)
        g_scaled = g.astype(np.float32) * (avg_mean / g_mean)
        r_scaled = r.astype(np.float32) * (avg_mean / r_mean)
        return np.clip(cv2.merge((b_scaled, g_scaled, r_scaled)), 0, 255).astype(np.uint8)
    except (cv2.error, ValueError):
        return image


def apply_clahe_lab(image: np.ndarray, clip_limit: float = 2.0, grid_size: tuple[int, int] = (5, 5)) -> np.ndarray:
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l_clahe = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge((l_clahe, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    except (cv2.error, ValueError):
        return image


def create_candidate_mask(hsv: np.ndarray) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in STRAWBERRY_COLOR_RANGES:
        mask = cv2.inRange(hsv, lower, upper) | mask

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    object_mask = ((saturation > 35) & (value > 18) & (value < 245)).astype(np.uint8) * 255
    damaged_mask = ((saturation > 15) & (value > 12) & (value < 110)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(mask, object_mask)
    mask = cv2.bitwise_or(mask, damaged_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)
    return mask


def is_valid_strawberry_contour(contour: np.ndarray, image_h: int, image_w: int, min_area_ratio: float) -> bool:
    area = cv2.contourArea(contour)
    min_area = max(300, int(min_area_ratio * image_h * image_w))
    if area < min_area:
        return False

    _x, _y, w_box, h_box = cv2.boundingRect(contour)
    if w_box < 20 or h_box < 20:
        return False

    aspect_ratio = w_box / float(h_box)
    if aspect_ratio < 0.25 or aspect_ratio > 3.0:
        return False

    extent = area / float(w_box * h_box)
    return extent > 0.12


def create_grabcut_mask(roi: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    grabcut_mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)

    probable_fg = cv2.dilate(support_mask, KERNEL, iterations=2) > 0
    sure_fg = cv2.erode(support_mask, SMALL_KERNEL, iterations=1) > 0
    sure_bg = cv2.dilate(support_mask, GRABCUT_OUTER_KERNEL, iterations=1) == 0
    white_bg = (roi_hsv[:, :, 1] < 60) & (roi_hsv[:, :, 2] > 90)

    grabcut_mask[probable_fg] = cv2.GC_PR_FGD
    grabcut_mask[sure_fg] = cv2.GC_FGD
    grabcut_mask[sure_bg | white_bg] = cv2.GC_BGD
    grabcut_mask[0, :] = cv2.GC_BGD
    grabcut_mask[-1, :] = cv2.GC_BGD
    grabcut_mask[:, 0] = cv2.GC_BGD
    grabcut_mask[:, -1] = cv2.GC_BGD

    if not np.any((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)):
        grabcut_mask[support_mask > 0] = cv2.GC_PR_FGD
    return grabcut_mask


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255


def refine_mask(mask_res: np.ndarray, color_support: np.ndarray, roi: np.ndarray) -> np.ndarray:
    color_support = cv2.dilate(color_support, KERNEL, iterations=1)
    refined = mask_res & (color_support > 0).astype(np.uint8)

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    background_like = (roi_hsv[:, :, 1] < 70) & (roi_hsv[:, :, 2] > 70)
    refined[background_like] = 0

    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, SMALL_KERNEL)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, KERNEL)
    refined = largest_component_mask(refined.astype(np.uint8) * 255)
    blurred = cv2.GaussianBlur(refined, (15, 15), 0)
    return np.where(blurred > 127, 255, 0).astype(np.uint8)


def apply_transparent_mask(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b_channel, g_channel, r_channel = cv2.split(roi)
    return cv2.merge([b_channel, g_channel, r_channel, mask])


def estimate_segmentation_confidence(mask: np.ndarray, support_mask: np.ndarray, roi_shape: tuple[int, int, int]) -> float:
    mask_pixels = float(cv2.countNonZero(mask))
    if mask_pixels == 0:
        return 0.0

    roi_area = float(roi_shape[0] * roi_shape[1])
    fill_ratio = min(mask_pixels / max(roi_area, 1.0), 1.0)
    support_overlap = cv2.countNonZero(cv2.bitwise_and(mask, support_mask))
    overlap_ratio = support_overlap / mask_pixels

    confidence = (0.65 * overlap_ratio) + (0.35 * min(fill_ratio / 0.45, 1.0))
    return float(np.clip(confidence, 0.0, 1.0))


def preprocess(image: np.ndarray, config: dict) -> PreprocessResult | None:
    if image is None or image.size == 0:
        return None

    image = remove_alpha(image)
    image = center_crop(image, config.get("crop_width"), config.get("crop_height"))
    if image is None or image.size == 0:
        return None

    image_h, image_w = image.shape[:2]
    corrected = apply_grayworld(image)
    enhanced = apply_clahe_lab(corrected)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    candidate_mask = create_candidate_mask(hsv)

    contours, _hierarchy = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        contour
        for contour in contours
        if is_valid_strawberry_contour(
            contour,
            image_h,
            image_w,
            float(config.get("min_strawberry_area_ratio", 0.0003)),
        )
    ]
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(contour)
    pad = 20
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_w, x + w_box + pad)
    y2 = min(image_h, y + h_box + pad)

    roi = image[y1:y2, x1:x2]
    roi_support = candidate_mask[y1:y2, x1:x2]
    if roi.size == 0 or cv2.countNonZero(roi_support) == 0:
        return None

    try:
        grabcut_mask = create_grabcut_mask(roi, roi_support)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(roi, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    mask_res = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        1,
        0,
    ).astype(np.uint8)
    final_mask = refine_mask(mask_res, roi_support, roi)

    min_pixels = int(config.get("min_mask_pixels", 300))
    if cv2.countNonZero(final_mask) < min_pixels:
        return None

    confidence = estimate_segmentation_confidence(final_mask, roi_support, roi.shape)
    transparent_roi = apply_transparent_mask(roi, final_mask)
    return PreprocessResult(
        roi=transparent_roi,
        mask=final_mask,
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
    )
