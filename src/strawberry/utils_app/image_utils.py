from __future__ import annotations

from io import BytesIO
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}


def validate_image(content: bytes, content_type: str | None = None) -> bool:
    if not content:
        return False
    if content_type and content_type.lower() not in SUPPORTED_IMAGE_TYPES:
        return False
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def read_image(content: bytes) -> np.ndarray | None:
    try:
        array = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        return remove_alpha(image)
    except (cv2.error, ValueError):
        return None


def remove_alpha(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        return np.clip((bgr * alpha) + (white * (1.0 - alpha)), 0, 255).astype(np.uint8)
    return image


def center_crop(image: np.ndarray, crop_width: int | None = None, crop_height: int | None = None) -> np.ndarray:
    height, width = image.shape[:2]
    if not crop_width or not crop_height:
        return image

    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)
    x1 = max(0, (width - crop_width) // 2)
    y1 = max(0, (height - crop_height) // 2)
    return image[y1 : y1 + crop_height, x1 : x1 + crop_width]


def resize(image: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        size = (size, size)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(remove_alpha(image), cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def numpy_to_tensor(image: np.ndarray, image_size: int = 224, device: str | torch.device = "cpu") -> torch.Tensor:
    image = remove_alpha(image)
    image = resize(image, image_size)
    rgb = bgr_to_rgb(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb - mean) / std
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
    return tensor.to(device)


def ensure_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
