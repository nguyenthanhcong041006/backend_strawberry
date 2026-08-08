import cv2
import numpy as np
from PIL import Image

try:
    from rembg import new_session, remove
except ImportError:
    new_session = None
    remove = None


class SmoothU2NetMasker:
    def __init__(self) -> None:
        self.sessions = {}

    def _get_session(self, model_name: str):
        if new_session is None:
            raise RuntimeError("rembg is not installed. Please install it to use U2Net models.")
        if model_name not in self.sessions:
            self.sessions[model_name] = new_session(model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        return self.sessions[model_name]

    def predict_alpha(self, image_bgr: np.ndarray, model_name: str) -> np.ndarray:
        session = self._get_session(model_name)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgba = remove(
            Image.fromarray(image_rgb),
            session=session,
            alpha_matting=True,
            post_process_mask=True,
            alpha_matting_foreground_threshold=150,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=11,
        )
        return np.array(rgba.convert("RGBA"))[:, :, 3].astype(np.uint8)

    def border_normalizer(self, mask: np.ndarray) -> np.ndarray:
        """Thresholds the grayscale mask strictly to 0 or 255 and keeps only the largest component."""
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary_mask
            
        largest_contour = max(contours, key=cv2.contourArea)
        cleaned_mask = np.zeros_like(binary_mask)
        cv2.drawContours(cleaned_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        return cleaned_mask
