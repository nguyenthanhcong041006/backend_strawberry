import cv2
import numpy as np

REFERENCE_SIZE = (1920, 1080)

class TiltedRoi:
    def __init__(self, fruit_id: int, center: tuple[int, int], axes: tuple[int, int], angle_deg: int):
        self.fruit_id = fruit_id
        self.center = center
        self.axes = axes
        self.angle_deg = angle_deg

    def scaled_center(self, width: int, height: int) -> tuple[int, int]:
        scale_x = width / REFERENCE_SIZE[0]
        scale_y = height / REFERENCE_SIZE[1]
        return round(self.center[0] * scale_x), round(self.center[1] * scale_y)

    def scaled_axes(self, width: int, height: int) -> tuple[int, int]:
        scale_x = width / REFERENCE_SIZE[0]
        scale_y = height / REFERENCE_SIZE[1]
        return round(self.axes[0] * scale_x), round(self.axes[1] * scale_y)

class TiltedAvocadoRoiMap:
    """Defines regions where avocados are located in the frame."""
    ROIS = (
        TiltedRoi(1, (540, 790), (330, 105), 58),
        TiltedRoi(2, (1105, 805), (310, 120), 61),
        TiltedRoi(3, (1625, 825), (320, 100), 60),
        TiltedRoi(4, (590, 345), (335, 120), 61),
        TiltedRoi(5, (1060, 335), (325, 125), 52),
        TiltedRoi(6, (1585, 295), (285, 100), 58),
    )

    def __init__(self, threshold_px: int = 50) -> None:
        self.threshold_px = threshold_px

    def build_bounding_boxes(self, width: int, height: int) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Returns fruit_id and (x, y, w, h) for each ROI."""
        boxes = []
        for roi in self.ROIS:
            core = np.zeros((height, width), dtype=np.uint8)
            center = roi.scaled_center(width, height)
            axes = roi.scaled_axes(width, height)
            cv2.ellipse(core, center, axes, roi.angle_deg, 0, 360, 255, -1, cv2.LINE_AA)
            
            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.threshold_px * 2 + 1, self.threshold_px * 2 + 1),
            )
            accepted = cv2.dilate(core, dilation_kernel, iterations=1)
            x, y, w, h = cv2.boundingRect(accepted)
            boxes.append((roi.fruit_id, (x, y, w, h)))
        return boxes


class ImageEnhancer:
    def __init__(self, use_grayworld: bool, use_clahe: bool, clahe_clip_limit: float, clahe_grid_size: int):
        self.use_grayworld = use_grayworld
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = max(2, clahe_grid_size)

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        enhanced = image_bgr.copy()
        if self.use_grayworld:
            enhanced = self.apply_grayworld(enhanced)
        if self.use_clahe:
            enhanced = self.apply_clahe_lab(
                enhanced,
                clip_limit=self.clahe_clip_limit,
                grid_size=(self.clahe_grid_size, self.clahe_grid_size),
            )
        return enhanced

    @staticmethod
    def apply_grayworld(image_bgr: np.ndarray) -> np.ndarray:
        channels = cv2.split(image_bgr.astype(np.float32))
        means = np.array([channel.mean() for channel in channels], dtype=np.float32)
        if np.any(means <= 1e-6):
            return image_bgr

        target = float(means.mean())
        corrected = [
            channel * (target / channel_mean)
            for channel, channel_mean in zip(channels, means)
        ]
        return np.clip(cv2.merge(corrected), 0, 255).astype(np.uint8)

    @staticmethod
    def apply_clahe_lab(
        image_bgr: np.ndarray,
        clip_limit: float = 2.0,
        grid_size: tuple[int, int] = (5, 5),
    ) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        lightness = clahe.apply(lightness)
        return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
