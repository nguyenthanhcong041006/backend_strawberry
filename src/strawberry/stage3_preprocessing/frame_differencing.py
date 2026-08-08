"""
Frame differencing for image sequences captured every 15 minutes.

Main idea:
1. Read images in frame order.
2. Compare the current image with the previous frame or the latest stable frame.
3. Compute motion_ratio = changed pixels / total pixels.
4. If motion_ratio exceeds the threshold, mark the frame as needing a new mask.
5. If segmented PNG masks are available, validate each mask using alpha/shape/color checks.
6. Export a CSV report, binary motion masks, and red motion overlays for inspection.
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = PROJECT_ROOT / "src" / "strawberry" / "stage3_preprocessing" / "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    configs = json.load(f)



active_dataset = configs["active_dataset"]
config = configs["datasets"][active_dataset]

print("RAW DATASET =", configs["datasets"][active_dataset])
print("HAS frame_diff =", "frame_diff" in configs["datasets"][active_dataset])
print("CONFIG ID =", id(configs["datasets"][active_dataset]))

if "frame_diff" not in config:
    raise ValueError("Missing frame_diff config for dataset")
# Image formats accepted when scanning the input folder.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default paths for running the script on this project's sample_data.
# For other datasets, pass --input-dir, --mask-dir, and --output-csv from the command line.
DEFAULT_INPUT_DIR = PROJECT_ROOT / configs["datasets"][active_dataset]["output_dir"]

# 
DEFAULT_MASK_DIR = PROJECT_ROOT / "data" / "02_processed" / "strawberry" / "segmented_18-03-2026"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "data" / "02_processed" / "strawberry" / "frame_differencing_results_18-03-2026"
DEFAULT_OUTPUT_CSV = DEFAULT_RESULT_DIR / "frame_differencing_report_18-03-2026.csv"
DEFAULT_DEBUG_DIR = DEFAULT_RESULT_DIR / "motion_masks"
DEFAULT_OVERLAY_DIR = DEFAULT_RESULT_DIR / "motion_overlays"

# HSV color ranges used to check whether a segmented mask looks fruit/object-like.
# These ranges are not used for motion detection; they are only used when validating PNG masks.
STRAWBERRY_COLOR_RANGES = [
    (np.array([0, 25, 18]), np.array([25, 255, 255])),
    (np.array([160, 25, 18]), np.array([180, 255, 255])),
    (np.array([5, 20, 15]), np.array([45, 255, 245])),
    (np.array([35, 25, 15]), np.array([100, 255, 245])),
]

# avocado color range
AVOCADO_COLOR_RANGES = [
    (np.array([25, 40, 10]), np.array([85, 255, 220])),    # dark to medium green avocado skin
    (np.array([0, 30, 10]), np.array([40, 255, 180])),     # brown / ripening spots and edges
    (np.array([15, 30, 20]), np.array([60, 255, 255])),    # olive-yellow transition areas
]

# Morphology kernels remove small noise and fill small gaps in motion/mask regions.
MOTION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MASK_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


@dataclass
class FrameDiffResult:
    """Frame differencing result for one image."""

    frame_path: Path
    compare_path: Path | None
    motion_ratio: float
    changed_pixels: int
    mean_delta: float
    largest_component_ratio: float
    motion_detected: bool
    regenerate_mask: bool
    motion_mask: np.ndarray | None
    reason: str


@dataclass
class MaskValidationResult:
    """Validation result for one segmented PNG mask."""

    mask_path: Path
    valid: bool
    alpha_ratio: float
    fruit_color_ratio: float
    border_touch_ratio: float
    largest_component_ratio: float
    overlap_motion_ratio: float | None
    reason: str


def natural_sort_key(path):
    """Sort filenames naturally, so frame-2 comes before frame-10."""

    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def list_image_paths(input_dir):
    """Collect all image files from the input folder and sort them by frame order."""

    return sorted(
        [
            path
            for path in Path(input_dir).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_sort_key,
    )


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    """Read an image with OpenCV and raise a clear error if it cannot be loaded."""

    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def resize_for_analysis(image, max_side):
    """
    Resize large images to speed up analysis.

    Images smaller than max_side are kept unchanged. This resize is only used for
    motion calculation and does not overwrite the original image.
    """

    if not max_side:
        return image

    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale >= 1.0:
        return image

    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_for_diff(image, max_side=900):
    """
    Normalize an image before frame comparison.

    - Drop the alpha channel if the image is BGRA PNG.
    - Resize to reduce processing cost.
    - Convert to grayscale because motion only needs brightness differences.
    - Apply CLAHE to reduce false motion from lighting changes.
    - Apply Gaussian blur to reduce small camera/compression noise.
    """

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    image = resize_for_analysis(image, max_side=max_side)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize local brightness before differencing so sunlight/exposure changes
    # are less likely to be counted as object motion.
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    gray = clahe.apply(gray)

    return cv2.GaussianBlur(gray, (5, 5), 0)


def threshold_frame_delta(delta, pixel_threshold):
    """
    Convert the delta image into a binary motion mask.

    delta stores the per-pixel difference between two frames. Pixels greater than
    pixel_threshold are treated as changed. If pixel_threshold <= 0, OpenCV's
    Otsu thresholding automatically chooses the threshold.
    """

    if pixel_threshold and pixel_threshold > 0:
        _, motion_mask = cv2.threshold(delta, pixel_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, motion_mask = cv2.threshold(
            delta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Opening removes tiny noise; closing reconnects broken motion regions.
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, MOTION_KERNEL)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, MOTION_KERNEL)
    return motion_mask


def largest_component_ratio(binary_mask):
    """Return the largest connected component area divided by total mask area."""

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
    if num_labels <= 1:
        return 0.0

    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return largest_area / float(binary_mask.size)


def compute_frame_difference(
    current_image,
    compare_image,
    motion_threshold=0.015,
    component_threshold=0.006,
    pixel_threshold=25,
    max_side=900,
):
    """
    Measure motion between current_image and compare_image.

    Returns:
    - motion_ratio: percentage of pixels that changed.
    - changed_pixels: number of changed pixels after thresholding.
    - mean_delta: average pixel difference between the two frames.
    - component_ratio: largest connected motion region ratio.
    - motion_detected: True if motion_ratio or component_ratio exceeds its threshold.
    - motion_mask: binary image showing which pixels changed.
    """

    current_gray = preprocess_for_diff(current_image, max_side=max_side)
    compare_gray = preprocess_for_diff(compare_image, max_side=max_side)

    # If the images have different sizes, resize the comparison image to match the current one.
    if current_gray.shape != compare_gray.shape:
        compare_gray = cv2.resize(
            compare_gray,
            (current_gray.shape[1], current_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    # absdiff creates the delta image: brighter pixels mean stronger frame difference.
    delta = cv2.absdiff(compare_gray, current_gray)
    motion_mask = threshold_frame_delta(delta, pixel_threshold)

    changed_pixels = int(cv2.countNonZero(motion_mask))
    motion_ratio = changed_pixels / float(motion_mask.size)
    mean_delta = float(np.mean(delta))
    component_ratio = largest_component_ratio(motion_mask)

    # Two conditions are used:
    # - motion_ratio catches changes spread across the image.
    # - component_ratio catches a localized object/hand entering the scene.
    motion_detected = (
        motion_ratio >= motion_threshold or component_ratio >= component_threshold
    )

    return motion_ratio, changed_pixels, mean_delta, component_ratio, motion_detected, motion_mask


def create_motion_overlay(current_image, motion_mask, alpha=0.45):
    """
    Create a visual result image for manual checking.

    Changed pixels are painted red on top of the current frame. This is easier
    to inspect than the black/white motion mask alone.
    """

    display_image = resize_for_analysis(current_image, max(motion_mask.shape[:2]))
    if display_image.shape[:2] != motion_mask.shape:
        display_image = cv2.resize(
            display_image,
            (motion_mask.shape[1], motion_mask.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    red_layer = np.zeros_like(display_image)
    red_layer[:, :, 2] = 255
    motion_area = motion_mask > 0
    overlay = display_image.copy()
    blended_pixels = (
        display_image[motion_area].astype(np.float32) * (1.0 - alpha)
        + red_layer[motion_area].astype(np.float32) * alpha
    )
    overlay[motion_area] = np.clip(blended_pixels, 0, 255).astype(np.uint8)
    return overlay

def create_fruit_color_mask(bgr_image):
    """
    Create a fruit/strawberry-like color mask from a BGR image.

    This is used for mask validation: if the alpha mask contains too little
    fruit/object-like color, the mask may be cutting background or a wrong object.
    """

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    ranges = AVOCADO_COLOR_RANGES if active_dataset == "avocado" else STRAWBERRY_COLOR_RANGES
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    object_like = ((saturation > 35) & (value > 18) & (value < 245)).astype("uint8") * 255
    damaged_like = ((saturation > 15) & (value > 12) & (value < 110)).astype("uint8") * 255
    mask = cv2.bitwise_or(mask, object_like)
    mask = cv2.bitwise_or(mask, damaged_like)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, MASK_KERNEL)


def extract_alpha_mask(mask_image):
    """
    Extract a binary foreground mask from a segmented image.

    - BGRA PNG: use the alpha channel.
    - Grayscale image: pixels > 0 are foreground.
    - BGR image without alpha: convert to grayscale, then pixels > 0 are foreground.
    """

    if mask_image.ndim == 3 and mask_image.shape[2] == 4:
        return (mask_image[:, :, 3] > 0).astype("uint8") * 255

    if mask_image.ndim == 2:
        return (mask_image > 0).astype("uint8") * 255

    gray = cv2.cvtColor(mask_image[:, :, :3], cv2.COLOR_BGR2GRAY)
    return (gray > 0).astype("uint8") * 255


def border_touch_ratio(alpha_mask):
    """
    Measure how much foreground touches the image border.

    Too much border contact may mean the ROI cuts through the object or the mask
    leaked into the background.
    """

    border_pixels = np.concatenate(
        [
            alpha_mask[0, :],
            alpha_mask[-1, :],
            alpha_mask[:, 0],
            alpha_mask[:, -1],
        ]
    )
    return float(np.count_nonzero(border_pixels)) / float(border_pixels.size)


def validate_mask(
    mask_path,
    motion_mask=None,
    min_alpha_ratio=0.03,
    max_alpha_ratio=0.90,
    min_fruit_color_ratio=0.20,
    max_border_touch_ratio=0.35,
    max_motion_overlap_ratio=0.55,
):
    """
    Check whether a segmented PNG mask looks reasonable.

    Signs of a bad mask:
    - Empty or too-small alpha: the object was not captured.
    - Too-large alpha: the mask may include background or a hand.
    - Low fruit/object-like color inside alpha: the mask may target the wrong region.
    - Too much border contact: the ROI/mask may be cut off or leaking into the background.
    - If motion_mask has the same size, high overlap with motion may indicate the mask captured a moving object.
    """

    mask_image = read_image(mask_path, flags=cv2.IMREAD_UNCHANGED)
    alpha_mask = extract_alpha_mask(mask_image)
    alpha_pixels = int(cv2.countNonZero(alpha_mask))
    alpha_ratio = alpha_pixels / float(alpha_mask.size)

    reasons = []
    if alpha_pixels == 0:
        reasons.append("empty_alpha")
    if alpha_ratio < min_alpha_ratio:
        reasons.append("alpha_too_small")
    if alpha_ratio > max_alpha_ratio:
        reasons.append("alpha_too_large")

    # Check whether the area inside alpha looks like fruit/object pixels.
    if mask_image.ndim == 3:
        bgr = mask_image[:, :, :3]
        fruit_mask = create_fruit_color_mask(bgr)
        fruit_inside_alpha = cv2.bitwise_and(fruit_mask, fruit_mask, mask=alpha_mask)
        fruit_color_ratio = (
            cv2.countNonZero(fruit_inside_alpha) / float(alpha_pixels)
            if alpha_pixels
            else 0.0
        )
    else:
        fruit_color_ratio = 1.0

    if fruit_color_ratio < min_fruit_color_ratio:
        reasons.append("low_fruit_color_support")

    touch_ratio = border_touch_ratio(alpha_mask)
    if touch_ratio > max_border_touch_ratio:
        reasons.append("touches_border_too_much")

    component_ratio = largest_component_ratio(alpha_mask)
    if component_ratio < min_alpha_ratio:
        reasons.append("fragmented_or_tiny_mask")

    # Only compute overlap when the mask and motion_mask have the same coordinate system.
    # Current sample segmentation masks are cropped ROIs while motion_mask is full-frame, so do not force resize.
    overlap_ratio = None
    if motion_mask is not None and alpha_pixels > 0 and motion_mask.shape == alpha_mask.shape:
        moving_alpha = cv2.bitwise_and(alpha_mask, alpha_mask, mask=motion_mask)
        overlap_ratio = cv2.countNonZero(moving_alpha) / float(alpha_pixels)
        if overlap_ratio > max_motion_overlap_ratio:
            reasons.append("mask_overlaps_motion")

    return MaskValidationResult(
        mask_path=Path(mask_path),
        valid=not reasons,
        alpha_ratio=alpha_ratio,
        fruit_color_ratio=fruit_color_ratio,
        border_touch_ratio=touch_ratio,
        largest_component_ratio=component_ratio,
        overlap_motion_ratio=overlap_ratio,
        reason="ok" if not reasons else "|".join(reasons),
    )


def find_masks_for_frame(mask_dir, frame_path):
    """Find masks whose names start with the frame stem, such as frame_x_strawberry_y.png."""

    if not mask_dir:
        return []

    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        return []

    return sorted(mask_dir.glob(f"{frame_path.stem}_*.png"), key=natural_sort_key)


def build_report_row(diff_result, mask_result):
    """Convert frame diff and mask validation results into one CSV row."""

    row = {
        "frame_path": str(diff_result.frame_path),
        "compare_path": str(diff_result.compare_path) if diff_result.compare_path else "",
        "motion_ratio": f"{diff_result.motion_ratio:.6f}",
        "changed_pixels": diff_result.changed_pixels,
        "mean_delta": f"{diff_result.mean_delta:.3f}",
        "largest_motion_component_ratio": f"{diff_result.largest_component_ratio:.6f}",
        "motion_detected": diff_result.motion_detected,
        "regenerate_mask": diff_result.regenerate_mask,
        "reason": diff_result.reason,
        "mask_path": "",
        "mask_valid": "",
        "mask_alpha_ratio": "",
        "mask_fruit_color_ratio": "",
        "mask_border_touch_ratio": "",
        "mask_largest_component_ratio": "",
        "mask_motion_overlap_ratio": "",
        "mask_reason": "",
    }

    # If a frame has multiple strawberry masks, each mask gets its own CSV row.
    if mask_result is not None:
        row.update(
            {
                "mask_path": str(mask_result.mask_path),
                "mask_valid": mask_result.valid,
                "mask_alpha_ratio": f"{mask_result.alpha_ratio:.6f}",
                "mask_fruit_color_ratio": f"{mask_result.fruit_color_ratio:.6f}",
                "mask_border_touch_ratio": f"{mask_result.border_touch_ratio:.6f}",
                "mask_largest_component_ratio": f"{mask_result.largest_component_ratio:.6f}",
                "mask_motion_overlap_ratio": (
                    f"{mask_result.overlap_motion_ratio:.6f}"
                    if mask_result.overlap_motion_ratio is not None
                    else ""
                ),
                "mask_reason": mask_result.reason,
            }
        )
    return row


def write_report(output_csv, rows):
    """Write the report as CSV so it can be opened with Excel or Pandas."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_path",
        "compare_path",
        "motion_ratio",
        "changed_pixels",
        "mean_delta",
        "largest_motion_component_ratio",
        "motion_detected",
        "regenerate_mask",
        "reason",
        "mask_path",
        "mask_valid",
        "mask_alpha_ratio",
        "mask_fruit_color_ratio",
        "mask_border_touch_ratio",
        "mask_largest_component_ratio",
        "mask_motion_overlap_ratio",
        "mask_reason",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_frame_sequence(
    input_dir,
    mask_dir=None,
    output_csv=DEFAULT_OUTPUT_CSV,
    debug_dir=DEFAULT_DEBUG_DIR,
    overlay_dir=DEFAULT_OVERLAY_DIR,
    motion_threshold=0.015,
    component_threshold=0.006,
    pixel_threshold=25,
    reference_strategy="previous",
    max_side=900,
):
    """
    Main processing pipeline.

    For each frame:
    1. Read the current image.
    2. If it is the first frame, require an initial mask.
    3. Otherwise, compare it with the reference frame and compute motion_ratio.
    4. If motion is detected, set regenerate_mask=True.
    5. If mask_dir is provided, validate that frame's masks.
    6. Write all results to a CSV report.
    """

    image_paths = list_image_paths(input_dir)
    if not image_paths:
        raise ValueError(f"No images found in: {input_dir}")

    # When provided, debug_dir stores black/white motion masks.
    debug_dir = Path(debug_dir) if debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # When provided, overlay_dir stores frames with changed areas painted red.
    overlay_dir = Path(overlay_dir) if overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    results = []
    rows = []
    compare_path = None
    compare_image = None

    for frame_path in image_paths:
        current_image = read_image(frame_path, flags=cv2.IMREAD_COLOR)

        if compare_image is None:
            # The first frame has no previous frame for comparison, so it needs an initial mask.
            diff_result = FrameDiffResult(
                frame_path=frame_path,
                compare_path=None,
                motion_ratio=0.0,
                changed_pixels=0,
                mean_delta=0.0,
                largest_component_ratio=0.0,
                motion_detected=False,
                regenerate_mask=True,
                motion_mask=None,
                reason="first_frame_needs_initial_mask",
            )
        else:
            # Measure motion between the current frame and the current reference frame.
            (
                motion_ratio,
                changed_pixels,
                mean_delta,
                component_ratio,
                motion_detected,
                motion_mask,
            ) = compute_frame_difference(
                current_image=current_image,
                compare_image=compare_image,
                motion_threshold=motion_threshold,
                component_threshold=component_threshold,
                pixel_threshold=pixel_threshold,
                max_side=max_side,
            )

            diff_result = FrameDiffResult(
                frame_path=frame_path,
                compare_path=compare_path,
                motion_ratio=motion_ratio,
                changed_pixels=changed_pixels,
                mean_delta=mean_delta,
                largest_component_ratio=component_ratio,
                motion_detected=motion_detected,
                regenerate_mask=motion_detected,
                motion_mask=motion_mask,
                reason="motion_detected" if motion_detected else "stable",
            )

            # Save the black/white motion mask for manual debugging if requested.
            if debug_dir:
                debug_path = debug_dir / f"{frame_path.stem}_motion.png"
                cv2.imwrite(str(debug_path), motion_mask)

            # Save a red overlay on top of the current frame for easier visual checking.
            if overlay_dir:
                overlay_path = overlay_dir / f"{frame_path.stem}_overlay.png"
                overlay = create_motion_overlay(current_image, motion_mask)
                cv2.imwrite(str(overlay_path), overlay)

        # Find and validate every mask belonging to the current frame.
        mask_results = []
        for mask_path in find_masks_for_frame(mask_dir, frame_path):
            mask_results.append(validate_mask(mask_path, diff_result.motion_mask))

        # If any mask is invalid, the whole frame should regenerate its mask.
        invalid_masks = [item for item in mask_results if not item.valid]
        if invalid_masks:
            diff_result.regenerate_mask = True
            diff_result.reason = f"{diff_result.reason}|invalid_mask"

        # previous: always compare the next frame with the current frame.
        # last_stable: if the current frame has motion, keep the old reference to avoid using a disturbed frame as the new baseline.
        if reference_strategy == "previous" or not diff_result.motion_detected:
            compare_path = frame_path
            compare_image = current_image

        results.append(diff_result)
        if mask_results:
            for mask_result in mask_results:
                rows.append(build_report_row(diff_result, mask_result))
        else:
            rows.append(build_report_row(diff_result, None))

    write_report(output_csv, rows)
    return results, rows


def parse_args():
    """Define command-line options for running this file directly."""

    parser = argparse.ArgumentParser(
        description=(
            "Detect motion between 15-minute frames and flag frames/masks that "
            "need a new segmentation mask."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--motion-threshold", type=float, default=0.015)
    parser.add_argument("--component-threshold", type=float, default=0.006)
    parser.add_argument("--pixel-threshold", type=int, default=45)
    parser.add_argument(
        "--reference-strategy",
        choices=["previous", "last_stable"],
        default="previous",
        help="previous compares with the last frame; last_stable keeps the last no-motion frame as reference.",
    )
    parser.add_argument("--max-side", type=int, default=900)
    return parser.parse_args()


def main():
    """Entry point when running: python frame_differencing.py ..."""

    args = parse_args()
    processed_root = PROJECT_ROOT / "data" / "02_processed" / active_dataset

    if active_dataset == "strawberry":
        cropped_folders = sorted(
            [
                folder
                for folder in processed_root.iterdir()
                if folder.is_dir()
                and re.match(r"^cropped_\d{2}-\d{2}-\d{4}$", folder.name) is not None
            ]
        )
    else:
        cropped_folders = sorted(
            [
                folder
                for folder in processed_root.iterdir()
                if folder.is_dir()
                and folder.name == "cropped_avocado"
            ]
        )

    if not cropped_folders:
        print(f"No cropped folders found in: {processed_root}")
        return

    all_results = []
    all_rows = []
    output_csvs = []
    
    for input_dir in cropped_folders:
        date_str = input_dir.name.replace("cropped_", "")
        # mask_prefix = frame_diff_cfg.get("mask_prefix", "segmented")
        mask_dir = processed_root / f"segmented_{date_str}"
        result_dir = processed_root / f"frame_differencing_results_{date_str}"
        output_csv = result_dir / f"frame_differencing_report_{date_str}.csv"
        debug_dir = result_dir / "motion_masks"
        overlay_dir = result_dir / "motion_overlays"

        print(f"Processing input folder: {input_dir}")
        print(f"Using mask folder: {mask_dir}")
        print(f"Output CSV: {output_csv}")
        print(f"Debug motion masks: {debug_dir}")
        print(f"Overlay images: {overlay_dir}")

        results, rows = analyze_frame_sequence(
            input_dir=input_dir,
            mask_dir=mask_dir,
            output_csv=output_csv,
            debug_dir=debug_dir,
            overlay_dir=overlay_dir,
            motion_threshold=args.motion_threshold,
            component_threshold=args.component_threshold,
            pixel_threshold=args.pixel_threshold,
            reference_strategy=args.reference_strategy,
            max_side=args.max_side,
        )
        all_results.extend(results)
        all_rows.extend(rows)
        output_csvs.append(output_csv)

    # Print a quick summary after processing finishes.
    motion_count = sum(result.motion_detected for result in all_results)
    regenerate_count = sum(result.regenerate_mask for result in all_results)
    invalid_mask_count = sum(
        1 for row in all_rows if str(row.get("mask_valid", "")).lower() == "false"
    )

    print("=" * 60)
    print(f"Frames analyzed: {len(all_results)}")
    print(f"Motion frames: {motion_count}")
    print(f"Frames needing new mask: {regenerate_count}")
    print(f"Invalid masks: {invalid_mask_count}")
    print("Reports saved to:")
    for output_csv in output_csvs:
        print(f"- {output_csv}")


if __name__ == "__main__":
    main()