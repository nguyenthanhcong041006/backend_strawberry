
import csv
import json
import re
import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Project / config setup  (relative paths, identical pattern to other scripts)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    configs = json.load(f)

active_dataset = configs["active_dataset"]
config = configs["datasets"][active_dataset]

PROCESSED_DIR = PROJECT_ROOT / "data" / "02_processed" / active_dataset
CROPPED_ROOT = PROCESSED_DIR

MASK_PREFIX = config["frame_diff"].get("mask_prefix", "segmented")


# ---------------------------------------------------------------------------
# Segmentation constants  (unchanged from original backup)
# ---------------------------------------------------------------------------
STRAWBERRY_COLOR_RANGES = [
    (np.array([0, 25, 18]), np.array([25, 255, 255])),    # red/dark red/orange
    (np.array([160, 25, 18]), np.array([180, 255, 255])), # wrapped red
    (np.array([5, 20, 15]), np.array([45, 255, 245])),    # brown/damaged fruit
    (np.array([35, 25, 15]), np.array([100, 255, 245])),  # green calyx/leaves
]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grabcut_outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))


# ---------------------------------------------------------------------------
# Segmentation functions  (unchanged from original backup)
# ---------------------------------------------------------------------------

def create_strawberry_candidate_mask(hsv):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in STRAWBERRY_COLOR_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dark_or_saturated_object = ((saturation > 35) & (value > 18) & (value < 245)).astype("uint8") * 255
    dark_damaged_object = ((saturation > 15) & (value > 12) & (value < 110)).astype("uint8") * 255
    mask = cv2.bitwise_or(mask, dark_or_saturated_object)
    mask = cv2.bitwise_or(mask, dark_damaged_object)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def is_valid_strawberry_contour(cnt, img_h, img_w):
    area = cv2.contourArea(cnt)
    min_area = max(300, int(0.0003 * img_h * img_w))
    if area < min_area:
        return False

    x, y, w_box, h_box = cv2.boundingRect(cnt)
    if w_box < 20 or h_box < 20:
        return False

    aspect_ratio = w_box / float(h_box)
    if aspect_ratio < 0.25 or aspect_ratio > 3.0:
        return False

    extent = area / float(w_box * h_box)
    return extent > 0.12


def create_grabcut_mask(roi, roi_support):
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    grabcut_mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)

    probable_fg = cv2.dilate(roi_support, kernel, iterations=2) > 0
    sure_fg = cv2.erode(roi_support, small_kernel, iterations=1) > 0
    sure_bg = cv2.dilate(roi_support, grabcut_outer_kernel, iterations=1) == 0
    white_bg = (roi_hsv[:, :, 1] < 60) & (roi_hsv[:, :, 2] > 90)  # edit remove background

    grabcut_mask[probable_fg] = cv2.GC_PR_FGD
    grabcut_mask[sure_fg] = cv2.GC_FGD
    grabcut_mask[sure_bg | white_bg] = cv2.GC_BGD

    grabcut_mask[0, :] = cv2.GC_BGD
    grabcut_mask[-1, :] = cv2.GC_BGD
    grabcut_mask[:, 0] = cv2.GC_BGD
    grabcut_mask[:, -1] = cv2.GC_BGD

    if not np.any((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)):
        grabcut_mask[roi_support > 0] = cv2.GC_PR_FGD

    return grabcut_mask


def refine_foreground_mask(mask_res, color_support, roi):
    color_support = cv2.dilate(color_support, kernel, iterations=1)
    refined = mask_res & (color_support > 0).astype("uint8")

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    background_like = (roi_hsv[:, :, 1] < 70) & (roi_hsv[:, :, 2] > 70)  # edit remove background
    refined[background_like] = 0

    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, small_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, 8)
    if num_labels <= 1:
        return refined

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype("uint8")


def apply_mask_to_roi(roi, mask_res):
    b_channel, g_channel, r_channel = cv2.split(roi)
    alpha_channel = mask_res * 255
    return cv2.merge([b_channel, g_channel, r_channel, alpha_channel])


# ---------------------------------------------------------------------------
# Helper: natural sort
# ---------------------------------------------------------------------------

def natural_sort_key(path):
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(p) if p.isdigit() else p for p in parts]


# ---------------------------------------------------------------------------
# Core: segment a single image and save results to output_dir
# ---------------------------------------------------------------------------

def segment_image(img_path: Path, output_dir: Path, clear_old: bool = True) -> int:
    """
    Segment one image and save transparent PNG crops.

    Args:
        clear_old: if True, delete any existing masks for this frame before
                   segmenting (used when regenerate_mask=True or no CSV).
                   If False, keep existing masks and only add/overwrite with
                   new results.

    Returns the number of strawberry objects saved.
    """
    base_name = img_path.stem

    # Optionally remove previously generated masks for this frame
    if clear_old:
        for old_mask in output_dir.glob(f"{base_name}_strawberry_*.png"):
            old_mask.unlink()

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] Cannot read image: {img_path}")
        return 0

    h, w = img.shape[:2]
    print(f"  --- Segmenting: {base_name} ({w}x{h}) ---")

    #convert image to HSV color space and create masks for strawberry-colored regions
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = create_strawberry_candidate_mask(hsv)

    #find contours of the thresholded image to get bounding boxes for potential strawberries
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if is_valid_strawberry_contour(cnt, h, w)]
    contours = sorted(contours, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

    strawberry_idx = 1

    for cnt in contours:
        #use bounding box to define ROI for GrabCut
        x, y, w_box, h_box = cv2.boundingRect(cnt)

        #expand bounding box
        pad = 20  # increase bounding box by 20 pixels on each side
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_end = min(w, x + w_box + pad)
        y_end = min(h, y + h_box + pad)

        #crop ROI from original image
        roi = img[y_start:y_end, x_start:x_end]
        roi_color_support = thresh[y_start:y_end, x_start:x_end]

        #GrabCut => mask => apply mask
        mask = create_grabcut_mask(roi, roi_color_support)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        #segmentation with GrabCut
        cv2.grabCut(roi, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        #filter mask to keep only foreground (strawberry) pixels
        mask_res = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        mask_res = refine_foreground_mask(mask_res, roi_color_support, roi)
        if np.count_nonzero(mask_res) < max(200, int(0.0002 * h * w)):
            continue

        #apply mask and create transparent image with alpha channel
        strawberry_transparent = apply_mask_to_roi(roi, mask_res)

        # save images with name format: {base_name}_strawberry_{index}.png
        output_filename = f"{base_name}_strawberry_{strawberry_idx}.png"
        cv2.imwrite(str(output_dir / output_filename), strawberry_transparent)
        strawberry_idx += 1

    saved = strawberry_idx - 1
    print(f"  -> Done {base_name}: segmented {saved} strawberries\n")
    return saved


# ---------------------------------------------------------------------------
# Helper: load frame_differencing CSV and return {frame_stem -> regenerate}
# ---------------------------------------------------------------------------

def load_regenerate_map(csv_path: Path) -> dict:
    """
    Read the frame_differencing CSV and build a dict:
        frame_stem -> True/False  (whether to regenerate the mask)

    A frame needs regeneration if ANY row for that stem has regenerate_mask=True.
    """
    regen_map = {}
    if not csv_path.exists():
        return regen_map

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_path_str = row.get("frame_path", "")
            regen_str = row.get("regenerate_mask", "").strip().lower()
            if not frame_path_str:
                continue
            stem = Path(frame_path_str).stem
            regen = regen_str == "true"
            # If already True, keep True; otherwise update
            regen_map[stem] = regen_map.get(stem, False) or regen

    return regen_map


# ---------------------------------------------------------------------------
# Process one cropped folder
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_cropped_folder(cropped_dir: Path) -> None:
    """
    Segment all images in *cropped_dir*, respecting frame_differencing results.

    Expected naming conventions:
      - cropped folder  : data/02_processed/strawberry/cropped_<date>
      - segmented folder: data/02_processed/strawberry/segmented_<date>
      - diff CSV        : data/02_processed/strawberry/frame_differencing_results_<date>/
                          frame_differencing_report_<date>.csv
    """
    # Derive the date suffix from the folder name (e.g. "18-03-2026")
    folder_name = cropped_dir.name                    # e.g. "cropped_18-03-2026"
    date_suffix = folder_name[len("cropped_"):]       # e.g. "18-03-2026"

    output_dir = PROCESSED_DIR / f"{MASK_PREFIX}_{date_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate the frame_differencing CSV for this date (if any)
    csv_path = (
        PROCESSED_DIR
        / f"frame_differencing_results_{date_suffix}"
        / f"frame_differencing_report_{date_suffix}.csv"
    )

    regen_map = load_regenerate_map(csv_path)
    has_csv = csv_path.exists()

    if has_csv:
        print(f"[INFO] Found frame_differencing CSV: {csv_path}")
    else:
        print(f"[INFO] No frame_differencing CSV found for {date_suffix} — will segment ALL images.")

    # Collect and sort images
    image_paths = sorted(
        [p for p in cropped_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key,
    )

    if not image_paths:
        print(f"[WARN] No images found in: {cropped_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Cropped folder : {cropped_dir}")
    print(f"Output folder  : {output_dir}")
    print(f"Total images   : {len(image_paths)}")
    print(f"{'='*60}")

    total_segmented = 0

    for img_path in image_paths:
        stem = img_path.stem

        # regenerate=True  → wipe old masks and run fresh segmentation
        # regenerate=False → run segmentation but keep any existing masks
        #                    (segment_image will still overwrite if new objects found)
        # No CSV           → always run segmentation (first-time run)
        if has_csv:
            regenerate = regen_map.get(stem, True)
            if not regenerate:
                print(f"  [INFO] {stem} — regenerate_mask=False, segmenting without clearing old masks")

        count = segment_image(img_path, output_dir, clear_old=not has_csv or regen_map.get(stem, True))
        total_segmented += count

    print(f"\n{'='*60}")
    print(f"Done [{date_suffix}]: segmented={total_segmented} objects from {len(image_paths)} frames")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main: find all 'cropped*' subfolders inside the configured output_dir
# ---------------------------------------------------------------------------

def main() -> None:
    # CROPPED_ROOT = data/02_processed/cropped_strawberry  (from config output_dir)
    if not CROPPED_ROOT.exists():
        print(f"[ERROR] Cropped root directory does not exist: {CROPPED_ROOT}")
        return

    # Find every subfolder whose name starts with 'cropped' inside CROPPED_ROOT
    # e.g. cropped_18-03-2026, cropped_19-03-2026, ...
    cropped_dirs = sorted(
        [d for d in CROPPED_ROOT.iterdir() if d.is_dir() and d.name.startswith("cropped")],
        key=lambda d: d.name,
    )

    if not cropped_dirs:
        print(f"[WARN] No 'cropped*' subfolders found in: {CROPPED_ROOT}")
        return

    print(f"Found {len(cropped_dirs)} cropped folder(s) to process.")
    for d in cropped_dirs:
        print(f"  - {d.name}")

    for cropped_dir in cropped_dirs:
        process_cropped_folder(cropped_dir)

    print("=" * 60)
    print("Segmentation complete for all cropped folders.")


if __name__ == "__main__":
    main()
