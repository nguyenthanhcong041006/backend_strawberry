# Preprocessing Guide

This guide explains how to run the Stage 3 preprocessing pipeline for **strawberry** and **avocado** datasets. It covers configuration, folder structure, script order, and common path mistakes.

All scripts read settings from:

```text
src/stage3_preprocessing/config.json
```

Run every command from the project root:

```text
Strawberry-RUL-prediction/
```

Example:

```bash
python src/stage3_preprocessing/crop_images.py
```

---

## 1. Quick Start

1. Open `src/stage3_preprocessing/config.json`.
2. Set `active_dataset` to the fruit you want to process:
   - `"strawberry"`
   - `"avocado"`
3. Adjust `crop.width` and `crop.height` if needed.
4. Place raw images in the correct input folder (see Section 4).
5. Run the scripts in order:

```bash
python src/stage3_preprocessing/crop_images.py
python src/stage3_preprocessing/frame_differencing.py
python src/stage3_preprocessing/segmentation.py
python src/stage3_preprocessing/assign_id.py
```

Or run the full pipeline:

```bash
python src/stage3_preprocessing/main_preprocessing.py
```

Note: `main_preprocessing.py` also runs later steps (`eol.py`, `label_rul.py`, `manifests.py`, `split_data.py`). Use individual scripts above if you only need crop, segment, QC, and ID assignment.

---

## 2. Configuration File

File: `src/stage3_preprocessing/config.json`

### 2.1 Switch fruit type

Change the top-level field:

```json
"active_dataset": "avocado"
```

Allowed values in the current config:

| Value | Description |
| --- | --- |
| `"strawberry"` | Frame-based strawberry experiment (date folders) |
| `"avocado"` | Webcam-based avocado experiment (flat image files) |

Every script (`crop_images.py`, `segmentation.py`, `frame_differencing.py`, `assign_id.py`) reads this value at startup. You must change it before running the pipeline for a different fruit.

### 2.2 Crop size

Inside each dataset block, edit:

```json
"crop": {
    "width": 1777,
    "height": 1052
}
```

Both values are in pixels. The crop step performs a **center crop**:

- The script takes the middle region of each image.
- The source image must be **at least** as large as `width` x `height`.
- If the image is smaller, that file is skipped with an error message.

Change `width` and `height` freely to match your camera setup or region of interest.

### 2.3 Other config fields

| Field | Meaning |
| --- | --- |
| `processed_dir` | Root folder for all processed outputs (default: `data/02_processed`) |
| `input_dir` | Raw input folder for the active dataset |
| `output_dir` | Crop output folder for the active dataset |
| `image_type` | `"frames"` for strawberry, `"webcam"` for avocado |
| `mask_dir` | Used by some scripts; see path notes in Section 4 |
| `frame_diff` | Default thresholds for frame differencing |

Current config summary:

| Dataset | input_dir | output_dir | image_type |
| --- | --- | --- | --- |
| strawberry | `data/01_raw` | `data/02_processed/cropped_strawberry` | `frames` |
| avocado | `data/01_raw/data/output` | `data/02_processed/cropped_avocado` | `webcam` |

---

## 3. Pipeline Overview

```text
Raw images
    |
    v
crop_images.py          Center-crop to fixed width/height
    |
    v
frame_differencing.py   Detect motion between frames, validate masks, write CSV report
    |
    v
segmentation.py       Detect each fruit, remove background, save PNG + mask, if regenerate_mask = True (after run frame_differencing.py) => regenerate new mask
    |
    v
assign_id.py            Group segmented images into F01..F06 folders
```

### What each script does

| Script | Purpose |
| --- | --- |
| `crop_images.py` | Reads raw images, center-crops them, saves JPG files |
| `frame_differencing.py` | Compares consecutive frames, flags motion, validates segmentation quality |
| `segmentation.py` | Finds up to 6 fruits per frame, segments each one, assigns grid position 1-6, regenerate new mask if regenerate_mask = True |
| `assign_id.py` | Copies segmented PNGs into per-fruit folders (`F01`..`F06`) with standardized names |

---

## 4. Folder Structure

Understanding folder names is important. A wrong path is the most common cause of "No images found" errors.

### 4.1 Project layout

```text
Strawberry-RUL-prediction/
|-- data/
|   |-- 01_raw/                          Raw input data
|   |-- 02_processed/                    All preprocessing outputs
|-- src/
|   |-- stage3_preprocessing/
|       |-- config.json
|       |-- crop_images.py
|       |-- segmentation.py
|       |-- frame_differencing.py
|       |-- assign_id.py
|       |-- ...
|-- docs/
    |-- PREPROCESSING_GUIDE.md           This file
```

### 4.2 Strawberry paths

**Raw input** (two possible sources):

```text
data/01_raw/
    18-03-2026/
        frame-10_14-41-29.jpg
        frame-11_14-56-29.jpg
        ...
    19-03-2026/
	...
    21-03-2026/
	...
```

Date folders must match the pattern `DD-MM-YYYY` (example: `18-03-2026`).

The crop script can also read from existing frame folders under `data/02_processed/` named `frames_{DD-MM-YYYY}` if present.

**After crop** (written by `crop_images.py`):

```text
data/02_processed/cropped_strawberry/
    cropped_18-03-2026/
        frame-10_14-41-29.jpg
        ...
    cropped_19-03-2026/
	...
    cropped_21-03-2026/
	...
```

**Important path note for strawberry**

`segmentation.py` does **not** read from `cropped_strawberry/`. It scans `data/02_processed/` directly and looks for folders named:   (note: "cropped_strawberry" is just an example for a 																	new strawberry dataset.)

```text
cropped_{DD-MM-YYYY}
```

Example:

```text
data/02_processed/cropped_18-03-2026/
```

If you use the default config as-is, crop output goes to `cropped_strawberry/cropped_18-03-2026/`, but segmentation expects `cropped_18-03-2026/` directly under `02_processed`.

**Recommended fix for strawberry:** set `output_dir` to `data/02_processed` in `config.json`:

```json
"strawberry": {
    "input_dir": "data/01_raw",
    "output_dir": "data/02_processed",
    ...
}
```

Then crop output becomes `data/02_processed/cropped_18-03-2026/`, which matches what `segmentation.py` expects.

**After segmentation:**

```text
data/02_processed/
    segmented_18-03-2026/
        frame-12_15-11-29_strawberry_1.png
        frame-12_15-11-29_strawberry_2.png
        ...
    mask_18-03-2026/
        frame-12_15-11-29_strawberry_1_mask.png
        ...
```

**After assign_id:**

```text
data/02_processed/assigned_18-03-2026/
    F01/
        frame-12_15-11-29_F01.png
    F02/
    F03/
    F04/
    F05/
    F06/
```

**After frame differencing:**

```text
data/02_processed/frame_differencing_results_18-03-2026/
    frame_differencing_report_18-03-2026.csv
    motion_masks/
    motion_overlays/
```

### 4.3 Avocado paths

**Raw input:**

```text
data/01_raw/data/output/
    webcam_2026-06-14_20-30-44.jpg
    webcam_2026-06-14_20-45-44.jpg
    ...
```

File names must match:

```text
webcam_YYYY-MM-DD_HH-MM-SS.jpg
```

**After crop:**

```text
data/02_processed/cropped_avocado/
    webcam_2026-06-14_20-30-44.jpg
    webcam_2026-06-14_20-45-44.jpg
    ...
```

Crop keeps the same file names. Images are center-cropped in place into this folder.

**After segmentation:**

```text
data/02_processed/segmented_avocado/
    webcam_2026-06-14_20-30-44_avocado_1.png
    webcam_2026-06-14_20-30-44_avocado_2.png
    ...
    webcam_2026-06-14_20-30-44_avocado_6.png

data/02_processed/mask_avocado/
    webcam_2026-06-14_20-30-44_avocado_1_mask.png
    ...
```

Each source frame produces up to 6 segmented PNG files (one per fruit position).

**After assign_id:**

```text
data/02_processed/assigned_avocado/
    F01/
        webcam_2026-06-14_20-30-44_F01.png
    F02/
    ...
    F06/
```

**After frame differencing:**

```text
data/02_processed/frame_differencing_results_avocado/
    frame_differencing_report_avocado.csv
    motion_masks/
    motion_overlays/
```

**Important path note for avocado frame differencing**

Inside `frame_differencing.py`, the default mask folder is built from `mask_dir` in config (`"segmented"`), which resolves to:

```text
data/02_processed/segmented
```

That folder does not exist. The real segmented folder for avocado is:

```text
data/02_processed/segmented_avocado
```

When running frame differencing for avocado, pass the mask folder explicitly:

```bash
python src/stage3_preprocessing/frame_differencing.py --mask-dir data/02_processed/segmented_avocado
```

---

## 5. Step-by-Step Instructions

### Step 1: Crop images

```bash
python src/stage3_preprocessing/crop_images.py
```

**Input**

| Dataset | Source |
| --- | --- |
| Strawberry | `data/01_raw/{DD-MM-YYYY}/` or `data/02_processed/frames_{DD-MM-YYYY}/` |
| Avocado | `data/01_raw/data/output/` |

**Output**

| Dataset | Destination |
| --- | --- |
| Strawberry | `{output_dir}/cropped_{DD-MM-YYYY}/` |
| Avocado | `data/02_processed/cropped_avocado/` |

**Behavior**

- Performs center crop to `crop.width` x `crop.height`.
- Skips unreadable files or files smaller than the target size.
- Strawberry: processes one folder per date.
- Avocado: processes all matching webcam JPG files in the input folder.

---

### Step 2: Frame differencing

```bash
python src/stage3_preprocessing/frame_differencing.py
```

For avocado, also set the mask folder:

```bash
python src/stage3_preprocessing/frame_differencing.py 
```

**Purpose**

- Compares each frame with the previous frame (or last stable frame).
- Computes motion ratio and flags frames that likely need a new mask.
- Validates existing segmented PNGs (alpha size, fruit color, border contact).
- Writes a CSV report plus optional debug images.

**Input**

All folders under `data/02_processed/` whose names start with `cropped_`:

```text
cropped_18-03-2026/
cropped_19-03-2026/
cropped_avocado/
```

**Output** (one result folder per cropped input folder)

```text
frame_differencing_results_{date_or_name}/
    frame_differencing_report_{date_or_name}.csv
    motion_masks/
    motion_overlays/
```

**Key CSV columns**

| Column | Meaning |
| --- | --- |
| `motion_detected` | True if the frame changed significantly |
| `regenerate_mask` | True if a new segmentation mask is recommended |
| `mask_valid` | True if the existing segmented PNG passed validation |
| `mask_reason` | Reason when validation failed |

**Useful options**

```bash
python src/stage3_preprocessing/frame_differencing.py \
    --input-dir data/02_processed/cropped_avocado \
    --mask-dir data/02_processed/segmented_avocado \
    --motion-threshold 0.015 \
    --pixel-threshold 45 \
    --reference-strategy previous
```

Default thresholds can also be changed in `config.json` under `frame_diff`.

---

### Step 3: Segmentation

```bash
python src/stage3_preprocessing/segmentation.py
```

**Input**

| Dataset | Source folder |
| --- | --- |
| Strawberry | `data/02_processed/cropped_{DD-MM-YYYY}/` (all matching date folders) |
| Avocado | `data/02_processed/cropped_avocado/` |

**Output**

| Dataset | Segmented PNGs | Binary masks |
| --- | --- | --- |
| Strawberry | `segmented_{DD-MM-YYYY}/` | `mask_{DD-MM-YYYY}/` |
| Avocado | `segmented_avocado/` | `mask_avocado/` |

**Output file naming**

```text
{frame_name}_{fruit_type}_{position}.png
{frame_name}_{fruit_type}_{position}_mask.png
```

Examples:

```text
frame-12_15-11-29_strawberry_3.png
webcam_2026-06-14_20-30-44_avocado_1.png
```

Segmented PNGs are BGRA images (transparent background). Masks are full-frame binary images.

**Fruit position grid (2 rows x 3 columns)**

The script assigns position 1-6 from the fruit center `(cX, cY)`:

| Column | Left third | Middle third | Right third |
| --- | --- | --- | --- |
| Avocado bottom row | 1 | 2 | 3 |
| Avocado top row | 4 | 5 | 6 |
| Strawberry top row | 1 | 2 | 3 |
| Strawberry bottom row | 4 | 5 | 6 |

Avocado numbering matches a bottom-to-top layout:

```text
4   5   6     (top row)
1   2   3     (bottom row)
```

**Optional filters**

Process only some frames:

```bash 
# By frame number (strawberry frame-XX names)
python src/stage3_preprocessing/segmentation.py --start-frame 10 --end-frame 20
python src/stage3_preprocessing/segmentation.py --only-frame 3 4 5

# By filename stem (useful for avocado webcam names)
python src/stage3_preprocessing/segmentation.py --start-name "webcam_2026-06-14_20-30-44" --end-name "webcam_2026-06-14_21-00-44"

# Keep old outputs when reprocessing
python src/stage3_preprocessing/segmentation.py --keep-existing
```

---

### Step 4: Assign fruit IDs

```bash
python src/stage3_preprocessing/assign_id.py
```

**Input**

| Dataset | Source |
| --- | --- |
| Strawberry | `data/02_processed/segmented_{DD-MM-YYYY}/` |
| Avocado | `data/02_processed/segmented_avocado/` |

**Output**

| Dataset | Destination |
| --- | --- |
| Strawberry | `data/02_processed/assigned_{DD-MM-YYYY}/` |
| Avocado | `data/02_processed/assigned_avocado/` |

**Naming change**

Input:

```text
webcam_2026-06-14_20-30-44_avocado_3.png
```

Output:

```text
assigned_avocado/F03/webcam_2026-06-14_20-30-44_F03.png
```

Each physical fruit position (1-6) becomes a fixed folder `F01`..`F06`. All time points for the same grid position are collected together.

---

## 6. Full Pipeline Script

`main_preprocessing.py` runs all steps in sequence:

1. `extracting_frames.py` - extract frames from video (if used)
2. `crop_images.py`
3. `segmentation.py`
4. `frame_differencing.py`
5. `assign_id.py`
6. `eol.py`
7. `manifests.py`
8. `label_rul.py`
9. `split_data.py`

Run:

```bash
python src/stage3_preprocessing/main_preprocessing.py
```

Make sure `active_dataset` in `config.json` is correct before running. The full pipeline is intended for strawberry workflows that start from video. For avocado webcam images, running Steps 1-4 manually is usually enough.

---

## 7. Requirements

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Preprocessing scripts require at least:

- Python 3
- OpenCV (`opencv-python`)
- NumPy

---

## 8. Common Problems

### "No images found" in crop step

- Check `active_dataset` in `config.json`.
- Avocado: confirm files exist in `data/01_raw/data/output/` and match `webcam_YYYY-MM-DD_HH-MM-SS.jpg`.
- Strawberry: confirm date folders exist in `data/01_raw/` with format `DD-MM-YYYY`.

### "No cropped folders found" in segmentation (strawberry)

- Segmentation looks for `data/02_processed/cropped_{DD-MM-YYYY}/`.
- If crop wrote to `cropped_strawberry/cropped_{date}/`, either move the folders or set strawberry `output_dir` to `data/02_processed`.

### "Cannot find any matching images" in segmentation (avocado)

- Confirm cropped files exist in `data/02_processed/cropped_avocado/`.
- Check filename filters (`--start-name`, `--end-name`, etc.).

### Crop skipped: image smaller than target size

- Reduce `crop.width` and `crop.height` in `config.json`, or use higher-resolution source images.

### Frame differencing mask validation always empty (avocado)

- Pass `--mask-dir data/02_processed/segmented_avocado`.
- Run `segmentation.py` before frame differencing.

### Wrong fruit ID in output filename

- The grid index comes from fruit center position in the cropped image.
- If the camera or tray moved, adjust crop region or verify the 2x3 layout still matches the expected positions.

---

## 9. Checklist Before Running

1. Set `active_dataset` to `"strawberry"` or `"avocado"`.
2. Set `crop.width` and `crop.height` for your camera view.
3. Put raw files in the correct input folder.
4. For strawberry, confirm `output_dir` aligns with what `segmentation.py` expects.
5. Run scripts in order: crop, segmentation, frame differencing, assign_id.
6. For avocado frame differencing, pass `--mask-dir data/02_processed/segmented_avocado`.
7. Check outputs under `data/02_processed/` before moving to labeling or model training.

---

## 10. Related Documentation

- `docs/PREPROCESSING_SPEC.md` - higher-level preprocessing design and QC rules
- `docs/DATA_PROTOCOL.md` - dataset organization and naming conventions
- `docs/LABELING_PROTOCOL.md` - steps after preprocessing
