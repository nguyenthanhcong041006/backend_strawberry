# Preprocessing Specification

Stage 2 converts raw recordings into fruit-ID-separated image sequences and traceable reports. The goal is not only to produce clean images, but also to preserve data integrity for labeling, EDA, and leakage-safe model development.

## Inputs

Expected inputs from Stage 1:

- Raw videos or image captures.
- Sensor logs with timestamped temperature and humidity.
- Optional daily firmness records for avocado.
- Experiment metadata describing fruit type, ROI layout, recording window, and camera setup.

## Outputs

Expected Stage 2 outputs:

```text
data/02_processed/<experiment_id>/
  frames/
  cropped/
  masks/
  assigned/
    F01/
    F02/
    F03/
    F04/
    F05/
    F06/
  manifests/
    frame_manifest.csv
    numeric_mapping.csv
    excluded_frames.csv
    preprocessing_summary.json

output/reports/processed/
output/graphs/processed/
```

Existing sample folders may use names such as `assigned_18-03-2026/strawberry_1`. New or refactored outputs should prefer experiment IDs and `F01`-style fruit IDs.

## Workflow

1. Extract frames from raw video or collect timestamped captures.
2. Verify the frame is readable and not blank/black.
3. Crop or locate the 3x2 box area.
4. Apply fixed ROI mapping so each physical fruit keeps one `fruit_id`.
5. Segment or mask the fruit region.
6. Use frame differencing and mask validation to flag disturbances.
7. Map environmental values by timestamp.
8. Map avocado firmness by fruit-day when available.
9. Write manifests and reports.
10. Prepare label inputs but do not finalize EOL labels without review.

## Current Script Mapping

| Script | Current Purpose | Standardization Need |
| --- | --- | --- |
| `src/stage3_preprocessing/extracting_frames.py` | Extracts frames at a fixed interval | Refactor hard-coded paths into CLI args |
| `src/stage3_preprocessing/crop_images.py` | Center-crops raw frames | Refactor hard-coded paths and crop dimensions into config |
| `src/stage3_preprocessing/segmentation.py` | Segments strawberry candidates into PNGs | Already has CLI args; add experiment-aware output naming |
| `src/stage3_preprocessing/frame_differencing.py` | Flags motion/unstable frames and suspicious masks | Already has CLI args; use as QC report generator |
| `src/stage3_preprocessing/assign_id.py` | Groups images by detected strawberry ID | Refactor hard-coded paths; align with ROI-based `F01` IDs |
| `src/stage3_preprocessing/label_rul.py` | Creates prototype RUL labels | Replace single global EOL with fruit-specific EOL anchors |
| `src/stage3_preprocessing/split_data.py` | Prototype 4/1/1 fruit split | Add final LOOCV split generation |

## Frame Quality Checks

A frame must fail QC if it is:

- unreadable by OpenCV/Pillow;
- mostly black or blank;
- missing from the manifest;
- duplicated unexpectedly;
- blocked by a hand, probe, measuring device, or other interruption;
- too motion-heavy for a stable fruit observation;
- missing corresponding sensor data;
- missing fruit ID after ROI assignment.

Frame differencing is useful for finding daily measurement interruptions. It should generate both a machine-readable report and visual overlays for manual review.

## Mask Quality Checks

Mask reports should include:

- mask path;
- alpha/foreground ratio;
- fruit-color support ratio where applicable;
- border-touch ratio;
- largest component ratio;
- motion overlap if coordinate systems match;
- pass/fail result;
- reason code.

Do not silently accept failed masks. Either regenerate them or exclude the affected frame from the model-ready manifest.

## Fruit-ID Assignment

Fruit ID is assigned by fixed ROI position, not by frame-level visual detection alone. Segmentation may find candidate fruit regions, but the final ID must preserve the physical fruit's original 3x2 position.

This prevents leakage and keeps sequence labels biologically meaningful.

## Numeric Mapping

Each processed sample must carry:

```text
experiment_id,fruit_type,fruit_id,timestamp,temperature_c,humidity_pct
```

For avocado, also include:

```text
firmness_avg,firmness_n,firmness_date
```

For strawberry, leave firmness fields empty/`NA`.

## Split Policy

Final evaluation must be fruit-ID safe.

Allowed:

- LOOCV by fruit ID: train on all but one fruit, test on the held-out fruit.
- Prototype 4/1/1 fruit-level split for early code testing.

Not allowed:

- random frame-level split;
- placing frames from the same fruit in both train and test;
- using augmented frames from a held-out fruit during training;
- tuning preprocessing thresholds on the held-out fruit without recording it as a validation decision.

## Acceptance Criteria

Stage 2 is complete only when:

- output folders are separated by fruit ID;
- every model-ready image has a manifest row;
- every manifest row has timestamp and numeric mapping;
- excluded frames are logged;
- suspicious masks are reviewed or regenerated;
- counts are consistent across raw, processed, excluded, and accepted files;
- a reviewer can trace any processed image back to raw data.

