# Labeling Protocol

Labeling defines the end-of-life (EOL) timestamp for each fruit and converts every valid observation into an hourly RUL target.

## Core Formula

```text
RUL_hours = EOL_timestamp_for_that_fruit - sample_timestamp
```

RUL is measured in hours, not days.

Each fruit has its own EOL timestamp. Do not use one global EOL for all fruits unless the experiment explicitly proves that all fruits reached EOL at the same time.

## EOL Basis

Avocado EOL:

- the current MVP defines EOL as the first externally visible mold/spoilage onset in fruit-level segmented images;
- daily firmness is used as supporting validation, especially when it has collapsed to zero or near zero;
- this endpoint represents visible shelf-life/spoilage failure, not a guarantee of internal eating-quality or food-safety state;
- the chosen timestamp is reviewed against the surrounding time-lapse frames and recorded with confidence and notes.

Strawberry EOL:

- visual EOL anchor is used unless a numeric quality proxy is introduced later;
- firmness must remain missing/optional rather than imputed.

Every label row should include an `eol_basis` value such as:

```text
visual
firmness_visual
manual_review
visible_mold_onset
visible_spoilage_onset
firmness_zero_visual_confirmed
```

## Approval Flow

1. Hung proposes the EOL anchor for each fruit after recording ends.
2. Hai reviews the proposed anchor for paper consistency and dataset evidence.
3. Gate checker approves the final anchor before labels are generated.

Do not train final models on labels that have not passed this flow.

## Label Manifest

Recommended `labels.csv` fields:

```text
experiment_id
fruit_type
fruit_id
roi_id
image_path
raw_path
timestamp
eol_timestamp
rul_hours
temperature_c
humidity_pct
firmness_avg
firmness_available
valid_frame
exclude_reason
eol_basis
label_status
```

Recommended `eol_anchors.csv` fields:

```text
experiment_id,fruit_id,eol_timestamp,eol_basis,proposed_by,reviewed_by,approved_by,status,notes
```

For the avocado MVP, the annotation tool writes `data/02_processed/avocado/labels/eol_annotations.csv` with:

```text
fruit_id
eol_timestamp
eol_image_path
eol_basis
eol_confidence
visual_note
firmness_latest
firmness_date
first_firmness_le5_date
first_firmness_zero_date
label_status
updated_at
```

## Post-EOL Handling

If valid frames exist after the selected EOL timestamp, they must be handled consistently:

- set `rul_hours = 0` for post-EOL frames if they are kept for EOL-buffer analysis; or
- exclude them from model-ready labels and log the reason.

The chosen rule must be recorded in the labeling report.

This is a pending team decision before final training: either keep post-EOL frames as a zero-RUL buffer or exclude them from model-ready labels.

## Label Quality Checks

Before accepting labels:

- no valid row has negative RUL unless explicitly kept as post-EOL buffer and clamped to zero;
- each fruit has exactly one approved EOL anchor;
- every label row maps to a valid processed image;
- every valid image maps to exactly one label row;
- `firmness_avg` is present for avocado labels when available;
- strawberry firmness fields are empty/`NA`;
- RUL decreases over time for each fruit;
- splits are generated after labels and still preserve fruit-ID isolation.

## Leakage Warnings

Labeling can accidentally leak information when:

- frames from one fruit are split across train and test;
- EOL decisions are adjusted after seeing model performance;
- augmented versions of a test fruit are used in training;
- preprocessing thresholds are tuned using held-out fruit performance without recording the decision.

Final evaluation must use approved labels and fruit-level splits.
