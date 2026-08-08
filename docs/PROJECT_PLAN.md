# Project Plan

This plan coordinates the shared early pipeline for fruit RUL prediction. It focuses on data acquisition, preprocessing, labeling, and EDA. Model development is intentionally kept as a placeholder until the data foundation is stable.

## Operating Principles

- One source of truth lives in GitHub through Markdown docs and tracked reports.
- Raw data remains unchanged.
- Every derived file must be traceable.
- Fruit IDs are physical entities, not temporary filenames.
- Split by fruit ID before model evaluation.
- Review twice per week and record decisions.
- Prefer small reviewed outputs over large unverified batches.

## Phase 1: Data Acquisition

Owner: Hung  
Reviewers: Gate checker, Hai for metadata completeness

Tasks:

- Confirm experiment metadata template.
- Confirm 3x2 ROI-to-fruit mapping before recording.
- Capture images/video every 15 minutes across the full recording period.
- Record temperature and humidity from the box/room sensor.
- For avocado, measure firmness once per day per fruit using five points and average the values.
- Record capture interruptions and measurement periods.

Acceptance:

- raw captures exist and are readable;
- sensor logs cover the recording period;
- metadata identifies experiment, fruit type, ROI layout, and recording window;
- avocado firmness records map by fruit and day;
- known capture failures are documented.

## Phase 2: Preprocessing

Owner: Cong  
Reviewer: Gate checker, with Hung for technical integration

Tasks:

- Extract frames or collect timestamped images.
- Crop/locate the 3x2 box region.
- Segment fruit regions and generate masks where needed.
- Assign fruit images to stable fruit IDs.
- Run frame differencing to detect hands/devices and unstable frames.
- Create processed manifests and exclusion logs.
- Map environmental and optional firmness values.

Acceptance:

- fruit-ID folders exist for all six ROI positions;
- model-ready images have manifest rows;
- invalid frames are excluded from model-ready paths and logged;
- numeric fields are mapped or explicitly missing;
- preprocessing summary reports counts and failures.

## Phase 2.5: Labeling

Owner: Hung proposes, Hai reviews, gate checker approves

Tasks:

- Inspect the final 1-2 days before recording ends.
- Propose per-fruit EOL anchors.
- Review visual and firmness evidence for avocado.
- Approve final EOL anchors.
- Generate `labels.csv` with hourly RUL values.

Acceptance:

- each fruit has one approved EOL timestamp;
- each valid frame has one label row;
- RUL is measured in hours;
- label status is approved before final training;
- post-EOL handling is documented.

## Parallel Track: EDA

Owner: Hai  
Reviewers: Gate checker, Hung and Cong as needed

Tasks:

- Build dataset inventory.
- Summarize raw, processed, excluded, and labeled counts.
- Analyze temporal visual and numeric trends.
- Analyze spatial/ROI variation.
- Prepare paper-ready graphs and summaries.

Acceptance:

- EDA outputs are stored in `output/graphs/eda/` and `output/reports/eda/`;
- dataset limitations are documented;
- legacy and current datasets are separated clearly;
- figures can be regenerated or traced to scripts/notebooks.

## Phase 3: Model Development Placeholder

Owner: Hung, with support as needed

Strawberry:

- prototype CNN/GRU/LSTM/EfficientNet-style models;
- use legacy strawberry data only for pipeline testing;
- do not report legacy results as final evidence for the new dataset.

Avocado:

- placeholder for attention-based multimodal approaches;
- possible ViT/Mamba/MBT-style fusion using visual sequences, environment, and firmness;
- begin only after early-stage data rules are stable.

## Phase 4: Evaluation and XAI Placeholder

Final evaluation must:

- use fruit-level LOOCV;
- report regression metrics such as MAE, RMSE, MAPE with care near zero, and R2;
- preserve held-out fruit isolation;
- produce explainability outputs only after labels and splits are approved.

## Review Rhythm

Twice per week:

1. Each member reports completed work with links to files.
2. Blockers are recorded in `docs/PROGRESS_TRACKER.md`.
3. Gate checker approves or requests changes.
4. New tasks are assigned with expected evidence.

Recommended GitHub practice:

- Keep `docs/PROGRESS_TRACKER.md` as the human-readable project board.
- Open GitHub Issues for tasks that need discussion, code changes, or review history.
- Link issue numbers in the tracker when used.
