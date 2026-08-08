# Fruit RUL Prediction

This repository predicts fruit Remaining Useful Life (RUL) from time-ordered images plus timestamp-aligned sensor data.

Current active scope: strawberry only. Avocado code/data remain in the repo for future work and should not be changed while the current strawberry training pipeline is being rebuilt.

## Current Strawberry Direction

The strawberry pipeline now trains sequence-based RUL models with fruit-level leakage control:

- Model-ready samples are built as ordered image sequences, not isolated random frames.
- Temperature and humidity are mapped from real sensor logs by timestamp.
- Sensor values are never fabricated. If a real sensor reading cannot be matched, the sample keeps missing-sensor flags.
- Final validation uses Leave-One-Fruit-Out Cross-Validation (LOOCV).
- A small lab sweep is used to choose sequence length and fusion mode before training the large A/B/C/D models.

The current selected config is based on the lab sweep in `notebooks/strawberry/results/`:

```text
seq_len = 8
fusion_mode = image_only
temporal_pooling = last_mean_max
```

Note: the current split metadata is treated as trusted real sensor data. The latest small-model LOOCV sweep found image-only slightly ahead of early temp/humidity fusion, so the default config prioritizes the visual sequence signal.

## Model Ideas Kept

The old training codebase was removed, but the original A/B/C/D architecture ideas are preserved:

| Model | Visual backbone | Attention | Temporal layer |
| --- | --- | --- | --- |
| A | EfficientNet-B0 | CBAM | GRU |
| B | MobileNetV2 | CBAM | LSTM |
| C | EfficientNet-B0 | CBAM | LSTM |
| D | MobileNetV2 | CBAM | GRU |

All four variants now share one clean implementation under `src/strawberry/training/`.

## Repository Structure

```text
configs/
  strawberry_training.json          Default strawberry LOOCV training config

data/
  01_raw/
    strawberry/                     Raw strawberry videos/images and real sensor logs
    avocado/                        Avocado raw data, kept untouched
  02_processed/
    strawberry/final/               Final labels, metadata, sensor mapping report
  03_split/
    strawberry/                     Fruit-ID-safe split artifacts

docs/
  DATA_PROTOCOL.md                  Data schema and sensor mapping rules
  PREPROCESSING_SPEC.md             Preprocessing expectations
  PREPROCESSING_GUIDE.md            Preprocessing operator notes
  LABELING_PROTOCOL.md              EOL/RUL labeling rules
  PROJECT_PLAN.md                   Team plan
  PROGRESS_TRACKER.md               Milestone tracker

models/
  strawberry/runs/                  New LOOCV checkpoints
  strawberry/production/            Optional production checkpoint location
  avocado/                          Avocado model area, kept untouched

notebooks/strawberry/
  00_lab_overview.ipynb
  01_data_audit.ipynb
  02_ml_baseline_fusion_search.ipynb
  03_deep_fusion_ablation_plan.ipynb
  lab_utils.py                      Notebook helper shim over the new training package
  results/lab_seq_fusion_search.*   Small-model sequence/fusion search results

scripts/
  run_strawberry_lab.py             Run the small model lab sweep
  run_strawberry_training.py        Train A/B/C/D with LOOCV

src/
  app.py                            FastAPI entry point
  avocado/                          Avocado-specific code, not modified for strawberry-only work
  strawberry/
    stage3_preprocessing/           Strawberry preprocessing and real sensor mapping
    training/                       New strawberry training/evaluation package
```

## Real Sensor Mapping

Preprocessing now requires a real strawberry sensor CSV. Expected columns:

```text
timestamp,temperature_c,humidity_pct
```

Accepted aliases:

- `temperature` -> `temperature_c`
- `humidity_rh` -> `humidity_pct`

Run preprocessing with:

```powershell
.\.venv\Scripts\python src\strawberry\stage3_preprocessing\main_preprocessing.py --sensor-csv data\01_raw\strawberry\sensor_readings.csv
```

or set `sensor_csv` in `src/strawberry/stage3_preprocessing/config.json`.

The mapper writes:

```text
data/02_processed/strawberry/final/sensor_mapping_report.csv
```

If no real sensor CSV is available, preprocessing stops with a clear error instead of inventing environment values.

## Training

Run the small lab sweep:

```powershell
.\.venv\Scripts\python scripts\run_strawberry_lab.py
```

Train all four strawberry models with LOOCV:

```powershell
.\.venv\Scripts\python scripts\run_strawberry_training.py
```

Equivalent module command:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python -m strawberry.training.cli train --config configs\strawberry_training.json
```

Train a subset:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python -m strawberry.training.cli train --config configs\strawberry_training.json --models C D
```

Summarize a completed run:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python -m strawberry.training.cli evaluate --run-root output\runs\strawberry\strawberry_loocv_seq3_late_env_branch
```

Outputs:

```text
models/strawberry/runs/<run_name>/<model_key>/holdout_Fxx/best_model.pt
output/runs/strawberry/<run_name>/fold_results.csv
output/runs/strawberry/<run_name>/model_summary.csv
output/runs/strawberry/<run_name>/sequence_index.csv
```

## Backend

The API loads the new checkpoint format from:

```text
models/strawberry/production/model_D/best_model.pt
```

Run:

```powershell
.\.venv\Scripts\python -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8000
```

`POST /api/predict` accepts the image field `file`. If the frontend has real sensor values for the current capture, include optional form fields:

```text
temperature_c
humidity_pct
```

If these are omitted, the model receives missing-sensor flags rather than fabricated environment readings.

## Minimum Definition of Done

A strawberry training run is acceptable only when:

- Metadata has fruit ID, timestamp, image path, EOL timestamp, and RUL hours.
- Temperature/humidity come from real sensor mapping or are explicitly marked missing.
- Sequences do not cross fruit IDs or long capture gaps.
- Train/validation/test separation is by fruit ID.
- LOOCV fold metrics and predictions are saved.
- Avocado files remain untouched unless avocado work is explicitly requested.
