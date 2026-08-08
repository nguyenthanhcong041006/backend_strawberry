# macOS Setup Guide

This guide is for running the current strawberry-only RUL workflow on macOS. Avocado files remain in the repository for future work, but the commands below target strawberry only.

## 1. Setup

```bash
cd ~/Desktop/Strawberry-RUL-prediction
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch is missing or incompatible:

```bash
pip install torch torchvision
```

## 2. Real Sensor Data

Strawberry preprocessing requires a real sensor CSV. Expected columns:

```text
timestamp,temperature_c,humidity_pct
```

Accepted aliases:

```text
temperature -> temperature_c
humidity_rh -> humidity_pct
```

Run preprocessing with:

```bash
python3 src/strawberry/stage3_preprocessing/main_preprocessing.py \
  --sensor-csv data/01_raw/strawberry/sensor_readings.csv
```

If the file is missing, preprocessing stops instead of generating fake temperature/humidity.

## 3. Lab Sweep

Run the small LOOCV sequence/fusion sweep:

```bash
python3 scripts/run_strawberry_lab.py
```

Outputs:

```text
notebooks/strawberry/results/lab_seq_fusion_search.csv
notebooks/strawberry/results/lab_seq_fusion_search.json
```

Current selected setting:

```text
seq_len = 8
fusion_mode = image_only
temporal_pooling = last_mean_max
```

## 4. Training

The A/B/C/D ideas are preserved in one shared training package:

| Model | Backbone | Attention | Temporal |
| --- | --- | --- | --- |
| A | EfficientNet-B0 | CBAM | GRU |
| B | MobileNetV2 | CBAM | LSTM |
| C | EfficientNet-B0 | CBAM | LSTM |
| D | MobileNetV2 | CBAM | GRU |

Train all models with LOOCV:

```bash
PYTHONPATH=src python3 -m strawberry.training.cli train \
  --config configs/strawberry_training.json
```

Train one model for a quick check:

```bash
PYTHONPATH=src python3 -m strawberry.training.cli train \
  --config configs/strawberry_training.json \
  --models D
```

Outputs:

```text
models/strawberry/runs/<run_name>/<model_key>/holdout_Fxx/best_model.pt
output/runs/strawberry/<run_name>/fold_results.csv
output/runs/strawberry/<run_name>/model_summary.csv
output/runs/strawberry/<run_name>/sequence_index.csv
```

Summarize a completed run:

```bash
PYTHONPATH=src python3 -m strawberry.training.cli evaluate \
  --run-root output/runs/strawberry/strawberry_loocv_seq3_late_env_branch
```

## 5. Backend API

Place the chosen production checkpoint at:

```text
models/strawberry/production/model_D/best_model.pt
```

Run the API:

```bash
PYTHONPATH=src python3 -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Predict with real sensor values:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "file=@data/03_split/strawberry/test/images/F06/<image>.png" \
  -F "temperature_c=<real_sensor_temperature_c>" \
  -F "humidity_pct=<real_sensor_humidity_pct>"
```

Predict without realtime sensor values:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "file=@data/03_split/strawberry/test/images/F06/<image>.png"
```

When sensor fields are omitted, the predictor sends missing-sensor flags to the model. It does not invent environment values.

## 6. Troubleshooting

### `python: command not found`

Use `python3` on macOS:

```bash
python3 --version
```

### PyTorch MPS memory issues

Use a smaller model subset first:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src python3 -m strawberry.training.cli train \
  --config configs/strawberry_training.json \
  --models D
```

### `ModuleNotFoundError`

Run commands from the repository root and set `PYTHONPATH=src`.

### Backend cannot find checkpoint

Check:

```bash
ls models/strawberry/production/model_D/best_model.pt
```

## 7. References

| Document | Path |
| --- | --- |
| README | `README.md` |
| Data Protocol | `docs/DATA_PROTOCOL.md` |
| Preprocessing Guide | `docs/PREPROCESSING_GUIDE.md` |
| Labeling Protocol | `docs/LABELING_PROTOCOL.md` |
| Strawberry Training Config | `configs/strawberry_training.json` |
| Strawberry Training Code | `src/strawberry/training/` |
| Strawberry Lab | `notebooks/strawberry/` |
