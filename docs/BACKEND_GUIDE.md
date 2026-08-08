# Backend Guide: Strawberry RUL Prediction API

This guide explains how to run the FastAPI backend for the current strawberry-only MVP.

The backend loads the new strawberry training checkpoint format:

```text
models/strawberry/production/model_D/best_model.pt
```

This checkpoint should be copied from a completed LOOCV run, for example:

```text
models/strawberry/runs/strawberry_loocv_seq3_late_env_branch/D/holdout_Fxx/best_model.pt
```

## 1. Important Files

```text
src/app.py                                      FastAPI entry point
src/strawberry/api/routes.py                   API routes
src/strawberry/config_app/config.json          Backend config
src/strawberry/services/preprocessing.py       Image preprocessing/segmentation
src/strawberry/services/predictor.py           New checkpoint loader and RUL predictor
src/strawberry/services/postprocess.py         Response formatting
src/strawberry/schemas/response.py             Standard JSON responses
models/strawberry/production/model_D/best_model.pt
```

## 2. Install Dependencies

Use the project virtual environment if available:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The backend needs:

```text
fastapi
uvicorn
python-multipart
torch
torchvision
opencv-python
```

`python-multipart` is required because the frontend uploads images with `multipart/form-data`.

## 3. Check the Model File

Before running the API, confirm the production checkpoint exists:

```powershell
Test-Path models\strawberry\production\model_D\best_model.pt
```

Expected result:

```text
True
```

If it is missing, train the models first and copy the chosen fold/checkpoint into the production path.

## 4. Run the Backend

From the project root:

```powershell
.\.venv\Scripts\python -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8000
```

For phone testing on the same network:

```powershell
.\.venv\Scripts\python -m uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000
```

## 5. Health Check

```powershell
curl.exe http://127.0.0.1:8000/api/health
```

Expected:

```json
{"success": true, "message": "API is running"}
```

## 6. Predict Endpoint

Endpoint:

```text
POST /api/predict
```

Required multipart form field:

```text
file
```

Optional real sensor fields:

```text
temperature_c
humidity_pct
```

Example with image only:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/predict `
  -F "file=@data\03_split\strawberry\test\images\F06\<image>.png"
```

Example with real sensor values from the capture:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/predict `
  -F "file=@data\03_split\strawberry\test\images\F06\<image>.png" `
  -F "temperature_c=<real_sensor_temperature_c>" `
  -F "humidity_pct=<real_sensor_humidity_pct>"
```

If `temperature_c` and `humidity_pct` are omitted, the predictor sends missing-sensor flags to the model instead of fabricating environment values.

## 7. Response Shape

Successful response:

```json
{
  "success": true,
  "remaining_useful_life": 123.45,
  "confidence": 0.85
}
```

Invalid image response:

```json
{
  "success": false,
  "message": "Invalid image"
}
```

## 8. Common Problems

### `Model checkpoint not found`

Check:

```powershell
Test-Path models\strawberry\production\model_D\best_model.pt
```

Also verify:

```text
src/strawberry/config_app/config.json
```

### `ModuleNotFoundError`

Run the backend from the repository root with `--app-dir src`, not from inside `src/strawberry`.

### The frontend gets `Invalid image`

Common reasons:

- The uploaded file is not an image.
- The form-data field name is not `file`.
- The image does not contain a visible strawberry.
- The segmentation step cannot isolate the strawberry clearly.

## 9. Minimal Checklist

```text
[ ] Python dependencies are installed
[ ] models\strawberry\production\model_D\best_model.pt exists
[ ] Backend starts on http://127.0.0.1:8000
[ ] /api/health returns success true
[ ] Frontend sends multipart/form-data field file
[ ] Frontend sends temperature_c/humidity_pct only when values come from a real sensor
```
