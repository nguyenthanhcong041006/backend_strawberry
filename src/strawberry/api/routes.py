from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile

from schemas import response
from services.postprocess import format_result
from services.preprocessing import preprocess
from utils_app.image_utils import read_image, validate_image


router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"success": True, "message": "API is running"}


@router.get("/predict")
async def predict_help() -> dict:
    return {
        "success": True,
        "message": "Use POST /api/predict with multipart/form-data field 'file'. Optional real sensor fields: temperature_c, humidity_pct.",
    }


@router.post("/predict")
async def predict_strawberry_rul(
    request: Request,
    file: UploadFile = File(...),
    temperature_c: float | None = Form(default=None),
    humidity_pct: float | None = Form(default=None),
) -> dict:
    content = await file.read()
    if not validate_image(content, file.content_type):
        return response.error("Invalid image")

    image = read_image(content)
    if image is None:
        return response.error("Invalid image")

    config = request.app.state.config
    preprocess_result = preprocess(image, config)
    if preprocess_result is None:
        return response.error("Invalid image")

    predictor = request.app.state.predictor
    raw_rul, model_confidence = predictor.predict(
        preprocess_result.roi,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
    )
    confidence = min(model_confidence, preprocess_result.confidence)
    result = format_result(raw_rul, confidence)
    if result["confidence"] < float(config.get("confidence_threshold", 0.5)):
        return response.error("Invalid image")

    return response.success(
        remaining_useful_life=result["remaining_useful_life"],
        confidence=result["confidence"],
    )

