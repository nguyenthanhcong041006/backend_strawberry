import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIGS = json.load(f)

ACTIVE_DATASET = CONFIGS["active_dataset"]
DATASET_CFG = CONFIGS["datasets"][ACTIVE_DATASET]
PROCESSED_DIR = PROJECT_ROOT / DATASET_CFG["output_dir"]
FINAL_DIR = PROCESSED_DIR / "final"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEMP_RANGE_C = (0.0, 60.0)
HUMIDITY_RANGE_PCT = (0.0, 100.0)

SENSOR_CASING_HSV_LOWER = np.array([35, 30, 20])
SENSOR_CASING_HSV_UPPER = np.array([95, 255, 160])
SENSOR_CASING_MIN_AREA = 800
SENSOR_CASING_ASPECT_RANGE = (0.8, 3.5)
SENSOR_CASING_PADDING_RATIO = 0.15


@dataclass(frozen=True)
class SensorReading:
    frame_path: Path
    timestamp: datetime
    temperature_c: Optional[float]
    humidity_pct: Optional[float]
    confidence: float
    raw_text: str


def _load_paddle_ocr():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install PaddleOCR and PaddlePaddle before "
            "running sensor OCR, for example: pip install paddlepaddle paddleocr"
        ) from exc

    try:
        return PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
    except TypeError:
        return PaddleOCR(use_angle_cls=True, lang="en")


def parse_frame_timestamp(frame_path: Path) -> datetime:
    date_match = re.search(r"frames_(\d{2}-\d{2}-\d{4})", str(frame_path.parent))
    time_match = re.search(r"frame-\d+_(\d{2})-(\d{2})-(\d{2})", frame_path.stem)
    if not date_match or not time_match:
        raise ValueError(f"Cannot parse timestamp from frame path: {frame_path}")

    hh, mm, ss = time_match.groups()
    return datetime.strptime(f"{date_match.group(1)} {hh}:{mm}:{ss}", "%d-%m-%Y %H:%M:%S")


def iter_frame_paths(processed_dir: Path = PROCESSED_DIR) -> Iterable[Path]:
    for frame_dir in sorted(processed_dir.glob("frames_*")):
        if not frame_dir.is_dir():
            continue
        for image_path in sorted(frame_dir.iterdir(), key=_natural_sort_key):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield image_path


def _natural_sort_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def locate_sensor_casing(image: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SENSOR_CASING_HSV_LOWER, SENSOR_CASING_HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[tuple[float, int, int, int, int]] = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < SENSOR_CASING_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h if h else 0
        if not (SENSOR_CASING_ASPECT_RANGE[0] < aspect < SENSOR_CASING_ASPECT_RANGE[1]):
            continue
        if best is None or area > best[0]:
            best = (area, x, y, w, h)

    if best is None:
        return None

    _, x, y, w, h = best
    pad_x = int(w * SENSOR_CASING_PADDING_RATIO)
    pad_y = int(h * SENSOR_CASING_PADDING_RATIO)
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return x1, y1, x2, y2


def build_candidate_rois(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    h, w = image.shape[:2]
    rois: list[tuple[str, np.ndarray]] = []

    casing_box = locate_sensor_casing(image)
    if casing_box is not None:
        x1, y1, x2, y2 = casing_box
        casing_crop = image[y1:y2, x1:x2]
        if casing_crop.size:
            rois.append(("sensor_lcd_auto", casing_crop))

    specs = [
        ("full", 0.00, 0.00, 1.00, 1.00),
        ("top_left", 0.00, 0.00, 0.40, 0.35),
        ("top_right", 0.60, 0.00, 1.00, 0.35),
        ("bottom_left", 0.00, 0.60, 0.45, 1.00),
        ("bottom_right", 0.55, 0.60, 1.00, 1.00),
        ("left_band", 0.00, 0.15, 0.35, 0.85),
        ("right_band", 0.65, 0.15, 1.00, 0.85),
        ("center", 0.25, 0.20, 0.75, 0.80),
    ]

    for name, x1r, y1r, x2r, y2r in specs:
        x1, y1 = int(w * x1r), int(h * y1r)
        x2, y2 = int(w * x2r), int(h * y2r)
        crop = image[y1:y2, x1:x2]
        if crop.size:
            rois.append((name, crop))
    return rois


def preprocess_for_ocr(crop: np.ndarray, aggressive: bool = False) -> list[np.ndarray]:

    scaled = _resize_for_ocr(crop, aggressive=aggressive)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    return [
        scaled,
        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
    ]


def _resize_for_ocr(
    image: np.ndarray,
    min_side: int = 640,
    max_side: int = 1600,
    aggressive: bool = False,
) -> np.ndarray:

    if aggressive:
        min_side = max(min_side, 900)
        max_side = max(max_side, 2400)

    h, w = image.shape[:2]
    short_side = max(1, min(h, w))
    long_side = max(h, w)
    scale = max(1.0, min_side / short_side)
    if long_side * scale > max_side:
        scale = max_side / long_side
    if abs(scale - 1.0) < 0.01:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def run_paddle_ocr(ocr, image: np.ndarray) -> list[tuple[str, float]]:
    result = ocr.ocr(image, cls=True)
    return _flatten_paddle_result(result)


def _flatten_paddle_result(result) -> list[tuple[str, float]]:
    texts: list[tuple[str, float]] = []

    def walk(node):
        if node is None:
            return
        if isinstance(node, dict):
            if "rec_texts" in node:
                scores = node.get("rec_scores", [0.0] * len(node["rec_texts"]))
                for text, score in zip(node["rec_texts"], scores):
                    texts.append((str(text), float(score or 0.0)))
                return
            for value in node.values():
                walk(value)
            return
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[1], (list, tuple)) and len(node[1]) >= 2:
                text, score = node[1][0], node[1][1]
                if isinstance(text, str):
                    texts.append((text, float(score or 0.0)))
                    return
            for item in node:
                walk(item)

    walk(result)
    return texts


def extract_sensor_values(text_items: Iterable[tuple[str, float]]) -> tuple[Optional[float], Optional[float], float, str]:
    items = [(clean_ocr_text(text), float(score or 0.0)) for text, score in text_items if str(text).strip()]
    raw_text = " | ".join(text for text, _ in items)
    joined = " ".join(text for text, _ in items)
    normalized = _normalize_sensor_text(joined)

    temperature = _extract_labeled_value(
        normalized,
        labels=("temp", "temperature", "temperat", "celsius", "degc"),
        value_range=TEMP_RANGE_C,
    )
    humidity = _extract_labeled_value(
        normalized,
        labels=("humid", "humidity", "hum", "rh"),
        value_range=HUMIDITY_RANGE_PCT,
    )

    numbers_meta = _extract_numbers_with_meta(normalized)
    if temperature is None or humidity is None:
        inferred_temp, inferred_humidity = _infer_unlabeled_values(numbers_meta)
        temperature = temperature if temperature is not None else inferred_temp
        humidity = humidity if humidity is not None else inferred_humidity

    confidence = _score_reading(temperature, humidity, items)
    return temperature, humidity, confidence, raw_text


def clean_ocr_text(text: str) -> str:
    text = str(text).strip()
    replacements = {
        "，": ".",
        ",": ".",
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _normalize_sensor_text(text: str) -> str:
    text = text.lower()
    text = text.replace("°", " deg")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_labeled_value(text: str, labels: tuple[str, ...], value_range: tuple[float, float]) -> Optional[float]:
    label_pattern = "|".join(re.escape(label) for label in labels)
    patterns = [
        rf"(?:{label_pattern})\D{{0,8}}(-?\d{{1,3}}(?:\.\d{{1,2}})?)",
        rf"(-?\d{{1,3}}(?:\.\d{{1,2}})?)\D{{0,8}}(?:{label_pattern})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = _to_float(match.group(1))
            if value is not None and value_range[0] <= value <= value_range[1]:
                return value
    return None


def _extract_numbers(text: str) -> list[float]:
    values = []
    for token in re.findall(r"-?\d{1,3}(?:\.\d{1,2})?", text):
        value = _to_float(token)
        if value is not None:
            values.append(value)
    return values


def _extract_numbers_with_meta(text: str) -> list[tuple[float, bool]]:
    results: list[tuple[float, bool]] = []
    for token in re.findall(r"-?\d{1,3}(?:\.\d{1,2})?", text):
        had_decimal = "." in token
        value = _to_float(token)
        if value is not None:
            results.append((value, had_decimal))
    return results


def _infer_unlabeled_values(numbers_meta: list[tuple[float, bool]]) -> tuple[Optional[float], Optional[float]]:

    from collections import Counter

    humidity_candidates = [value for value, _ in numbers_meta if 20.0 <= value <= HUMIDITY_RANGE_PCT[1]]

    direct_decimal_candidates = [value for value, had_decimal in numbers_meta if had_decimal and 5.0 <= value <= 45.0]

    corrected_candidates = []
    for value, had_decimal in numbers_meta:
        if had_decimal:
            continue
        if 100 <= value <= 999:
            corrected = round(value / 10, 1)
            if 5.0 <= corrected <= 45.0:
                corrected_candidates.append(corrected)

    temperature = None
    if direct_decimal_candidates:
        temperature = Counter(direct_decimal_candidates).most_common(1)[0][0]
    elif corrected_candidates:
        temperature = Counter(corrected_candidates).most_common(1)[0][0]
    else:
        fallback_candidates = [value for value, had_decimal in numbers_meta if not had_decimal and 5.0 <= value <= 45.0]
        if fallback_candidates:
            temperature = Counter(fallback_candidates).most_common(1)[0][0]

    humidity = None
    if humidity_candidates:
        for value, _count in Counter(humidity_candidates).most_common():
            if temperature is None or abs(value - temperature) > 0.09:
                humidity = value
                break

    return temperature, humidity


def _to_float(value: str) -> Optional[float]:
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def _score_reading(temperature: Optional[float], humidity: Optional[float], items: list[tuple[str, float]]) -> float:
    if temperature is None and humidity is None:
        return 0.0
    base = 0.45 if temperature is not None and humidity is not None else 0.25
    text_score = max((score for _, score in items), default=0.0)
    return round(min(1.0, base + 0.55 * text_score), 3)


def ocr_frame(frame_path: Path, ocr=None) -> SensorReading:
    if ocr is None:
        ocr = _load_paddle_ocr()

    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError(f"Cannot read image: {frame_path}")

    best = (None, None, 0.0, "")
    for name, crop in build_candidate_rois(image):
        
        is_dedicated_lcd_crop = name == "sensor_lcd_auto"

        all_items: list[tuple[str, float]] = []
        for variant in preprocess_for_ocr(crop, aggressive=is_dedicated_lcd_crop):
            all_items.extend(run_paddle_ocr(ocr, variant))
        reading = extract_sensor_values(all_items)
        if reading[2] > best[2]:
            best = reading
        if best[0] is not None and best[1] is not None and best[2] >= 0.80:
            break

    return SensorReading(
        frame_path=frame_path,
        timestamp=parse_frame_timestamp(frame_path),
        temperature_c=best[0],
        humidity_pct=best[1],
        confidence=best[2],
        raw_text=best[3],
    )


def collect_sensor_readings(
    processed_dir: Path = PROCESSED_DIR,
    limit: Optional[int] = None,
    min_confidence: float = 0.35,
) -> dict[datetime, SensorReading]:
    ocr = _load_paddle_ocr()
    readings: dict[datetime, SensorReading] = {}

    for idx, frame_path in enumerate(iter_frame_paths(processed_dir), start=1):
        if limit is not None and idx > limit:
            break
        try:
            reading = ocr_frame(frame_path, ocr=ocr)
        except Exception as exc:
            print(f"[WARN] OCR failed for {frame_path}: {exc}")
            continue

        if reading.confidence >= min_confidence and (
            reading.temperature_c is not None or reading.humidity_pct is not None
        ):
            readings[reading.timestamp] = reading
            print(
                "[OK] "
                f"{reading.timestamp} temp={reading.temperature_c} "
                f"humidity={reading.humidity_pct} conf={reading.confidence} "
                f"raw='{reading.raw_text}'"
            )
        else:
            print(f"[WARN] No reliable sensor reading for {frame_path.name}: {reading.raw_text}")

    return readings


def update_label_files_with_readings(
    readings: dict[datetime, SensorReading],
    final_dir: Path = FINAL_DIR,
) -> int:
    label_files = sorted(final_dir.glob("F*/labels.csv"))
    if not label_files:
        print(f"[WARN] No labels.csv files found under {final_dir}")
        return 0

    updated_rows = 0
    for labels_csv in label_files:
        df = pd.read_csv(labels_csv)
        if "timestamp" not in df.columns:
            print(f"[WARN] Skipping {labels_csv}: missing timestamp column")
            continue

        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        for row_idx, ts in timestamps.items():
            if pd.isna(ts):
                continue
            reading = readings.get(ts.to_pydatetime().replace(microsecond=0))
            if reading is None:
                continue
            if reading.temperature_c is not None:
                df.at[row_idx, "temperature_c"] = reading.temperature_c
            if reading.humidity_pct is not None:
                df.at[row_idx, "humidity_pct"] = reading.humidity_pct
            updated_rows += 1

        df.to_csv(labels_csv, index=False)
        print(f"[OK] Updated {labels_csv}")

    return updated_rows


def generate_sensor_ocr_data(
    processed_dir: Path = PROCESSED_DIR,
    final_dir: Path = FINAL_DIR,
    limit: Optional[int] = None,
    min_confidence: float = 0.35,
) -> None:
    readings = collect_sensor_readings(
        processed_dir=processed_dir,
        limit=limit,
        min_confidence=min_confidence,
    )
    updated_rows = update_label_files_with_readings(readings, final_dir=final_dir)
    print(f"Sensor OCR complete: readings={len(readings)}, updated_label_rows={updated_rows}")


def self_test() -> None:
    samples = [
        ([("Temp: 22.4 C", 0.96), ("Humidity: 61.8 %", 0.95)], 22.4, 61.8),
        ([("T 23.0C RH 58%", 0.90)], 23.0, 58.0),
        ([("22.1 63.5", 0.80)], 22.1, 63.5),
    ]
    for items, expected_temp, expected_humidity in samples:
        temp, humidity, confidence, raw_text = extract_sensor_values(items)
        assert temp == expected_temp, (items, temp, raw_text)
        assert humidity == expected_humidity, (items, humidity, raw_text)
        assert confidence > 0.0

    for folder in (PROCESSED_DIR / "frames_18-03-2026", PROCESSED_DIR / "frames_19-03-2026"):
        if folder.exists():
            first_image = next(
                (p for p in sorted(folder.iterdir(), key=_natural_sort_key) if p.suffix.lower() in IMAGE_EXTENSIONS),
                None,
            )
            if first_image is not None:
                timestamp = parse_frame_timestamp(first_image)
                image = cv2.imread(str(first_image))
                assert image is not None, first_image
                assert build_candidate_rois(image), first_image
                assert isinstance(timestamp, datetime)

    print("sensor_ocr self-test passed")


def main(limit: Optional[int] = None, min_confidence: float = 0.35) -> None:
    generate_sensor_ocr_data(limit=limit, min_confidence=min_confidence)


def cli() -> None:
    parser = argparse.ArgumentParser(description="OCR temperature and humidity from strawberry frame images.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of frames processed.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Minimum confidence for accepting a reading.")
    parser.add_argument("--self-test", action="store_true", help="Run parser and image plumbing tests without PaddleOCR.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    main(limit=args.limit, min_confidence=args.min_confidence)


if __name__ == "__main__":
    cli()