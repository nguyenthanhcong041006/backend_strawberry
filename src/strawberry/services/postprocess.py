from __future__ import annotations


def format_result(remaining_useful_life: float, confidence: float) -> dict[str, float]:
    rul = max(0.0, round(float(remaining_useful_life), 2))
    formatted_confidence = round(max(0.0, min(float(confidence), 1.0)), 2)
    return {
        "remaining_useful_life": rul,
        "confidence": formatted_confidence,
    }
