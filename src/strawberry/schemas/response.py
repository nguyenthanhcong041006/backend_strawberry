from __future__ import annotations

from typing import Any


def success(remaining_useful_life: float, confidence: float) -> dict[str, Any]:
    return {
        "success": True,
        "remaining_useful_life": remaining_useful_life,
        "confidence": confidence,
    }


def error(message: str = "Invalid image") -> dict[str, Any]:
    return {
        "success": False,
        "message": message,
    }
