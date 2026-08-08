from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
STRAWBERRY_DIR = SRC_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(STRAWBERRY_DIR) not in sys.path:
    sys.path.insert(0, str(STRAWBERRY_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routes import router
from services.predictor import FruitRULPredictor


def load_config() -> dict:
    candidates = [
        STRAWBERRY_DIR / "config_app" / "config.json",
        PROJECT_ROOT / "src" / "strawberry" / "config_app" / "config.json",
        Path.cwd() / "src" / "strawberry" / "config_app" / "config.json",
        Path.cwd() / "config_app" / "config.json",
    ]
    for config_path in candidates:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8-sig") as config_file:
                return json.load(config_file)
    raise FileNotFoundError(f"Config file not found in any candidate path: {candidates}")



def create_app() -> FastAPI:
    config = load_config()
    app = FastAPI(title="Fruit RUL Prediction API", version="1.0.0")
    app.state.config = config
    app.state.predictor = FruitRULPredictor(config=config, project_root=PROJECT_ROOT)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")

    @app.get("/")
    async def root() -> dict:
        return {
            "success": True,
            "message": "Fruit RUL Prediction API is running",
            "docs": "/docs",
            "health": "/api/health",
            "predict": "POST /api/predict",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

