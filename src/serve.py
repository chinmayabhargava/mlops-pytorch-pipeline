"""
FastAPI serving app for the trained image classifier.

Run with:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_CHECKPOINT_PATH   Path to the .pt checkpoint produced by src/train.py
                            (default: ./checkpoints/model.pt)

Endpoints:
    GET  /health   -> 200 if the model is loaded, 503 otherwise
    POST /predict  -> multipart/form-data upload under field "file"; returns
                       predicted class and per-class probabilities

Example:
    curl -X POST http://localhost:8000/predict -F "file=@cat.jpg"
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import get_eval_transform
from src.model import get_model


def _checkpoint_path() -> str:
    raw = os.environ.get("MODEL_CHECKPOINT_PATH", "checkpoints/model.pt")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


# Holds the loaded model + preprocessing pipeline. Populated at startup by load_model().
_state: Dict[str, object] = {
    "model": None,
    "transform": None,
    "classes": None,
    "device": None,
    "in_channels": None,
}


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_path: str


def load_model(checkpoint_path: str | None = None) -> torch.nn.Module:
    """Load a checkpoint produced by src/train.py and prepare it for inference."""
    path = checkpoint_path or _checkpoint_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
        in_channels=checkpoint["in_channels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_eval_transform(
        dataset_name=checkpoint["dataset_name"],
        architecture=checkpoint["architecture"],
        image_size=checkpoint.get("image_size"),
    )

    _state["model"] = model
    _state["transform"] = transform
    _state["classes"] = checkpoint["classes"]
    _state["device"] = device
    _state["in_channels"] = checkpoint["in_channels"]
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"[startup] Warning: {e}")
    yield


app = FastAPI(title="Image Classifier Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = _state["model"] is not None
    payload = HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        checkpoint_path=_checkpoint_path(),
    )
    if not loaded:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health.")

    contents = await file.read()
    mode = "L" if _state["in_channels"] == 1 else "RGB"
    try:
        image = Image.open(io.BytesIO(contents)).convert(mode)
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    transform = _state["transform"]
    tensor = transform(image).unsqueeze(0).to(_state["device"])

    with torch.no_grad():
        logits = _state["model"](tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    classes: List[str] = _state["classes"]
    prob_dict = {classes[i]: round(float(probs[i]), 6) for i in range(len(classes))}
    top_idx = int(torch.argmax(probs).item())

    return PredictionResponse(
        predicted_class=classes[top_idx],
        predicted_index=top_idx,
        probabilities=prob_dict,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
