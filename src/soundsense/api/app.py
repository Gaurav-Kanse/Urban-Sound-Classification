"""FastAPI inference service."""

import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from soundsense.config import CLASS_NAMES, PATHS, audio_config_dict
from soundsense.inference.predictor import Predictor

MAX_BYTES = 20 * 1024 * 1024
CHECKPOINT = Path(os.getenv("SOUNDSENSE_MODEL", str(PATHS["models"] / "cnn_best.pt")))
app = FastAPI(title="SoundSense API", version="1.0.0", description="Urban sound classification")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_predictor: Predictor | None = None


class Prediction(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    top_three: list[dict]


class BatchPredictionItem(BaseModel):
    filename: str | None = None
    predicted_class: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    top_three: list[dict] | None = None
    error: str | None = None


class BatchPrediction(BaseModel):
    predictions: list[BatchPredictionItem]


def get_predictor() -> Predictor:
    """Create the process-wide predictor lazily, once per worker."""
    global _predictor
    if _predictor is None:
        if not CHECKPOINT.is_file():
            raise HTTPException(
                status_code=503, detail="Model checkpoint is unavailable; train the CNN first."
            )
        _predictor = Predictor(CHECKPOINT)
    return _predictor


async def save_upload(upload: UploadFile) -> str:
    """Persist a bounded upload to a temporary file for librosa decoding."""
    suffix = Path(upload.filename or "audio.wav").suffix.lower()
    if suffix not in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}:
        raise HTTPException(
            status_code=415, detail="Supported audio extensions: WAV, MP3, OGG, FLAC, M4A."
        )
    content = await upload.read(MAX_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Audio upload exceeds the 20 MiB limit.")
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(content)
        return handle.name
    finally:
        handle.close()


@app.get("/")
def root() -> dict:
    return {
        "name": "SoundSense",
        "description": "Urban sound classification",
        "classes": CLASS_NAMES,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_available": CHECKPOINT.is_file()}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "architecture": "4-stage CNN, channels 16/32/64/128",
        "input": "128-band log-Mel",
        "classes": CLASS_NAMES,
        "audio_config": audio_config_dict(),
        "model_available": CHECKPOINT.is_file(),
    }


@app.post("/predict", response_model=Prediction)
async def predict(file: Annotated[UploadFile, File()]) -> dict:
    path = await save_upload(file)
    try:
        return get_predictor().predict_file(path)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        Path(path).unlink(missing_ok=True)


@app.post("/predict/batch", response_model=BatchPrediction)
async def predict_batch(files: Annotated[list[UploadFile], File()]) -> dict[str, list[dict]]:
    if not files or len(files) > 10:
        raise HTTPException(status_code=400, detail="Submit between one and ten audio files.")
    paths: list[str] = []
    try:
        for upload in files:
            paths.append(await save_upload(upload))
        predictor = get_predictor()
        results = []
        for upload, path in zip(files, paths, strict=True):
            try:
                results.append({"filename": upload.filename, **predictor.predict_file(path)})
            except ValueError as exc:
                results.append({"filename": upload.filename, "error": str(exc)})
        return {"predictions": results}
    finally:
        for path in paths:
            Path(path).unlink(missing_ok=True)
