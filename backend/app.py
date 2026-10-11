"""FashionVision prediction API and React website serving."""

from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from ml.photo_classifier import MODEL_PATH, classify_image
from ml import mobilenet_classifier


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "react" / "public" / "frontend"
REACT_DIST = ROOT / "frontend" / "react" / "dist"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = FastAPI(title="Fashion Classifier", version="2.0")
app.mount("/frontend", StaticFiles(directory=FRONTEND), name="frontend")
app.mount("/assets", StaticFiles(directory=REACT_DIST / "assets", check_dir=False), name="react-assets")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(REACT_DIST / "index.html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok" if MODEL_PATH.exists() else "model_missing",
        "photo_model_ready": MODEL_PATH.exists(),
    }


def _decode_and_classify(payload: bytes) -> dict[str, object]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            return classify_image(image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a readable image.",
        ) from exc


@app.post("/api/predict")
async def predict(request: Request) -> dict[str, object]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, WebP, or BMP image.",
        )

    content_length = request.headers.get("content-length")
    if (
        content_length
        and content_length.isdigit()
        and int(content_length) > MAX_UPLOAD_BYTES
    ):
        raise HTTPException(
            status_code=413,
            detail="Images must be 15 MB or smaller.",
        )

    payload = await request.body()
    if not payload:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Images must be 15 MB or smaller.",
        )
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "The real-photo model is missing. Run: "
                "python scripts/download_photo_model.py"
            ),
        )

    return await run_in_threadpool(_decode_and_classify, payload)


async def _mobilenet_upload(request: Request) -> tuple[bytes, str]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, WebP, or BMP image.",
        )

    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Images must be 15 MB or smaller.",
        )

    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Images must be 15 MB or smaller.",
            )
        payload.extend(chunk)

    if not payload:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )
    return bytes(payload), content_type


@app.post("/api/predict/mobilenet")
async def predict_mobilenet(request: Request) -> dict[str, object]:
    payload, content_type = await _mobilenet_upload(request)
    try:
        return await run_in_threadpool(
            mobilenet_classifier.classify,
            bytes(payload),
            content_type,
        )
    except mobilenet_classifier.InvalidImage as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except mobilenet_classifier.ModelUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail="MobileNet is unavailable. Contact the server operator.",
        ) from error


@app.post("/api/explain/mobilenet")
async def explain_mobilenet(request: Request) -> dict[str, object]:
    payload, content_type = await _mobilenet_upload(request)
    try:
        return await run_in_threadpool(
            mobilenet_classifier.explain,
            payload,
            content_type,
        )
    except mobilenet_classifier.InvalidImage as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except mobilenet_classifier.ModelUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail="MobileNet explanation is unavailable. Contact the server operator.",
        ) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=int(os.environ.get("FASHION_APP_PORT", "8765")),
        reload=False,
    )