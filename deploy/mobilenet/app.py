"""MobileNetV3 and Grad-CAM API for the separate Vercel project."""

from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml import mobilenet_classifier

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = FastAPI(title="FashionVision MobileNet API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fashion-vision.vercel.app"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_readiness_lock = Lock()
_readiness = {"mobilenet_model_ready": False, "gradcam_ready": False}


def _check_readiness() -> dict[str, bool]:
    # Successful probes are cached per process; merely finding a file is not readiness.
    with _readiness_lock:
        if not all(_readiness.values()):
            _readiness.update(mobilenet_model_ready=False, gradcam_ready=False)
            pixels = np.zeros((1, 160, 160, 3), dtype=np.float32)
            try:
                mobilenet_classifier.service.predict(pixels)
                _readiness["mobilenet_model_ready"] = True
                mobilenet_classifier.service.explain(pixels)
                _readiness["gradcam_ready"] = True
            except mobilenet_classifier.ModelUnavailable:
                # The service logs diagnostic details; keep public responses generic.
                _readiness["gradcam_ready"] = False
        return dict(_readiness)


def _invalidate_readiness(*, prediction_failed: bool) -> None:
    with _readiness_lock:
        if prediction_failed:
            _readiness["mobilenet_model_ready"] = False
        _readiness["gradcam_ready"] = False


@app.get("/", include_in_schema=False)
@app.get("/api/health")
async def health() -> JSONResponse:
    readiness = await run_in_threadpool(_check_readiness)
    ready = all(readiness.values())
    return JSONResponse(
        {
            "status": "ok" if ready else "unavailable",
            **readiness,
            "photo_model_ready": False,
            "model": mobilenet_classifier.MODEL_ID,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        },
        status_code=200 if ready else 503,
        headers={"Cache-Control": "no-store"},
    )


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
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
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
        _invalidate_readiness(prediction_failed=True)
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
        _invalidate_readiness(prediction_failed=True)
        raise HTTPException(
            status_code=503,
            detail="MobileNet explanation is unavailable. Contact the server operator.",
        ) from error
