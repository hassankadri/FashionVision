"""CLIP-only FastAPI entrypoint for the separate Vercel project."""

import io
import logging
from threading import RLock

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from ml import photo_classifier
from scripts.download_photo_model import FILES, TARGET, sha256

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = FastAPI(title="FashionVision CLIP API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fashion-vision.vercel.app"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_logger = logging.getLogger(__name__)
_model_lock = RLock()
_verified = False
_prediction_succeeded = False


def _ensure_ready() -> None:
    global _verified
    if _verified:
        return
    for name, (_url, expected_hash) in FILES.items():
        path = TARGET / name
        if not path.is_file():
            raise FileNotFoundError(
                "The real-photo model is missing. Run: python scripts/download_photo_model.py"
            )
        if sha256(path) != expected_hash:
            raise RuntimeError(f"SHA-256 verification failed for {name}")
    # Use the canonical session settings and tokenizer without changing inference.
    photo_classifier._prompt_inputs()
    photo_classifier._session()
    _verified = True


def _decode_and_classify(payload: bytes) -> dict[str, object]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            return photo_classifier.classify_image(image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a readable image.",
        ) from exc


def _predict(payload: bytes) -> dict[str, object]:
    global _verified, _prediction_succeeded
    # Serialize inference per worker to bound concurrent image/tensor memory.
    with _model_lock:
        try:
            _ensure_ready()
            result = _decode_and_classify(payload)
            _prediction_succeeded = True
            return result
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            _verified = _prediction_succeeded = False
            raise HTTPException(
                status_code=503,
                detail="The real-photo model is missing. Run: python scripts/download_photo_model.py",
            ) from exc
        except Exception as exc:
            _verified = _prediction_succeeded = False
            _logger.exception("CLIP inference unavailable")
            raise HTTPException(status_code=503, detail="CLIP is unavailable. Contact the server operator.") from exc


def _check_readiness() -> bool:
    global _verified, _prediction_succeeded
    with _model_lock:
        try:
            _ensure_ready()
            if not _prediction_succeeded:
                photo_classifier.classify_image(Image.new("RGB", (224, 224)))
                _prediction_succeeded = True
            return True
        except Exception:
            _verified = _prediction_succeeded = False
            _logger.exception("CLIP readiness check failed")
            return False


@app.get("/", include_in_schema=False)
@app.get("/api/health")
async def health() -> JSONResponse:
    ready = await run_in_threadpool(_check_readiness)
    status = "ok" if ready else (
        "model_missing" if not photo_classifier.MODEL_PATH.is_file() else "model_unavailable"
    )
    return JSONResponse(
        {"status": status, "photo_model_ready": ready},
        status_code=200 if ready else 503,
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/predict")
async def predict(request: Request) -> dict[str, object]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, WebP, or BMP image.")

    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 15 MB or smaller.")

    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 15 MB or smaller.")
    return await run_in_threadpool(_predict, payload)
