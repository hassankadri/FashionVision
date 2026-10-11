"""Optional, cached inference for the verified family-aware MobileNet checkpoint."""
import hashlib
import io
import json
import logging
from pathlib import Path
from threading import Lock
import time

import numpy as np
from PIL import Image

from fashion_rgb_data.pipeline import _decode

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/mobilenet-family-v1"
MODEL_SHA256 = "3d095fbaffef382d7be0068db178a713b79558c06ae97fb043e4f0319b0479a2"
CONFIG_SHA256 = "2e5086533c5e4780fb2dcff2d55a6c45e0d323b9cb54d2873fca73dbb020369f"
MODEL_ID = "mobilenetv3small-family-v1-77a4872e94084b1fbda881dab6316972"
CLASS_NAMES = ("T-shirts", "Shirts", "Kurtas", "Pants", "Closed shoes", "Handbags",
               "Watches", "Sunglasses", "Backpacks", "Belts")
MAX_PIXELS = 16_000_000
MAX_DIMENSION = 8192
FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP", "image/bmp": "BMP"}


class InvalidImage(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


def preprocess(payload, mime):
    try:
        # Inspect dimensions before allocating decoded pixels.
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            if image.format != FORMATS.get(mime):
                raise ValueError("Image format does not match Content-Type")
            if min(width, height) < 1 or max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
                raise ValueError("Decoded image dimensions exceed the allowed limit")
        # Exact saved inference policy: EXIF, alpha-on-white, RGB, bilinear,
        # float32 [0,255]. Normalization lives inside the saved model.
        image, _ = _decode(payload)
        return np.asarray(image.resize((160, 160), Image.Resampling.BILINEAR), dtype=np.float32)[None]
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise InvalidImage("Unreadable image, mismatched format, or invalid decoded dimensions.") from error


def _load_verified():
    for name, expected in (("best.keras", MODEL_SHA256), ("configuration.json", CONFIG_SHA256)):
        with (MODEL_DIR / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Verified MobileNet asset hash mismatch")
    config = json.loads((MODEL_DIR / "configuration.json").read_text())
    if tuple(config["class_names"]) != CLASS_NAMES or config["image_size"] != 160:
        raise ValueError("MobileNet class order or input size mismatch")
    import keras  # Optional dependency; CLIP-only startup never imports TensorFlow.
    model = keras.models.load_model(MODEL_DIR / "best.keras", compile=False)
    if model.input_shape != (None, 160, 160, 3) or model.output_shape != (None, 10):
        raise ValueError("MobileNet shape mismatch")
    return model


class MobileNetService:
    def __init__(self):
        self._lock = Lock()
        self._model = None
        self._failed = False
        self._gradcam = None

    def predict(self, pixels):
        # Serialize first load and inference; concurrent cold requests cannot
        # load multiple copies. Cache failures until process restart as well.
        with self._lock:
            if self._failed:
                raise ModelUnavailable("MobileNet is unavailable.")
            if self._model is None:
                try:
                    self._model = _load_verified()
                except Exception as error:
                    self._failed = True
                    logging.getLogger(__name__).exception("MobileNet loading failed")
                    raise ModelUnavailable("MobileNet is unavailable.") from error
            try:
                started = time.perf_counter()
                scores = np.asarray(self._model(pixels, training=False).numpy())
                elapsed = (time.perf_counter() - started) * 1000
                if (scores.shape != (1, 10) or not np.isfinite(scores).all()
                        or (scores < 0).any() or (scores > 1).any()
                        or not np.allclose(scores.sum(1), 1, atol=1e-5)):
                    raise ValueError("Invalid model output")
            except Exception as error:
                logging.getLogger(__name__).exception("MobileNet inference failed")
                raise ModelUnavailable("MobileNet inference is unavailable.") from error
        probabilities = [{"class_id": i, "label": name, "probability": float(scores[0, i])}
                         for i, name in enumerate(CLASS_NAMES)]
        top3 = sorted(probabilities, key=lambda p: p["probability"], reverse=True)[:3]
        return {"prediction": top3[0]["label"], "probabilities": probabilities,
                "top3": top3, "model": MODEL_ID, "inference_time_ms": elapsed,
                "score_type": "uncalibrated_softmax"}

    def explain(self, pixels):
        prediction = self.predict(pixels)  # Reuse verified loading and output validation.
        with self._lock:
            try:
                from .mobilenet_gradcam import GradCAM, LAYER_NAME, overlay_png
                if self._gradcam is None:
                    self._gradcam = GradCAM(self._model)
                result = self._gradcam.compute(pixels)
                expected = [p["probability"] for p in prediction["probabilities"]]
                if not np.allclose(result["probabilities"], expected, rtol=1e-4, atol=1e-6):
                    raise ValueError("Explanation graph differs from inference")
                if CLASS_NAMES[result["class_id"]] != prediction["prediction"]:
                    raise ValueError("Explanation targets a different prediction")
                return {"prediction": prediction["prediction"], "model": MODEL_ID,
                        "explained_class": CLASS_NAMES[result["class_id"]],
                        "target_layer": LAYER_NAME, "target_score": "pre_softmax_logit",
                        "heatmap_overlay_base64": overlay_png(pixels, result["heatmap"]),
                        "overlay_size": [160, 160], "overlay_format": "image/png",
                        "attribution_status": result["status"]}
            except Exception as error:
                logging.getLogger(__name__).exception("MobileNet explanation failed")
                raise ModelUnavailable("MobileNet explanation is unavailable.") from error


service = MobileNetService()


def classify(payload, mime):
    return service.predict(preprocess(payload, mime))


def explain(payload, mime):
    return service.explain(preprocess(payload, mime))
