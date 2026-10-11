"""Prediction sources and original-test identity checks. No fitting occurs here."""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

from . import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"


def file_identity(path):
    path = Path(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"filename": path.name, "sha256": digest}


def load_test_set():
    image_path = RAW / "t10k-images-idx3-ubyte.gz"
    label_path = RAW / "t10k-labels-idx1-ubyte.gz"
    with gzip.open(image_path, "rb") as handle:
        images = handle.read()
    with gzip.open(label_path, "rb") as handle:
        labels = handle.read()
    if (struct.unpack(">IIII", images[:16]) != (2051, 10000, 28, 28)
            or struct.unpack(">II", labels[:8]) != (2049, 10000)
            or len(images) != 16 + 10000 * 784 or len(labels) != 10008):
        raise ValueError("Expected original 10,000-image Fashion-MNIST test IDX files")
    X = np.frombuffer(images, dtype=np.uint8, offset=16).reshape(10000, 28, 28)
    y = np.frombuffer(labels, dtype=np.uint8, offset=8)
    return X, y, {"name": "Fashion-MNIST original test split",
                  "files": [file_identity(image_path), file_identity(label_path)]}


def verify_test_identity(y_true, test_indices, full=False):
    _, labels, provenance = load_test_set()
    indices = np.asarray(test_indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != len(y_true)
            or not len(indices) or np.any((indices < 0) | (indices >= 10000))
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("test_indices must identify unique original test rows for every prediction")
    if full and (len(indices) != 10000 or not np.array_equal(np.sort(indices), np.arange(10000))):
        raise ValueError("Full evaluation requires each of the 10,000 untouched test rows exactly once")
    if not np.array_equal(np.asarray(y_true), labels[indices]):
        raise ValueError("y_true does not match original test labels at supplied indices")
    provenance.update({"verified_original_labels": True,
                       "scope": "full-test" if full else "test-subset",
                       "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                       "indices_encoding": "ordered little-endian int64", "samples": len(indices)})
    return provenance


def load_precomputed(path):
    """JSON bundle contract is documented in experiments/evaluation_v1/README.md."""
    with Path(path).open(encoding="utf-8") as handle:
        bundle = json.load(handle)
    required = {"y_true", "class_names", "model_identifier", "inference_variant"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Precomputed bundle requires {sorted(required)}")
    return bundle, file_identity(path)


def generate_predictions(predict, X, *, model_identifier, inference_variant,
                         class_names=CLASS_NAMES, artifacts=()):
    """Call an already-loaded, inference-only callable once. No model selection.

    The caller supplies a frozen model's predict_proba/score function (including
    transform-only preprocessing if needed). Loading time is excluded. Do not
    pass a training callable. Raw IDX pixels require the model's own existing
    stateless normalization; this helper never guesses preprocessing.
    """
    start = time.perf_counter()
    values = np.asarray(predict(X))
    duration = time.perf_counter() - start
    return {"values": values.tolist(), "class_names": list(class_names),
            "model_identifier": model_identifier, "inference_variant": inference_variant,
            "inference_duration_seconds": duration,
            "timing_origin": "measured_predict_call",
            "artifacts": [file_identity(path) for path in artifacts]}
