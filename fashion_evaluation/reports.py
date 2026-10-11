"""Versioned report construction and deterministic, exclusive serialization."""
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import uuid

from . import CLASS_NAMES, SCHEMA_VERSION
from .metrics import calculate_metrics
from .predictions import ROOT, file_identity, verify_test_identity

REPORT_DIR = ROOT / "experiments/evaluation_v1/reports"


def build_report(bundle, *, prediction_origin="precomputed", source_identity=None,
                 full=False, timestamp=None):
    if prediction_origin not in ("precomputed", "new_inference"):
        raise ValueError("Unknown prediction origin")
    for key in ("model_identifier", "inference_variant"):
        if not isinstance(bundle.get(key), str) or not bundle[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if "class_names" not in bundle:
        raise ValueError("class_names must explicitly declare prediction column order")
    metrics = calculate_metrics(bundle["y_true"], values=bundle.get("values"),
                                predicted_labels=bundle.get("predicted_labels"),
                                class_names=bundle["class_names"],
                                value_kind=bundle.get("value_kind", "probabilities"))
    indices = bundle.get("test_indices")
    if full and indices is None:
        raise ValueError("Full evaluation requires original test_indices")
    dataset = (verify_test_identity(bundle["y_true"], indices, full) if indices is not None else
               {"name": "Fashion-MNIST-compatible supplied labels; dataset identity UNKNOWN",
                "scope": "unverified-prediction-set", "verified_original_labels": False})
    duration = bundle.get("inference_duration_seconds")
    if duration is not None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Inference duration must be finite and nonnegative, or null")
    if prediction_origin == "new_inference" and duration is None:
        raise ValueError("New inference must supply measured duration")
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scikit-learn", "joblib", "tensorflow", "keras"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema": SCHEMA_VERSION,
        "created_at_utc": timestamp or datetime.now(timezone.utc).isoformat(),
        "metric_origin": "newly_calculated_from_predictions",
        "prediction_origin": prediction_origin,
        "historical_metrics_imported": False,
        "model": {"identifier": bundle["model_identifier"], "inference_variant": bundle["inference_variant"],
                  "artifacts": bundle.get("artifacts", []),
                  "artifact_provenance_origin": "caller_supplied; hashes identify files, not verified prediction lineage"},
        "prediction_source": source_identity,
        "dataset": dataset,
        "class_names": list(CLASS_NAMES),
        "metric_policy": {"zero_division": 0, "macro_classes": 10,
                          "ranking_ties": "lower canonical class ID first",
                          "confusion_matrix_axes": "rows=true, columns=predicted",
                          "top3": "ranking of supplied scores/probabilities; null for label-only predictions"},
        "metrics": metrics,
        "timing": {"inference_duration_seconds": duration,
                   "average_latency_seconds_per_sample": None if duration is None else duration / metrics["evaluated_samples"],
                   "origin": ("UNKNOWN" if duration is None else
                              "measured_predict_call" if prediction_origin == "new_inference" else "caller_reported_unverified"),
                   "scope": "aggregate prediction-call duration / N; not individual request latency; excludes model loading and metric computation"},
        "evaluation_dependency_versions": versions,
        "prediction_dependency_versions": bundle.get("prediction_dependency_versions"),
        "evaluator_sources": [file_identity(p) for p in sorted(Path(__file__).parent.glob("*.py"))],
    }


def serialize_report(report):
    # Timestamps/timing are supplied report data. Serialization itself is stable.
    return json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_report(report, filename=None):
    name = filename or f"evaluation_v1_{uuid.uuid4().hex}.json"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json", name):
        raise ValueError("Report filename must be a plain JSON filename, not a path")
    if REPORT_DIR.resolve() != REPORT_DIR.absolute():
        raise ValueError("Report directory cannot redirect through symlinks/junctions")
    payload = serialize_report(report)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / name
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return destination
