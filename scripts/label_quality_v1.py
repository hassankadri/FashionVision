"""Separate label-quality experiment. Flags are suspicions, never corrections.

No filtered-model training, official-test evaluation, or historical writes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from fashion_evaluation import CLASS_NAMES
from fashion_evaluation.metrics import label_vector
# Reuse read-only IDX loading and exclusive JSON publication, not v3 training.
from scripts.classical_v3 import RAW, atomic_json_new, load_split, sha256

REPORT_DIR = ROOT / "experiments" / "label_quality_v1" / "runs"
SCHEMA = "fashion-label-quality/1.0"
REVIEW_STATUSES = ("UNREVIEWED", "CONFIRMED_WRONG", "AMBIGUOUS", "LABEL_OK")


def require_cleanlab():
    try:
        from cleanlab.filter import find_label_issues
    except ImportError as error:
        raise RuntimeError("Cleanlab is unavailable; label-quality analysis cannot run. "
                           "No substitute ranking or dependency installation is performed.") from error
    return find_label_issues


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def reserve_validation(labels, seed=42):
    """Freeze 10% of the original training split BEFORE any issue analysis."""
    labels = label_vector(labels, "training labels")
    training, validation = train_test_split(
        np.arange(len(labels)), test_size=0.1, stratify=labels, random_state=seed)
    return training, validation


def make_pipeline(seed):
    return Pipeline([
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=seed)),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
    ])


def validate_probabilities(probabilities, count):
    values = np.asarray(probabilities)
    if (values.shape != (count, 10) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
            or not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("OOF probabilities must be finite N x 10 probability rows")
    return values


def generate_oof(X, labels, sample_indices, seed=42):
    """Each row is predicted by one clone fitted on the other two folds only."""
    labels = label_vector(labels, "OOF labels")
    indices = np.asarray(sample_indices)
    if (len(X) != len(labels) or indices.shape != labels.shape or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) != len(indices) or np.any(indices < 0)):
        raise ValueError("OOF features, labels and unique original indices must align")
    if np.any(np.bincount(labels, minlength=10) < 3):
        raise ValueError("Three-fold OOF requires at least three examples per canonical class")
    template = make_pipeline(seed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    probabilities = np.empty((len(labels), 10), dtype=float)
    fold_ids = np.full(len(labels), -1, dtype=int)
    folds = []
    with threadpool_limits(limits=1):
        for fold, (fit_rows, held_out) in enumerate(cv.split(X, labels)):
            estimator = clone(template)
            estimator.fit(X[fit_rows], labels[fit_rows])
            if not np.array_equal(estimator.classes_, np.arange(10)):
                raise ValueError("OOF classifier columns must follow canonical class IDs 0..9")
            probabilities[held_out] = validate_probabilities(
                estimator.predict_proba(X[held_out]), len(held_out))
            fold_ids[held_out] = fold
            folds.append({"fold": fold, "fit_count": len(fit_rows), "held_out_count": len(held_out),
                          "fit_indices_sha256": index_hash(indices[fit_rows]),
                          "held_out_indices_sha256": index_hash(indices[held_out])})
    if np.any(fold_ids < 0):
        raise ValueError("Every analysis row must have an out-of-fold prediction")
    return validate_probabilities(probabilities, len(labels)), fold_ids, folds


def rank_suspicions(labels, probabilities):
    # The original Cleanlab algorithm; only its INPUT methodology is corrected.
    finder = require_cleanlab()
    return finder(labels=labels, pred_probs=probabilities,
                  return_indices_ranked_by="self_confidence", n_jobs=1)


def analyze_training(pixels, labels, *, mode="smoke", seed=42):
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    labels = label_vector(labels, "training labels")
    if len(pixels) != len(labels):
        raise ValueError("Training pixels and labels must align")
    development, validation = reserve_validation(labels, seed)
    analysis_indices = development
    if mode == "smoke" and len(development) > 300:
        analysis_indices, _ = train_test_split(
            development, train_size=300, stratify=labels[development], random_state=seed)
    # No validation pixels/labels are passed to either OOF or Cleanlab.
    X = pixels[analysis_indices].reshape(len(analysis_indices), -1).astype(np.float32) / 255.0
    y = labels[analysis_indices]
    probabilities, fold_ids, folds = generate_oof(X, y, analysis_indices, seed)
    probabilities = validate_probabilities(probabilities, len(y))
    ranked = np.asarray(rank_suspicions(y, probabilities))
    if (ranked.ndim != 1 or (ranked.size and ranked.dtype.kind not in "iu")
            or len(np.unique(ranked)) != len(ranked)
            or np.any((ranked < 0) | (ranked >= len(y)))):
        raise ValueError("Cleanlab must return unique ranked positions in the analysis pool")
    candidates = []
    for rank, position in enumerate(ranked.astype(int), 1):
        given, predicted = int(y[position]), int(probabilities[position].argmax())
        candidates.append({
            "sample_index": int(analysis_indices[position]), "analysis_position": int(position),
            "given_label": given, "given_class": CLASS_NAMES[given],
            "predicted_label": predicted, "predicted_class": CLASS_NAMES[predicted],
            "self_confidence": float(probabilities[position, given]), "rank": rank,
            "assessment": "SUSPECTED_LABEL_ISSUE", "review_status": "UNREVIEWED",
            "reviewed_by": None, "review_notes": None,
        })
    return {
        "schema": SCHEMA, "analysis_id": uuid.uuid4().hex, "mode": mode, "seed": seed,
        "class_names": list(CLASS_NAMES), "sample_index_namespace": "original_training_split",
        "development_indices": development.tolist(), "validation_indices": validation.tolist(),
        "validation_indices_sha256": index_hash(validation),
        "analysis_indices": analysis_indices.tolist(), "given_labels": y.tolist(),
        "oof_probabilities": probabilities.tolist(), "oof_fold_ids": fold_ids.tolist(), "folds": folds,
        "candidates": candidates, "automatic_removals": [],
        "official_test": {"split": "original Fashion-MNIST test", "count": 10000,
                          "used_for_issue_detection_or_strategy_selection": False},
        "comparison_training_config": {
            "normalization": "float32 / 255", "pca": {"n_components": 0.95, "svd_solver": "full"},
            "classifier": "LogisticRegression", "C": 1.0, "max_iter": 1000, "random_state": seed,
        },
        "methodology": {"oof_folds": 3, "shuffle": True, "fold_local_pca": True,
                        "ranking": "cleanlab self_confidence; lower means more suspect",
                        "strategy_selection": "reserved validation only; never official test",
                        "training_comparison_executed": False},
    }


def review_manifest(analysis):
    return {"schema": SCHEMA, "analysis_id": analysis["analysis_id"],
            "class_names": list(CLASS_NAMES), "sample_index_namespace": "original_training_split",
            "validation_indices_sha256": analysis["validation_indices_sha256"],
            "allowed_statuses": list(REVIEW_STATUSES), "items": deepcopy(analysis["candidates"])}


def plan_strategy(analysis, strategy="no_filtering", *, top_n=None, threshold=None, reviewed=None):
    """Return an explicit selection PLAN; never change pixels, labels or files.

    Compare plans from the SAME analysis with the SAME training configuration.
    This is not a training runner or an automatic strategy chooser.
    """
    if tuple(analysis["class_names"]) != CLASS_NAMES:
        raise ValueError("Canonical class ordering required")
    pool = analysis["analysis_indices"]
    validation = analysis["validation_indices"]
    if (len(set(pool)) != len(pool) or len(set(validation)) != len(validation)
            or set(pool) & set(validation)
            or index_hash(validation) != analysis["validation_indices_sha256"]):
        raise ValueError("Analysis and validation indices must be unique/disjoint")
    candidates = analysis["candidates"]
    candidate_ids = [item["sample_index"] for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids) or not set(candidate_ids) <= set(pool):
        raise ValueError("Candidates must be unique original indices inside the analysis pool")
    if strategy == "no_filtering" and top_n is None and threshold is None and reviewed is None:
        removed = []
    elif strategy == "top_n" and threshold is None and reviewed is None:
        if type(top_n) is not int or not 0 <= top_n <= len(candidates):
            raise ValueError("top_n must be an explicit integer within the candidate count")
        removed = candidate_ids[:top_n]
    elif strategy == "confidence_threshold" and top_n is None and reviewed is None:
        if (type(threshold) not in (int, float) or not np.isfinite(threshold)
                or not 0 <= threshold <= 1):
            raise ValueError("An explicit self-confidence threshold in [0,1] is required")
        removed = [item["sample_index"] for item in candidates if item["self_confidence"] < threshold]
    elif strategy == "manually_reviewed" and top_n is None and threshold is None:
        expected = review_manifest(analysis)
        if not isinstance(reviewed, dict) or any(reviewed.get(k) != v for k, v in expected.items() if k != "items"):
            raise ValueError("Review manifest must belong to the same analysis/split/classes")
        items = reviewed.get("items", [])
        if len(items) != len(candidates):
            raise ValueError("Review manifest must preserve every candidate exactly once and in rank order")
        removed = []
        editable = {"review_status", "reviewed_by", "review_notes"}
        for original, item in zip(candidates, items):
            if ({k: v for k, v in item.items() if k not in editable}
                    != {k: v for k, v in original.items() if k not in editable}):
                raise ValueError("Review cannot change original sample/label/rank/score metadata")
            status = item.get("review_status")
            if status not in REVIEW_STATUSES:
                raise ValueError("Unknown review status")
            if status != "UNREVIEWED" and any(
                    not isinstance(item.get(k), str) or not item[k].strip()
                    for k in ("reviewed_by", "review_notes")):
                raise ValueError("Human review requires reviewer identity and notes")
            if status == "CONFIRMED_WRONG":
                removed.append(item["sample_index"])
    else:
        raise ValueError("Unknown strategy or unexpected strategy parameters")
    removed_set = set(removed)
    kept = [index for index in pool if index not in removed_set]
    labels = np.asarray(analysis["given_labels"])
    if set(labels[[i not in removed_set for i in pool]].tolist()) != set(range(10)):
        raise ValueError("A controlled plan must retain all ten classes")
    return {"schema": SCHEMA, "analysis_id": analysis["analysis_id"], "mode": analysis["mode"],
            "strategy": strategy, "top_n": top_n, "self_confidence_threshold": threshold,
            "training_indices": kept, "proposed_excluded_indices": removed,
            "validation_indices": list(validation), "class_names": list(CLASS_NAMES),
            "validation_indices_sha256": analysis["validation_indices_sha256"],
            "official_test": deepcopy(analysis["official_test"]),
            "training_config": deepcopy(analysis["comparison_training_config"]),
            "status": "PLAN_ONLY", "data_modified": False, "retraining_executed": False}


def run(mode="smoke", seed=42):
    require_cleanlab()  # Fail before loading data, fitting or creating any output.
    start = time.perf_counter()
    pixels, labels = load_split("train")
    if len(labels) != 60000:
        raise ValueError("Expected the original 60,000 training rows")
    analysis = analyze_training(pixels, labels, mode=mode, seed=seed)
    analysis.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - start,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scikit-learn", "cleanlab", "threadpoolctl")},
        "python_version": sys.version, "script_sha256": sha256(Path(__file__)),
        "dataset_hashes": {p.name: sha256(p) for p in sorted(RAW.glob("*.gz"))},
    })
    # Test files are hashed for provenance only; their rows are never loaded.
    analysis["official_test"]["file_sha256"] = {
        name: digest for name, digest in analysis["dataset_hashes"].items() if name.startswith("t10k-")}
    if REPORT_DIR.resolve() != REPORT_DIR.absolute():
        raise ValueError("Output directory cannot redirect through symlinks/junctions")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / f"label_quality_{mode}_{analysis['analysis_id']}"
    destination.mkdir(exist_ok=False)
    atomic_json_new(destination / "analysis.json", analysis)
    atomic_json_new(destination / "review_manifest.json", review_manifest(analysis))
    atomic_json_new(destination / "baseline_plan.json", plan_strategy(analysis))
    atomic_json_new(destination / "status.json", {"status": "COMPLETE", "schema": SCHEMA,
                    "analysis_id": analysis["analysis_id"], "training_comparison_executed": False})
    return analysis, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analysis, destination = run(args.mode, args.seed)
    print(f"Saved {args.mode} label-quality analysis: {destination}")
    print(f"{len(analysis['candidates'])} SUSPECTED label issues; zero automatic removals")


if __name__ == "__main__":
    main()
