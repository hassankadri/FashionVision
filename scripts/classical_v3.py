"""Separate classical experiment: fold-local PCA, train-only selection.

Defaults to a small smoke run. Full training requires --mode full. No historical
model is loaded or saved; reports are exclusively created outside outputs/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time
import uuid

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
# Preserve direct-script invocation as well as imports from the test suite.
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from fashion_evaluation import CLASS_NAMES as EVALUATOR_CLASS_NAMES
from fashion_evaluation.metrics import label_vector
from fashion_evaluation.predictions import verify_test_identity
from fashion_evaluation import reports as unified_evaluation

RAW = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "experiments" / "classical_v3" / "runs"
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def atomic_json_new(destination, value):
    """Publish an immutable JSON file; never replace an existing result.

    A same-directory hard link publishes the fully flushed temporary inode
    atomically and fails if the destination exists (including on POSIX).
    Interrupted .tmp files are not committed results and are never read.
    """
    payload = unified_evaluation.serialize_report(value)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, destination)
    temporary.unlink()


class RunJournal:
    """Append-only metadata/results, not a model checkpoint or resume system."""
    def __init__(self, mode, seed):
        if REPORT_DIR.resolve() != REPORT_DIR.absolute():
            raise ValueError("Run directory must not redirect through a symlink/junction")
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = REPORT_DIR / f"classical_v3_{mode}_{uuid.uuid4().hex}"
        self.path.mkdir(exist_ok=False)
        self.context = {
            "mode": mode, "random_seed": seed, "class_names": CLASS_NAMES,
            "evaluator_schema": unified_evaluation.SCHEMA_VERSION,
            "dataset_hashes": {p.name: sha256(p) for p in sorted(RAW.glob("*.gz"))},
            "resume_support": "NOT SUPPORTED",
        }
        self.models = {}
        self.sequence = 0
        self.state("RUNNING")

    def state(self, status, **details):
        # Immutable numbered snapshots avoid overwriting even run-status files.
        destination = self.path / f"state_{self.sequence:04d}.json"
        atomic_json_new(destination, {**self.context, "status": status,
                                     "models": self.models, **details})
        self.sequence += 1

    def cv_complete(self, name, record):
        stem = f"model_{len(self.models) + 1:02d}"
        filename = f"{stem}_cv.json"
        atomic_json_new(self.path / filename, {
            **self.context, "model_name": name, "status": "CV_COMPLETE",
            "prediction_report_reference": None, "cv": record,
        })
        self.models[name] = {"status": "CV_COMPLETE", "stem": stem, "cv_reference": filename}
        self.state("RUNNING")

    def complete_model(self, name, record, evaluation):
        stem = self.models[name]["stem"]
        report_name = f"{stem}_evaluation.json"
        result_name = f"{stem}_result.json"
        atomic_json_new(self.path / report_name, evaluation["unified_report"])
        atomic_json_new(self.path / result_name, {
            **self.context, "model_name": name, "status": "COMPLETE",
            "selected_hyperparameters": record["selected_hyperparameters"],
            "cv_accuracy_mean": record["cv_accuracy_mean"],
            "cv_accuracy_std": record["cv_accuracy_std"],
            "prediction_report_reference": report_name,
            "prediction_bundle": evaluation["prediction_bundle"],
            "predicted_labels": evaluation["predicted_labels"],
        })
        self.models[name].update(status="COMPLETE", result_reference=result_name,
                                 evaluation_reference=report_name)
        self.state("RUNNING")

    def fail(self, error):
        self.state("PARTIAL" if self.models else "FAILED",
                   error={"type": type(error).__name__, "message": str(error)})

    def finish(self, report):
        if set(self.models) != set(report["models"]) or not self.models or any(
                model["status"] != "COMPLETE" for model in self.models.values()):
            raise ValueError("Cannot complete a run with incomplete models")
        atomic_json_new(self.path / "summary.json", report)
        self.state("COMPLETE", summary_reference="summary.json")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_split(split):
    """Read the original IDX boundary; no processed data or Cleanlab filtering."""
    prefix, count = {"train": ("train", 60000), "test": ("t10k", 10000)}[split]
    with gzip.open(RAW / f"{prefix}-images-idx3-ubyte.gz", "rb") as handle:
        image_bytes = handle.read()
    with gzip.open(RAW / f"{prefix}-labels-idx1-ubyte.gz", "rb") as handle:
        label_bytes = handle.read()
    if struct.unpack(">IIII", image_bytes[:16]) != (2051, count, 28, 28):
        raise ValueError(f"Unexpected {split} image IDX header")
    if struct.unpack(">II", label_bytes[:8]) != (2049, count):
        raise ValueError(f"Unexpected {split} label IDX header")
    if len(image_bytes) != 16 + count * 784 or len(label_bytes) != 8 + count:
        raise ValueError(f"Unexpected {split} IDX payload length")
    pixels = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(count, 784)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
    if not np.array_equal(np.unique(labels), np.arange(10)):
        raise ValueError("Unexpected class labels")
    return pixels, labels


def select_rows(pixels, labels, size, seed):
    indices = np.arange(len(labels))
    if size is not None and size < len(labels):
        indices, _ = train_test_split(indices, train_size=size, stratify=labels, random_state=seed)
    # Dividing by a constant is stateless. No learned preprocessing occurs here.
    return pixels[indices].astype(np.float32) / 255.0, labels[indices], indices


def make_searches(mode, seed):
    smoke = mode == "smoke"
    folds = 2 if smoke else 3
    configs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=seed),
            {"classifier__C": [1.0] if smoke else [0.01, 0.1, 1, 10]}),
        "Random Forest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {"classifier__n_estimators": [10] if smoke else [100, 150],
             "classifier__max_depth": [5] if smoke else [15, 25]}),
        "Hist Gradient Boosting": (
            HistGradientBoostingClassifier(random_state=seed),
            {"classifier__max_iter": [5] if smoke else [75, 100],
             "classifier__learning_rate": [0.1] if smoke else [0.1, 0.2]}),
    }
    return {
        name: RandomizedSearchCV(
            Pipeline([("pca", PCA(n_components=0.95, svd_solver="full", random_state=seed)),
                      ("classifier", model)]),
            param_distributions=params,
            n_iter=1 if smoke else 3,
            scoring="accuracy",
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
            random_state=seed, n_jobs=1, refit=True, error_score="raise",
        ) for name, (model, params) in configs.items()
    }


def fit_candidates(X_train, y_train, mode, seed, *, on_fitted=None):
    """Accepts training data only; CV clones and fits PCA for each fold."""
    searches = make_searches(mode, seed)
    records = {}
    for name, search in searches.items():
        start = time.perf_counter()
        search.fit(X_train, y_train)
        duration = time.perf_counter() - start
        results = search.cv_results_
        candidates = []
        for index, params in enumerate(results["params"]):
            candidates.append({
                "parameters": params,
                "cv_accuracy_mean": float(results["mean_test_score"][index]),
                "cv_accuracy_std": float(results["std_test_score"][index]),
                "cv_fold_accuracies": [float(results[f"split{fold}_test_score"][index])
                                       for fold in range(search.n_splits_)],
            })
        best = candidates[search.best_index_]
        records[name] = {
            "cv_accuracy": float(search.best_score_),
            "cv_accuracy_mean": best["cv_accuracy_mean"],
            "cv_accuracy_std": best["cv_accuracy_std"],
            "cv_fold_accuracies": best["cv_fold_accuracies"],
            "selected_hyperparameters": search.best_params_,
            "candidates": candidates,
            "training_duration_seconds": duration,
            "training_duration_scope": "all CV candidates plus final training-only refit",
            "refit_duration_seconds": float(search.refit_time_),
            "refit_pca_components": int(search.best_estimator_.named_steps["pca"].n_components_),
        }
        if on_fitted is not None:
            on_fitted(name, records[name])
    # Dict insertion order provides a deterministic tie-break. No test scores.
    selected = max(records, key=lambda name: records[name]["cv_accuracy_mean"])
    return searches, records, selected


def evaluate_candidates(searches, X_test, y_test, test_indices, *, full=False, on_evaluated=None):
    """Retain final predictions and delegate every metric to the shared evaluator.

    The prediction bundle has one evaluator input representation (probabilities
    when supported). Actual predict() labels are retained alongside that bundle
    and checked against its argmax; no policy or model selection happens here.
    """
    if tuple(CLASS_NAMES) != EVALUATOR_CLASS_NAMES:
        raise ValueError("v3 and evaluator class ordering differ")
    truth = label_vector(y_test, "y_true")
    if len(X_test) != len(truth):
        raise ValueError("Test features and labels must have matching lengths")
    verify_test_identity(truth, test_indices, full=full)
    evaluations = {}
    for name, search in searches.items():
        # Predict only: neither refit nor selection may happen in this stage.
        model = search.best_estimator_
        if not np.array_equal(model.classes_, np.arange(10)):
            raise ValueError(f"{name}: model classes must be in canonical 0..9 order")
        start = time.perf_counter()
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        duration = time.perf_counter() - start
        predicted = label_vector(predicted, "predicted_labels")
        if predicted.shape != truth.shape:
            raise ValueError(f"{name}: prediction and label lengths differ")
        bundle = {
            "y_true": truth.tolist(), "test_indices": np.asarray(test_indices).tolist(),
            "class_names": list(CLASS_NAMES), "model_identifier": f"classical-v3/{name}",
            "inference_variant": "fitted-pca-classifier-single-pass",
            "inference_duration_seconds": duration,
            "prediction_dependency_versions": {"sklearn": sklearn.__version__, "numpy": np.__version__,
                                               "python": platform.python_version()},
        }
        if probabilities is not None:
            probabilities = np.asarray(probabilities)
            if (probabilities.shape != (len(truth), 10) or probabilities.dtype.kind not in "iuf"
                    or not np.isfinite(probabilities).all()):
                raise ValueError(f"{name}: expected finite probability matrix [{len(truth)}, 10]")
            if not np.array_equal(predicted, probabilities.argmax(axis=1)):
                raise ValueError(f"{name}: predict labels disagree with probability argmax")
            bundle["values"] = probabilities.tolist()
            bundle["value_kind"] = "probabilities"
        else:
            bundle["predicted_labels"] = predicted.tolist()
        # Shared code validates probability bounds/sums and original test identity.
        standardized = unified_evaluation.build_report(
            bundle, prediction_origin="new_inference", full=full,
            source_identity={"kind": "in-memory final v3 predictions", "script_sha256": sha256(Path(__file__))})
        standardized["timing"]["scope"] = (
            "predict plus predict_proba when supported, including fitted PCA transforms; "
            "excludes training, loading, validation and metric calculation; total / N is aggregate cost")
        evaluations[name] = {
            **standardized["metrics"],
            "test_accuracy": standardized["metrics"]["accuracy"],
            "predicted_labels": predicted.tolist(),
            "prediction_bundle": bundle,
            "unified_report": standardized,
        }
        if on_evaluated is not None:
            on_evaluated(name, evaluations[name])
    return evaluations


def index_identity(indices):
    return {"count": len(indices),
            "indices_sha256": hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
            "indices_encoding": "ordered little-endian int64 original split row indices"}


def run_experiment(mode="smoke", seed=42, *, journal=None):
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    start = time.perf_counter()
    train_pixels, train_labels = load_split("train")
    X_train, y_train, train_indices = select_rows(
        train_pixels, train_labels, 300 if mode == "smoke" else None, seed)
    # Bound native worker counts too; no environment variables are changed.
    with threadpool_limits(limits=1):
        fit_options = {} if journal is None else {"on_fitted": journal.cv_complete}
        searches, records, selected = fit_candidates(X_train, y_train, mode, seed, **fit_options)
        # Intentionally load test data only AFTER all tuning/model selection.
        test_pixels, test_labels = load_split("test")
        X_test, y_test, test_indices = select_rows(
            test_pixels, test_labels, 100 if mode == "smoke" else None, seed)
        def completed(name, evaluation):
            evaluation["unified_report"]["experiment"] = {
                "name": "classical-v3-fold-local-pca", "mode": mode, "random_seed": seed,
                "full_benchmark": mode == "full", "selected_model_by_training_cv": selected,
                "cv_accuracy_mean": records[name]["cv_accuracy_mean"],
                "selected_hyperparameters": records[name]["selected_hyperparameters"],
            }
            journal.complete_model(name, records[name], evaluation)

        eval_options = {} if journal is None else {"on_evaluated": completed}
        evaluations = evaluate_candidates(searches, X_test, y_test, test_indices,
                                          full=mode == "full", **eval_options)
    for name in records:
        records[name]["evaluation"] = evaluations[name]
        evaluations[name]["unified_report"]["experiment"] = {
            "name": "classical-v3-fold-local-pca", "mode": mode, "random_seed": seed,
            "full_benchmark": mode == "full", "selected_model_by_training_cv": selected,
            "cv_accuracy_mean": records[name]["cv_accuracy_mean"],
            "selected_hyperparameters": records[name]["selected_hyperparameters"],
        }
    return {
        "experiment": "classical-v3-fold-local-pca",
        "mode": mode,
        "full_benchmark": mode == "full",
        "metrics_scope": "SMOKE ONLY: not benchmark accuracy" if mode == "smoke" else "original full test split",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "script_sha256": sha256(Path(__file__)),
        "class_names": CLASS_NAMES,
        "dataset": {
            "name": "Fashion-MNIST; original raw IDX train/test files; no label filtering",
            "raw_file_sha256": {p.name: sha256(p) for p in sorted(RAW.glob("*.gz"))},
            "original_split_counts": {"train": len(train_labels), "test": len(test_labels)},
            "training_selection": index_identity(train_indices),
            "test_selection": index_identity(test_indices),
            "sampling": "stratified train_test_split within each original split, seed above; all rows in full mode",
        },
        "methodology": {
            "preprocessing": "flatten, float32 / 255 (stateless), then fold-local PCA(0.95, full SVD)",
            "cv": {"type": "StratifiedKFold", "folds": 2 if mode == "smoke" else 3, "shuffle": True},
            "search_candidates_per_model": 1 if mode == "smoke" else 3,
            "selection": "highest training CV mean accuracy; ties use declared model order",
            "threshold_selection": "none",
            "cv_std": "population standard deviation across validation-fold accuracies; not a confidence interval",
            "cv_caveat": "selection scores, not nested-CV unbiased generalization estimates",
            "threads": 1,
        },
        "selected_model_by_cv": selected,
        "models": records,
        "total_duration_seconds": time.perf_counter() - start,
    }


def save_report(report):
    # Fixed new namespace, no caller-supplied output path, no model serialization.
    # Reject symlink/junction redirection before creating directories or files.
    if REPORT_DIR.resolve() != REPORT_DIR.absolute():
        raise ValueError("Report directory must not redirect through a symlink/junction")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"classical_v3_{report['mode']}_{uuid.uuid4().hex}.json"
    destination = REPORT_DIR / name
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return destination


def run_durable_experiment(mode="smoke", seed=42):
    journal = RunJournal(mode, seed)
    print(f"Run directory: {journal.path}", flush=True)
    try:
        report = run_experiment(mode, seed, journal=journal)
        journal.finish(report)
    except BaseException as error:
        try:
            journal.fail(error)
        except BaseException as status_error:
            # Disk loss can also prevent writing failure status. Never mask the
            # original failure or manufacture a COMPLETE marker.
            print(f"Unable to write failure status: {status_error}", file=sys.stderr)
        raise
    return report, journal.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report, run_path = run_durable_experiment(args.mode, args.seed)
    print(f"Saved {args.mode} summary: {run_path / 'summary.json'}")
    print(report["metrics_scope"])
    for name, record in report["models"].items():
        print(f"{name}: CV={record['cv_accuracy_mean']:.6f} +/- {record['cv_accuracy_std']:.6f}; "
              f"test={record['evaluation']['test_accuracy']:.6f}")


if __name__ == "__main__":
    main()
