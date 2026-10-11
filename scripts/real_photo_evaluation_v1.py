"""Independent, manually labeled photo evaluation; never trains or saves a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts import mobilenet_fashion_v1 as baseline
from scripts import mobilenet_family_v1 as family
from fashion_rgb_data.pipeline import _decode, _pixel_hash, EXTENSIONS

CLASS_NAMES = baseline.CLASS_NAMES
MODEL_RUN = ROOT / "models/mobilenet-family-v1"
MODEL_SHA256 = "3d095fbaffef382d7be0068db178a713b79558c06ae97fb043e4f0319b0479a2"
SCHEMA = "independent-real-photo-evaluation/1.0"
SPECIAL = {"AMBIGUOUS", "OUT_OF_SCOPE"}


def known_benchmark_hashes():
    # Check the pinned benchmark before trusting its exclusion inventory.
    family.read_preparation()
    hashes = {"file_sha256": set(), "pixel_sha256": set()}
    for split in ("train", "validation", "test"):
        for line in (family.MANIFESTS / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            for key in hashes:
                hashes[key].add(row[key])
    return hashes


def scan(root, forbidden):
    """Folder labels are human annotations, never predictions. Fail closed on errors."""
    root = Path(root).resolve()
    info = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if (info.get("manually_labeled") is not True or info.get("independent_of_benchmark") is not True
            or any(not isinstance(info.get(k), str) or not info[k].strip()
                   for k in ("dataset_name", "source", "labeler"))):
        raise ValueError("dataset.json requires provenance and explicit manual-label/independence declarations")
    rows, excluded, errors = [], [], []
    seen = {key: set() for key in forbidden}
    for folder in sorted(root.iterdir()):
        if folder.name == "dataset.json":
            continue
        if folder.is_symlink() or not folder.is_dir() or folder.name not in (*CLASS_NAMES, *SPECIAL):
            errors.append({"path": folder.name, "error": "Unknown class or invalid folder"})
            continue
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                errors.append({"path": relative, "error": "Redirected path"})
                continue
            if path.is_dir():
                continue
            try:
                if path.suffix.lower() not in EXTENSIONS:
                    raise ValueError("Unsupported image extension")
                raw = path.read_bytes()
                image, metadata = _decode(raw)
                row = {"relative_path": relative, "original_path": str(path.resolve()),
                       "file_sha256": hashlib.sha256(raw).hexdigest(),
                       "pixel_sha256": _pixel_hash(image), "image_metadata": metadata}
                for key in forbidden:
                    if row[key] in forbidden[key]:
                        raise ValueError("Exact overlap with existing benchmark")
                    if row[key] in seen[key]:
                        raise ValueError("Duplicate independent image")
                for key in forbidden:
                    seen[key].add(row[key])
                if folder.name in SPECIAL:
                    excluded.append({**row, "status": folder.name, "review_status": "UNREVIEWED"})
                else:
                    rows.append({**row, "class_id": CLASS_NAMES.index(folder.name),
                                 "class_name": folder.name})
            except (OSError, ValueError, SyntaxError) as error:
                errors.append({"path": relative, "error": str(error)})
    return {"provenance": info, "provenance_sha256": baseline.sha256(root / "dataset.json"),
            "rows": rows, "excluded": excluded, "errors": errors}


def load_checkpoint():
    complete = json.loads((MODEL_RUN / "COMPLETE.json").read_text())
    config = json.loads((MODEL_RUN / "configuration.json").read_text())
    if complete["status"] != "COMPLETE" or tuple(config["class_names"]) != CLASS_NAMES:
        raise ValueError("Checkpoint completion/class ordering mismatch")
    for name, expected in complete["files"].items():
        if baseline.sha256(MODEL_RUN / name) != expected:
            raise ValueError(f"Checkpoint artifact changed: {name}")
    if baseline.sha256(MODEL_RUN / "best.keras") != MODEL_SHA256:
        raise ValueError("Not the verified family-aware checkpoint")
    if baseline.sha256(baseline.__file__) != config["shared_architecture_helpers_sha256"]:
        raise ValueError("Saved inference preprocessing helpers changed")
    import keras
    model = keras.models.load_model(MODEL_RUN / "best.keras", compile=False)
    if model.input_shape != (None, 160, 160, 3) or model.output_shape != (None, 10):
        raise ValueError("Unexpected model dimensions")
    return model


def predict(model, rows):
    if not rows:
        raise ValueError("No in-scope labeled images")
    return np.concatenate([model(images, training=False).numpy()
                           for images, _ in baseline.make_dataset(rows, training=False)])


def results(rows, probabilities):
    if not rows:
        raise ValueError("No in-scope labeled images")
    for row in rows:
        label = row["class_id"]
        if type(label) is not int or not 0 <= label < 10 or row["class_name"] != CLASS_NAMES[label]:
            raise ValueError("Class ordering mismatch")
    metrics = baseline.evaluate_predictions(rows, probabilities)
    predictions = []
    for row, scores in zip(rows, probabilities):
        predicted = int(np.argmax(scores))
        predictions.append({**row, "predicted_class_id": predicted,
                            "predicted_class_name": CLASS_NAMES[predicted],
                            "probabilities": scores.tolist(), "correct": predicted == row["class_id"],
                            "review_status": "UNREVIEWED"})
    return metrics, predictions


def publish(directory, scan_result, probabilities):
    """Exclusive creation: existing files are never replaced; COMPLETE is written last."""
    directory = Path(directory)
    metrics, predictions = results(scan_result["rows"], probabilities)
    baseline.write_json(directory / "predictions.json", predictions)
    baseline.write_json(directory / "incorrect_predictions.json", [p for p in predictions if not p["correct"]])
    baseline.write_json(directory / "excluded_review.json", scan_result["excluded"])
    baseline.write_json(directory / "evaluation.json", {
        "schema": SCHEMA, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": MODEL_SHA256, "class_names": CLASS_NAMES,
        "sample_count": len(predictions), "metrics": metrics,
        "missing_classes": [c for c in CLASS_NAMES if not any(r["class_name"] == c for r in predictions)],
        "macro_policy": "All ten fixed classes; zero division=0; absent classes are not evidence of performance",
        "excluded_count": len(scan_result["excluded"]), "provenance": scan_result["provenance"],
        "dependencies": {n: importlib.metadata.version(n) for n in ("tensorflow", "keras", "numpy", "pillow", "scikit-learn")},
        "limitations": ["Independence provenance is human-declared; hashes cannot prove collection independence",
                        "Near-duplicate/product-family overlap is not automatically ruled out",
                        "Ambiguous/out-of-scope folders are human annotations, not an automatic rejection detector"]})
    baseline.write_json(directory / "COMPLETE.json", {
        "status": "COMPLETE", "files": {p.name: baseline.sha256(p) for p in directory.iterdir() if p.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    audit = scan(args.dataset, known_benchmark_hashes())
    run = ROOT / "experiments/real_photo_evaluation_v1/runs" / uuid.uuid4().hex
    run.mkdir(parents=True, exist_ok=False)
    baseline.write_json(run / "input_audit.json", audit)
    print(f"RUN_DIRECTORY={run}", flush=True)
    if audit["errors"] or not audit["rows"]:
        baseline.write_json(run / "NOT_RUN.json", {"status": "NOT_RUN", "reason": "Input errors or no labeled in-scope images"})
        raise SystemExit("Evaluation NOT RUN; inspect input_audit.json")
    try:
        publish(run, audit, predict(load_checkpoint(), audit["rows"]))
    except Exception as error:
        baseline.write_json(run / "FAILED.json", {"status": "FAILED", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
