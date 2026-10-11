"""One two-stage MobileNetV3 baseline on the immutable Phase 3B manifests."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fashion_rgb_data.pipeline import _decode

MANIFESTS = ROOT / "experiments/fashion_products_v1/runs/fashion_products_2ec9a5867e8c4c259b77106889d9c375"
PREPARATION_SHA256 = "5306aca6075045f77c66d88c71ead8c48328ef3232bf6f70bc2a6400215884fe"
CLASS_NAMES = ("T-shirts", "Shirts", "Kurtas", "Pants", "Closed shoes", "Handbags",
               "Watches", "Sunglasses", "Backpacks", "Belts")
SEED = 42
IMAGE_SIZE = 160
BATCH_SIZE = 32
SCHEMA = "fashion-products-mobilenet/1.0"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Publish new reports only; never replace a historical report."""
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def read_preparation():
    if sha256(MANIFESTS / "preparation.json") != PREPARATION_SHA256:
        raise ValueError("The fixed Phase 3B preparation has changed")
    preparation = json.loads((MANIFESTS / "preparation.json").read_text())
    complete = json.loads((MANIFESTS / "COMPLETE.json").read_text())
    if complete["status"] != "COMPLETE":
        raise ValueError("Phase 3B is not complete")
    if tuple(c["class_name"] for c in preparation["classes"]) != CLASS_NAMES:
        raise ValueError("Class ordering differs from Phase 3B")
    if [c["class_id"] for c in preparation["classes"]] != list(range(10)):
        raise ValueError("Noncanonical class IDs")
    for split in ("train", "validation", "test"):
        if sha256(MANIFESTS / f"{split}.jsonl") != complete["files"][f"{split}.jsonl"]:
            raise ValueError(f"Fixed {split} manifest hash mismatch")
    return preparation, complete


def load_manifest(path, dataset_root, split, expected_counts, expected_sha256):
    """Keep manifest order/identity; reject edits, bad labels, and missing images."""
    path, dataset_root = Path(path), Path(dataset_root).resolve()
    if split not in ("train", "validation", "test") or sha256(path) != expected_sha256:
        raise ValueError("Invalid split or modified manifest")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    seen = {key: set() for key in ("image_id", "file_sha256", "pixel_sha256")}
    for row in rows:
        label = row["class_id"]
        if (type(label) is not int or not 0 <= label < len(CLASS_NAMES)
                or row["class_name"] != CLASS_NAMES[label] or row["split"] != split):
            raise ValueError("Class or split mapping mismatch")
        source = (dataset_root / row["relative_path"]).resolve()
        if (not source.is_relative_to(dataset_root) or not source.is_file()
                or source != Path(row["original_path"]).resolve()
                or source.stem != row["image_id"]
                or str(row["source_metadata"]["id"]) != row["image_id"]):
            raise ValueError("Missing image or changed original image identity")
        for key, values in seen.items():
            if row[key] in values:
                raise ValueError(f"Duplicate {key} in manifest")
            values.add(row[key])
    if Counter(row["class_id"] for row in rows) != Counter(expected_counts):
        raise ValueError("Manifest class counts differ from fixed preparation")
    return rows


def assert_disjoint(*splits):
    for key in ("image_id", "file_sha256", "pixel_sha256"):
        seen = set()
        for rows in splits:
            values = {row[key] for row in rows}
            if seen & values:
                raise ValueError(f"Cross-split leakage: {key}")
            seen.update(values)


def training_class_weights(rows):
    counts = Counter(row["class_id"] for row in rows)
    if set(counts) != set(range(len(CLASS_NAMES))):
        raise ValueError("All ten classes must occur in training")
    return {label: len(rows) / (len(CLASS_NAMES) * counts[label]) for label in range(10)}


def build_model(weights="imagenet"):
    import keras
    backbone = keras.applications.MobileNetV3Small(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), alpha=1.0, include_top=False,
        weights=weights, pooling="avg", include_preprocessing=True,
        name="mobilenetv3small_backbone")
    backbone.trainable = False
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="rgb_pixels_0_255")
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomRotation(0.03, fill_mode="reflect", seed=SEED + 1),
        keras.layers.RandomZoom(0.1, fill_mode="reflect", seed=SEED + 2),
        keras.layers.RandomContrast(0.1, value_range=(0, 255), seed=SEED + 3),
    ], name="training_only_augmentation")
    # Keep pretrained batch-normalization statistics fixed during fine-tuning.
    features = backbone(augmentation(inputs), training=False)
    features = keras.layers.Dropout(0.2, seed=SEED, name="head_dropout")(features)
    outputs = keras.layers.Dense(10, activation="softmax", name="class_probabilities")(features)
    return keras.Model(inputs, outputs, name="fashion_products_mobilenet_v1"), backbone


def enable_fine_tuning(backbone):
    import keras
    start = next(i for i, layer in enumerate(backbone.layers)
                 if layer.name == "expanded_conv_8_expand")
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= start and not isinstance(layer, keras.layers.BatchNormalization)
    return [layer.name for layer in backbone.layers if layer.trainable and layer.weights]


def make_dataset(rows, training=False):
    import tensorflow as tf
    from PIL import Image
    paths = [row["original_path"] for row in rows]
    hashes = [row["file_sha256"] for row in rows]
    labels = np.array([row["class_id"] for row in rows], dtype=np.int32)

    def decode(path, expected_hash):
        raw = Path(path.decode("utf-8")).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash.decode("ascii"):
            raise ValueError("Original image bytes differ from Phase 3B")
        image, _ = _decode(raw)
        return np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR),
                          dtype=np.float32)

    def load(path, expected_hash, label):
        image = tf.numpy_function(decode, [path, expected_hash], tf.float32)
        image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, hashes, labels))
    if training:
        dataset = dataset.shuffle(len(rows), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(load, num_parallel_calls=4, deterministic=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 4
    return dataset.batch(BATCH_SIZE).with_options(options).prefetch(1)


def evaluate_predictions(rows, probabilities):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    labels = np.array([row["class_id"] for row in rows], dtype=np.int32)
    probabilities = np.asarray(probabilities)
    if (probabilities.shape != (len(rows), 10) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Malformed probability matrix")
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=list(range(10)), zero_division=0)
    result = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "top3_accuracy": float(np.mean(np.any(
            np.argsort(-probabilities, axis=1, kind="stable")[:, :3] == labels[:, None], axis=1))),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(10))).tolist(),
        "per_class": [{"class_id": i, "class_name": CLASS_NAMES[i], "precision": float(precision[i]),
                       "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i in range(10)],
    }
    for prefix, weights in (("macro", None), ("weighted", support)):
        for key, values in (("precision", precision), ("recall", recall), ("f1", f1)):
            result[f"{prefix}_{key}"] = float(np.average(values, weights=weights))
    return result


def run():
    run_dir = ROOT / "experiments/mobilenet_fashion_v1/runs" / f"mobilenet_v1_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"RUN_DIRECTORY={run_dir}", flush=True)
    # Cache downloaded ImageNet initialization only within this new experiment.
    os.environ["KERAS_HOME"] = str(run_dir / "keras_cache")
    os.environ["MPLCONFIGDIR"] = str(run_dir / "matplotlib_cache")
    os.environ["KERAS_BACKEND"] = "tensorflow"
    try:
        train_and_evaluate(run_dir)
    except BaseException as error:
        write_json(run_dir / "FAILED.json", {"status": "FAILED", "error": repr(error),
                                             "traceback": traceback.format_exc()})
        raise


def train_and_evaluate(run_dir):
    import tensorflow as tf
    import keras
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    preparation, complete = read_preparation()

    def load(split):
        return load_manifest(MANIFESTS / f"{split}.jsonl", preparation["dataset_root"], split,
                             {c["class_id"]: c["split_counts"][split] for c in preparation["classes"]},
                             complete["files"][f"{split}.jsonl"])

    train, validation = load("train"), load("validation")
    assert_disjoint(train, validation)
    class_weights = training_class_weights(train)
    config = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "model": "MobileNetV3Small", "weights": "imagenet", "alpha": 1.0,
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "class_names": CLASS_NAMES,
        "data_counts": preparation["split_counts"], "manifest_directory": str(MANIFESTS),
        "manifest_hashes": complete["files"], "dataset_root": preparation["dataset_root"],
        "source_inventory_sha256": preparation["source_inventory_sha256"],
        "class_weights": class_weights, "class_weight_formula": "N_train / (10 * train_class_count)",
        "stage1": {"max_epochs": 5, "learning_rate": 0.001, "backbone": "frozen"},
        "stage2": {"max_epochs": 8, "learning_rate": 0.0001,
                   "unfreeze_from": "expanded_conv_8_expand", "batch_normalization": "frozen"},
        "optimizer": "Adam", "loss": "sparse_categorical_crossentropy",
        "selection": "maximum unweighted validation accuracy across both stages; earliest tie",
        "early_stopping": {"monitor": "val_accuracy", "patience": 2, "min_delta": 0},
        "augmentation": {"horizontal_flip": True, "rotation_fraction": 0.03,
                         "zoom_fraction": 0.1, "contrast_factor": 0.1, "fill_mode": "reflect"},
        "preprocessing": "Phase 3A RGB decode; Pillow bilinear resize; float32 [0,255]; built-in MobileNetV3 [-1,1] rescaling",
        "validation_test": "deterministic, no augmentation/resampling/class weighting",
        "test_policy": "load test images only after both stages and checkpoint selection are finished; evaluate once",
        "deterministic_ops": True, "tf_threads": {"intra": 4, "inter": 2, "data": 4},
        "devices": [str(device) for device in tf.config.list_physical_devices()],
        "python": platform.python_version(), "executable": sys.executable,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("tensorflow", "keras", "numpy", "scikit-learn", "pillow")},
        "script_sha256": sha256(__file__),
    }
    write_json(run_dir / "configuration.json", config)
    print("Building ImageNet-pretrained MobileNetV3Small", flush=True)
    model, backbone = build_model()
    write_json(run_dir / "initialization.json", {
        "total_parameters": model.count_params(),
        "stage1_trainable_parameters": sum(int(np.prod(v.shape)) for v in model.trainable_weights),
        "pretrained_assets": {p.relative_to(run_dir).as_posix(): sha256(p)
                              for p in (run_dir / "keras_cache").rglob("*.h5")},
    })
    train_ds, val_ds = make_dataset(train, True), make_dataset(validation)
    history = []

    class Recorder(keras.callbacks.Callback):
        best = -float("inf")
        best_epoch = None
        stage = None

        def on_epoch_begin(self, epoch, logs=None):
            self.started = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            record = {"stage": self.stage, "stage_epoch": epoch + 1,
                      "global_epoch": len(history) + 1, "seconds": time.perf_counter() - self.started,
                      **{key: float(value) for key, value in logs.items()}}
            if not all(np.isfinite(record[key]) for key in ("loss", "accuracy", "val_loss", "val_accuracy")):
                raise ValueError("Nonfinite training metrics")
            history.append(record)
            if record["val_accuracy"] > self.best:
                # Only this run's live checkpoint is replaced; previous experiments are never opened for writing.
                temporary = run_dir / "best.pending.keras"
                self.model.save(temporary)
                os.replace(temporary, run_dir / "best.keras")
                self.best = record["val_accuracy"]
                self.best_epoch = record["global_epoch"]
            write_json(run_dir / f"epoch_{record['global_epoch']:02d}.json", record)
            print(json.dumps(record), flush=True)

    recorder = Recorder()
    stage_durations = {}
    training_started = time.perf_counter()
    for stage in ("stage1", "stage2"):
        if stage == "stage2":
            tuned = enable_fine_tuning(backbone)
            write_json(run_dir / "fine_tuning.json", {
                "trainable_backbone_layers": tuned,
                "trainable_parameters": sum(int(np.prod(v.shape)) for v in model.trainable_weights),
                "initialization": "best stage1 validation checkpoint restored by EarlyStopping",
            })
        model.compile(optimizer=keras.optimizers.Adam(config[stage]["learning_rate"]),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
        recorder.stage = stage
        early = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=2,
                                              restore_best_weights=True, verbose=1)
        print(f"Starting {stage}", flush=True)
        started = time.perf_counter()
        model.fit(train_ds, validation_data=val_ds, epochs=config[stage]["max_epochs"],
                  class_weight=class_weights, callbacks=[recorder, early], verbose=2)
        stage_durations[stage] = time.perf_counter() - started
    training_seconds = time.perf_counter() - training_started
    write_json(run_dir / "training_history.json", {"epochs": history, "stage_seconds": stage_durations,
                                                   "training_seconds": training_seconds})
    selection = {"best_validation_accuracy": recorder.best, "best_global_epoch": recorder.best_epoch,
                 "selected_model": "best.keras", "selected_model_sha256": sha256(run_dir / "best.keras"),
                 "criterion": config["selection"], "test_evaluated": False}
    write_json(run_dir / "selection.json", selection)
    # Selection is durably recorded before opening any test images.
    print("Training finished; checkpoint fixed. Starting the single test evaluation.", flush=True)
    best = keras.models.load_model(run_dir / "best.keras", compile=False)
    test = load("test")
    assert_disjoint(train, validation, test)
    test_ds = make_dataset(test)
    # Warm up on reserved validation data only. Test timings include decoding, batching and inference.
    best(next(iter(val_ds))[0], training=False).numpy()
    probabilities, inference_compute_seconds = [], 0.0
    started = time.perf_counter()
    for images, labels in test_ds:
        compute_started = time.perf_counter()
        probabilities.append(best(images, training=False).numpy())
        inference_compute_seconds += time.perf_counter() - compute_started
    inference_seconds = time.perf_counter() - started
    probabilities = np.concatenate(probabilities)
    metrics = evaluate_predictions(test, probabilities)
    with (run_dir / "test_predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, probabilities=probabilities,
                            y_true=np.array([row["class_id"] for row in test]),
                            predicted_labels=probabilities.argmax(axis=1),
                            image_ids=np.array([row["image_id"] for row in test]),
                            relative_paths=np.array([row["relative_path"] for row in test]),
                            class_names=np.array(CLASS_NAMES))
    report = {
        "schema": SCHEMA, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics, "data_counts": preparation["split_counts"], "class_names": CLASS_NAMES,
        "probability_shape": list(probabilities.shape), "selection": selection,
        "manifest_hashes": complete["files"], "training_seconds": training_seconds,
        "stage_seconds": stage_durations, "parameter_count": best.count_params(),
        "saved_model_bytes": (run_dir / "best.keras").stat().st_size,
        "saved_model_sha256": sha256(run_dir / "best.keras"),
        "inference": {"samples": len(test), "batch_size": BATCH_SIZE,
                      "end_to_end_seconds": inference_seconds,
                      "amortized_end_to_end_ms_per_image": 1000 * inference_seconds / len(test),
                      "model_compute_seconds": inference_compute_seconds,
                      "amortized_model_compute_ms_per_image": 1000 * inference_compute_seconds / len(test),
                      "single_request_latency": "NOT MEASURED", "warmup": "one validation batch"},
        "limitations": preparation["limitations"] + [
            "One seed and one configuration; no repeated tuning or uncertainty estimates",
            "Training accuracy uses augmentation/dropout; validation/test do not",
            "Batched CPU throughput is not single-image deployment latency"],
    }
    write_json(run_dir / "evaluation.json", report)
    write_json(run_dir / "COMPLETE.json", {"status": "COMPLETE", "schema": SCHEMA,
                                          "files": {p.name: sha256(p) for p in run_dir.iterdir() if p.is_file()}})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run()
