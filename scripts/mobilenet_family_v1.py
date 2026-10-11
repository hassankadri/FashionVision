"""One fresh ImageNet MobileNetV3Small run on the frozen inferred-family benchmark."""
from datetime import datetime, timezone
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
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts import mobilenet_fashion_v1 as baseline
from scripts.mobilenet_fashion_v1 import (
    CLASS_NAMES, SEED, IMAGE_SIZE, BATCH_SIZE, sha256, write_json, load_manifest,
    assert_disjoint, training_class_weights, build_model, enable_fine_tuning,
    make_dataset, evaluate_predictions,
)

MANIFESTS = ROOT / "experiments/fashion_products_family_v1/runs/family_v1_a1540a6a473445bea044159eb9ae66a6"
FAMILY_COMPLETE_SHA256 = "24a5520f30a2b889e579f1297cacf2543e512a0c77be1727ba26d173a656b130"
RECIPE = ROOT / "experiments/mobilenet_fashion_v1/runs/mobilenet_v1_e636ace068d64476ac35580eead11462/configuration.json"
RECIPE_SHA256 = "c5ae7b02bcb52797c6545e10442adb1af2731ae30f4e242d75cfb5141421104d"
SCHEMA = "fashion-products-family-mobilenet/1.0"
STRATEGY_FIELDS = (
    "model", "weights", "alpha", "seed", "image_size", "batch_size", "class_names",
    "class_weight_formula", "stage1", "stage2", "optimizer", "loss", "selection",
    "early_stopping", "augmentation", "preprocessing", "validation_test",
    "test_policy", "deterministic_ops", "tf_threads",
)


def assert_original_strategy(configuration):
    # Read settings only. Never load the old checkpoint or old predictions.
    if sha256(RECIPE) != RECIPE_SHA256:
        raise ValueError("Original recipe provenance changed")
    original = json.loads(RECIPE.read_text())
    if sha256(baseline.__file__) != original["script_sha256"]:
        raise ValueError("Shared original architecture/training helpers changed")
    for key in STRATEGY_FIELDS:
        if json.loads(json.dumps(configuration[key])) != original[key]:
            raise ValueError(f"Training strategy differs from original: {key}")
    if configuration["versions"] != original["versions"]:
        raise ValueError("Dependency versions differ from the original baseline")


def validate_group_manifest(rows, groups, split):
    expected = {identity: group for group in groups.values() if group["split"] == split
                for identity in group["image_ids"]}
    if len(rows) != len(expected) or {r["image_id"] for r in rows} != set(expected):
        raise ValueError("Manifest membership differs from frozen family groups")
    for row in rows:
        group = expected[row["image_id"]]
        if (row["group_id"] != group["group_id"] or row["split"] != group["split"]
                or row["class_id"] != group["class_id"]
                or row["group_identity_status"] != "INFERRED_UNVERIFIED"):
            raise ValueError("Family group/class/split mismatch")
    return rows


def assert_group_disjoint(*splits):
    assert_disjoint(*splits)
    seen = set()
    for rows in splits:
        groups = {r["group_id"] for r in rows}
        if seen & groups:
            raise ValueError("Cross-split family-group leakage")
        seen.update(groups)


def read_preparation():
    if sha256(MANIFESTS / "COMPLETE.json") != FAMILY_COMPLETE_SHA256:
        raise ValueError("Frozen family benchmark completion hash changed")
    complete = json.loads((MANIFESTS / "COMPLETE.json").read_text())
    if complete["status"] != "COMPLETE":
        raise ValueError("Family benchmark is incomplete")
    for name, expected in complete["files"].items():
        if sha256(MANIFESTS / name) != expected:
            raise ValueError(f"Frozen family benchmark file changed: {name}")
    configuration = json.loads((MANIFESTS / "configuration.json").read_text())
    audit = json.loads((MANIFESTS / "audit.json").read_text())
    if not audit["ready_for_training"] or tuple(configuration["class_names"]) != CLASS_NAMES:
        raise ValueError("Family benchmark readiness/class ordering mismatch")
    if audit["split_counts"] != {"train": 20364, "validation": 2545, "test": 2546}:
        raise ValueError("Frozen family sample counts changed")
    records = [json.loads(line) for line in (MANIFESTS / "groups.jsonl").read_text().splitlines()]
    groups = {group["group_id"]: group for group in records}
    members = [identity for group in records for identity in group["image_ids"]]
    if len(groups) != len(records) or len(members) != len(set(members)) or len(members) != 25455:
        raise ValueError("Invalid family group partition")
    classes = [{"class_id": c["class_id"], "class_name": c["class_name"],
                "split_counts": {split: c[split] for split in ("train", "validation", "test")}}
               for c in audit["per_class"]]
    preparation = {"dataset_root": configuration["dataset_root"], "classes": classes,
                   "split_counts": audit["split_counts"], "groups": groups,
                   "limitations": audit["limitations"]}
    return preparation, complete


def run():
    run_dir = ROOT / "experiments/mobilenet_family_v1/runs" / f"mobilenet_family_v1_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"RUN_DIRECTORY={run_dir}", flush=True)
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
        rows = load_manifest(MANIFESTS / f"{split}.jsonl", preparation["dataset_root"], split,
                             {c["class_id"]: c["split_counts"][split] for c in preparation["classes"]},
                             complete["files"][f"{split}.jsonl"])
        return validate_group_manifest(rows, preparation["groups"], split)

    train, validation = load("train"), load("validation")
    assert_group_disjoint(train, validation)
    class_weights = training_class_weights(train)
    config = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "model": "MobileNetV3Small", "weights": "imagenet", "alpha": 1.0,
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "class_names": CLASS_NAMES,
        "data_counts": preparation["split_counts"], "manifest_directory": str(MANIFESTS),
        "manifest_hashes": complete["files"], "dataset_root": preparation["dataset_root"],
        "family_benchmark_complete_sha256": FAMILY_COMPLETE_SHA256,
        "initialization_policy": "fresh official ImageNet weights; old trained checkpoint never loaded",
        "original_recipe_sha256": RECIPE_SHA256,
        "shared_architecture_helpers_sha256": sha256(baseline.__file__),
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
    assert_original_strategy(config)
    write_json(run_dir / "configuration.json", config)
    print("Building ImageNet-pretrained MobileNetV3Small", flush=True)
    model, backbone = build_model(weights="imagenet")
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
    assert_group_disjoint(train, validation, test)
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
                            class_names=np.array(CLASS_NAMES),
                            group_ids=np.array([row["group_id"] for row in test]))
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
