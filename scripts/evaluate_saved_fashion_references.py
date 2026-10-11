"""Calculate subset metrics from saved reference predictions; NO inference rerun.

Run from repository root: python -B -m scripts.evaluate_saved_fashion_references
"""
import json
from pathlib import Path

import numpy as np

from fashion_evaluation import CLASS_NAMES
from fashion_evaluation.predictions import ROOT, file_identity, load_test_set
from fashion_evaluation.reports import build_report, write_report


def read(relative):
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    images, labels, _ = load_test_set()
    lookup = {image.tobytes(): i for i, image in enumerate(images)}
    if len(lookup) != 10000:
        raise ValueError("Reference identity mapping requires unique test images")
    single_path = "tests/browser/validation_cases.json"
    six_path = "tests/browser/tta_validation_cases.json"
    five_path = "tests/fixtures/browser_five_view_tta.json"
    single = read(single_path)
    six = read(six_path)
    five = read(five_path)
    if five["input_fixture_sha256"] != file_identity(ROOT / six_path)["sha256"]:
        raise ValueError("Five-view fixture input hash mismatch")
    variants = [
        ("cnn-single-pass-reference", single, [c["expected_probs"] for c in single], [single_path]),
        ("cnn-standalone-six-view-tta-reference", six, [c["expected_avg_probs"] for c in six], [six_path]),
        ("cnn-browser-five-view-tta-reference", [six[c["source_case"]] for c in five["cases"]],
         [c["expected_probs"] for c in five["cases"]], [five_path, six_path]),
    ]
    for variant, cases, probabilities, sources in variants:
        indices = []
        for case in cases:
            pixels = np.asarray(case["pixels"])
            if pixels.shape != (784,) or not np.isfinite(pixels).all():
                raise ValueError("Malformed reference pixels")
            index = lookup.get(np.rint(pixels * 255).astype(np.uint8).tobytes())
            if index is None or int(labels[index]) != case["true_label"]:
                raise ValueError("Reference image/label does not match original test set")
            np.testing.assert_allclose(pixels, images[index].reshape(-1).astype(np.float32) / 255,
                                       rtol=0, atol=1e-7)
            indices.append(index)
        bundle = {"y_true": [c["true_label"] for c in cases], "values": probabilities,
                  "class_names": list(CLASS_NAMES), "test_indices": indices,
                  "model_identifier": "repository CNN reference predictions",
                  "inference_variant": variant,
                  "artifacts": [file_identity(ROOT / "frontend/react/public/frontend/model_weights.json")],
                  "inference_duration_seconds": None}
        report = build_report(bundle, source_identity={
            "kind": "repository-recorded reference prediction vectors; not rerun",
            "files": [file_identity(ROOT / p) for p in sources]})
        destination = write_report(report)
        print(f"{variant}: {destination}")
        print(json.dumps({key: report["metrics"][key] for key in
                          ("evaluated_samples", "accuracy", "top_3_accuracy", "macro_f1", "weighted_f1")}))


if __name__ == "__main__":
    main()
