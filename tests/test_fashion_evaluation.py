import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fashion_evaluation import CLASS_NAMES
from fashion_evaluation.metrics import calculate_metrics
from fashion_evaluation.predictions import (
    generate_predictions,
    load_precomputed,
    verify_test_identity,
)
from fashion_evaluation import reports


class FashionEvaluationTests(unittest.TestCase):
    def bundle(self):
        return {
            "y_true": [0, 0, 0, 1],
            "predicted_labels": [0, 1, 1, 1],
            "class_names": list(CLASS_NAMES),
            "model_identifier": "synthetic-test",
            "inference_variant": "label-only",
        }

    def test_known_confusion_and_macro_weighted_metrics(self):
        m = calculate_metrics([0, 0, 0, 1], predicted_labels=[0, 1, 1, 1])
        expected = np.zeros((10, 10), dtype=int)
        expected[:2, :2] = [[1, 2], [0, 1]]

        self.assertEqual(m["confusion_matrix"], expected.tolist())
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["top_1_accuracy"], 0.5)
        self.assertIsNone(m["top_3_accuracy"])
        self.assertAlmostEqual(m["macro_precision"], (1 + 1 / 3) / 10)
        self.assertAlmostEqual(m["macro_recall"], (1 / 3 + 1) / 10)
        self.assertAlmostEqual(m["macro_f1"], 0.1)
        self.assertAlmostEqual(m["weighted_precision"], 5 / 6)
        self.assertAlmostEqual(m["weighted_recall"], 0.5)
        self.assertAlmostEqual(m["weighted_f1"], 0.5)
        self.assertEqual(m["per_class"][0]["support"], 3)
        self.assertEqual(m["per_class"][0]["f1"], 0.5)

    def test_top1_top3_probabilities_and_scores(self):
        values = np.zeros((3, 10))
        values[:, :4] = [0.4, 0.3, 0.2, 0.1]
        m = calculate_metrics([0, 2, 3], values=values)

        self.assertAlmostEqual(m["top_1_accuracy"], 1 / 3)
        self.assertAlmostEqual(m["top_3_accuracy"], 2 / 3)
        self.assertEqual(
            m,
            calculate_metrics(
                [0, 2, 3],
                values=values * 10 - 3,
                value_kind="scores",
            ),
        )

    def test_class_column_order_and_ties_are_explicit(self):
        values = np.ones((1, 10)) / 10
        m = calculate_metrics(
            [2], values=values[:, ::-1], class_names=CLASS_NAMES[::-1]
        )

        self.assertEqual(m["top_1_accuracy"], 0)
        self.assertEqual(m["top_3_accuracy"], 1)

        scores = np.eye(10)[[2]]
        self.assertEqual(
            calculate_metrics(
                [2], values=scores[:, ::-1], class_names=CLASS_NAMES[::-1]
            )["accuracy"],
            1,
        )

    def test_malformed_shapes_class_counts_and_nonfinite_values(self):
        for values in (
            [0.1] * 10,
            np.ones((2, 10)),
            np.ones((1, 9)),
            np.ones((1, 11)),
            np.full((1, 10), np.nan),
            np.full((1, 10), np.inf),
            np.ones((1, 10)),
            np.full((1, 10), -0.1),
        ):
            with self.subTest(shape=np.shape(values)), self.assertRaises(ValueError):
                calculate_metrics([0], values=values)

        for names in (CLASS_NAMES[:9], CLASS_NAMES + ("extra",), ("Bag",) * 10):
            with self.assertRaises(ValueError):
                calculate_metrics([0], predicted_labels=[0], class_names=names)

    def test_malformed_labels_and_ambiguous_inputs(self):
        for truth in ([], [10], [-1], [0.5], [[0]], [True]):
            with self.assertRaises(ValueError):
                calculate_metrics(truth, predicted_labels=[0])

        for kwargs in (
            {},
            {"predicted_labels": [0, 1]},
            {"predicted_labels": [0], "values": np.eye(10)[[0]]},
        ):
            with self.assertRaises(ValueError):
                calculate_metrics([0], **kwargs)

    def test_prediction_generation_is_one_inference_call_and_timed(self):
        predictor = Mock(return_value=np.eye(10)[[0, 1]])
        X = np.zeros((2, 784))

        with patch(
            "fashion_evaluation.predictions.time.perf_counter",
            side_effect=[10.0, 10.5],
        ):
            generated = generate_predictions(
                predictor,
                X,
                model_identifier="fake-frozen",
                inference_variant="single",
            )

        predictor.assert_called_once_with(X)
        generated["y_true"] = [0, 1]
        report = reports.build_report(generated, prediction_origin="new_inference")

        self.assertEqual(report["timing"]["inference_duration_seconds"], 0.5)
        self.assertEqual(report["timing"]["average_latency_seconds_per_sample"], 0.25)

    def test_precomputed_origin_and_unknown_latency(self):
        report = reports.build_report(self.bundle())

        self.assertEqual(report["prediction_origin"], "precomputed")
        self.assertEqual(report["metric_origin"], "newly_calculated_from_predictions")
        self.assertFalse(report["historical_metrics_imported"])
        self.assertIsNone(report["timing"]["inference_duration_seconds"])

        for duration in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                reports.build_report(
                    {**self.bundle(), "inference_duration_seconds": duration}
                )

    def test_full_test_requires_exact_coverage_and_matching_labels(self):
        labels = np.tile(np.arange(10), 1000)

        with patch(
            "fashion_evaluation.predictions.load_test_set",
            return_value=(None, labels, {}),
        ):
            provenance = verify_test_identity(labels, np.arange(10000), full=True)
            self.assertEqual(provenance["scope"], "full-test")

            for truth, indices in (
                (labels[:10], np.arange(10)),
                (labels, np.zeros(10000, dtype=int)),
                (np.zeros(10000, dtype=int), np.arange(10000)),
            ):
                with self.assertRaises(ValueError):
                    verify_test_identity(truth, indices, full=True)

        with self.assertRaises(ValueError):
            reports.build_report(self.bundle(), full=True)

    def test_serialization_is_deterministic_for_fixed_report_data(self):
        first = reports.build_report(
            self.bundle(), timestamp="2026-09-28T00:00:00+00:00"
        )
        second = reports.build_report(
            self.bundle(), timestamp="2026-09-28T00:00:00+00:00"
        )

        self.assertEqual(reports.serialize_report(first), reports.serialize_report(second))
        self.assertEqual(json.loads(reports.serialize_report(first)), first)

    def test_report_overwrite_and_path_traversal_protection(self):
        with TemporaryDirectory() as directory, patch.object(
            reports, "REPORT_DIR", Path(directory)
        ):
            report = reports.build_report(self.bundle())
            path = reports.write_report(report, "example.json")
            content = path.read_bytes()

            with self.assertRaises(FileExistsError):
                reports.write_report(report, "example.json")

            self.assertEqual(path.read_bytes(), content)

            for name in ("../escape.json", "sub/report.json", "C:\\outputs\\report.json"):
                with self.assertRaises(ValueError):
                    reports.write_report(report, name)

    def test_precomputed_file_contract(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            path.write_text(json.dumps(self.bundle()), encoding="utf-8")
            bundle, identity = load_precomputed(path)

            self.assertEqual(bundle, self.bundle())
            self.assertEqual(len(identity["sha256"]), 64)

            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_precomputed(path)


if __name__ == "__main__":
    unittest.main()