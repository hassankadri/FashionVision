"""Methodology tests for new experiments; no production artifact is loaded."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

spec = importlib.util.spec_from_file_location(
    "classical_v3",
    Path(__file__).resolve().parents[1] / "scripts/classical_v3.py",
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


class ClassicalV3Tests(unittest.TestCase):
    def test_all_searches_contain_unfitted_pca_and_stratified_cv(self):
        for mode in ("smoke", "full"):
            searches = v3.make_searches(mode, 42)
            self.assertEqual(
                set(searches),
                {"Logistic Regression", "Random Forest", "Hist Gradient Boosting"},
            )
            for search in searches.values():
                self.assertIsInstance(search.estimator, Pipeline)
                self.assertEqual(list(search.estimator.named_steps), ["pca", "classifier"])
                pca = search.estimator.named_steps["pca"]
                self.assertIsInstance(pca, PCA)
                self.assertFalse(hasattr(pca, "components_"))
                self.assertEqual(search.cv.__class__.__name__, "StratifiedKFold")
                self.assertTrue(search.cv.shuffle)
                self.assertEqual(search.cv.random_state, 42)
                self.assertTrue(
                    all(
                        key.startswith("classifier__")
                        for key in search.param_distributions
                    )
                )

    def test_pca_fit_rows_are_exact_training_folds_then_refit(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(40, 6))

        # Row identities are checked at PCA fit, not at prediction time.
        X[:, 0] = np.arange(40)
        y = np.tile(np.arange(10), 4)
        original = PCA.fit_transform

        for name, search in v3.make_searches("smoke", 42).items():
            with self.subTest(model=name):
                observed = []

                def spy(pca, values, *args, **kwargs):
                    observed.append(set(values[:, 0].astype(int)))
                    return original(pca, values, *args, **kwargs)

                expected = [set(train) for train, _ in search.cv.split(X, y)]
                with patch.object(PCA, "fit_transform", spy), threadpool_limits(limits=1):
                    search.fit(X, y)

                self.assertEqual(observed, expected + [set(range(40))])
                # Original template remains unfitted; clones did the work.
                self.assertFalse(
                    hasattr(search.estimator.named_steps["pca"], "components_")
                )

    def test_test_split_is_loaded_only_after_training_selection(self):
        events = []
        train = np.full((40, 784), 12, dtype=np.uint8)
        test = np.full((20, 784), 200, dtype=np.uint8)
        y_train = np.tile(np.arange(10), 4)
        y_test = np.tile(np.arange(10), 2)

        def load(split):
            events.append(split)
            return (train, y_train) if split == "train" else (test, y_test)

        def fit(X, y, mode, seed):
            self.assertEqual(events, ["train"])
            np.testing.assert_array_equal(y, y_train)
            np.testing.assert_array_equal(X, train.astype(np.float32) / 255)
            events.append("selected-by-cv")
            return (
                {},
                {"cv-winner": {"cv_accuracy_mean": 0.9, "selected_hyperparameters": {}}},
                "cv-winner",
            )

        def evaluate(searches, X, y, indices, *, full):
            self.assertEqual(events, ["train", "selected-by-cv", "test"])
            np.testing.assert_array_equal(X, test.astype(np.float32) / 255)
            np.testing.assert_array_equal(y, y_test)
            np.testing.assert_array_equal(indices, np.arange(len(y_test)))
            self.assertFalse(full)
            return {"cv-winner": {"test_accuracy": 0.0, "unified_report": {}}}

        with (
            patch.object(v3, "load_split", side_effect=load),
            patch.object(v3, "fit_candidates", side_effect=fit),
            patch.object(v3, "evaluate_candidates", side_effect=evaluate),
        ):
            report = v3.run_experiment()

        self.assertEqual(report["selected_model_by_cv"], "cv-winner")
        self.assertFalse(report["full_benchmark"])

    def test_selection_uses_cv_even_when_test_winner_differs(self):
        searches = {}

        for name, score in [("cv-winner", 0.9), ("other", 0.1)]:
            search = Mock(
                best_score_=score,
                best_index_=0,
                n_splits_=2,
                refit_time_=0.01,
                best_params_={"classifier__C": 1},
            )
            search.cv_results_ = {
                "params": [{}],
                "mean_test_score": [score],
                "std_test_score": [0.0],
                "split0_test_score": [score],
                "split1_test_score": [score],
            }
            search.best_estimator_.named_steps = {"pca": Mock(n_components_=2)}
            search.best_estimator_.classes_ = np.arange(10)
            search.best_estimator_.predict.return_value = np.array(
                [1, 0] if name == "cv-winner" else [0, 1]
            )
            search.best_estimator_.predict_proba.return_value = np.eye(10)[
                search.best_estimator_.predict.return_value
            ]
            searches[name] = search

        with patch.object(v3, "make_searches", return_value=searches):
            fitted, _, selected = v3.fit_candidates(
                np.zeros((4, 3)), np.array([0, 1, 0, 1]), "smoke", 42
            )

        # Synthetic selection test isolates dataset identity; real identity is
        # exercised by the integration tests below and the evaluator suite.
        with (
            patch.object(v3, "verify_test_identity"),
            patch.object(
                v3.unified_evaluation,
                "verify_test_identity",
                return_value={"scope": "synthetic"},
            ),
        ):
            evaluation = v3.evaluate_candidates(
                fitted, np.ones((2, 3)), np.array([0, 1]), np.arange(2)
            )

        self.assertEqual(selected, "cv-winner")
        self.assertEqual(evaluation["other"]["test_accuracy"], 1.0)

        for search in searches.values():
            search.fit.assert_called_once()
            search.best_estimator_.fit.assert_not_called()

    def test_sampling_is_deterministic_and_keeps_source_split(self):
        X = np.arange(100 * 4, dtype=np.uint8).reshape(100, 4)
        y = np.tile(np.arange(10), 10)
        first = v3.select_rows(X, y, 30, 42)
        second = v3.select_rows(X, y, 30, 42)

        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

        np.testing.assert_array_equal(first[1], y[first[2]])
        np.testing.assert_array_equal(np.bincount(first[1]), np.full(10, 3))
        np.testing.assert_array_equal(
            v3.select_rows(X, y, None, 42)[2], np.arange(100)
        )

    def test_report_creation_is_exclusive_and_outside_protected_roots(self):
        self.assertEqual(v3.REPORT_DIR, v3.ROOT / "experiments/classical_v3/runs")

        with (
            TemporaryDirectory() as directory,
            patch.object(v3, "REPORT_DIR", Path(directory)),
            patch.object(v3.uuid, "uuid4", return_value=Mock(hex="fixed")),
        ):
            report = {"mode": "smoke", "value": 1}
            path = v3.save_report(report)
            before = path.read_bytes()

            with self.assertRaises(FileExistsError):
                v3.save_report({"mode": "smoke", "value": 2})

            self.assertEqual(path.read_bytes(), before)

    def fixture_search(self, probability=True):
        # Frozen fake model, real test labels/indices: no training needed.
        y = v3.load_split("test")[1][:2]
        methods = ["classes_", "predict", "fit"] + (
            ["predict_proba"] if probability else []
        )
        model = Mock(spec=methods)
        model.classes_ = np.arange(10)
        model.predict.return_value = y.copy()

        if probability:
            model.predict_proba.return_value = np.eye(10)[y]

        return {"test-model": Mock(best_estimator_=model)}, model, y

    def test_unified_integration_retains_predictions_and_times_both_calls(self):
        searches, model, y = self.fixture_search()

        with (
            patch.object(v3.time, "perf_counter", side_effect=[1.0, 1.4]),
            patch.object(
                v3.unified_evaluation,
                "calculate_metrics",
                wraps=v3.unified_evaluation.calculate_metrics,
            ) as metrics,
        ):
            result = v3.evaluate_candidates(
                searches, np.zeros((2, 784)), y, np.arange(2)
            )["test-model"]

        metrics.assert_called_once()
        bundle = result["prediction_bundle"]

        self.assertEqual(bundle["y_true"], y.tolist())
        self.assertEqual(result["predicted_labels"], y.tolist())
        self.assertEqual(bundle["test_indices"], [0, 1])
        self.assertEqual(bundle["class_names"], v3.CLASS_NAMES)
        self.assertEqual(np.shape(bundle["values"]), (2, 10))
        self.assertAlmostEqual(bundle["inference_duration_seconds"], 0.4)
        self.assertEqual(result["unified_report"]["prediction_origin"], "new_inference")
        self.assertEqual(result["unified_report"]["dataset"]["scope"], "test-subset")
        self.assertEqual(result["test_accuracy"], 1.0)
        self.assertEqual(result["top_3_accuracy"], 1.0)

        # Retained bundle can be passed to the unchanged evaluator directly.
        replay = v3.unified_evaluation.build_report(bundle)
        self.assertEqual(replay["metrics"], result["unified_report"]["metrics"])
        model.predict.assert_called_once()
        model.predict_proba.assert_called_once()
        model.fit.assert_not_called()

    def test_label_only_models_use_shared_evaluator_without_top3(self):
        searches, model, y = self.fixture_search(probability=False)
        result = v3.evaluate_candidates(
            searches, np.zeros((2, 784)), y, np.arange(2)
        )["test-model"]

        self.assertEqual(result["prediction_bundle"]["predicted_labels"], y.tolist())
        self.assertNotIn("values", result["prediction_bundle"])
        self.assertIsNone(result["top_3_accuracy"])
        model.fit.assert_not_called()

    def test_invalid_outputs_are_rejected_before_metric_calculation(self):
        for problem in (
            "classes",
            "label_length",
            "label_nan",
            "probability_length",
            "probability_classes",
            "nan",
            "inf",
            "argmax",
        ):
            with self.subTest(problem=problem):
                searches, model, y = self.fixture_search()

                if problem == "classes":
                    model.classes_ = np.arange(10)[::-1]
                elif problem == "label_length":
                    model.predict.return_value = y[:1]
                elif problem == "label_nan":
                    model.predict.return_value = np.array([float("nan"), 0])
                elif problem == "probability_length":
                    model.predict_proba.return_value = np.eye(10)[y[:1]]
                elif problem == "probability_classes":
                    model.predict_proba.return_value = np.zeros((2, 9))
                elif problem in ("nan", "inf"):
                    model.predict_proba.return_value[0, 0] = float(problem)
                else:
                    model.predict.return_value = (y + 1) % 10

                with patch.object(v3.unified_evaluation, "calculate_metrics") as metrics:
                    with self.assertRaises(ValueError):
                        v3.evaluate_candidates(searches, np.zeros((2, 784)), y, np.arange(2))
                    metrics.assert_not_called()

    def test_test_identity_and_full_count_are_checked_before_inference(self):
        searches, model, y = self.fixture_search()
        cases = [
            (np.zeros((1, 784)), y, [0, 1], False),
            (np.zeros((2, 784)), y, [0], False),
            (np.zeros((2, 784)), y, [1, 0], False),
            (np.zeros((2, 784)), y, [0, 1], True),
        ]

        for X, truth, indices, full in cases:
            with self.subTest(indices=indices, full=full), self.assertRaises(ValueError):
                v3.evaluate_candidates(searches, X, truth, indices, full=full)

        model.predict.assert_not_called()
        model.predict_proba.assert_not_called()

    def persistence_searches(self, failure_stage):
        pixels, y = v3.load_split("test")
        searches = {}

        for index, name in enumerate(("model-1", "model-2")):
            search = Mock(
                best_score_=0.9 - index * 0.1,
                best_index_=0,
                n_splits_=2,
                refit_time_=0.01,
                best_params_={"classifier__C": 1},
            )
            search.cv_results_ = {
                "params": [{}],
                "mean_test_score": [0.9 - index * 0.1],
                "std_test_score": [0.0],
                "split0_test_score": [0.9],
                "split1_test_score": [0.9],
            }
            model = search.best_estimator_
            model.named_steps = {"pca": Mock(n_components_=2)}
            model.classes_ = np.arange(10)
            model.predict.return_value = y[:2]
            model.predict_proba.return_value = np.eye(10)[y[:2]]

            if index == 1:
                if failure_stage == "fit":
                    search.fit.side_effect = RuntimeError("simulated later fit failure")
                else:
                    model.predict.side_effect = RuntimeError("simulated later prediction failure")

            searches[name] = search

        return searches, pixels[:2], y[:2]

    def test_completed_model_survives_later_model_failure(self):
        searches, pixels, y = self.persistence_searches("predict")

        with (
            TemporaryDirectory() as directory,
            patch.object(v3, "REPORT_DIR", Path(directory)),
            patch.object(v3, "make_searches", return_value=searches),
            patch.object(v3, "load_split", return_value=(pixels, y)),
        ):
            with self.assertRaisesRegex(RuntimeError, "later prediction"):
                v3.run_durable_experiment()

            run = next(Path(directory).iterdir())
            state = json.loads(sorted(run.glob("state_*.json"))[-1].read_text())
            self.assertEqual(state["status"], "PARTIAL")
            self.assertEqual(state["models"]["model-1"]["status"], "COMPLETE")

            result = json.loads((run / "model_01_result.json").read_text())
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["model_name"], "model-1")
            self.assertTrue((run / result["prediction_report_reference"]).is_file())
            self.assertEqual(len(result["prediction_bundle"]["y_true"]), 2)
            self.assertEqual(state["models"]["model-2"]["status"], "CV_COMPLETE")
            self.assertFalse((run / "summary.json").exists())
            self.assertFalse(
                any(
                    json.loads(p.read_text())["status"] == "COMPLETE"
                    for p in run.glob("state_*.json")
                )
            )

    def test_later_fit_failure_preserves_cv_without_claiming_evaluation_complete(self):
        searches, pixels, y = self.persistence_searches("fit")

        with (
            TemporaryDirectory() as directory,
            patch.object(v3, "REPORT_DIR", Path(directory)),
            patch.object(v3, "make_searches", return_value=searches),
            patch.object(v3, "load_split", return_value=(pixels, y)) as load,
        ):
            with self.assertRaisesRegex(RuntimeError, "later fit"):
                v3.run_durable_experiment()

            run = next(Path(directory).iterdir())
            state = json.loads(sorted(run.glob("state_*.json"))[-1].read_text())
            result = json.loads((run / "model_01_cv.json").read_text())

            self.assertEqual(result["status"], "CV_COMPLETE")
            self.assertIsNone(result["prediction_report_reference"])
            self.assertEqual(state["status"], "PARTIAL")
            load.assert_called_once_with("train")

    def test_first_failure_is_failed_not_complete(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(v3, "REPORT_DIR", Path(directory)),
            patch.object(
                v3, "run_experiment", side_effect=RuntimeError("first failure")
            ),
        ):
            with self.assertRaises(RuntimeError):
                v3.run_durable_experiment()

            run = next(Path(directory).iterdir())
            state = json.loads(sorted(run.glob("state_*.json"))[-1].read_text())
            self.assertEqual(state["status"], "FAILED")
            self.assertEqual(state["models"], {})

    def test_atomic_publication_refuses_overwrite_and_ignores_temporary_files(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "historical.json"
            v3.atomic_json_new(path, {"original": True})
            before = path.read_bytes()

            with self.assertRaises(FileExistsError):
                v3.atomic_json_new(path, {"replacement": True})

            self.assertEqual(path.read_bytes(), before)
            incomplete = Path(directory) / "incomplete.json"

            with patch.object(v3.os, "link", side_effect=OSError("interrupted publication")):
                with self.assertRaises(OSError):
                    v3.atomic_json_new(incomplete, {"status": "COMPLETE"})

            self.assertFalse(incomplete.exists())
            self.assertTrue(list(Path(directory).glob("*.tmp")))
            self.assertEqual(list(Path(directory).glob("*.json")), [path])

    def test_run_cannot_complete_with_unfinished_models(self):
        with TemporaryDirectory() as directory, patch.object(v3, "REPORT_DIR", Path(directory)):
            journal = v3.RunJournal("smoke", 42)
            journal.cv_complete("model-1", {"cv_accuracy_mean": 0.9})

            with self.assertRaises(ValueError):
                journal.finish({"models": {"model-1": {}}})

            self.assertFalse((journal.path / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()