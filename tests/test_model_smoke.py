"""Read-only loading smoke tests. Never fit, save, download, or export models."""

import importlib.util
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np
from PIL import Image

from ml import photo_classifier as photo


ROOT = Path(__file__).resolve().parents[1]


class ModelLoadingSmokeTests(unittest.TestCase):
    def test_clip_assets_load_and_run_in_memory(self):
        session = photo._session()
        self.assertEqual(
            {x.name for x in session.get_inputs()},
            {"input_ids", "attention_mask", "pixel_values"},
        )

        labels, prompts, ids, mask = photo._prompt_inputs()
        self.assertEqual(len(prompts), 23)
        self.assertEqual(set(labels), set(photo.CLASS_NAMES))
        self.assertEqual(ids.shape, (23, 77))
        self.assertEqual(mask.shape, ids.shape)
        self.assertEqual(ids.dtype, np.dtype("int64"))

        result = photo.classify_image(Image.new("RGB", (224, 224), "white"))
        self.assertEqual(
            set(result),
            {"prediction", "confidence", "uncertain", "predictions", "model"},
        )

        scores = [row["probability"] for row in result["predictions"]]
        self.assertEqual(len(scores), 10)
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(sum(scores), 1.0, places=6)
        self.assertEqual(scores, sorted(scores, reverse=True))
        # A blank synthetic image tests loading and output shape, not accuracy.

    @unittest.skipUnless(
        importlib.util.find_spec("joblib") and importlib.util.find_spec("sklearn"),
        "joblib and scikit-learn are required to load classical artifacts",
    )
    def test_both_classical_model_and_pca_pairs_load(self):
        import joblib

        for suffix in ("", "_v2"):
            with self.subTest(version=suffix or "v1"):
                pca = joblib.load(ROOT / "outputs" / f"pca_transformer{suffix}.joblib")
                model = joblib.load(
                    ROOT / "outputs" / f"best_classical_model{suffix}.joblib"
                )

                self.assertEqual(type(pca).__name__, "PCA")
                self.assertEqual(type(model).__name__, "HistGradientBoostingClassifier")
                self.assertEqual(pca.n_features_in_, 784)
                self.assertEqual(model.n_features_in_, pca.n_components_)
                np.testing.assert_array_equal(model.classes_, np.arange(10))

                prediction = model.predict(
                    pca.transform(np.zeros((1, 784), dtype=np.float32))
                )
                self.assertEqual(prediction.shape, (1,))
                self.assertIn(int(prediction[0]), range(10))

    def test_all_keras_archives_are_readable_without_tensorflow(self):
        for name in ("cnn_best.keras", "cnn_checkpoint.keras", "cnn_v2_best.keras"):
            with (
                self.subTest(model=name),
                zipfile.ZipFile(ROOT / "outputs" / name) as archive,
            ):
                self.assertIsNone(archive.testzip())
                self.assertIn("model.weights.h5", archive.namelist())

                config = json.loads(archive.read("config.json"))
                self.assertEqual(config["class_name"], "Sequential")

                final_layer = config["config"]["layers"][-1]
                self.assertEqual(final_layer["config"]["units"], 10)
                self.assertEqual(final_layer["config"]["activation"], "softmax")

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow"),
        (
            "TensorFlow is not installed; Keras runtime loading cannot execute "
            "(no install in Phase 1A)"
        ),
    )
    def test_all_keras_models_load_in_tensorflow(self):
        import tensorflow as tf

        for name in ("cnn_best.keras", "cnn_checkpoint.keras", "cnn_v2_best.keras"):
            with self.subTest(model=name):
                model = tf.keras.models.load_model(
                    ROOT / "outputs" / name,
                    compile=False,
                )
                output = model(
                    np.zeros((1, 28, 28, 1), dtype=np.float32),
                    training=False,
                ).numpy()

                self.assertEqual(output.shape, (1, 10))
                self.assertTrue(np.isfinite(output).all())
                self.assertAlmostEqual(float(output.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()