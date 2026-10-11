"""Synthetic fixtures only: no real-photo metrics, downloads, or training."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from scripts import real_photo_evaluation_v1 as evaluation


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'dataset.json').write_text(
            json.dumps({
                'dataset_name': 'synthetic test fixture',
                'source': 'Pillow fixtures',
                'labeler': 'test',
                'manually_labeled': True,
                'independent_of_benchmark': True,
            })
        )
        self.forbidden = {'file_sha256': set(), 'pixel_sha256': set()}

    def image(self, folder, color):
        directory = self.root / folder
        directory.mkdir(exist_ok=True)
        path = directory / 'fixture.png'
        Image.new('RGB', (13, 11), color).save(path)
        return path

    def test_class_mapping_and_special_examples(self):
        for i, name in enumerate(evaluation.CLASS_NAMES):
            self.image(name, (i * 20, 10, 20))
        self.image('AMBIGUOUS', (1, 2, 3))
        self.image('OUT_OF_SCOPE', (4, 5, 6))

        audit = evaluation.scan(self.root, self.forbidden)
        self.assertFalse(audit['errors'])
        self.assertEqual(len(audit['rows']), 10)
        self.assertEqual(
            {r['status'] for r in audit['excluded']},
            evaluation.SPECIAL,
        )
        for row in audit['rows']:
            self.assertEqual(
                evaluation.CLASS_NAMES[row['class_id']],
                row['class_name'],
            )
        self.assertTrue(
            all(r['review_status'] == 'UNREVIEWED' for r in audit['excluded'])
        )

    def test_corrupt_unknown_duplicate_and_benchmark_overlap(self):
        path = self.image('T-shirts', (10, 20, 30))
        self.image('Shirts', (10, 20, 30))
        self.image('Unknown', (0, 0, 0))
        (path.parent / 'broken.jpg').write_bytes(b'not an image')

        audit = evaluation.scan(self.root, self.forbidden)
        self.assertEqual(len(audit['errors']), 3)
        self.forbidden['file_sha256'].add(evaluation.baseline.sha256(path))
        self.assertTrue(
            any(
                'overlap' in r['error']
                for r in evaluation.scan(self.root, self.forbidden)['errors']
            )
        )

    def test_provenance_required(self):
        (self.root / 'dataset.json').write_text('{}')
        with self.assertRaises(ValueError):
            evaluation.scan(self.root, self.forbidden)

    def test_metrics_incorrect_review_and_overwrite_protection(self):
        self.image('T-shirts', (1, 10, 20))
        self.image('Shirts', (2, 10, 20))
        audit = evaluation.scan(self.root, self.forbidden)

        p = np.zeros((2, 10))
        p[:, 0] = 1
        metrics, predictions = evaluation.results(audit['rows'], p)
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['top3_accuracy'], 1)
        self.assertAlmostEqual(metrics['macro_f1'], 1 / 15)
        self.assertEqual(np.asarray(metrics['confusion_matrix']).sum(), 2)
        self.assertEqual(sum(not r['correct'] for r in predictions), 1)

        output = self.root / 'output'
        output.mkdir()
        evaluation.publish(output, audit, p)
        before = {f.name: f.read_bytes() for f in output.iterdir()}
        with self.assertRaises(FileExistsError):
            evaluation.publish(output, audit, p)
        self.assertEqual(
            before,
            {f.name: f.read_bytes() for f in output.iterdir()},
        )
        self.assertEqual(
            json.loads(
                (output / 'incorrect_predictions.json').read_text()
            )[0]['review_status'],
            'UNREVIEWED',
        )

    def test_invalid_probabilities_and_labels(self):
        self.image('T-shirts', (1, 10, 20))
        rows = evaluation.scan(self.root, self.forbidden)['rows']
        for p in (
            np.zeros((1, 9)),
            np.full((1, 10), np.nan),
            np.zeros((2, 10)),
            np.ones((1, 10)),
        ):
            with self.assertRaises(ValueError):
                evaluation.results(rows, p)
        rows[0]['class_name'] = 'Shirts'
        with self.assertRaises(ValueError):
            evaluation.results(rows, np.full((1, 10), .1))

    def test_exact_preprocessing_and_inference_only(self):
        self.image('T-shirts', (22, 100, 240))
        rows = evaluation.scan(self.root, self.forbidden)['rows']
        expected = next(
            iter(evaluation.baseline.make_dataset(rows, training=False))
        )[0].numpy()

        class InferenceSpy:
            def __call__(self, images, training):
                self.training = training
                np.testing.assert_array_equal(images.numpy(), expected)
                import tensorflow as tf
                return tf.constant(
                    np.full((len(images), 10), .1),
                    dtype=tf.float32,
                )

        model = InferenceSpy()
        p = evaluation.predict(model, rows)
        self.assertFalse(model.training)
        self.assertEqual(p.shape, (1, 10))
        self.assertEqual(expected.shape, (1, 160, 160, 3))
        np.testing.assert_array_equal(expected[0, 0, 0], [22, 100, 240])


if __name__ == '__main__':
    unittest.main()