
"""Focused construction/manifest checks; never train or download pretrained weights."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from scripts import mobilenet_fashion_v1 as baseline


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'train.jsonl'
        self.rows = []
        for label, name in enumerate(baseline.CLASS_NAMES):
            path = self.root / f'{label}.png'
            Image.new('RGB', (11, 13), (label * 20, 30, 60)).save(path)
            self.rows.append({
                'image_id': str(label),
                'relative_path': path.name,
                'original_path': str(path),
                'class_id': label,
                'class_name': name,
                'split': 'train',
                'source_metadata': {'id': str(label)},
                'file_sha256': baseline.sha256(path),
                'pixel_sha256': str(label),
            })

    def load(self):
        self.path.write_text(
            '\n'.join((json.dumps(row) for row in self.rows)),
            encoding='utf-8',
        )
        return baseline.load_manifest(
            self.path,
            self.root,
            'train',
            dict.fromkeys(range(10), 1),
            baseline.sha256(self.path),
        )

    def test_manifest_identity_order_weights_and_deterministic_rgb(self):
        rows = self.load()
        self.assertEqual(rows, self.rows)
        self.assertEqual(baseline.training_class_weights(rows), dict.fromkeys(range(10), 1.0))
        before = {row['original_path']: baseline.sha256(row['original_path']) for row in rows}
        first = next(iter(baseline.make_dataset(rows)))
        second = next(iter(baseline.make_dataset(rows)))
        self.assertEqual(first[0].shape, (10, 160, 160, 3))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], np.arange(10))
        # MobileNet's built-in rescaling handles normalization.
        self.assertGreater(float(np.max(first[0])), 1.0)
        self.assertEqual(before, {p: baseline.sha256(p) for p in before})

    def test_class_order_and_source_id_mismatches_rejected(self):
        self.rows[0]['class_name'] = 'Shirts'
        with self.assertRaisesRegex(ValueError, 'mapping'):
            self.load()
        self.rows[0]['class_name'] = 'T-shirts'
        self.rows[0]['source_metadata']['id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.load()

    def test_missing_images_manifest_changes_and_leakage_rejected(self):
        self.load()
        with self.assertRaisesRegex(ValueError, 'modified manifest'):
            baseline.load_manifest(self.path, self.root, 'train', {}, 'wrong hash')
        with self.assertRaisesRegex(ValueError, 'leakage'):
            baseline.assert_disjoint(self.rows, self.rows)
        self.rows[0]['relative_path'] = 'missing.png'
        with self.assertRaisesRegex(ValueError, 'Missing image'):
            self.load()

    def test_fixed_repository_manifests_counts_classes_and_disjointness(self):
        preparation, complete = baseline.read_preparation()
        splits = []
        for split in ('train', 'validation', 'test'):
            rows = baseline.load_manifest(
                baseline.MANIFESTS / f'{split}.jsonl',
                preparation['dataset_root'],
                split,
                {c['class_id']: c['split_counts'][split] for c in preparation['classes']},
                complete['files'][f'{split}.jsonl'],
            )
            self.assertEqual(len(rows), preparation['split_counts'][split])
            splits.append(rows)
        baseline.assert_disjoint(*splits)
        self.assertEqual([len(rows) for rows in splits], [20363, 2546, 2546])


class ConstructionTests(unittest.TestCase):
    def test_model_output_freezing_fine_tuning_and_saved_class_order(self):
        import keras
        keras.utils.set_random_seed(42)
        model, backbone = baseline.build_model(weights=None)
        self.assertFalse(backbone.trainable)
        # Only the Dense head's kernel and bias are trainable.
        self.assertEqual(len(model.trainable_weights), 2)
        images = np.full((2, 160, 160, 3), 120, dtype=np.float32)
        first = model(images, training=False).numpy()
        self.assertEqual(first.shape, (2, 10))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first.sum(axis=1), 1, atol=1e-06)
        np.testing.assert_array_equal(first, model(images, training=False).numpy())
        tuned = baseline.enable_fine_tuning(backbone)
        self.assertIn('expanded_conv_8_expand', tuned)
        self.assertIn('conv_1', tuned)
        self.assertNotIn('expanded_conv_7_expand', tuned)
        for layer in backbone.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                self.assertFalse(layer.trainable)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'construction.keras'
            model.save(path)
            restored = keras.models.load_model(path, compile=False)
            np.testing.assert_allclose(
                first,
                restored(images, training=False).numpy(),
                atol=1e-07,
            )
        self.assertEqual(model.output_names, ['class_probabilities'])
        self.assertEqual(baseline.CLASS_NAMES[0], 'T-shirts')
        self.assertEqual(baseline.CLASS_NAMES[-1], 'Belts')


if __name__ == '__main__':
    unittest.main()
