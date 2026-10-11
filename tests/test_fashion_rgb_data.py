
"""Only temporary synthetic unit-test images; no real-world dataset is supplied."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, PngImagePlugin

from fashion_rgb_data import ClassMapping, RGBDataset, Transform, scan_dataset
from fashion_rgb_data.__main__ import main


class FashionRGBDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'dataset'
        self.paths = {}
        for split_id, split in enumerate(('train', 'validation', 'test')):
            for class_id, name in enumerate(('dress', 'shoe')):
                folder = self.root / split / name
                folder.mkdir(parents=True)
                y, x = np.mgrid[:12, :16]
                pixels = np.stack(
                    (
                        x * 8 + class_id * 7,
                        y * 10 + split_id * 11,
                        (x + y) * 3 + split_id * 9 + class_id * 5,
                    ),
                    axis=-1,
                ).astype(np.uint8)
                path = folder / 'example.png'
                Image.fromarray(pixels).save(path)
                self.paths[split, name] = path

    def hashes(self):
        return {
            p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.rglob('*')
            if p.is_file()
        }

    def test_layout_default_mapping_identity_and_audit_determinism(self):
        before = self.hashes()
        first = scan_dataset(self.root)
        self.assertTrue(first['ready'])
        self.assertEqual(first, scan_dataset(self.root))
        self.assertEqual(
            first['class_mapping'],
            {'class_names': ['dress', 'shoe'], 'folders': {'dress': 0, 'shoe': 1}},
        )
        self.assertEqual(len(first['records']), 6)
        for record in first['records']:
            self.assertEqual(record['file_sha256'], before[record['relative_path']])
            self.assertEqual(record['original_mode'], 'RGB')
            self.assertEqual(record['original_size'], [16, 12])
            self.assertEqual(len(record['pixel_sha256']), 64)
        self.assertEqual(before, self.hashes())
        self.assertFalse(first['data_modified'])
        json.dumps(first, allow_nan=False)

    def test_configurable_mapping_and_folder_aliases(self):
        mapping = ClassMapping(('Footwear', 'Clothing'), {'shoe': 0, 'dress': 1, 'tops': 1})
        alias = self.root / 'train' / 'tops'
        alias.mkdir()
        Image.new('RGB', (5, 7), (7, 91, 131)).save(alias / 'top.png')
        audit = scan_dataset(self.root, mapping)
        self.assertTrue(audit['ready'])
        self.assertEqual(
            audit['statistics']['train']['class_counts'],
            {'Footwear': 1, 'Clothing': 2},
        )
        for record in audit['records']:
            self.assertEqual(
                record['class_id'],
                mapping.folders[record['source_class_folder']],
            )
        with self.assertRaises(TypeError):
            mapping.folders['shoe'] = 1

    def test_invalid_mapping_rejected(self):
        cases = [
            ((), {}),
            (('x', 'x'), {'a': 0}),
            (('x',), {'../a': 0}),
            (('x',), {'a': True}),
            (('x',), {'a': 1}),
            (('x', 'y'), {'a': 0}),
        ]
        for names, folders in cases:
            with self.subTest(names=names, folders=folders), self.assertRaises(ValueError):
                ClassMapping(names, folders)

    def test_validation_cannot_introduce_an_unmapped_class(self):
        folder = self.root / 'validation' / 'unknown'
        folder.mkdir()
        Image.new('RGB', (3, 3), 'green').save(folder / 'new.png')
        audit = scan_dataset(self.root)
        self.assertFalse(audit['ready'])
        self.assertIn('unmapped_class_folder', [e['code'] for e in audit['errors']])
        self.assertNotIn('unknown', audit['class_mapping']['class_names'])
        with self.assertRaises(ValueError):
            RGBDataset(audit, 'train')

    def test_missing_and_empty_splits_are_reported(self):
        other = Path(self.temporary.name) / 'incomplete'
        (other / 'train' / 'dress').mkdir(parents=True)
        audit = scan_dataset(other)
        self.assertFalse(audit['ready'])
        self.assertIn('missing_or_redirected_split', [e['code'] for e in audit['errors']])
        self.assertEqual(audit['statistics']['train']['valid_images'], 0)

    def test_rgb_resize_and_explicit_normalization(self):
        path = self.paths['validation', 'dress']
        Image.new('L', (5, 9), 128).save(path)
        dataset = RGBDataset(
            scan_dataset(self.root),
            'validation',
            transform=Transform(size=(7, 11), mean=(0.5, 0.25, 0), std=(0.5, 0.25, 1)),
        )
        sample = dataset[0]
        self.assertEqual(sample['image'].shape, (7, 11, 3))
        self.assertEqual(sample['image'].dtype, np.float32)
        expected = (np.float32(128 / 255) - np.array([0.5, 0.25, 0])) / [0.5, 0.25, 1]
        np.testing.assert_allclose(sample['image'][0, 0], expected, rtol=1e-05)
        self.assertEqual(sample['identity']['original_mode'], 'L')
        self.assertEqual(sample['identity']['relative_path'], 'validation/dress/example.png')

    def test_alpha_is_composited_white(self):
        Image.new('RGBA', (6, 8), (0, 0, 0, 0)).save(self.paths['test', 'dress'])
        sample = RGBDataset(scan_dataset(self.root), 'test')[0]
        np.testing.assert_array_equal(
            sample['image'],
            np.ones((224, 224, 3), dtype=np.float32),
        )

    def test_exif_orientation_is_applied_in_memory(self):
        path = self.paths['train', 'dress'].with_suffix('.jpg')
        image = Image.new('RGB', (8, 4), (120, 30, 80))
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)
        audit = scan_dataset(self.root)
        record = next(
            r for r in audit['records']
            if r['relative_path'].endswith('example.jpg')
        )
        self.assertEqual(record['original_size'], [8, 4])
        self.assertEqual(record['rgb_size'], [4, 8])

    def test_only_training_can_augment_reproducibly_per_epoch(self):
        before = self.hashes()
        audit = scan_dataset(self.root)
        config = Transform(size=(12, 16), brightness_range=(0.6, 1.4))
        first = RGBDataset(audit, 'train', transform=config, augment=True, seed=42)
        second = RGBDataset(audit, 'train', transform=config, augment=True, seed=42)
        np.testing.assert_array_equal(
            first.get(0, epoch=3)['image'],
            second.get(0, epoch=3)['image'],
        )
        # Sample access order must not affect augmentation.
        first.get(1, epoch=3)
        np.testing.assert_array_equal(
            first.get(0, epoch=3)['image'],
            second.get(0, epoch=3)['image'],
        )
        self.assertFalse(np.array_equal(
            first.get(0, epoch=3)['image'],
            first.get(0, epoch=4)['image'],
        ))
        for split in ('validation', 'test'):
            with self.assertRaises(ValueError):
                RGBDataset(audit, split, augment=True)
            dataset = RGBDataset(audit, split, transform=config)
            np.testing.assert_array_equal(
                dataset.get(0, epoch=0)['image'],
                dataset.get(0, epoch=9)['image'],
            )
        self.assertEqual(before, self.hashes())

    def test_corrupt_image_reported_and_not_silently_dropped(self):
        path = self.root / 'train' / 'dress' / 'broken.jpg'
        path.write_bytes(b'not an image')
        audit = scan_dataset(self.root)
        self.assertFalse(audit['ready'])
        record = next((r for r in audit['records'] if r['relative_path'].endswith('broken.jpg')))
        self.assertEqual(record['status'], 'INVALID')
        self.assertEqual(record['file_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(audit['statistics']['train']['invalid_images'], 1)
        with self.assertRaises(ValueError):
            RGBDataset(audit, 'train')

    def test_forced_training_flip_preserves_label_and_source_identity(self):
        audit = scan_dataset(self.root)
        config = Transform(
            size=(12, 16),
            horizontal_flip_probability=1,
            brightness_range=(1, 1),
        )
        plain = RGBDataset(audit, 'train', transform=config)
        augmented = RGBDataset(audit, 'train', transform=config, augment=True)
        expected, actual = (plain[0], augmented[0])
        np.testing.assert_array_equal(actual['image'], expected['image'][:, ::-1, :])
        self.assertEqual(actual['class_id'], expected['class_id'])
        self.assertEqual(actual['identity'], expected['identity'])
        actual['identity']['original_size'][0] = 999
        audit['records'][0]['original_size'][0] = 888
        self.assertEqual(augmented[0]['identity']['original_size'], [16, 12])

    def test_identical_bytes_across_splits_are_leakage(self):
        self.paths['test', 'dress'].write_bytes(self.paths['train', 'dress'].read_bytes())
        audit = scan_dataset(self.root)
        self.assertFalse(audit['ready'])
        self.assertTrue(any((g['kind'] == 'file_sha256' and g['splits'] == [
            'test',
            'train',
        ] for g in audit['cross_split_leakage'])))

    def test_same_pixels_different_encoding_metadata_are_leakage(self):
        source = self.paths['train', 'shoe']
        destination = self.paths['validation', 'shoe']
        info = PngImagePlugin.PngInfo()
        info.add_text('unit-test', 'different container, identical decoded pixels')
        with Image.open(source) as image:
            image.save(destination, pnginfo=info, compress_level=0)
        self.assertNotEqual(source.read_bytes(), destination.read_bytes())
        audit = scan_dataset(self.root)
        self.assertFalse(audit['ready'])
        self.assertEqual({g['kind'] for g in audit['cross_split_leakage']}, {'pixel_sha256'})

    def test_within_split_duplicates_and_conflicting_labels_are_distinguished(self):
        copy = self.root / 'train' / 'dress' / 'copy.png'
        copy.write_bytes(self.paths['train', 'dress'].read_bytes())
        audit = scan_dataset(self.root)
        self.assertTrue(audit['ready'])
        self.assertTrue(audit['duplicates'])
        self.assertEqual(audit['cross_split_leakage'], [])
        self.paths['train', 'shoe'].write_bytes(copy.read_bytes())
        audit = scan_dataset(self.root)
        self.assertFalse(audit['ready'])
        self.assertTrue(audit['duplicate_label_conflicts'])

    def test_statistics_and_missing_classes(self):
        extra = self.root / 'train' / 'dress' / 'extra.png'
        Image.new('RGB', (3, 6), 'purple').save(extra)
        mapping = ClassMapping(('dress', 'shoe', 'bag'), {'dress': 0, 'shoe': 1, 'bag': 2})
        audit = scan_dataset(self.root, mapping)
        train = audit['statistics']['train']
        self.assertEqual(train['class_counts'], {'dress': 2, 'shoe': 1, 'bag': 0})
        self.assertEqual(train['missing_classes'], ['bag'])
        self.assertIsNone(train['max_to_min_ratio'])
        self.assertFalse(audit['ready'])
        train = scan_dataset(self.root)['statistics']['train']
        self.assertEqual(train['max_to_min_ratio'], 2.0)
        self.assertAlmostEqual(train['class_proportions']['dress'], 2 / 3)

    def test_changed_source_image_refused_after_audit(self):
        dataset = RGBDataset(scan_dataset(self.root), 'train')
        Image.new('RGB', (16, 12), 'orange').save(self.paths['train', 'dress'])
        with self.assertRaisesRegex(ValueError, 'changed after audit'):
            dataset[0]

    def test_nested_files_and_ignored_sidecars_preserve_paths(self):
        nested = self.root / 'train' / 'dress' / 'originals'
        nested.mkdir()
        Image.new('RGB', (2, 8), 'yellow').save(nested / 'nested.png')
        (nested / 'notes.txt').write_text('unit fixture', encoding='utf-8')
        audit = scan_dataset(self.root)
        self.assertTrue(audit['ready'])
        self.assertIn('train/dress/originals/notes.txt', audit['ignored_non_image_files'])
        self.assertIn(
            'train/dress/originals/nested.png',
            [r['relative_path'] for r in audit['records']],
        )

    def test_invalid_transform_configuration_rejected(self):
        for kwargs in (
            {'size': (0, 20)},
            {'std': (1, 0, 1)},
            {'mean': (0, float('nan'), 0)},
            {'horizontal_flip_probability': 2},
            {'brightness_range': (1.2, 0.8)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Transform(**kwargs)

    def test_cli_is_read_only_and_returns_failure_for_leakage(self):
        before = self.hashes()
        with (
            patch('sys.argv', ['fashion_rgb_data', str(self.root)]),
            patch('builtins.print') as output,
        ):
            self.assertEqual(main(), 0)
            self.assertTrue(json.loads(output.call_args.args[0])['ready'])
        self.assertEqual(before, self.hashes())
        self.paths['test', 'dress'].write_bytes(self.paths['train', 'dress'].read_bytes())
        with patch('sys.argv', ['fashion_rgb_data', str(self.root)]), patch('builtins.print'):
            self.assertEqual(main(), 1)


if __name__ == '__main__':
    unittest.main()
