
"""Family-loader safeguards plus the unchanged baseline architecture smoke test."""

from copy import deepcopy
import json
import unittest

from scripts import mobilenet_family_v1 as family
from scripts import mobilenet_fashion_v1 as original
from test_mobilenet_fashion_v1 import ConstructionTests  # weights=None; no training/downloads.


class FamilyManifestTests(unittest.TestCase):
    def test_original_architecture_transforms_and_metrics_are_reused(self):
        for name in (
            'build_model',
            'enable_fine_tuning',
            'make_dataset',
            'evaluate_predictions',
            'training_class_weights',
        ):
            self.assertIs(getattr(family, name), getattr(original, name))
        recipe = json.loads(family.RECIPE.read_text())
        family.assert_original_strategy(recipe)
        changed = deepcopy(recipe)
        changed['stage2']['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'stage2'):
            family.assert_original_strategy(changed)

    def test_frozen_family_manifests_counts_order_groups_and_new_weights(self):
        preparation, complete = family.read_preparation()
        splits = []
        for split in ('train', 'validation', 'test'):
            rows = family.load_manifest(
                family.MANIFESTS / f'{split}.jsonl',
                preparation['dataset_root'],
                split,
                {c['class_id']: c['split_counts'][split] for c in preparation['classes']},
                complete['files'][f'{split}.jsonl'],
            )
            family.validate_group_manifest(rows, preparation['groups'], split)
            splits.append(rows)
        family.assert_group_disjoint(*splits)
        self.assertEqual([len(rows) for rows in splits], [20364, 2545, 2546])
        weights = family.training_class_weights(splits[0])
        self.assertEqual(weights[5], 20364 / (10 * 1404))
        self.assertNotEqual(weights[5], 20363 / (10 * 1403))
        self.assertEqual(
            tuple((c['class_name'] for c in preparation['classes'])),
            family.CLASS_NAMES,
        )

    def test_group_id_class_and_membership_changes_rejected(self):
        groups = {'g': {'group_id': 'g', 'image_ids': ['1'], 'split': 'train', 'class_id': 0}}
        row = {
            'image_id': '1',
            'group_id': 'g',
            'class_id': 0,
            'split': 'train',
            'group_identity_status': 'INFERRED_UNVERIFIED',
        }
        self.assertEqual(family.validate_group_manifest([row], groups, 'train'), [row])
        for key, value in (('group_id', 'wrong'), ('class_id', 1), ('split', 'test')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                family.validate_group_manifest([{**row, key: value}], groups, 'train')
        with self.assertRaisesRegex(ValueError, 'membership'):
            family.validate_group_manifest([], groups, 'train')

    def test_group_overlap_rejected_even_when_images_differ(self):
        a = [{'image_id': '1', 'file_sha256': 'a', 'pixel_sha256': 'b', 'group_id': 'family'}]
        b = [{'image_id': '2', 'file_sha256': 'c', 'pixel_sha256': 'd', 'group_id': 'family'}]
        with self.assertRaisesRegex(ValueError, 'family-group leakage'):
            family.assert_group_disjoint(a, b)


if __name__ == '__main__':
    unittest.main()
