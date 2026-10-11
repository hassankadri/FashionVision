
"""Phase 2D safeguards. Mocked ranking is not a verified Cleanlab execution."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from scripts import label_quality_v1 as q


class LabelQualityTests(unittest.TestCase):
    def synthetic(self):
        rng = np.random.default_rng(42)
        return (
            rng.integers(0, 256, size=(400, 12), dtype=np.uint8),
            np.tile(np.arange(10), 40),
        )

    def fixture(self):
        pixels, labels = self.synthetic()
        with patch.object(q, 'rank_suspicions', return_value=np.array([0, 1, 2, 3])):
            return q.analyze_training(pixels, labels)

    def test_validation_reserved_before_oof_and_cleanlab_and_never_passed_to_them(self):
        pixels, labels = self.synthetic()
        events = []
        original_split = q.reserve_validation
        original_oof = q.generate_oof
        reserved = {}

        def split(y, seed):
            events.append('reserve')
            training, validation = original_split(y, seed)
            reserved.update(training=training, validation=validation)
            return (training, validation)

        def oof(X, y, indices, seed):
            self.assertEqual(events, ['reserve'])
            events.append('oof')
            self.assertFalse(set(indices) & set(reserved['validation']))
            self.assertTrue(set(indices) <= set(reserved['training']))
            np.testing.assert_array_equal(y, labels[indices])
            np.testing.assert_array_equal(X, pixels[indices].astype(np.float32) / 255)
            reserved['analysis'] = indices.copy()
            return original_oof(X, y, indices, seed)

        def rank(y, probabilities):
            self.assertEqual(events, ['reserve', 'oof'])
            events.append('cleanlab')
            np.testing.assert_array_equal(y, labels[reserved['analysis']])
            self.assertEqual(probabilities.shape, (300, 10))
            return np.array([0, 2])
        with (
            patch.object(q, 'reserve_validation', side_effect=split),
            patch.object(q, 'generate_oof', side_effect=oof),
            patch.object(q, 'rank_suspicions', side_effect=rank),
        ):
            report = q.analyze_training(pixels, labels)
        self.assertEqual(events, ['reserve', 'oof', 'cleanlab'])
        self.assertEqual(report['validation_indices'], reserved['validation'].tolist())

    def test_pca_fits_only_exact_oof_training_folds_without_global_refit(self):
        X = np.random.default_rng(42).normal(size=(100, 8))
        X[:, 0] = np.arange(100)
        y = np.tile(np.arange(10), 10)
        observed = []
        original = PCA.fit_transform

        def spy(estimator, rows, *args, **kwargs):
            observed.append(set(rows[:, 0].astype(int)))
            return original(estimator, rows, *args, **kwargs)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        expected = [set(train) for train, _ in cv.split(X, y)]
        with patch.object(PCA, 'fit_transform', spy):
            probabilities, ids, folds = q.generate_oof(X, y, np.arange(100))
        self.assertEqual(observed, expected)
        self.assertEqual(len(folds), 3)
        self.assertEqual(probabilities.shape, (100, 10))
        for fold, (train, held_out) in enumerate(cv.split(X, y)):
            np.testing.assert_array_equal(np.flatnonzero(ids == fold), held_out)
            self.assertFalse(observed[fold] & set(held_out))
        self.assertFalse(hasattr(q.make_pipeline(42).named_steps['pca'], 'components_'))

    def test_canonical_original_index_mapping_and_default_unreviewed(self):
        report = self.fixture()
        _, labels = self.synthetic()
        manifest = q.review_manifest(report)
        self.assertEqual(manifest['class_names'], list(q.CLASS_NAMES))
        self.assertEqual(manifest['sample_index_namespace'], 'original_training_split')
        for rank, item in enumerate(manifest['items'], 1):
            position = item['analysis_position']
            self.assertEqual(item['sample_index'], report['analysis_indices'][position])
            self.assertEqual(item['given_label'], labels[item['sample_index']])
            self.assertEqual(item['given_class'], q.CLASS_NAMES[item['given_label']])
            prob = report['oof_probabilities'][position]
            self.assertEqual(item['predicted_label'], int(np.argmax(prob)))
            self.assertEqual(item['self_confidence'], prob[item['given_label']])
            self.assertEqual(item['rank'], rank)
            self.assertEqual(item['review_status'], 'UNREVIEWED')
            self.assertEqual(item['assessment'], 'SUSPECTED_LABEL_ISSUE')

    def test_seed_reproduces_split_oof_and_ranking_inputs(self):
        pixels, labels = self.synthetic()
        with patch.object(q, 'rank_suspicions', return_value=np.array([0, 1])):
            first = q.analyze_training(pixels, labels)
            second = q.analyze_training(pixels, labels)
        for key in (
            'development_indices',
            'validation_indices',
            'analysis_indices',
            'oof_probabilities',
            'oof_fold_ids',
            'candidates',
        ):
            self.assertEqual(first[key], second[key])
        self.assertNotEqual(
            q.reserve_validation(labels, 43)[1].tolist(),
            first['validation_indices'],
        )

    def test_default_plan_never_removes_candidates_or_modifies_analysis(self):
        analysis = self.fixture()
        before = deepcopy(analysis)
        plan = q.plan_strategy(analysis)
        self.assertEqual(plan['training_indices'], analysis['analysis_indices'])
        self.assertEqual(plan['proposed_excluded_indices'], [])
        self.assertEqual(analysis['automatic_removals'], [])
        self.assertEqual(plan['status'], 'PLAN_ONLY')
        self.assertFalse(plan['data_modified'])
        self.assertFalse(plan['retraining_executed'])
        self.assertEqual(before, analysis)

    def test_controlled_strategies_share_validation_test_and_training_config(self):
        analysis = self.fixture()
        reviewed = q.review_manifest(analysis)
        for item, status in zip(reviewed['items'], q.REVIEW_STATUSES):
            item.update(
                review_status=status,
                reviewed_by='test-reviewer',
                review_notes='synthetic test only',
            )
        plans = [
            q.plan_strategy(analysis),
            q.plan_strategy(analysis, 'top_n', top_n=2),
            q.plan_strategy(analysis, 'confidence_threshold', threshold=0.5),
            q.plan_strategy(analysis, 'manually_reviewed', reviewed=reviewed),
        ]
        self.assertEqual(len(plans[1]['proposed_excluded_indices']), 2)
        expected = [
            item['sample_index']
            for item in analysis['candidates']
            if item['self_confidence'] < 0.5
        ]
        self.assertEqual(plans[2]['proposed_excluded_indices'], expected)
        self.assertEqual(
            plans[3]['proposed_excluded_indices'],
            [reviewed['items'][1]['sample_index']],
        )
        for plan in plans:
            for key in (
                'validation_indices',
                'validation_indices_sha256',
                'official_test',
                'class_names',
                'training_config',
            ):
                self.assertEqual(plan[key], plans[0][key])
            self.assertFalse(
                set(plan['proposed_excluded_indices']) & set(plan['validation_indices'])
            )
        self.assertFalse(
            plans[0]['official_test']['used_for_issue_detection_or_strategy_selection']
        )

    def test_review_rejects_changed_identity_metadata_and_unattributed_confirmations(self):
        analysis = self.fixture()
        for problem in (
            'analysis_id',
            'class_names',
            'index',
            'given_label',
            'status',
            'unattributed',
        ):
            reviewed = q.review_manifest(analysis)
            if problem == 'analysis_id':
                reviewed['analysis_id'] = 'different-analysis'
            elif problem == 'class_names':
                reviewed['class_names'].reverse()
            elif problem == 'index':
                reviewed['items'][0]['sample_index'] = analysis['validation_indices'][0]
            elif problem == 'given_label':
                reviewed['items'][0]['given_label'] = 99
            elif problem == 'status':
                reviewed['items'][0]['review_status'] = 'AUTOMATICALLY_WRONG'
            else:
                reviewed['items'][0]['review_status'] = 'CONFIRMED_WRONG'
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                q.plan_strategy(analysis, 'manually_reviewed', reviewed=reviewed)

    def test_strategy_rejects_validation_candidates_and_implicit_parameters(self):
        analysis = self.fixture()
        for strategy, kwargs in (
            ('top_n', {}),
            ('top_n', {'top_n': -1}),
            ('confidence_threshold', {'threshold': float('nan')}),
            ('manually_reviewed', {}),
            ('no_filtering', {'top_n': 1}),
        ):
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                q.plan_strategy(analysis, strategy, **kwargs)
        analysis['candidates'][0]['sample_index'] = analysis['validation_indices'][0]
        with self.assertRaises(ValueError):
            q.plan_strategy(analysis, 'top_n', top_n=1)

    def test_plans_reject_changed_validation_and_removal_of_a_whole_class(self):
        analysis = self.fixture()
        analysis['validation_indices'].reverse()
        with self.assertRaises(ValueError):
            q.plan_strategy(analysis)
        analysis['validation_indices'].reverse()
        labels = np.asarray(analysis['given_labels'])
        analysis['candidates'] = [
            {'sample_index': analysis['analysis_indices'][position]}
            for position in np.flatnonzero(labels == 0)
        ]
        with self.assertRaisesRegex(ValueError, 'retain all ten classes'):
            q.plan_strategy(analysis, 'top_n', top_n=len(analysis['candidates']))

    def test_cleanlab_adapter_uses_only_supplied_oof_probabilities(self):
        finder = Mock(return_value=np.array([1]))
        labels = np.arange(10)
        probabilities = np.eye(10)
        with patch.object(q, 'require_cleanlab', return_value=finder):
            np.testing.assert_array_equal(q.rank_suspicions(labels, probabilities), [1])
        finder.assert_called_once_with(
            labels=labels,
            pred_probs=probabilities,
            return_indices_ranked_by='self_confidence',
            n_jobs=1,
        )

    def test_invalid_oof_values_and_rank_positions_fail(self):
        for values in (
            np.zeros((5, 9)),
            np.full((5, 10), np.nan),
            np.full((5, 10), np.inf),
            np.full((5, 10), -0.1),
            np.ones((5, 10)),
        ):
            with self.assertRaises(ValueError):
                q.validate_probabilities(values, 5)
        pixels, labels = self.synthetic()
        for ranked in ([300], [-1], [0, 0], [0.5]):
            with (
                patch.object(q, 'rank_suspicions', return_value=np.asarray(ranked)),
                self.assertRaises(ValueError),
            ):
                q.analyze_training(pixels, labels)

    def test_missing_cleanlab_fails_before_data_or_output(self):
        with (
            patch.object(q, 'require_cleanlab', side_effect=RuntimeError('Cleanlab unavailable')),
            patch.object(q, 'load_split') as load,
            patch.object(q, 'atomic_json_new') as write,
        ):
            with self.assertRaisesRegex(RuntimeError, 'Cleanlab unavailable'):
                q.run()
        load.assert_not_called()
        write.assert_not_called()

    def test_real_data_smoke_oof_and_persistence_with_explicitly_mocked_cleanlab(self):
        # Run real out-of-fold scoring with mocked Cleanlab ranking only.
        protected = [p for root in (
            'models',
            'outputs',
            'data/raw',
            'experiments/classical_v3',
        ) for p in (q.ROOT / root).rglob('*') if p.is_file()]
        before = {p: q.sha256(p) for p in protected}
        original_labels = q.load_split('train')[1]
        real_version = q.importlib.metadata.version

        def version(name):
            return 'MOCKED-NOT-CLEANLAB' if name == 'cleanlab' else real_version(name)
        with (
            TemporaryDirectory() as directory,
            patch.object(q, 'REPORT_DIR', Path(directory)),
            patch.object(q, 'require_cleanlab', return_value=Mock(return_value=np.array([0, 1]))),
            patch.object(q.importlib.metadata, 'version', side_effect=version),
            patch.object(q, 'load_split', wraps=q.load_split) as load,
        ):
            analysis, destination = q.run()
            load.assert_called_once_with('train')
            self.assertEqual(len(analysis['development_indices']), 54000)
            self.assertEqual(len(analysis['validation_indices']), 6000)
            np.testing.assert_array_equal(
                np.bincount(original_labels[analysis['validation_indices']]),
                np.full(10, 600),
            )
            self.assertEqual(np.shape(analysis['oof_probabilities']), (300, 10))
            self.assertEqual(analysis['automatic_removals'], [])
            self.assertEqual(
                set(analysis['development_indices']) | set(analysis['validation_indices']),
                set(range(60000)),
            )
            status = json.loads((destination / 'status.json').read_text())
            self.assertEqual(status['status'], 'COMPLETE')
            self.assertEqual(
                set(analysis['official_test']['file_sha256']),
                {'t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'},
            )
            saved = json.loads((destination / 'review_manifest.json').read_text())
            self.assertTrue(
                all(item['review_status'] == 'UNREVIEWED' for item in saved['items'])
            )
            path = destination / 'analysis.json'
            original_hash = q.sha256(path)
            with self.assertRaises(FileExistsError):
                q.atomic_json_new(path, {'overwrite': True})
            self.assertEqual(q.sha256(path), original_hash)
        self.assertEqual(before, {p: q.sha256(p) for p in protected})
        self.assertEqual(q.REPORT_DIR, q.ROOT / 'experiments/label_quality_v1/runs')

    @unittest.skipUnless(
        importlib.util.find_spec('cleanlab'),
        'Cleanlab is not installed; real ranking integration cannot execute',
    )

    def test_real_cleanlab_integration(self):
        pixels, labels = self.synthetic()
        report = q.analyze_training(pixels, labels)
        self.assertEqual(np.shape(report['oof_probabilities']), (300, 10))
        self.assertTrue(all((c['review_status'] == 'UNREVIEWED' for c in report['candidates'])))


if __name__ == '__main__':
    unittest.main()
