
"""API regression and real checkpoint smoke checks; no training or artifact writes."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

import backend.app as backend
from ml import mobilenet_classifier as mobile


def png(mode='RGB', size=(19, 23)):
    buffer = io.BytesIO()
    Image.new(
        mode,
        size,
        (20, 80, 140, 90) if mode == 'RGBA' else (20, 80, 140),
    ).save(buffer, format='PNG')
    return buffer.getvalue()


class MobileNetApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(backend.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def post(self, payload=None, **headers):
        return self.client.post(
            '/api/predict/mobilenet',
            content=png() if payload is None else payload,
            headers={'content-type': 'image/png', **headers},
        )

    def test_invalid_uploads_never_load_model(self):
        with patch.object(mobile, '_load_verified') as load:
            for payload, headers, status in [
                (png(), {'content-type': 'text/plain'}, 415),
                (b'', {}, 400),
                (b'not an image', {}, 400),
                (png(), {'content-type': 'image/jpeg'}, 400),
                (png(size=(8193, 1)), {}, 400),
                (b'x', {'content-length': str(backend.MAX_UPLOAD_BYTES + 1)}, 413),
                (b'x' * (backend.MAX_UPLOAD_BYTES + 1), {'content-length': '1'}, 413),
            ]:
                with self.subTest(status=status, headers=headers):
                    self.assertEqual(self.post(payload, **headers).status_code, status)
            with patch.object(mobile, 'MAX_PIXELS', 10):
                self.assertEqual(self.post().status_code, 400)
            load.assert_not_called()

    def test_unavailable_and_corrupt_models_fail_safely_and_cache_failure(self):
        for error in (
            FileNotFoundError('missing private path'),
            ValueError('corrupt checkpoint'),
            ImportError('missing tensorflow'),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(mobile, 'service', mobile.MobileNetService()),
                patch.object(mobile, '_load_verified', side_effect=error) as load,
                self.assertLogs(mobile.__name__, level='ERROR'),
            ):
                for _ in range(2):
                    response = self.post()
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(
                        response.json(),
                        {'detail': 'MobileNet is unavailable. Contact the server operator.'},
                    )
                load.assert_called_once()

    def test_nonfinite_model_output_is_not_returned(self):
        model = Mock(return_value=Mock(numpy=Mock(return_value=np.full((1, 10), np.nan))))
        with (
            patch.object(mobile, 'service', mobile.MobileNetService()),
            patch.object(mobile, '_load_verified', return_value=model),
            self.assertLogs(mobile.__name__, level='ERROR'),
        ):
            self.assertEqual(self.post().status_code, 503)

    def test_clip_endpoint_is_not_routed_to_mobilenet(self):
        expected = {
            'prediction': 'Pullover',
            'confidence': 0.8,
            'uncertain': False,
            'predictions': [],
            'model': 'CLIP test fixture',
        }
        with (
            patch.object(backend, 'classify_image', return_value=expected) as clip,
            patch.object(backend, 'MODEL_PATH', Mock(exists=Mock(return_value=True))),
            patch.object(mobile, 'classify') as classify,
        ):
            response = self.client.post(
                '/api/predict',
                content=png(),
                headers={'content-type': 'image/png'},
            )
            self.assertEqual(response.json(), expected)
            self.assertEqual(response.status_code, 200)
            clip.assert_called_once()
            classify.assert_not_called()

    def test_startup_does_not_import_tensorflow(self):
        completed = subprocess.run(
            [
                sys.executable,
                '-B',
                '-c',
                "import app,sys; assert 'tensorflow' not in sys.modules; assert 'keras' not in sys.modules",
            ],
            cwd=mobile.ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class ServiceTests(unittest.TestCase):
    def test_concurrent_requests_load_once_and_use_inference_mode(self):
        probabilities = np.arange(1, 11, dtype=np.float32)[None] / 55
        model = Mock(return_value=Mock(numpy=Mock(return_value=probabilities)))
        service = mobile.MobileNetService()
        with patch.object(mobile, '_load_verified', return_value=model) as load:
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(pool.map(service.predict, [np.zeros((1, 160, 160, 3))] * 4))
            load.assert_called_once()
        self.assertTrue(
            all(call.kwargs == {'training': False} for call in model.call_args_list)
        )
        for response in responses:
            self.assertEqual(response['prediction'], 'Belts')
            self.assertEqual(
                [p['label'] for p in response['probabilities']],
                list(mobile.CLASS_NAMES),
            )
            self.assertEqual([p['class_id'] for p in response['top3']], [9, 8, 7])
            self.assertEqual(response['score_type'], 'uncalibrated_softmax')
            self.assertGreaterEqual(response['inference_time_ms'], 0)

    def test_missing_and_changed_assets_are_rejected(self):
        with TemporaryDirectory() as directory:
            with patch.object(mobile, 'MODEL_DIR', Path(directory)):
                with self.assertRaises(FileNotFoundError):
                    mobile._load_verified()
                (Path(directory) / 'best.keras').write_bytes(b'synthetic invalid artifact')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    mobile._load_verified()

    def test_exact_training_preprocessing_including_alpha_and_exif(self):
        from scripts import mobilenet_fashion_v1 as baseline
        payloads = [png(), png('RGBA')]
        buffer = io.BytesIO()
        image = Image.fromarray(np.arange(18 * 27 * 3, dtype=np.uint8).reshape(18, 27, 3))
        exif = Image.Exif()
        exif[274] = 6
        image.save(buffer, format='JPEG', exif=exif)
        payloads.append(buffer.getvalue())
        with TemporaryDirectory() as directory:
            for i, payload in enumerate(payloads):
                path = Path(directory) / f'{i}.image'
                path.write_bytes(payload)
                rows = [{
                    'original_path': str(path),
                    'file_sha256': hashlib.sha256(payload).hexdigest(),
                    'class_id': 0,
                }]
                expected = next(iter(baseline.make_dataset(rows, training=False)))[0].numpy()
                actual = mobile.preprocess(payload, 'image/jpeg' if i == 2 else 'image/png')
                np.testing.assert_array_equal(actual, expected)


class RealCheckpointTests(unittest.TestCase):
    def test_verified_checkpoint_loads_and_real_api_predicts_without_modification(self):
        path = mobile.MODEL_DIR / 'best.keras'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(before, mobile.MODEL_SHA256)
        model = mobile._load_verified()
        self.assertEqual(model.count_params(), 944890)
        self.assertEqual(model.output_shape, (None, 10))
        service = mobile.MobileNetService()
        with (
            patch.object(mobile, 'service', service),
            patch.object(mobile, '_load_verified', return_value=model) as load,
            TestClient(backend.app) as client,
        ):
            responses = [client.post(
                '/api/predict/mobilenet',
                content=png(),
                headers={'content-type': 'image/png'},
            ) for _ in range(2)]
            load.assert_called_once()
        for response in responses:
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result['model'], mobile.MODEL_ID)
            self.assertEqual(
                [p['label'] for p in result['probabilities']],
                list(mobile.CLASS_NAMES),
            )
            scores = [p['probability'] for p in result['probabilities']]
            self.assertTrue(np.isfinite(scores).all())
            self.assertAlmostEqual(sum(scores), 1, places=5)
            self.assertEqual(
                result['top3'],
                sorted(result['probabilities'], key=lambda p: p['probability'], reverse=True)[:3],
            )
            self.assertEqual(result['prediction'], result['top3'][0]['label'])
        self.assertEqual(
            responses[0].json()['probabilities'],
            responses[1].json()['probabilities'],
        )
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()
