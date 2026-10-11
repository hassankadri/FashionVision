
"""Focused Grad-CAM tests: real checkpoint gradients, no training or model saves."""

import base64
import hashlib
import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

import backend.app as backend
from ml import mobilenet_classifier as mobile
from ml.mobilenet_gradcam import GradCAM, overlay_png


def fixture():
    pixels = np.random.default_rng(42).integers(0, 256, (80, 120, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    return buffer.getvalue()


class UploadTests(unittest.TestCase):
    def test_invalid_uploads_share_prediction_validation(self):
        with (
            TestClient(backend.app) as client,
            patch.object(mobile, '_load_verified') as load,
        ):
            for payload, headers, status in [
                (b'', {'content-type': 'image/png'}, 400),
                (b'broken', {'content-type': 'image/png'}, 400),
                (fixture(), {'content-type': 'text/plain'}, 415),
                (fixture(), {'content-type': 'image/jpeg'}, 400),
                (
                    b'x',
                    {
                        'content-type': 'image/png',
                        'content-length': str(backend.MAX_UPLOAD_BYTES + 1),
                    },
                    413,
                ),
            ]:
                with self.subTest(status=status):
                    response = client.post(
                        '/api/explain/mobilenet',
                        content=payload,
                        headers=headers,
                    )
                    self.assertEqual(response.status_code, status)
            with patch.object(mobile, 'MAX_PIXELS', 10):
                self.assertEqual(
                    client.post(
                        '/api/explain/mobilenet',
                        content=fixture(),
                        headers={'content-type': 'image/png'},
                    ).status_code,
                    400,
                )
            load.assert_not_called()

    def test_unavailable_model_is_503(self):
        with (
            TestClient(backend.app) as client,
            patch.object(mobile, 'service', mobile.MobileNetService()),
            patch.object(mobile, '_load_verified', side_effect=FileNotFoundError()),
            self.assertLogs(mobile.__name__, level='ERROR'),
        ):
            response = client.post(
                '/api/explain/mobilenet',
                content=fixture(),
                headers={'content-type': 'image/png'},
            )
            self.assertEqual(response.status_code, 503)

    def test_zero_attribution_does_not_invent_color(self):
        pixels = mobile.preprocess(fixture(), 'image/png')
        data = base64.b64decode(
            overlay_png(pixels, np.zeros((5, 5), dtype=np.float32)),
            validate=True,
        )
        np.testing.assert_array_equal(
            np.asarray(Image.open(io.BytesIO(data))),
            pixels[0].astype(np.uint8),
        )


class RealGradCAMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mobile._load_verified()
        cls.cam = GradCAM(cls.model)
        cls.payload = fixture()
        cls.pixels = mobile.preprocess(cls.payload, 'image/png')

    def test_class_specific_gradients_and_prediction_parity(self):
        probabilities = self.model(self.pixels, training=False).numpy()[0]
        target = int(probabilities.argmax())
        first = self.cam.compute(self.pixels, target)
        other = self.cam.compute(self.pixels, (target + 1) % 10)
        np.testing.assert_allclose(
            first['probabilities'],
            probabilities,
            atol=1e-06,
            rtol=0.0001,
        )
        self.assertEqual(first['heatmap'].shape, (5, 5))
        self.assertEqual(first['gradients'].shape, (1, 5, 5, 576))
        self.assertTrue(np.isfinite(first['gradients']).all())
        self.assertGreater(np.abs(first['gradients']).max(), 0)
        self.assertFalse(np.allclose(
            first['gradients'],
            other['gradients'],
            atol=1e-09,
            rtol=1e-05,
        ))
        self.assertTrue((first['heatmap'] >= 0).all() and (first['heatmap'] <= 1).all())

    def test_gradient_agrees_with_finite_difference(self):
        import keras
        activations, _ = self.cam.extractor(self.pixels, training=False)
        result = self.cam.compute(self.pixels)
        index = np.unravel_index(
            np.abs(result['gradients']).argmax(),
            result['gradients'].shape,
        )
        backbone = self.model.get_layer('mobilenetv3small_backbone')
        tail = keras.Model(backbone.get_layer('conv_1').output, backbone.output)

        def score(values):
            features = tail(values, training=False).numpy()
            return float((features @ self.cam.head.kernel.numpy() + self.cam.head.bias.numpy())[
                0,
                result['class_id'],
            ])
        plus, minus = (activations.numpy(), activations.numpy())
        epsilon = 0.01
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (score(plus) - score(minus)) / (2 * epsilon)
        np.testing.assert_allclose(
            result['gradients'][index],
            numerical,
            atol=0.0002,
            rtol=0.03,
        )

    def test_real_endpoint_png_and_no_weight_or_prediction_changes(self):

        def digest():
            return hashlib.sha256(
                b''.join(w.numpy().tobytes() for w in self.model.weights)
            ).hexdigest()
        before = digest()
        probabilities = self.model(self.pixels, training=False).numpy()
        service = mobile.MobileNetService()
        with (
            patch.object(mobile, 'service', service),
            patch.object(mobile, '_load_verified', return_value=self.model),
            TestClient(backend.app) as client,
        ):
            response = client.post(
                '/api/explain/mobilenet',
                content=self.payload,
                headers={'content-type': 'image/png'},
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(
                result['prediction'],
                mobile.CLASS_NAMES[int(probabilities.argmax())],
            )
            self.assertEqual(result['explained_class'], result['prediction'])
            self.assertEqual(result['target_layer'], 'conv_1')
            decoded = base64.b64decode(result['heatmap_overlay_base64'], validate=True)
            with Image.open(io.BytesIO(decoded)) as image:
                self.assertEqual(image.format, 'PNG')
                self.assertEqual(image.mode, 'RGB')
                self.assertEqual(image.size, (160, 160))
            after_prediction = client.post(
                '/api/predict/mobilenet',
                content=self.payload,
                headers={'content-type': 'image/png'},
            ).json()
            np.testing.assert_allclose(
                [r['probability'] for r in after_prediction['probabilities']],
                probabilities[0],
                atol=1e-07,
            )
        self.assertEqual(digest(), before)
        self.assertEqual(
            hashlib.sha256((mobile.MODEL_DIR / 'best.keras').read_bytes()).hexdigest(),
            mobile.MODEL_SHA256,
        )


if __name__ == '__main__':
    unittest.main()
