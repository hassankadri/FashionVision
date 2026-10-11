"""Current API behavior, including limitations; all fault injection is in memory."""

import io
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf

import backend.app as backend
from ml import photo_classifier as photo


class ApiBaselineTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(backend.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
        self.png = buffer.getvalue()

    def post(self, payload, **headers):
        return self.client.post(
            "/api/predict",
            content=payload,
            headers={"content-type": "image/png", **headers},
        )

    def test_valid_health_response_schema(self):
        with patch.object(
            backend, "MODEL_PATH", Mock(exists=Mock(return_value=True))
        ):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "photo_model_ready": True})

    def test_unsupported_mime_type(self):
        response = self.post(
            self.png, **{"content-type": "application/octet-stream"}
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(
            response.json(),
            {"detail": "Upload a JPG, PNG, WebP, or BMP image."},
        )

    def test_empty_payload(self):
        response = self.post(b"")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "The uploaded image is empty."})

    def test_malformed_image_bytes(self):
        with (
            patch.object(
                backend, "MODEL_PATH", Mock(exists=Mock(return_value=True))
            ),
            patch.object(backend, "classify_image") as classify,
        ):
            response = self.post(b"not a PNG")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"detail": "The uploaded file is not a readable image."},
        )
        classify.assert_not_called()

    def test_oversized_content_length(self):
        response = self.post(
            b"x", **{"content-length": str(backend.MAX_UPLOAD_BYTES + 1)}
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"detail": "Images must be 15 MB or smaller."})

    def test_oversized_body_with_understated_content_length(self):
        # Check the 15 MiB limit after reading the body, entirely in memory.
        response = self.post(
            b"x" * (backend.MAX_UPLOAD_BYTES + 1),
            **{"content-length": "1"},
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"detail": "Images must be 15 MB or smaller."})

    def test_missing_onnx_asset(self):
        with patch.object(
            backend, "MODEL_PATH", Mock(exists=Mock(return_value=False))
        ):
            health = self.client.get("/api/health")
            response = self.post(self.png)

        self.assertEqual(health.status_code, 200)
        self.assertEqual(
            health.json(), {"status": "model_missing", "photo_model_ready": False}
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "detail": (
                    "The real-photo model is missing. Run: "
                    "python scripts/download_photo_model.py"
                )
            },
        )

    def test_missing_tokenizer_asset_currently_becomes_image_error(self):
        # Bypass caches without clearing them or removing any model assets.
        prompts = photo._prompt_inputs()

        for name in ("VOCAB_PATH", "MERGES_PATH"):
            with (
                self.subTest(asset=name),
                patch.object(
                    backend, "MODEL_PATH", Mock(exists=Mock(return_value=True))
                ),
                patch.object(photo, name, Mock(exists=Mock(return_value=False))),
                patch.object(photo, "_prompt_inputs", return_value=prompts),
                patch.object(photo, "_session", photo._session.__wrapped__),
            ):
                health = self.client.get("/api/health")
                response = self.post(self.png)

            # The baseline health check only checks ONNX availability.
            # FileNotFoundError is also handled as an image decoding error.
            self.assertEqual(
                health.json(), {"status": "ok", "photo_model_ready": True}
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                response.json(),
                {"detail": "The uploaded file is not a readable image."},
            )

    def test_corrupt_onnx_loader_failure_currently_returns_500(self):
        # Simulate an ONNX parser failure without modifying the model file.
        with (
            patch.object(photo, "_session", photo._session.__wrapped__),
            patch.object(
                photo.ort,
                "InferenceSession",
                side_effect=InvalidProtobuf("mock corrupt ONNX"),
            ) as load,
        ):
            response = self.post(self.png)

        load.assert_called_once()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.text, "Internal Server Error")


if __name__ == "__main__":
    unittest.main()