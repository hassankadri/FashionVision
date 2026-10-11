from __future__ import annotations

import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app


DEFAULT_REFERENCE = Path(
    r"C:\Users\amaan\Downloads\bcf22fb3-a1c5-4e92-a7eb-44da4c0735f9.png"
)
REFERENCE_PHOTO = Path(os.environ.get("FASHION_TEST_IMAGE", DEFAULT_REFERENCE))


class FashionAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_reports_photo_model_ready(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["photo_model_ready"])

    def test_non_image_payload_is_rejected(self) -> None:
        response = self.client.post(
            "/api/predict",
            content=b"not an image",
            headers={"content-type": "text/plain"},
        )

        self.assertEqual(response.status_code, 415)

    @unittest.skipUnless(
        REFERENCE_PHOTO.exists(), "reference pullover photo not available"
    )
    def test_reference_fleece_is_not_misclassified_as_shirt(self) -> None:
        response = self.client.post(
            "/api/predict",
            content=REFERENCE_PHOTO.read_bytes(),
            headers={"content-type": "image/png"},
        )

        self.assertEqual(response.status_code, 200, response.text)

        result = response.json()
        self.assertEqual(result["prediction"], "Pullover")

        scores = {
            item["label"]: item["probability"] for item in result["predictions"]
        }
        self.assertGreater(scores["Pullover"], scores["Shirt"])
        self.assertTrue(
            result["uncertain"], "Coat should remain visible as a close alternative"
        )


if __name__ == "__main__":
    unittest.main()