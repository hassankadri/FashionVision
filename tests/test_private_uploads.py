"""Offline transport/security checks; no cloud calls, model loading or training."""
import hashlib
import io
import json
import unittest
from unittest.mock import patch
from contextlib import contextmanager
from types import SimpleNamespace
from PIL import Image
from fastapi import HTTPException
from backend import private_uploads as uploads

TICKET = {"uploadId": "a" * 32, "receipt": "b" * 64}
CONFIG = {"redis_url": "https://example.upstash.io", "redis_token": "test",
          "blob_host": "examplestore.private.blob.vercel-storage.com",
          "secret": "c" * 64, "environment": "production", "prefix": uploads.PREFIX,
          "redis_prefix": "fv:", "frontend_origin": "https://fashion-vision.vercel.app"}


def png():
    output = io.BytesIO()
    Image.new("RGB", (32, 24), (20, 30, 40)).save(output, "PNG")
    return output.getvalue()


def record(size):
    return {"receiptHash": hashlib.sha256(TICKET["receipt"].encode()).hexdigest(),
            "purpose": "clip", "status": "processing", "size": size, "mime": "image/png",
            "pathname": uploads.PREFIX + TICKET["uploadId"] + ".png"}


class Response(io.BytesIO):
    status = 200
    def __init__(self, payload, mime="image/png", length=None):
        super().__init__(payload)
        self.headers = {"Content-Type": mime}
        if length is not None:
            self.headers["Content-Length"] = str(length)


class PrivateUploadChecks(unittest.TestCase):
    def test_preview_configuration_and_secret_validation(self):
        environment = {"PRIVATE_UPLOADS_ENABLED": "1", "VERCEL": "1", "VERCEL_ENV": "preview",
                       "BLOB_STORE_ID": "store_examplestore", "UPSTASH_REDIS_REST_URL": CONFIG["redis_url"],
                       "UPSTASH_REDIS_REST_TOKEN": "test", "UPLOAD_AUTH_SECRET": CONFIG["secret"],
                       "PRIVATE_UPLOAD_NAMESPACE": "test-private-uploads-oidc",
                       "UPLOAD_FRONTEND_ORIGIN": "https://fashion-vision-git-test-team.vercel.app"}
        with patch.dict("os.environ", environment, clear=True):
            config = uploads.settings()
            self.assertEqual(config["redis_prefix"], "fv:preview:test-private-uploads-oidc:")
            self.assertEqual(config["prefix"], "fashionvision-temp/preview/test-private-uploads-oidc/")
        for invalid in ({"UPLOAD_AUTH_SECRET": "YWJjZA==" * 6}, {"UPLOAD_AUTH_SECRET": "A" * 64},
                        {"UPLOAD_FRONTEND_ORIGIN": "https://fashion-vision.vercel.app"},
                        {"UPLOAD_FRONTEND_ORIGIN": environment["UPLOAD_FRONTEND_ORIGIN"] + "/"},
                        {"UPLOAD_FRONTEND_ORIGIN": "https://attacker.vercel.app"},
                        {"PRIVATE_UPLOAD_NAMESPACE": ""}, {"VERCEL_ENV": ""}):
            with patch.dict("os.environ", {**environment, **invalid}, clear=True), self.assertRaises(HTTPException):
                uploads.settings()

    def test_preview_claim_and_cleanup_are_isolated_from_production(self):
        config = {**CONFIG, "environment": "preview", "redis_prefix": "fv:preview:test-private-uploads-oidc:",
                  "prefix": "fashionvision-temp/preview/test-private-uploads-oidc/",
                  "frontend_origin": "https://fashion-vision-git-test-team.vercel.app"}
        preview_record = {**record(100), "pathname": config["prefix"] + TICKET["uploadId"] + ".png"}
        with patch.object(uploads, "redis", return_value=json.dumps(preview_record)) as redis:
            uploads.claim(TICKET, "clip", config)
            self.assertEqual(redis.call_args.args[0][3], config["redis_prefix"] + "upload:v1:" + TICKET["uploadId"])
        with patch.object(uploads, "redis", return_value=json.dumps(record(100))), self.assertRaises(HTTPException):
            uploads.claim(TICKET, "clip", config)
        with patch.dict("os.environ", {"FRONTEND_PREVIEW_BYPASS_SECRET": "test-bypass"}, clear=True), \
                patch.object(uploads, "_json_request", return_value={"status": "deleted"}) as request:
            self.assertTrue(uploads.cleanup(TICKET, config))
            self.assertEqual(request.call_args.args[0], config["frontend_origin"] + "/api/uploads/cleanup")
            self.assertEqual(request.call_args.args[1], TICKET)
            self.assertEqual(request.call_args.args[2]["x-vercel-protection-bypass"], "test-bypass")
            self.assertEqual(len(request.call_args.args[2]["X-FV-Cleanup-Proof"]), 64)
            self.assertTrue(uploads.cleanup(TICKET, CONFIG))
            self.assertEqual(request.call_args.args[0], "https://fashion-vision.vercel.app/api/uploads/cleanup")
            self.assertNotIn("x-vercel-protection-bypass", request.call_args.args[2])

    def test_disabled_fails_closed(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(HTTPException) as failure:
            uploads.settings()
        self.assertEqual(failure.exception.status_code, 503)

    def test_arbitrary_object_references_rejected(self):
        for value in ({**TICKET, "url": "https://example.com/image"}, {**TICKET, "pathname": "../private"},
                      {"uploadId": "../secret", "receipt": TICKET["receipt"]}, None):
            with self.subTest(value=value), self.assertRaises(HTTPException) as failure:
                uploads.receipt_body(value)
            self.assertEqual(failure.exception.status_code, 403)

    def test_claim_rejects_replay_wrong_purpose_and_paths(self):
        for value in ("", json.dumps({**record(100), "purpose": "gradcam"}),
                      json.dumps({**record(100), "pathname": "another-user.png"}),
                      json.dumps({**record(100), "status": "issued"})):
            with patch.object(uploads, "redis", return_value=value), self.assertRaises(HTTPException):
                uploads.claim(TICKET, "clip", CONFIG)

    def test_claim_uses_atomic_receipt_hash_and_operation(self):
        with patch.object(uploads, "redis", return_value=json.dumps(record(100))) as redis:
            self.assertEqual(uploads.claim(TICKET, "clip", CONFIG)["size"], 100)
        command = redis.call_args.args[0]
        self.assertEqual(command[0], "EVAL")
        self.assertEqual(command[-2:], [record(100)["receiptHash"], "clip"])
        self.assertIn("record.status ~= 'issued'", command[1])
        self.assertIn("redis.call('TIME')", command[1])

    def test_image_type_actual_format_and_dimension_validation(self):
        uploads.validate_image(png(), "image/png")
        for payload, mime in ((png(), "image/jpeg"), (b"broken", "image/png"), (png(), "image/svg+xml")):
            with self.assertRaises(HTTPException) as failure:
                uploads.validate_image(payload, mime)
            self.assertEqual(failure.exception.status_code, 400)
        large = io.BytesIO()
        Image.new("RGB", (8193, 1)).save(large, "PNG")
        with self.assertRaises(HTTPException):
            uploads.validate_image(large.getvalue(), "image/png")

    def test_private_read_uses_fixed_host_and_preserves_bytes(self):
        payload = png()
        with patch.object(uploads, "oidc_token", return_value="fresh-oidc"), \
                patch("urllib.request.OpenerDirector.open", return_value=Response(payload, length=len(payload))) as opening:
            self.assertEqual(uploads.fetch_image(record(len(payload)), CONFIG), payload)
        request = opening.call_args.args[0]
        self.assertEqual(request.full_url, "https://" + CONFIG["blob_host"] + "/" + record(1)["pathname"])
        self.assertEqual(request.get_header("Authorization"), "Bearer fresh-oidc")
        self.assertIsNone(uploads.NoRedirect().redirect_request(None, None, 302, "", {}, "https://attacker.example"))

    def test_private_read_rejects_size_and_mime_mismatch(self):
        for response in (Response(png(), "image/jpeg"), Response(png(), length=100), Response(png())):
            with patch.object(uploads, "oidc_token", return_value="fresh-oidc"), \
                    patch("urllib.request.OpenerDirector.open", return_value=response), self.assertRaises(HTTPException):
                uploads.fetch_image(record(100), CONFIG)

    def test_cleanup_on_fetch_validation_model_failure_and_success(self):
        payload = png()
        for failure in ("fetch", "validate", "model", None):
            with self.subTest(failure=failure), patch.object(uploads, "settings", return_value=CONFIG), \
                    patch.object(uploads, "claim", return_value=record(len(payload))), \
                    patch.object(uploads, "fetch_image", return_value=payload) as fetching, \
                    patch.object(uploads, "validate_image") as validation, \
                    patch.object(uploads, "cleanup", return_value=True) as cleaning:
                def handler(body, mime):
                    self.assertEqual(body, payload)
                    self.assertEqual(mime, "image/png")
                    if failure == "model":
                        raise RuntimeError("model unavailable")
                    return {"prediction": "unchanged", "confidence": 0.8}
                if failure == "fetch": fetching.side_effect = RuntimeError("fetch unavailable")
                if failure == "validate": validation.side_effect = HTTPException(400, "invalid")
                if failure:
                    with self.assertRaises((RuntimeError, HTTPException)):
                        uploads.run_inference(TICKET, "clip", handler)
                else:
                    self.assertEqual(uploads.run_inference(TICKET, "clip", handler),
                                     {"prediction": "unchanged", "confidence": 0.8})
                cleaning.assert_called_once_with(TICKET, CONFIG)

    def test_invalid_receipt_never_reads_or_deletes_other_objects(self):
        with patch.object(uploads, "settings", return_value=CONFIG), \
                patch.object(uploads, "claim", side_effect=HTTPException(403, "invalid")), \
                patch.object(uploads, "fetch_image") as fetching, patch.object(uploads, "cleanup") as cleaning:
            with self.assertRaises(HTTPException):
                uploads.run_inference(TICKET, "clip", lambda *_: {})
        fetching.assert_not_called()
        cleaning.assert_not_called()

    def test_cleanup_failure_does_not_replace_prediction(self):
        with patch.object(uploads, "_json_request", side_effect=OSError), \
                self.assertLogs(uploads._logger, level="WARNING"):
            self.assertFalse(uploads.cleanup(TICKET, CONFIG))

    def test_settings_use_connected_store_without_static_tokens(self):
        environment = {"PRIVATE_UPLOADS_ENABLED": "1", "VERCEL": "1", "VERCEL_ENV": "production", "BLOB_STORE_ID": "store_examplestore",
                       "UPSTASH_REDIS_REST_URL": CONFIG["redis_url"], "UPSTASH_REDIS_REST_TOKEN": "test",
                       "UPLOAD_AUTH_SECRET": CONFIG["secret"]}
        with patch.dict("os.environ", environment, clear=True):
            self.assertEqual(uploads.settings()["blob_host"], CONFIG["blob_host"])
            self.assertNotIn("blob_token", uploads.settings())
        for invalid in ({"BLOB_STORE_ID": "../private"}, {"BLOB_PRIVATE_HOST": "other.private.blob.vercel-storage.com"},
                        {"VERCEL": "0"}, {"PRIVATE_UPLOADS_ENABLED": "0"}):
            with patch.dict("os.environ", {**environment, **invalid}, clear=True), self.assertRaises(HTTPException):
                uploads.settings()

    def test_oidc_helper_resolves_current_token_each_time_without_static_fallback(self):
        values = iter(["fresh-request-one", "fresh-request-two"])
        helper = SimpleNamespace(get_vercel_oidc_token=lambda: next(values))
        with patch.dict("sys.modules", {"vercel.oidc.token": helper}):
            self.assertEqual(uploads.oidc_token(), "fresh-request-one")
            self.assertEqual(uploads.oidc_token(), "fresh-request-two")
        def unavailable():
            raise RuntimeError("must-never-appear-in-response")
        with patch.dict("sys.modules", {"vercel.oidc.token": SimpleNamespace(get_vercel_oidc_token=unavailable)}), \
                patch.dict("os.environ", {"BLOB_READ_WRITE_TOKEN": "unused-static-token"}):
            with self.assertRaises(HTTPException) as failure:
                uploads.oidc_token()
            self.assertEqual(failure.exception.status_code, 503)
            self.assertNotIn("must-never", failure.exception.detail)

    def test_request_context_resets_after_failure(self):
        active = []
        class Context:
            def __init__(self, headers):
                self.headers = headers
            @contextmanager
            def use(self):
                active.append(self.headers)
                try:
                    yield
                finally:
                    active.pop()
        headers = {"x-vercel-oidc-token": "fresh-runtime-token"}
        with patch.dict("sys.modules", {"vercel.headers": SimpleNamespace(HeadersContext=Context)}):
            with self.assertRaises(RuntimeError):
                with uploads.oidc_context(headers):
                    self.assertEqual(active, [headers])
                    raise RuntimeError("inference failed")
        self.assertEqual(active, [])

    def test_oidc_failure_still_cleans_up_claimed_object(self):
        with patch.object(uploads, "settings", return_value=CONFIG), \
                patch.object(uploads, "claim", return_value=record(100)), \
                patch.object(uploads, "oidc_token", side_effect=HTTPException(503, "OIDC unavailable")), \
                patch.object(uploads, "cleanup", return_value=True) as cleaning, \
                patch("urllib.request.OpenerDirector.open") as opening:
            with self.assertRaises(HTTPException):
                uploads.run_inference(TICKET, "clip", lambda *_: {})
            opening.assert_not_called()
            cleaning.assert_called_once_with(TICKET, CONFIG)

    def test_blob_route_carries_oidc_context_into_worker_and_resets_it(self):
        import asyncio
        from contextvars import ContextVar
        from fastapi import FastAPI, Request
        active = ContextVar("test_request_headers", default=None)
        class Context:
            def __init__(self, headers):
                self.headers = headers
            @contextmanager
            def use(self):
                reset = active.set(self.headers)
                try:
                    yield
                finally:
                    active.reset(reset)
        app = FastAPI()
        uploads.register_blob_routes(app, {"/api/predict/blob": ("clip", lambda *_: {})})
        endpoint = next(route.endpoint for route in app.routes if route.path == "/api/predict/blob")
        expected = {"prediction": "unchanged", "confidence": 0.8}
        scope = {"type": "http", "method": "POST", "path": "/api/predict/blob",
                 "headers": [(b"content-type", b"application/json"), (b"x-vercel-oidc-token", b"runtime-current")]}
        async def receive():
            return {"type": "http.request", "body": json.dumps(TICKET).encode(), "more_body": False}
        def worker(ticket, purpose, handler):
            self.assertEqual(active.get()["x-vercel-oidc-token"], "runtime-current")
            self.assertEqual(ticket, TICKET); self.assertEqual(purpose, "clip")
            return expected
        async def invoke():
            result = await endpoint(Request(scope, receive))
            self.assertEqual(json.loads(result.body), expected)
            self.assertIsNone(active.get())
        with patch.dict("sys.modules", {"vercel.headers": SimpleNamespace(HeadersContext=Context)}), \
                patch.dict("os.environ", {"PRIVATE_UPLOADS_ENABLED": "1"}), \
                patch.object(uploads, "run_inference", side_effect=worker):
            asyncio.run(invoke())

    def test_disabled_blob_route_needs_no_oidc_dependency(self):
        import asyncio
        from fastapi import FastAPI, Request
        app = FastAPI()
        uploads.register_blob_routes(app, {"/api/predict/blob": ("clip", lambda *_: {})})
        endpoint = next(route.endpoint for route in app.routes if route.path == "/api/predict/blob")
        scope = {"type": "http", "method": "POST", "path": "/api/predict/blob",
                 "headers": [(b"content-type", b"application/json")]}
        async def receive():
            return {"type": "http.request", "body": json.dumps(TICKET).encode(), "more_body": False}
        with patch.dict("os.environ", {"PRIVATE_UPLOADS_ENABLED": "0"}), \
                patch.object(uploads, "oidc_context") as context, \
                patch.object(uploads, "run_inference") as inference:
            with self.assertRaises(HTTPException) as failure:
                asyncio.run(endpoint(Request(scope, receive)))
            self.assertEqual(failure.exception.status_code, 503)
        context.assert_not_called(); inference.assert_not_called()


if __name__ == "__main__":
    unittest.main()
