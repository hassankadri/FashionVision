"""Authenticated temporary Blob transport; inference modules remain unchanged."""

from contextlib import contextmanager
import hashlib
import hmac
import io
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
import warnings

from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image

MAX_BYTES = 15 * 1024 * 1024
MIME_FORMATS = {"image/jpeg": ("jpg", "JPEG"), "image/png": ("png", "PNG"),
                "image/webp": ("webp", "WEBP"), "image/bmp": ("bmp", "BMP")}
PREFIX = "fashionvision-temp/v1/"
_logger = logging.getLogger(__name__)

CLAIM = """
local value=redis.call('GET',KEYS[1])
if not value then return '' end
local record=cjson.decode(value)
if record.receiptHash ~= ARGV[1] or record.purpose ~= ARGV[2] or record.status ~= 'issued'
   or tonumber(record.expires) <= tonumber(redis.call('TIME')[1]) then return '' end
record.status='processing'
redis.call('SET',KEYS[1],cjson.encode(record),'EX',172800)
return cjson.encode(record)
"""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def deployment():
    environment = os.environ.get("VERCEL_ENV")
    frontend = os.environ.get("UPLOAD_FRONTEND_ORIGIN", "")
    namespace = os.environ.get("PRIVATE_UPLOAD_NAMESPACE", "")
    if environment == "production":
        if namespace or frontend not in ("", "https://fashion-vision.vercel.app"):
            raise HTTPException(503, "Deployment routing is not configured.")
        return {"environment": environment, "frontend_origin": "https://fashion-vision.vercel.app",
                "prefix": PREFIX, "redis_prefix": "fv:"}
    if (environment != "preview" or not re.fullmatch(
            r"https://fashion-vision-(?!clip-)[a-z0-9]+(?:-[a-z0-9]+)*\.vercel\.app", frontend)
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", namespace) or len(namespace) > 64):
        raise HTTPException(503, "Deployment routing is not configured.")
    return {"environment": environment, "frontend_origin": frontend,
            "prefix": f"fashionvision-temp/preview/{namespace}/",
            "redis_prefix": f"fv:preview:{namespace}:"}


def settings():
    redis_url = os.environ.get("UPSTASH_REDIS_REST_URL", "")
    secret = os.environ.get("UPLOAD_AUTH_SECRET", "")
    store_id = os.environ.get("BLOB_STORE_ID", "").lower().removeprefix("store_")
    blob_host = f"{store_id}.private.blob.vercel-storage.com"
    expected_host = os.environ.get("BLOB_PRIVATE_HOST")
    try:
        parsed = urllib.parse.urlsplit(redis_url)
        valid_url = (parsed.scheme == "https" and (parsed.hostname or "").endswith(".upstash.io")
                     and not parsed.username and not parsed.password and not parsed.query
                     and not parsed.fragment and parsed.path in ("", "/") and parsed.port in (None, 443))
    except ValueError:
        valid_url = False
    if (os.environ.get("PRIVATE_UPLOADS_ENABLED") != "1" or not valid_url
            or not re.fullmatch(r"[a-f0-9]{64}", secret)
            or os.environ.get("VERCEL") != "1" or not re.fullmatch(r"[a-z0-9]+", store_id)
            or (expected_host and expected_host != blob_host)
            or not os.environ.get("UPSTASH_REDIS_REST_TOKEN")):
        raise HTTPException(status_code=503, detail="Private image uploads are not configured.")
    return {"redis_url": redis_url.rstrip("/"), "redis_token": os.environ["UPSTASH_REDIS_REST_TOKEN"],
            "blob_host": blob_host, "secret": secret, **deployment()}


@contextmanager
def oidc_context(headers):
    # Bind the runtime's current request token; AnyIO copies this context to its worker.
    try:
        from vercel.headers import HeadersContext
    except ImportError as error:
        raise HTTPException(503, "Private Blob OIDC support is unavailable.") from error
    with HeadersContext(headers).use():
        yield


def oidc_token():
    try:
        from vercel.oidc.token import get_vercel_oidc_token
        token = get_vercel_oidc_token()
        if not isinstance(token, str) or not token:
            raise ValueError("Missing OIDC credential")
        return token
    except Exception as error:
        # No static-token fallback and no credentials in response/log messages.
        raise HTTPException(503, "Private Blob OIDC authentication is unavailable.") from error


def _json_request(url, value, headers):
    request = urllib.request.Request(url, data=json.dumps(value).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=10) as response:
        content = response.read(65537)
        if len(content) > 65536:
            raise ValueError("Control response exceeds limit")
        return json.loads(content)


def redis(command, config):
    try:
        result = _json_request(config["redis_url"], command,
                               {"Authorization": f"Bearer {config['redis_token']}"})
        if "error" in result or "result" not in result:
            raise ValueError("Redis operation failed")
        return result["result"]
    except Exception as error:
        raise HTTPException(status_code=503, detail="Upload authorization is temporarily unavailable.") from error


def receipt_body(body):
    if (not isinstance(body, dict) or set(body) != {"uploadId", "receipt"}
            or not isinstance(body.get("uploadId"), str)
            or not isinstance(body.get("receipt"), str)
            or not re.fullmatch(r"[a-f0-9]{32}", body["uploadId"])
            or not re.fullmatch(r"[a-f0-9]{64}", body["receipt"])):
        raise HTTPException(status_code=403, detail="Invalid or expired upload authorization.")
    return body


def claim(ticket, purpose, config):
    receipt_body(ticket)
    digest = hashlib.sha256(ticket["receipt"].encode()).hexdigest()
    value = redis(["EVAL", CLAIM, 1, f"{config['redis_prefix']}upload:v1:{ticket['uploadId']}", digest, purpose], config)
    try:
        record = json.loads(value) if value else {}
        mime = record["mime"]
        if (record["receiptHash"] != digest or record["purpose"] != purpose or record["status"] != "processing"
                or type(record["size"]) is not int or not 0 < record["size"] <= MAX_BYTES
                or mime not in MIME_FORMATS
                or record["pathname"] != f"{config['prefix']}{ticket['uploadId']}.{MIME_FORMATS[mime][0]}"):
            raise ValueError("Invalid authorization record")
        return record
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=403, detail="Invalid or expired upload authorization.")


def fetch_image(record, config):
    # Neither a URL nor a pathname is accepted from the browser.
    url = f"https://{config['blob_host']}/{record['pathname']}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {oidc_token()}",
                                                  "Cache-Control": "no-store"})
    try:
        started = time.monotonic()
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=20) as response:
            if response.status != 200:
                raise ValueError("Private image is unavailable")
            mime = response.headers.get("Content-Type", "").split(";")[0].lower()
            length = response.headers.get("Content-Length")
            if mime != record["mime"] or (length is not None and int(length) != record["size"]):
                raise HTTPException(status_code=400, detail="Uploaded image metadata does not match its authorization.")
            output = bytearray()
            while True:
                chunk = response.read(min(65536, MAX_BYTES + 1 - len(output)))
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Images must be 15 MB or smaller.")
                if time.monotonic() - started > 60:
                    raise TimeoutError("Private image read timed out")
            if len(output) != record["size"]:
                raise HTTPException(status_code=400, detail="Uploaded image size does not match its authorization.")
            return bytes(output)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=503, detail="The authorized image could not be retrieved.") from error


def validate_image(payload, mime):
    try:
        if not 0 < len(payload) <= MAX_BYTES or mime not in MIME_FORMATS:
            raise ValueError("Unsupported image")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                if (image.format != MIME_FORMATS[mime][1] or min(width, height) < 1
                        or max(width, height) > 8192 or width * height > 16_000_000):
                    raise ValueError("Invalid image format or dimensions")
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a supported, readable image.") from error


def cleanup(ticket, config):
    proof = hmac.new(config["secret"].encode(),
                     f"cleanup:{ticket['uploadId']}:{ticket['receipt']}".encode(), hashlib.sha256).hexdigest()
    headers = {"X-FV-Cleanup-Proof": proof}
    if config["environment"] == "preview" and os.environ.get("FRONTEND_PREVIEW_BYPASS_SECRET"):
        headers["x-vercel-protection-bypass"] = os.environ["FRONTEND_PREVIEW_BYPASS_SECRET"]
    for _attempt in range(2):
        try:
            result = _json_request(config["frontend_origin"] + "/api/uploads/cleanup", ticket, headers)
            if result.get("status") == "deleted":
                return True
        except Exception:
            pass
    _logger.warning("Temporary image deletion not confirmed; scheduled cleanup required")
    return False


def run_inference(ticket, purpose, handler):
    config = settings()
    record = claim(ticket, purpose, config)
    try:
        payload = fetch_image(record, config)
        validate_image(payload, record["mime"])
        return handler(payload, record["mime"])
    finally:
        cleanup(ticket, config)


def register_blob_routes(app, handlers):
    for path, (purpose, handler) in handlers.items():
        def make_endpoint(selected_purpose, selected_handler):
            async def endpoint(request: Request):
                if request.headers.get("content-type", "").split(";")[0] != "application/json":
                    raise HTTPException(status_code=415, detail="Use application/json.")
                payload = bytearray()
                async for chunk in request.stream():
                    if len(payload) + len(chunk) > 2048:
                        raise HTTPException(status_code=413, detail="Request is too large.")
                    payload.extend(chunk)
                try:
                    ticket = receipt_body(json.loads(payload))
                except (ValueError, UnicodeDecodeError):
                    raise HTTPException(status_code=400, detail="Invalid upload request.")
                if os.environ.get("PRIVATE_UPLOADS_ENABLED") != "1":
                    raise HTTPException(503, "Private image uploads are not configured.")
                with oidc_context(dict(request.headers)):
                    result = await run_in_threadpool(run_inference, ticket, selected_purpose, selected_handler)
                return JSONResponse(result, headers={"Cache-Control": "no-store"})
            return endpoint
        app.add_api_route(path, make_endpoint(purpose, handler), methods=["POST"])
