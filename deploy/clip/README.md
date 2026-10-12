# FashionVision CLIP backend on Vercel Hobby

This directory is the root of a **separate Vercel project** for quantized CLIP
ViT-B/32 zero-shot inference. It does not serve or build React, import TensorFlow,
or change the MobileNet deployment. The canonical classifier, custom tokenizer,
prompts, preprocessing, class aggregation, confidence thresholds, and response
format in `../../ml/photo_classifier.py` remain unchanged.

## Exact project settings

Import the **whole FashionVision repository** as a new Vercel project. Do not
upload just this folder: the build requires canonical files outside it.

| Setting | Value |
| --- | --- |
| Plan | Hobby |
| Root Directory | `deploy/clip` |
| Framework Preset | FastAPI |
| Include source files outside Root Directory in Build Step | Enabled |
| Python | 3.12, selected by `.python-version` |
| Install Command | Default framework dependency installation from this directory's `requirements.txt`; leave override off |
| Build Command | `python stage_runtime.py` (configured in `vercel.json`) |
| Output Directory | Framework default; leave override off |
| Fluid Compute | Enabled; Active CPU enabled |
| Memory / CPU | Hobby's 2 GB / 1 vCPU |
| Maximum duration | 300 seconds, configured for `app.py` |

Set **`VERCEL_SUPPORT_LARGE_FUNCTIONS=0`** for Production and Preview to enforce
the standard Python 500 MB limit instead of relying on the Large Functions beta.
No model-download token or TensorFlow configuration is required. Raw inference
does not require Blob. Optional private-upload inference uses the connected store's
`BLOB_STORE_ID`, runtime OIDC and shared Redis/security settings; it does not require
a static Blob token. Keep `PRIVATE_UPLOADS_ENABLED=0` or absent. The official
lightweight OIDC dependencies are listed in `requirements.txt`; no full Vercel
platform SDK is added. Do not copy unrelated environment files or secrets.

Vercel discovers the exported `app` in this directory's root `app.py`. It is
an API-only FastAPI application with no React production-build dependency.

## Build-time model provisioning and integrity

The build stages byte-identical copies of the canonical classifier, downloader,
tokenizer assets, model attribution, and project rights notice inside this
project root. Paths then resolve locally using the existing modules' own
`__file__` logic. Only the CLIP files are staged.

The root repository already ignores the ONNX model. This directory's
`.gitignore` also ignores all generated runtime copies, including the ONNX
file. **Never force-add either model path to Git.**

When the verified model exists in a local checkout, staging reuses it. In a Git
deployment, the unchanged `scripts/download_photo_model.py` downloads the
missing quantized model from
[Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32)
during the build. It uses HTTPS, writes a partial file, verifies SHA-256, then
publishes the verified file. Existing tokenizer files are verified too.
A missing source, failed download, or hash mismatch fails the build.

The ONNX SHA-256 remains:

```text
0898a3facfdb27f0a041e57649b4989cfd094e4a0040d6ae75ed69917dfc7328
```

All three expected hashes come from the existing downloader. The adapter
rechecks all three assets before initializing inference in each worker.
There are **no request-time model downloads**, model conversions, or training.
The upstream URL uses `main`; if upstream bytes change, verification fails
closed until the original approved artifact is restored. Do not update the
checksum to bypass that failure.

## Bundle feasibility

Estimate prepared on October 12, 2026 for Python 3.12, Linux x86_64. The
uncompressed file sizes were read from the central directories of the
matching published PyPI wheels, including the full runtime dependency closure;
no packages were installed for this estimate.

| Inputs | Bytes | Decimal MB |
| --- | ---: | ---: |
| Quantized ONNX model | 153,695,702 | 153.70 |
| Vocabulary and merges | 1,386,947 | 1.39 |
| All 20 pinned runtime distributions | 153,072,714 | 153.07 |
| Known subtotal, before small application files and platform overhead | 308,155,363 | 308.16 |

The subtotal is approximately **293.88 MiB**. Allowing for application files,
compiled bytecode and Vercel runtime overhead, a provisional complete-bundle
budget is **about 330–380 MB**, below the standard 500 MB Python limit.
The complete Vercel bundle has **not been built or measured**; only Vercel's
packaging output can confirm the final size. Keep Large Functions disabled for
the first deployment so an oversized bundle fails rather than silently relying
on beta support.

The build reports actual application/assets plus installed runtime distribution
sizes and fails if those known inputs reach 500,000,000 bytes. This is a
conservative preflight, **not a substitute for Vercel's final size check**.
Python dependencies are not automatically tree-shaken; no test packages or
TensorFlow extras are declared.

Published wheels: [ONNX Runtime](https://pypi.org/project/onnxruntime/1.31.0/),
[NumPy](https://pypi.org/project/numpy/2.5.4/),
[Pillow](https://pypi.org/project/pillow/12.3.0/).

## Endpoints and inference behavior

- `POST /api/predict`: the existing raw image request contract, accepting
  JPEG, PNG, WebP and BMP. It returns the unchanged `prediction`,
  `confidence`, `uncertain`, `predictions`, and `model` fields.
- `GET /api/health` (also `GET /`): retains `status` and
  `photo_model_ready`. Readiness requires verified assets, successful ONNX
  initialization, tokenizer loading and actual inference. A failed probe returns
  HTTP 503 and `photo_model_ready: false`; success is cached per worker.
- `/docs` and `/openapi.json`: FastAPI's API documentation.

Supported image-validation status codes and messages remain unchanged.
Missing model assets return 503; other model availability/integrity failures
also return 503 instead of claiming readiness. Diagnostic logs contain backend
exceptions, not uploaded image bytes, filenames, or prediction results.
Inference is serialized within each worker to bound concurrent tensor memory.
No prediction algorithms, session optimization settings or accuracy claims
have been changed.

CORS allows only `https://fashion-vision.vercel.app` for GET/POST and
Content-Type. The frontend's `frontend/react/vercel.json` now configures exact
rewrites for `/api/predict` and `/api/health` to
`https://fashion-vision-clip.vercel.app`. React keeps its same-origin requests;
the routing takes effect on the next frontend deployment. MobileNet and
Grad-CAM paths are not rewritten to CLIP. See the
[frontend routing instructions](../../README.md#production-frontend-routing-vercel).

## Remaining limits before production use

**Uploads:** The API still permits 15 MiB, without compression. Small images
use the original raw endpoint. The additional `POST /api/predict/blob` accepts
only an opaque, single-use receipt and privately retrieves the authorized
object. No arbitrary fetch URL or public image endpoint is added.
The shared transport module is staged with no new Python dependencies.
Configure the private store, free Redis state and secrets before enabling
large uploads. See [private upload setup](../../docs/PRIVATE_UPLOADS.md).
Direct raw requests above Vercel's 4.5 MB cap remain rejected by the platform.

**Cold starts and memory:** Prior Windows profiling recorded approximately
343 MiB peak resident RAM, 519 MiB peak private commit, and 4.82 seconds from
process start through first CLIP prediction. These are preliminary local
measurements, not Linux/Vercel guarantees. Hobby has one vCPU; loading a 154 MB
artifact and ONNX initialization may be slower. Measure cold and warm requests,
memory, and timeout behavior on Vercel before treating the backend as production
ready. Large or extreme-dimension images can consume substantially more memory
than the profiling input even when their compressed upload is small.

**Build availability and free usage:** Builds need outbound access to the
public Hugging Face artifact. No cloud download or deployment has been attempted
during preparation. Hobby's free compute/request quotas still apply; successful
bundle sizing does not guarantee unlimited free inference.

**Attribution:** The staged model README preserves existing attribution.
Project rights remain in the staged NOTICE.md; third-party packages and model
assets retain their respective licenses. Required distribution notices remain
with the dependency files.

Vercel references:
[FastAPI entrypoints and build commands](https://vercel.com/docs/frameworks/backend/fastapi),
[Python packaging](https://vercel.com/docs/functions/runtimes/python),
[function limits](https://vercel.com/docs/functions/limitations),
[shared source access](https://vercel.com/docs/monorepos/monorepo-faq).
