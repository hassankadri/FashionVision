# FashionVision MobileNet backend on Vercel

This is a separate Vercel Python project for MobileNetV3Small and Grad-CAM.
It reuses the canonical inference implementation and checkpoint without changes.
CLIP, its downloader and tokenizer, the original FastAPI app and React are not
part of this deployment.

## Vercel project settings

Create a **new backend project** from the repository; keep the existing React
project and its configuration unchanged.

| Setting | Value |
| --- | --- |
| Root Directory | `deploy/mobilenet` |
| Framework Preset | FastAPI (`fastapi`) |
| Include source files outside of the Root Directory in the Build Step | **Enabled** |
| Build Command | `python stage_runtime.py` (configured in `vercel.json`) |
| Install Command | Framework default; installs this directory's `requirements.txt` |
| Output Directory | Framework default; leave the override disabled |
| Python | 3.12, selected by `.python-version` |
| Fluid Compute | Enabled by `fluid: true`; Active CPU must be enabled |
| Function duration | 300 seconds |
| Hobby memory | 2 GB / 1 vCPU; not increased by Large Functions |

Set these environment variables in the **new backend project**, for Production
and Preview:

| Variable | Value | Purpose |
| --- | --- | --- |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | `1` | Opt into the public beta supporting Python bundles up to 5 GB |
| `KERAS_BACKEND` | `tensorflow` | Select the backend used by the existing model |
| `KERAS_HOME` | `/tmp/keras` | Keep any Keras configuration/cache writes in writable temporary storage |

Raw inference does not require Blob. The prepared private-upload routes use
OIDC and the official lightweight Python OIDC helper, without a static Blob token.
Connect the existing **private** store to this backend project in the intended
environment and configure the shared Redis/security settings from
[private upload setup](../../docs/PRIVATE_UPLOADS.md). Keep
`PRIVATE_UPLOADS_ENABLED=0` or absent. Never put runtime tokens or server secrets
in React, a `VITE_*` variable or Git. Do not change the store to public.

Use Git import from the whole repository with the source-files option above
enabled. Uploading only this directory omits its canonical build inputs.
For a future CLI deployment, invoke the CLI from the repository root and link
to the new backend project with its Root Directory configured; do not upload
only this subfolder or relink the React project.

## Build and runtime layout

The build command runs after dependency installation. It reads only the
following canonical inputs from the repository:

- `ml/__init__.py`, `ml/mobilenet_classifier.py`, `ml/mobilenet_gradcam.py`
- `fashion_rgb_data/__init__.py`, `fashion_rgb_data/pipeline.py`
- `models/mobilenet-family-v1/best.keras`
- `models/mobilenet-family-v1/configuration.json`
- `NOTICE.md`

It checks the model and configuration against the SHA-256 constants in the
canonical classifier, then creates byte-identical copies under this directory.
All copies are verified. These generated paths are Git-ignored and should never
be edited as a second source of truth. At runtime both modules and model assets
are inside the Vercel project root; no parent-directory imports or external
filesystem paths are required. The existing classifier verifies asset hashes
again before loading the model.

The backend exports `app` from `app.py`, which Vercel discovers directly.
No React build, static mount, GPU, model download or training step is required.
TensorFlow 2.21.0 and Keras 3.15.1 are pinned to the existing model environment.

## API and health

- `POST /api/predict/mobilenet`
- `POST /api/explain/mobilenet`
- `GET /api/health` (also available at `/`)

Prediction and explanation accept the original raw image bytes and MIME types,
retain the 15 MiB allowance, and delegate to the unchanged classifier. Response
fields and error contracts remain the same.

Health verifies actual model initialization and inference, then runs a real
Grad-CAM probe on a synthetic model input. It returns HTTP 200 only when both
probes succeed, with `mobilenet_model_ready: true` and `gradcam_ready: true`.
Missing/corrupt assets, missing TensorFlow/Keras or inference/explanation
failures return HTTP 503. Successful readiness is cached per instance;
prediction/explanation failures invalidate it. Existing model-loading failures
remain cached by the classifier until instance restart. Health is not a
concurrency or memory stress test. Probe images are not stored.

`photo_model_ready` is always false because this backend does not load CLIP.
The original combined backend's health endpoint remains unchanged. CORS allows
only `https://fashion-vision.vercel.app`, GET/POST and Content-Type. CORS is not
authentication. The React app currently uses same-origin `/api` URLs; this
new backend is not connected to the live website by these files.

## Limitations before deployment

1. **Private uploads require configuration and deployment.** The added
   `POST /api/predict/mobilenet/blob` and `POST /api/explain/mobilenet/blob`
   interfaces accept single-use receipts, privately retrieve validated images,
   and call the existing classifiers unchanged. The shared transport module
   is staged without new Python dependencies. Set the frontend's
   `MOBILENET_API_ORIGIN` only after this backend is deployed and verified.
   See [private upload setup](../../docs/PRIVATE_UPLOADS.md). Raw requests above
   Vercel's 4.5 MB cap remain unsupported; the application limit stays 15 MiB.
2. **Large Functions eligibility and Linux packaging remain unverified.**
   TensorFlow exceeds the standard Python bundle budget. The beta permits up
   to 5 GB but does not provide additional RAM. Verify the actual Linux bundle
   and project eligibility in the eventual backend deployment.
3. **Cold starts and concurrency need Linux validation.** Previous Windows CPU
   profiling measured about 298 MiB peak resident memory for MobileNet and
   331 MiB with Grad-CAM, with first results around 15.7 s and 9.7 s respectively.
   Those are preliminary single-process measurements on a six-logical-CPU
   machine, not Vercel performance guarantees. Avoid frequent health polling.
4. **MobileNet production connection remains pending.** Large-image routing
   uses the frontend gateway and the configured `MOBILENET_API_ORIGIN`.
   Small-image MobileNet and Grad-CAM paths still need their own production
   routing; neither is redirected to CLIP. This backend does not provide
   `POST /api/predict` for CLIP.
5. **Free usage is finite.** Confirm the new project's Hobby eligibility, active
   CPU/memory usage and the existing private Blob store's quotas. No account
   settings, credentials or Blob resources have been inspected or changed.

Official references:
[FastAPI](https://vercel.com/docs/frameworks/backend/fastapi),
[Python runtime](https://vercel.com/docs/functions/runtimes/python),
[Monorepo shared files](https://vercel.com/docs/monorepos/monorepo-faq),
[Function limits](https://vercel.com/docs/functions/limitations),
[Private Blob storage](https://vercel.com/docs/vercel-blob/private-storage).
