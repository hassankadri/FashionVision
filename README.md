# FashionVision AI/ML

Fashion image classification with a React interface, FastAPI backend, and three inference paths:

- **MobileNetV3Small:** a trained ten-class fashion-product model with Grad-CAM explanations.
- **CLIP ViT-B/32:** zero-shot photo classification using a quantized ONNX model.
- **Browser CNN:** Fashion-MNIST gallery inference in JavaScript with five-view shift augmentation.

## Structure

```text
FashionVision-publish/
├── app.py                       local API launcher
├── backend/                     FastAPI routes and React serving
├── frontend/react/
│   ├── src/                     React UI
│   └── public/frontend/         browser CNN, weights and sample gallery
├── ml/                          MobileNet, CLIP and Grad-CAM inference
├── models/
│   ├── mobilenet-family-v1/     active checkpoint and original metadata
│   └── clip-vit-base-patch32/   tokenizer; ONNX downloaded separately
├── scripts/                     dataset preparation, training and evaluation
├── fashion_rgb_data/            RGB decoding and dataset utilities
├── fashion_evaluation/          metrics and reporting utilities
├── tests/                       regression tests and reference fixtures
├── docs/                        API and dependency documentation
└── requirements*.txt, constraints-artifacts.txt
```

## Setup

Run these commands from the repository root in a Python environment:

```bash
python -m pip install -r requirements-mobilenet.txt
python scripts/download_photo_model.py
npm ci --prefix frontend/react
npm run build --prefix frontend/react
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8765
```

The API serves the React production bundle at `/`. JavaScript and CSS are served
at `/assets`; shared browser CNN resources are served at `/frontend`.
`python app.py` starts the API on loopback, using `FASHION_APP_PORT` (default 8765).

For React development, start the API on port 8765 and run
`npm run dev --prefix frontend/react`. Vite proxies API requests to that port
and serves the CNN resources from its own public directory.

The approximately 146 MB CLIP ONNX model is excluded from Git. Its downloader
verifies SHA-256 and uses the bundled tokenizer files. The trained MobileNet
checkpoint and original configuration, evaluation and source provenance are
included. [CLIP provenance](models/clip-vit-base-patch32/README.md) and
[MobileNet model notes](models/mobilenet-family-v1/README.md) describe these assets.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | CLIP model-file availability |
| `POST /api/predict` | CLIP photo classification |
| `POST /api/predict/mobilenet` | Trained MobileNet classification |
| `POST /api/explain/mobilenet` | MobileNet prediction and Grad-CAM overlay |

Prediction endpoints accept raw image bytes with the corresponding image
Content-Type. See [API documentation](docs/MOBILENET_API.md) for limits,
preprocessing, response fields and interpretation.

MobileNet softmax scores and CLIP comparison scores are not calibrated
confidence. Grad-CAM is coarse class attribution, not segmentation. Saved
benchmark results do not establish independent natural-photo accuracy.

## Training and evaluation

`scripts/` contains Fashion-MNIST loading, classical/CNN baselines and
optimization, RGB product preparation, family-aware split preparation,
MobileNet training, label-quality analysis and evaluation.
`fashion_rgb_data/` provides dataset utilities and the RGB decoder used by
MobileNet inference; `fashion_evaluation/` provides metrics and reports.

Datasets, raw Fashion-MNIST IDX files, historical classical/CNN artifacts and
the original product/family benchmark manifests are not bundled. MobileNet
training scripts require the matching manifests and recipe artifacts they
verify by hash. The saved run's `COMPLETE.json` and executed
`training_source.py` snapshot are also absent. Source code is provided, but
complete historical reproduction requires those matching inputs.

Original model metadata retains historical machine paths and source identities;
those paths are not used to load the deployed checkpoint. See
[dependency groups](docs/ENVIRONMENTS.md), [RGB pipeline documentation](fashion_rgb_data/README.md)
and [test documentation](tests/README.md).

## Deployment

Provision Python dependencies, download CLIP, and build React before serving
the website. Each process loads its own models; select worker count and memory
for the target host. Hosting configuration and a dataset redistribution policy must be selected
before public deployment or redistribution.

## Developer

FashionVision is developed and maintained by [Hassan Kadri](https://github.com/hassankadri).

## Copyright and Usage

Copyright © 2026 Hassan Kadri for original project material owned by Hassan Kadri.

This repository has no general open-source license. Public availability does not
grant unrestricted reuse rights. See the [Copyright and Rights Notice](NOTICE.md)
for usage terms.

Third-party models, datasets, and dependencies remain subject to their respective
licenses and terms.
