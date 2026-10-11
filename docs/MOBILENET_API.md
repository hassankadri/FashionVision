# MobileNet and Grad-CAM API

`POST /api/predict/mobilenet` and `POST /api/explain/mobilenet` accept raw
JPEG, PNG, WebP or BMP image bytes with the matching Content-Type.

## Model and preprocessing

The active MobileNetV3Small checkpoint and configuration are in
`models/mobilenet-family-v1/`. Their SHA-256 hashes are verified before loading
with `compile=False`. TensorFlow/Keras load lazily; one cached model and a lock
are shared by prediction and explanation. Loading failures are cached until
server restart.

Preprocessing applies EXIF orientation, composites transparency on white, converts
to RGB, and resizes to 160×160 with Pillow bilinear interpolation. Inputs are
float32 pixels in [0,255]; normalization is built into the model. Calls use
`training=False`.

Class order: T-shirts, Shirts, Kurtas, Pants, Closed shoes, Handbags, Watches,
Sunglasses, Backpacks, Belts. This differs from the CLIP/Fashion-MNIST vocabulary.

## Prediction response

- `prediction`: top class name.
- `probabilities`: ten `{class_id, label, probability}` entries in class order.
- `top3`: entries sorted by decreasing probability; ties use class order.
- `model`: saved run identifier.
- `inference_time_ms`: model-call time including output materialization;
  excludes loading, preprocessing, lock wait and network time.
- `score_type`: `uncalibrated_softmax`.

## Grad-CAM response

The explanation returns `prediction`, `explained_class`, `model`,
`target_layer`, `target_score`, `attribution_status`, `overlay_size`,
`overlay_format` and `heatmap_overlay_base64` (plain base64 PNG).

Grad-CAM differentiates the predicted class's pre-softmax logit with respect to
the backbone's `conv_1` activations. Spatially averaged gradients weight channels;
ReLU and per-image max normalization produce a 5×5 map. The red overlay is
bilinearly enlarged onto the 160×160 model-input view. A map without positive
attribution remains uncolored with `NO_POSITIVE_ATTRIBUTION`.
The explanation must reproduce inference probabilities or return 503.

This is coarse class attribution, not segmentation or causal proof. Scores are
not calibrated confidence, and independent natural-photo accuracy is not established.

## Limits and errors

Uploads are limited to 15 MiB, an 8,192-pixel maximum side and 16,000,000 pixels.
Multi-frame images are rejected. Unsupported MIME returns 415; oversized bytes
return 413; empty, corrupt, mismatched-format or oversized-dimension images
return 400. Missing dependencies or invalid model assets return a generic 503
with details logged on the server.

`POST /api/predict` uses CLIP separately. `GET /api/health` reports CLIP
model-file availability, not MobileNet readiness or full tokenizer validation.
