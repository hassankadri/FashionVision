# Regression tests

The Python suite covers API contracts, MobileNet preprocessing, Grad-CAM,
dataset preparation, metrics, label quality and historical model compatibility.
Run from the repository root with the required dependencies and artifacts.

```bash
python -B -m unittest discover -s tests -v
node tests/browser/inference.js
node --test tests/test_browser_baseline.cjs
```

The browser harness loads the React website's canonical CNN script without
rendering the UI. Browser inference uses five shift-only views; the standalone
reference implementation in `browser/inference.js` uses six views including
a horizontal flip. Their saved reference vectors remain distinct.
Reference vectors are behavior snapshots, not a measurement of full-test accuracy.

The complete suite is not self-contained: real CLIP inference requires the ONNX
download, some tests need external reference photos, and historical smoke tests
need the original classical/PCA and CNN artifacts. Dataset and training tests
also depend on matching recipe/manifests where specified.
`FASHION_TEST_IMAGE` can select the external pullover reference photo.

Tests do not train, download, save or export models. Synthetic fixtures check
mechanics and finite outputs, not natural-photo accuracy. API baseline tests
also document existing CLIP asset-error behavior: missing ONNX returns 503;
tokenizer failures can appear as 400; corrupt ONNX parsing can return 500.
