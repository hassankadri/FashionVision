# Dependencies

Run setup commands from the repository root in a virtual environment.

| File | Purpose |
|---|---|
| `requirements.txt` | FastAPI, CLIP ONNX inference, NumPy, Pillow and Uvicorn |
| `requirements-mobilenet.txt` | Runtime plus TensorFlow 2.21.0 and Keras 3.15.1 for MobileNet and Grad-CAM |
| `requirements-training.txt` | Classical/CNN training, label-quality tools and plotting |
| `requirements-test.txt` | Runtime, classical compatibility dependencies and HTTPX for tests |
| `requirements-legacy.txt` | Historical classical/PCA compatibility dependencies; also used by training |
| `constraints-artifacts.txt` | Producer-compatible scikit-learn/Keras version constraints |

For the complete application:

```bash
python -m venv .venv
# Activate .venv using your platform's shell.
python -m pip install -r requirements-mobilenet.txt
```

For research dependencies, use `requirements-training.txt`. TensorFlow API
tests additionally need the MobileNet runtime group. The full Python regression
suite needs test and training dependencies and its external artifacts; see
[tests](../tests/README.md). JavaScript reference checks use Node's built-in runner.

The shipped MobileNet configuration records Python 3.12.3, TensorFlow 2.21.0,
Keras 3.15.1, NumPy 2.5.3 and Pillow 12.3.0. The historical classical and CNN
artifact compatibility constraints are scikit-learn 1.8.0 and Keras 3.15.1;
those historical artifacts are not bundled. Matching producer versions is
necessary for serialized compatibility but does not guarantee successful loading.

Requirement ranges express supported dependency families rather than a complete
historical lock. Historical transitive versions and cross-platform compatibility
are not fully established. React dependencies are recorded in
`frontend/react/package-lock.json`.
