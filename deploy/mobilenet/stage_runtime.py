"""Stage byte-identical canonical inference files inside the Vercel project root."""

import ast
import hashlib
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
RUNTIME_FILES = (
    "backend/__init__.py",
    "backend/private_uploads.py",
    "ml/__init__.py",
    "ml/mobilenet_classifier.py",
    "ml/mobilenet_gradcam.py",
    "fashion_rgb_data/__init__.py",
    "fashion_rgb_data/pipeline.py",
    "models/mobilenet-family-v1/best.keras",
    "models/mobilenet-family-v1/configuration.json",
    "NOTICE.md",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_runtime() -> None:
    # Parse the existing verification constants without importing ML dependencies.
    source = REPOSITORY_ROOT / "ml/mobilenet_classifier.py"
    constants = {}
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("MODEL_SHA256", "CONFIG_SHA256"):
                    constants[target.id] = ast.literal_eval(node.value)
    expected = {
        "models/mobilenet-family-v1/best.keras": constants["MODEL_SHA256"],
        "models/mobilenet-family-v1/configuration.json": constants["CONFIG_SHA256"],
    }

    # Validate every input and the protected assets before creating build output.
    for relative in RUNTIME_FILES:
        source = REPOSITORY_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required runtime file missing: {relative}")
        if relative in expected and sha256(source) != expected[relative]:
            raise ValueError(f"SHA-256 verification failed: {relative}")

    for relative in RUNTIME_FILES:
        source = REPOSITORY_ROOT / relative
        destination = PROJECT_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(destination) != sha256(source):
            raise ValueError(f"Staged file differs from its source: {relative}")
        if relative in expected and sha256(destination) != expected[relative]:
            raise ValueError(f"Staged asset SHA-256 verification failed: {relative}")
    print(f"Staged and verified {len(RUNTIME_FILES)} runtime files; no model downloads or training.")


if __name__ == "__main__":
    stage_runtime()
