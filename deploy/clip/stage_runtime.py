"""Stage canonical CLIP files, provision verified assets, and report bundle inputs."""

import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
RUNTIME_FILES = (
    "backend/__init__.py",
    "backend/private_uploads.py",
    "ml/__init__.py",
    "ml/photo_classifier.py",
    "scripts/download_photo_model.py",
    "models/clip-vit-base-patch32/vocab.json",
    "models/clip-vit-base-patch32/merges.txt",
    "models/clip-vit-base-patch32/README.md",
    "NOTICE.md",
)
STANDARD_LIMIT_BYTES = 500_000_000


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_downloader(path: Path):
    spec = importlib.util.spec_from_file_location("clip_verified_downloader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_bundle_inputs() -> None:
    # Count installed runtime distributions, including their transitive dependencies,
    # rather than compressed wheels or unrelated packages in a development environment.
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    environment = default_environment()
    environment["extra"] = ""
    pending = [
        Requirement(line.strip())
        for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    distributions = set()
    dependency_paths = set()
    while pending:
        requirement = pending.pop()
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        distribution = importlib.metadata.distribution(requirement.name)
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name in distributions:
            continue
        if distribution.version not in requirement.specifier:
            raise RuntimeError(f"Installed dependency does not satisfy {requirement}")
        distributions.add(name)
        for relative in distribution.files or ():
            path = Path(distribution.locate_file(relative)).resolve()
            if path.is_file():
                dependency_paths.add(path)
        pending.extend(Requirement(value) for value in distribution.requires or ())

    project_paths = set()
    for directory, dirs, names in os.walk(PROJECT_ROOT):
        dirs[:] = [name for name in dirs if name not in {".git", ".venv", ".vercel", "__pycache__"}]
        project_paths.update((Path(directory) / name).resolve() for name in names)
    project_bytes = sum(path.stat().st_size for path in project_paths)
    dependency_bytes = sum(path.stat().st_size for path in dependency_paths - project_paths)
    total = project_bytes + dependency_bytes
    print(f"Application/assets: {project_bytes:,} bytes; installed runtime dependencies: {dependency_bytes:,} bytes.")
    print(f"Known uncompressed inputs: {total:,} bytes across {len(distributions)} distributions.")
    print("Vercel's final packaged size also includes runtime-generated files; its build output is authoritative.")
    if total >= STANDARD_LIMIT_BYTES:
        raise RuntimeError("Known inputs exceed the standard 500 MB Python limit; deployment stopped.")


def stage_runtime() -> None:
    # Validate every canonical input before writing deployment-local output.
    for relative in RUNTIME_FILES:
        if not (REPOSITORY_ROOT / relative).is_file():
            raise FileNotFoundError(f"Required source missing: {relative}")
    original = _load_downloader(REPOSITORY_ROOT / "scripts/download_photo_model.py")
    for name, (_url, expected) in original.FILES.items():
        source = original.TARGET / name
        if source.is_file() and sha256(source) != expected:
            raise RuntimeError(f"Canonical asset SHA-256 verification failed: {name}")

    for relative in RUNTIME_FILES:
        source = REPOSITORY_ROOT / relative
        destination = PROJECT_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(destination) != sha256(source):
            raise RuntimeError(f"Staged file differs from canonical source: {relative}")

    # Reuse the existing verified local model when present; a Git checkout omits it.
    local_model = original.TARGET / "model_quantized.onnx"
    staged = _load_downloader(PROJECT_ROOT / "scripts/download_photo_model.py")
    if local_model.is_file():
        shutil.copyfile(local_model, staged.TARGET / local_model.name)
    # The unchanged downloader validates existing files, downloads missing files
    # through HTTPS, verifies SHA-256, and only then replaces its partial download.
    if staged.main() != 0:
        raise RuntimeError("Verified CLIP model provisioning failed.")
    for name, (_url, expected) in staged.FILES.items():
        if sha256(staged.TARGET / name) != expected:
            raise RuntimeError(f"Staged asset SHA-256 verification failed: {name}")
    print("Canonical CLIP classifier/tokenizer preserved; all three assets SHA-256 verified.")


if __name__ == "__main__":
    stage_runtime()
    report_bundle_inputs()
