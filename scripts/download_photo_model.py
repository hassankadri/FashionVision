"""Download and verify the quantized CLIP model used for photo uploads."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "clip-vit-base-patch32"
BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
FILES = {
    "model_quantized.onnx": (
        f"{BASE}/onnx/model_quantized.onnx",
        "0898a3facfdb27f0a041e57649b4989cfd094e4a0040d6ae75ed69917dfc7328",
    ),
    "vocab.json": (
        f"{BASE}/vocab.json",
        "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
    ),
    "merges.txt": (
        f"{BASE}/merges.txt",
        "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, url: str, expected_hash: str | None) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    destination = TARGET / name
    if destination.exists() and (expected_hash is None or sha256(destination) == expected_hash):
        print(f"ready: {destination.relative_to(ROOT)}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {name} ...")
    try:
        urllib.request.urlretrieve(url, partial)
        if expected_hash is not None and sha256(partial) != expected_hash:
            raise RuntimeError(f"SHA-256 verification failed for {name}")
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()
    print(f"ready: {destination.relative_to(ROOT)}")


def main() -> int:
    try:
        for name, (url, expected_hash) in FILES.items():
            download(name, url, expected_hash)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
