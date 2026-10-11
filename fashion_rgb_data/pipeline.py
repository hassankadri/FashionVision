"""Read-only split auditing and RGB sample preparation using Pillow and NumPy."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import BytesIO
import os
from pathlib import Path
from types import MappingProxyType
import warnings

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

SPLITS = ("train", "validation", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
SCHEMA = "fashion-rgb-dataset/1.0"


@dataclass(frozen=True)
class ClassMapping:
    class_names: tuple[str, ...]
    folders: dict[str, int]

    def __post_init__(self):
        names = tuple(self.class_names)
        if (not names or any(not isinstance(n, str) or not n.strip() for n in names)
                or len(set(names)) != len(names)):
            raise ValueError("class_names must be nonempty unique strings in class-ID order")
        folders = dict(self.folders)
        for folder, label in folders.items():
            if (not isinstance(folder, str) or not folder or folder in (".", "..")
                    or any(c in folder for c in ("/", "\\", ":"))
                    or type(label) is not int or not 0 <= label < len(names)):
                raise ValueError("Map plain class folder names to valid integer class IDs")
        if set(folders.values()) != set(range(len(names))):
            raise ValueError("Every class ID must have at least one folder mapping")
        object.__setattr__(self, "class_names", names)
        object.__setattr__(self, "folders", MappingProxyType(folders))

    def to_dict(self):
        return {"class_names": list(self.class_names), "folders": dict(self.folders)}


def _redirected(path):
    return path.is_symlink() or path.resolve() != path.absolute()


def _decode(raw):
    # Verify container integrity AND force pixel decoding; never alter Pillow globals.
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multi-frame images are unsupported; select a frame explicitly upstream")
            image.load()
            metadata = {"original_mode": image.mode, "original_size": list(image.size), "format": image.format}
            oriented = ImageOps.exif_transpose(image)
            # Explicit alpha policy: composite onto white before conversion to RGB.
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                oriented = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
            rgb = oriented.convert("RGB")
            metadata["rgb_size"] = list(rgb.size)
            return rgb, metadata


def _pixel_hash(image):
    digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def scan_dataset(root, mapping=None):
    """Return a JSON-compatible audit. Never create, split, move or delete data.

    Default classes come ONLY from train directory names, sorted deterministically.
    Exact duplicate groups include byte hashes and decoded, EXIF-oriented RGB hashes.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Dataset root must be an existing directory")
    if mapping is None:
        train = root / "train"
        names = sorted(p.name for p in train.iterdir() if p.is_dir() and not _redirected(p)) if train.is_dir() and not _redirected(train) else []
        if not names:
            raise ValueError("No training class directories found; provide an explicit mapping for an incomplete layout")
        mapping = ClassMapping(tuple(names), {name: i for i, name in enumerate(names)})
    if not isinstance(mapping, ClassMapping):
        raise TypeError("mapping must be a ClassMapping")
    records, errors, ignored = [], [], []

    def error(code, path, **details):
        errors.append({"code": code, "path": path.relative_to(root).as_posix(), **details})

    for split in SPLITS:
        split_path = root / split
        if not split_path.is_dir() or _redirected(split_path):
            error("missing_or_redirected_split", split_path)
            continue
        for folder in sorted(split_path.iterdir()):
            if _redirected(folder):
                error("redirected_path", folder)
                continue
            if not folder.is_dir():
                error("file_outside_class_folder", folder)
                continue
            if folder.name not in mapping.folders:
                error("unmapped_class_folder", folder)
                continue
            label = mapping.folders[folder.name]
            def walk_error(problem):
                error("unreadable_directory", Path(problem.filename), detail=str(problem))
            for directory, directories, filenames in os.walk(folder, followlinks=False, onerror=walk_error):
                directories.sort()
                for name in list(directories):
                    child = Path(directory) / name
                    if _redirected(child):
                        error("redirected_path", child)
                        directories.remove(name)
                for name in sorted(filenames):
                    path = Path(directory) / name
                    if _redirected(path):
                        error("redirected_path", path)
                        continue
                    if path.suffix.lower() not in EXTENSIONS:
                        ignored.append(path.relative_to(root).as_posix())
                        continue
                    record = {"relative_path": path.relative_to(root).as_posix(), "split": split,
                              "source_class_folder": folder.name, "class_id": label,
                              "class_name": mapping.class_names[label], "status": "INVALID"}
                    records.append(record)
                    try:
                        raw = path.read_bytes()
                        record.update(byte_count=len(raw), file_sha256=hashlib.sha256(raw).hexdigest())
                        image, metadata = _decode(raw)
                        record.update(metadata, pixel_sha256=_pixel_hash(image), status="VALID")
                    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                            Image.DecompressionBombWarning) as problem:
                        record["error"] = f"{type(problem).__name__}: {problem}"
                        error("unreadable_or_corrupt_image", path, detail=record["error"])

    duplicates, leakage, conflicts = [], [], []
    for key in ("file_sha256", "pixel_sha256"):
        groups = defaultdict(list)
        for record in records:
            if key in record:
                groups[record[key]].append(record)
        for digest, group in sorted(groups.items()):
            if len(group) < 2:
                continue
            duplicate = {"kind": key, "sha256": digest,
                         "paths": sorted(r["relative_path"] for r in group),
                         "splits": sorted({r["split"] for r in group}),
                         "class_ids": sorted({r["class_id"] for r in group})}
            duplicates.append(duplicate)
            if len(duplicate["splits"]) > 1:
                leakage.append(duplicate)
            if len(duplicate["class_ids"]) > 1:
                conflicts.append(duplicate)

    statistics = {}
    for split in SPLITS:
        counts = [sum(r["status"] == "VALID" and r["split"] == split and r["class_id"] == i
                      for r in records) for i in range(len(mapping.class_names))]
        total = sum(counts)
        statistics[split] = {
            "valid_images": total, "invalid_images": sum(r["status"] != "VALID" and r["split"] == split for r in records),
            "class_counts": dict(zip(mapping.class_names, counts)),
            "class_proportions": dict(zip(mapping.class_names, [c / total if total else 0.0 for c in counts])),
            "missing_classes": [n for n, c in zip(mapping.class_names, counts) if c == 0],
            "max_to_min_ratio": max(counts) / min(counts) if min(counts) else None,
        }
        if not total:
            error("empty_split", root / split)
        elif split == "train" and any(c == 0 for c in counts):
            error("missing_training_class", root / split, classes=statistics[split]["missing_classes"])
    return {"schema": SCHEMA, "root": str(root), "class_mapping": mapping.to_dict(),
            "records": records, "statistics": statistics, "errors": errors,
            "ignored_non_image_files": sorted(ignored), "duplicates": duplicates,
            "cross_split_leakage": leakage, "duplicate_label_conflicts": conflicts,
            "ready": not errors and not leakage and not conflicts,
            "identity_policy": "SHA-256 of original bytes and full-resolution EXIF-oriented RGB pixels; alpha composited white",
            "data_modified": False}


@dataclass(frozen=True)
class Transform:
    size: tuple[int, int] = (224, 224)  # height, width; direct resize, no crop
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    horizontal_flip_probability: float = 0.5
    brightness_range: tuple[float, float] = (0.9, 1.1)

    def __post_init__(self):
        if len(self.size) != 2 or any(type(n) is not int or n <= 0 for n in self.size):
            raise ValueError("size must contain positive integer height and width")
        if (np.shape(self.mean) != (3,) or np.shape(self.std) != (3,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(np.asarray(self.std) <= 0)):
            raise ValueError("mean/std must contain three finite values with positive std")
        if (not np.isfinite(self.horizontal_flip_probability) or not 0 <= self.horizontal_flip_probability <= 1
                or len(self.brightness_range) != 2 or not np.isfinite(self.brightness_range).all()
                or not 0 < self.brightness_range[0] <= self.brightness_range[1]):
            raise ValueError("Invalid augmentation settings")
        for field in ("size", "mean", "std", "brightness_range"):
            object.__setattr__(self, field, tuple(getattr(self, field)))


class RGBDataset:
    """Read audited samples as HWC float32 arrays; no model/framework dependency.

    Training augmentation is opt-in, seeded per original path and epoch. Validation
    and test cannot enable augmentation. get(index, epoch=N) has no mutable RNG.
    """
    def __init__(self, audit, split, *, transform=None, augment=False, seed=42):
        if split not in SPLITS:
            raise ValueError("split must be train, validation or test")
        if not audit["ready"] or audit["errors"] or audit["cross_split_leakage"] or audit["duplicate_label_conflicts"]:
            raise ValueError("Dataset audit failed; inspect errors, leakage and label conflicts before loading")
        if augment and split != "train":
            raise ValueError("Augmentation is permitted only for training")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.root = Path(audit["root"])
        # Isolate identities from callers subsequently editing their report dictionaries.
        self.records = tuple(MappingProxyType(deepcopy(r)) for r in audit["records"] if r["split"] == split)
        self.transform = transform or Transform()
        self.augment, self.seed = bool(augment), seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, *, epoch=0):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        record = self.records[index]
        path = self.root / record["relative_path"]
        if _redirected(path) or not path.resolve().is_relative_to(self.root):
            raise ValueError("Sample path was redirected after audit")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["file_sha256"]:
            raise ValueError("Original image changed after audit; rescan before use")
        image, _ = _decode(raw)
        config = self.transform
        image = image.resize((config.size[1], config.size[0]), Image.Resampling.BILINEAR)
        if self.augment:
            identity = f"{self.seed}\0{epoch}\0{record['relative_path']}".encode("utf-8")
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest()[:8], "big"))
            if rng.random() < config.horizontal_flip_probability:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(*config.brightness_range)))
        values = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        values = (values - np.asarray(config.mean, dtype=np.float32)) / np.asarray(config.std, dtype=np.float32)
        return {"image": values, "class_id": record["class_id"], "class_name": record["class_name"],
                "identity": deepcopy(dict(record))}
