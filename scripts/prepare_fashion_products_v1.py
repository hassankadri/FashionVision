"""Audit original fashion product files, then create immutable reference manifests.

No training, image rewriting, relabeling, downloads, or original-file deletion.
"""
from collections import Counter, defaultdict
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid

import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from fashion_rgb_data.pipeline import EXTENSIONS, _decode, _pixel_hash

SCHEMA = "fashion-products-preparation/1.0"
FIELDS = ["id", "gender", "masterCategory", "subCategory", "articleType", "baseColour",
          "season", "year", "usage", "productDisplayName"]
RUNS = ROOT / "experiments/fashion_products_v1/runs"
# Product types only. Styles of closed shoe/pants are merged, not separate targets.
TASK = [
    ("T-shirts", "Apparel", "Topwear", ["Tshirts"]),
    ("Shirts", "Apparel", "Topwear", ["Shirts"]),
    ("Kurtas", "Apparel", "Topwear", ["Kurtas"]),
    ("Pants", "Apparel", "Bottomwear", ["Jeans", "Trousers", "Track Pants"]),
    ("Closed shoes", "Footwear", "Shoes", ["Casual Shoes", "Sports Shoes", "Formal Shoes"]),
    ("Handbags", "Accessories", "Bags", ["Handbags"]),
    ("Watches", "Accessories", "Watches", ["Watches"]),
    ("Sunglasses", "Accessories", "Eyewear", ["Sunglasses"]),
    ("Backpacks", "Accessories", "Bags", ["Backpacks"]),
    ("Belts", "Accessories", "Belts", ["Belts"]),
]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def write_lines(path, records):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def read_lines(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def inventory_digest(records):
    value = hashlib.sha256()
    for record in sorted(records, key=lambda r: r["path"]):
        value.update(json.dumps({k: record[k] for k in ("path", "sha256", "bytes")},
                                sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
    return value.hexdigest()


def parse_metadata(path):
    records, anomalies = [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        if next(reader) != FIELDS:
            raise ValueError("Unexpected CSV header; refusing to guess its schema")
        for row_number, values in enumerate(reader, 2):
            raw = list(values)
            if len(values) > len(FIELDS):
                anomalies.append({"row": row_number, "kind": "extra_trailing_name_fields",
                                  "raw_fields": raw, "action": "join fields 10 onward with commas in memory only"})
                values = values[:9] + [",".join(values[9:])]
            elif len(values) < len(FIELDS):
                anomalies.append({"row": row_number, "kind": "short_row", "raw_fields": raw,
                                  "action": "excluded; no guessed values"})
                records.append({"csv_row": row_number, "invalid": "short_row", "raw_fields": raw})
                continue
            record = dict(zip(FIELDS, values))
            record["csv_row"] = row_number
            if not record["id"].isascii() or not record["id"].isdigit():
                record["invalid"] = "invalid_id"
            records.append(record)
    return records, anomalies


def duplicate_components(images):
    parents = list(range(len(images)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    duplicate_groups = {}
    for key in ("sha256", "pixel_sha256"):
        groups = defaultdict(list)
        for i, image in enumerate(images):
            if key in image:
                groups[image[key]].append(i)
        duplicate_groups[key] = []
        for value, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            duplicate_groups[key].append({"hash": value, "paths": [images[i]["path"] for i in members]})
            for i in members[1:]:
                parents[find(i)] = find(members[0])
    components = defaultdict(list)
    for i, image in enumerate(images):
        components[find(i)].append(image["path"])
    return duplicate_groups, sorted((sorted(g) for g in components.values()), key=lambda g: g[0])


def audit(root):
    root = Path(root).resolve()
    rows, anomalies = parse_metadata(root / "styles.csv")
    if RUNS.resolve() != RUNS.absolute():
        raise ValueError("Output root must not redirect")
    RUNS.mkdir(parents=True, exist_ok=True)
    run = RUNS / f"fashion_products_{uuid.uuid4().hex}"
    run.mkdir(exist_ok=False)
    print(f"Audit directory: {run}", flush=True)
    inventory, images, decoded = [], [], {}
    for index, path in enumerate(sorted(p for p in root.rglob("*") if p.is_file()), 1):
        if path.resolve() != path.absolute():
            raise ValueError(f"Redirected original file: {path}")
        raw = path.read_bytes()
        identity = {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        inventory.append(identity)
        if path.suffix.lower() in EXTENSIONS:
            if identity["sha256"] not in decoded:
                try:
                    rgb, info = _decode(raw)
                    decoded[identity["sha256"]] = {"status": "READABLE", "pixel_sha256": _pixel_hash(rgb), **info}
                except Exception as error:
                    # Every failed decoder call is explicitly recorded, never silently skipped.
                    decoded[identity["sha256"]] = {"status": "UNREADABLE", "error": f"{type(error).__name__}: {error}"}
            images.append({**identity, "image_id": path.stem, **decoded[identity["sha256"]]})
        if index % 10000 == 0:
            print(f"Hashed {index} original files", flush=True)
    primary = [r for r in images if r["path"].startswith("images/")]
    mirror = [r for r in images if r["path"].startswith("myntradataset/images/")]
    primary_by_id = defaultdict(list)
    for image in primary:
        primary_by_id[image["image_id"]].append(image)
    counts = Counter(r["id"] for r in rows if "id" in r)
    duplicate_ids = {key: value for key, value in counts.items() if value > 1}
    valid_rows = {r["id"]: r for r in rows if "invalid" not in r and r["id"] not in duplicate_ids}
    missing = [r for r in rows if "id" in r and r["id"] not in primary_by_id]
    unreferenced = [r for r in primary if r["image_id"] not in counts]
    usable = [r for r in primary if r["status"] == "READABLE" and r["image_id"] in valid_rows
              and len(primary_by_id[r["image_id"]]) == 1]
    duplicates, components = duplicate_components([r for r in primary if r["status"] == "READABLE"])
    categories = {field: dict(sorted(Counter(r.get(field, "<INVALID_ROW>") for r in rows).items()))
                  for field in FIELDS[1:-1]}
    primary_hashes = {r["path"].removeprefix("images/"): r["sha256"] for r in primary}
    mirror_hashes = {r["path"].removeprefix("myntradataset/images/"): r["sha256"] for r in mirror}
    by_path = {r["path"]: r for r in inventory}
    mirror_report = {"image_count": len(mirror), "byte_identical_image_tree": bool(mirror) and primary_hashes == mirror_hashes,
                     "csv_identical": "myntradataset/styles.csv" in by_path and by_path["styles.csv"]["sha256"] == by_path["myntradataset/styles.csv"]["sha256"],
                     "different_or_missing_names": sorted(k for k in primary_hashes.keys() | mirror_hashes.keys() if primary_hashes.get(k) != mirror_hashes.get(k)),
                     "used_for_splits": False, "action": "left untouched; never double-counted as independent examples"}
    report = {"schema": SCHEMA, "dataset_name": "Fashion Product Images (Small)", "dataset_root": str(root),
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "metadata_rows": len(rows),
              "primary_image_files": len(primary), "all_image_files_including_nested": len(images),
              "usable_matched_images_before_task_selection_and_deduplication": len(usable),
              "missing_images": len(missing), "images_without_metadata": len(unreferenced),
              "corrupt_primary_images": [r for r in primary if r["status"] != "READABLE"],
              "corrupt_images_all_trees": [r for r in images if r["status"] != "READABLE"],
              "duplicate_metadata_ids": duplicate_ids,
              "ambiguous_image_ids": {k: [r["path"] for r in v] for k,v in primary_by_id.items() if len(v)>1},
              "invalid_metadata_rows": [r for r in rows if "invalid" in r],
              "csv_anomalies": anomalies, "category_fields": {k: len(v) for k,v in categories.items()},
              "mirror": mirror_report, "source_inventory_sha256": inventory_digest(inventory),
              "source_file_count": len(inventory), "styles_csv_sha256": by_path["styles.csv"]["sha256"],
              "duplicate_summary": {key: {"groups": len(groups), "files_in_groups": sum(len(g["paths"]) for g in groups),
                                               "extra_copies": sum(len(g["paths"])-1 for g in groups)} for key,groups in duplicates.items()},
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__},
              "script_sha256": digest(Path(__file__)), "model_training_executed": False}
    write_lines(run / "source_inventory.jsonl", inventory)
    write_lines(run / "image_audit.jsonl", primary)
    write_lines(run / "metadata.jsonl", rows)
    write_json(run / "categories.json", categories)
    write_json(run / "missing_images.json", missing)
    write_json(run / "images_without_metadata.json", unreferenced)
    write_json(run / "duplicates.json", {**duplicates, "components": [g for g in components if len(g)>1]})
    write_json(run / "audit.json", report)
    print(json.dumps({k: report[k] for k in ("metadata_rows", "primary_image_files", "usable_matched_images_before_task_selection_and_deduplication", "missing_images", "images_without_metadata", "duplicate_summary", "mirror")}, indent=2), flush=True)
    print(f"Corrupt primary images: {len(report['corrupt_primary_images'])}", flush=True)
    return run


def class_id(row):
    for i, (_, master, sub, articles) in enumerate(TASK):
        if row.get("masterCategory") == master and row.get("subCategory") == sub and row.get("articleType") in articles:
            return i
    return None


def prepare(run):
    run = Path(run).resolve()
    if not run.is_relative_to(RUNS.resolve()):
        raise ValueError("Preparation must use an audit under the versioned experiment directory")
    report = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    images = read_lines(run / "image_audit.jsonl")
    rows = read_lines(run / "metadata.jsonl")
    by_id = defaultdict(list)
    for row in rows:
        if "id" in row:
            by_id[row["id"]].append(row)
    image_id_counts = Counter(r["image_id"] for r in images)
    eligible, exclusions = {}, []
    for image in images:
        metadata = by_id.get(image["image_id"], [])
        reason = ("unreadable_image" if image["status"] != "READABLE" else
                  "missing_or_duplicate_metadata" if len(metadata) != 1 else
                  "invalid_metadata" if "invalid" in metadata[0] else
                  "ambiguous_image_id" if image_id_counts[image["image_id"]] != 1 else None)
        if reason:
            exclusions.append({"path": image["path"], "image_id": image["image_id"], "reason": reason})
        else:
            eligible[image["path"]] = (image, metadata[0], class_id(metadata[0]))
    _, components = duplicate_components([r for r in images if r["status"] == "READABLE"])
    selected = []
    for group in components:
        known = [eligible[path] for path in group if path in eligible]
        labels = {item[2] for item in known}
        if len(known) != len(group) or len(labels) > 1:
            for image, row, label in known:
                exclusions.append({"path": image["path"], "image_id": row["id"],
                                   "reason": "duplicate_group_with_conflicting_or_unresolved_metadata", "group_paths": group})
            continue
        if not known:
            continue
        if labels == {None}:
            for image, row, _ in known:
                exclusions.append({"path": image["path"], "image_id": row["id"], "reason": "outside_declared_task",
                                   "source_article_type": row["articleType"]})
            continue
        # Deduplicate manifests only: originals remain untouched. No group is split.
        known.sort(key=lambda item: (int(item[1]["id"]), item[0]["path"]))
        image, row, label = known[0]
        selected.append({"image_id": row["id"], "relative_path": image["path"],
                         "original_path": str(Path(report["dataset_root"]) / image["path"]),
                         "file_sha256": image["sha256"], "pixel_sha256": image["pixel_sha256"],
                         "class_id": label, "class_name": TASK[label][0], "source_metadata": row,
                         "duplicate_group_paths": group, "rgb_size": image["rgb_size"]})
        for duplicate, duplicate_row, _ in known[1:]:
            exclusions.append({"path": duplicate["path"], "image_id": duplicate_row["id"],
                               "reason": "duplicate_representative_selected", "representative_id": row["id"]})
    counts = Counter(item["class_id"] for item in selected)
    if set(counts) != set(range(len(TASK))) or min(counts.values()) < 500:
        raise ValueError(f"Task requires at least 500 usable unique images per class; observed {dict(counts)}")
    manifests = {"train": [], "validation": [], "test": []}
    for label in range(len(TASK)):
        items = sorted((r for r in selected if r["class_id"] == label), key=lambda r: int(r["image_id"]))
        order = np.random.default_rng(np.random.SeedSequence([42, label])).permutation(len(items))
        # Nearest integer allocation per class, remainder belongs to training.
        n_val = n_test = int(np.floor(len(items) * 0.1 + 0.5))
        for split, positions in (("validation", order[:n_val]), ("test", order[n_val:n_val+n_test]),
                                 ("train", order[n_val+n_test:])):
            manifests[split].extend({**items[int(i)], "split": split} for i in positions)
    for split in manifests:
        manifests[split].sort(key=lambda r: (r["class_id"], int(r["image_id"])))
    for identity in ("image_id", "file_sha256", "pixel_sha256"):
        sets = {s: {r[identity] for r in records} for s,records in manifests.items()}
        assert not sets["train"] & sets["validation"] and not sets["train"] & sets["test"] and not sets["validation"] & sets["test"]
        assert sum(map(len,sets.values())) == len(selected)
    assert len(exclusions)+len(selected)==len(images)
    classes = [{"class_id": i, "class_name": name, "source_masterCategory": master,
                "source_subCategory": sub, "source_articleTypes": articles, "image_count": counts[i],
                "split_counts": {s: sum(r["class_id"]==i for r in records) for s,records in manifests.items()}}
               for i,(name,master,sub,articles) in enumerate(TASK)]
    preparation = {"schema": SCHEMA, "dataset_root": report["dataset_root"], "classes": classes,
                   "selected_unique_images": len(selected), "split_counts": {s:len(v) for s,v in manifests.items()},
                   "split_method": {"seed":42, "stratification_field":"class_id", "target_ratios":[0.8,0.1,0.1],
                                    "algorithm":"sorted original IDs; per-class NumPy SeedSequence([42,class_id]) permutation; validation/test rounded to nearest integer, remainder train",
                                    "duplicate_policy":"one lowest-ID representative per connected byte/pixel hash group; exclude conflicting/unresolved groups"},
                   "imbalance_max_to_min": max(counts.values())/min(counts.values()),
                   "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)),
                   "csv_policy": "22 observed wide rows are documented in audit.json; only trailing name fields rejoined, never category fields",
                   "originals_modified": False, "model_training_executed": False,
                   "test_policy": "freeze these manifests for all future models; tune only on validation; test labels never select models or hyperparameters",
                   "source_inventory_sha256": report["source_inventory_sha256"],
                   "limitations": ["No perceptual/near-duplicate or product-family identity detection; group metadata unavailable",
                                   "Metadata labels are not independently human-verified; product photographs are not a real-world deployment benchmark"],
                   "versions": report["versions"], "preparation_script_sha256": digest(Path(__file__))}
    # Refuse any overwrite before publication. Status is published last.
    output_names = ["train.jsonl", "validation.jsonl", "test.jsonl", "exclusions.jsonl", "preparation.json", "COMPLETE.json"]
    if any((run/name).exists() for name in output_names):
        raise FileExistsError("This audit already contains split outputs; preserve the fixed split")
    for split, records in manifests.items():
        write_lines(run / f"{split}.jsonl", records)
    write_lines(run / "exclusions.jsonl", exclusions)
    write_json(run / "preparation.json", preparation)
    write_json(run / "COMPLETE.json", {"schema":SCHEMA, "status":"COMPLETE", "files":{name:digest(run/name) for name in output_names[:-1]},
                                      "source_inventory_sha256": report["source_inventory_sha256"]})
    print(json.dumps(preparation, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "prepare"))
    parser.add_argument("path", help="Dataset root for audit; resulting new run directory for prepare")
    args = parser.parse_args()
    if args.action == "audit":
        audit(args.path)
    else:
        prepare(args.path)
