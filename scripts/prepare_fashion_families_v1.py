"""Input-only inferred family grouping; a new benchmark, never a model experiment."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import re
import sys
import uuid

import numpy as np
from PIL import Image, __version__ as pillow_version
import scipy
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from fashion_rgb_data.pipeline import _decode, _pixel_hash
from scripts.prepare_fashion_products_v1 import FIELDS, digest, parse_metadata, write_json, write_lines

SOURCE = ROOT / "experiments/fashion_products_v1/runs/fashion_products_2ec9a5867e8c4c259b77106889d9c375"
RUNS = ROOT / "experiments/fashion_products_family_v1/runs"
PREPARATION_SHA256 = "5306aca6075045f77c66d88c71ead8c48328ef3232bf6f70bc2a6400215884fe"
SCHEMA = "fashion-products-inferred-families/1.0"
SEED = 42
SPLITS = ("train", "validation", "test")
RATIOS = np.array([0.8, 0.1, 0.1])
# These describe broad categories/styles, not distinctive model identifiers.
GENERIC_WORDS = set("a an the and with of for s men mens man women womens woman unisex boy boys girl girls kids kid children t shirt shirts tshirt tshirts tee tees top tops shoe shoes watch watches sunglass sunglasses bag bags handbag handbags backpack backpacks belt belts kurta kurtas trouser trousers pant pants jean jeans track casual formal sports sport solid plain printed print striped stripe stripes checked checks check leather cotton polyester regular slim fit fitted round neck v sleeve sleeves sleeveless short long full half dial washed wash denim canvas running training walking collection assorted pack set inch inches size large small medium".split())


def normalize(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def make_text_features(rows):
    # Vocabulary is taken from the selected inputs, without consulting any split or prediction.
    colors = sorted({normalize(r["source_metadata"]["baseColour"]) for r in rows},
                    key=lambda s: (-len(s), s))
    stop = GENERIC_WORDS | {token for color in colors for token in color.split()}
    for r in rows:
        for key in ("gender", "masterCategory", "subCategory", "articleType"):
            stop.update(normalize(r["source_metadata"][key]).split())
    features = []
    for row in rows:
        metadata = row["source_metadata"]
        name = normalize(metadata["productDisplayName"])
        colorless = " " + name + " "
        for color in colors:
            if color:
                colorless = colorless.replace(" " + color + " ", " ")
        colorless = " ".join(colorless.split())
        tokens = sorted(set(name.split()) - stop)
        features.append({
            "name": name, "colorless_name": colorless, "informative_tokens": tokens,
            "specific_name": len(tokens) >= 2,
            "compatible": (row["class_id"], normalize(metadata["gender"]),
                           normalize(metadata["articleType"])),
            "full_metadata": tuple(normalize(metadata[key]) for key in FIELDS if key != "id"),
        })
    return features, colors, sorted(stop)


def image_features(row):
    raw = Path(row["original_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["file_sha256"]:
        raise ValueError(f"Original bytes changed: {row['image_id']}")
    rgb, _ = _decode(raw)
    if _pixel_hash(rgb) != row["pixel_sha256"]:
        raise ValueError(f"Original RGB pixels changed: {row['image_id']}")
    gray = rgb.convert("L")
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    d_bits = (small[:, 1:] > small[:, :-1]).ravel()
    low = dctn(np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float),
               type=2, norm="ortho")[:8, :8].ravel()
    p_bits = low > np.median(low[1:])
    p_bits[0] = False
    return {"phash": int.from_bytes(np.packbits(p_bits).tobytes(), "big"),
            "dhash": int.from_bytes(np.packbits(d_bits).tobytes(), "big"),
            "rgb32": np.asarray(rgb.resize((32, 32), Image.Resampling.LANCZOS))}


def foreground_difference(first, second):
    a, b = first.astype(np.float32) / 255, second.astype(np.float32) / 255
    mask_a, mask_b = np.any(first < 240, axis=2), np.any(second < 240, axis=2)
    union = mask_a | mask_b
    if not union.any():
        return 1.0, 0.0  # Empty/white images are not family evidence.
    return float(np.sqrt(np.mean((a[union] - b[union]) ** 2))), float((mask_a & mask_b).sum() / union.sum())


def visual_rules(first, second, phash_distance, dhash_distance, rgb_first, rgb_second):
    """Accept only predeclared strong evidence, never a generic silhouette alone."""
    if first["compatible"] != second["compatible"]:
        return [], None
    rules, comparison = [], None
    if (first["specific_name"] and second["specific_name"]
            and first["colorless_name"] == second["colorless_name"]
            and phash_distance <= 4 and dhash_distance <= 4):
        rules.append("specific_colorless_name_plus_strong_visual")
    if phash_distance <= 2 and dhash_distance <= 2:
        if first["full_metadata"] == second["full_metadata"]:
            rules.append("full_metadata_plus_very_strong_visual")
        rmse, iou = foreground_difference(rgb_first, rgb_second)
        comparison = {"foreground_rgb_rmse_0_1": rmse, "foreground_iou": iou}
        if rmse <= 0.035 and iou >= 0.95:
            rules.append("near_identical_rgb_plus_very_strong_visual")
    return rules, comparison


def group_inputs(rows, images):
    text, colors, stop = make_text_features(rows)
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    edges = []
    def connect(i, j, rules, **evidence):
        parents[find(j)] = find(i)
        edges.append({"first_index": i, "second_index": j, "rules": rules, **evidence})

    for key in ("file_sha256", "pixel_sha256"):
        buckets = defaultdict(list)
        for i, row in enumerate(rows):
            buckets[row[key]].append(i)
        for members in buckets.values():
            for other in members[1:]:
                connect(members[0], other, [f"exact_{key}"])

    names, metadata_reviews = defaultdict(list), []
    for i, feature in enumerate(text):
        names[(feature["compatible"], feature["name"])].append(i)
    for (_, name), members in sorted(names.items()):
        if len(members) < 2:
            continue
        # High-frequency names are often generic catalog labels; no bulk union.
        if text[members[0]]["specific_name"] and len(members) <= 12:
            for other in members[1:]:
                rules = ["qualified_identical_name_max12"]
                if text[members[0]]["full_metadata"] == text[other]["full_metadata"]:
                    rules.append("qualified_identical_non_id_metadata")
                connect(members[0], other, rules, normalized_name=name)
        else:
            metadata_reviews.append({"normalized_name": name, "member_indices": members,
                                     "reason": "frequent_name_over12" if len(members) > 12 else "generic_or_insufficiently_specific_name",
                                     "action": "not_grouped_on_name_alone; strong visual rules still apply"})

    ph = np.array([f["phash"] for f in images], dtype=np.uint64)
    dh = np.array([f["dhash"] for f in images], dtype=np.uint64)
    candidates = []
    # All unordered pairs, including pairs within the old split. Original split is irrelevant here.
    for start in range(0, len(rows), 64):
        indices = np.arange(start, min(start + 64, len(rows)))
        distances = np.bitwise_count(ph[indices, None] ^ ph[None, :])
        x, y = np.nonzero(distances <= 6)
        mask = y > indices[x]
        x, y = x[mask], y[mask]
        d_distances = np.bitwise_count(dh[indices[x]] ^ dh[y])
        for k in np.flatnonzero(d_distances <= 6):
            i, j = int(indices[x[k]]), int(y[k])
            p, d = int(distances[x[k], y[k]]), int(d_distances[k])
            rules, comparison = visual_rules(text[i], text[j], p, d, images[i]["rgb32"], images[j]["rgb32"])
            candidate = {"first_index": i, "second_index": j, "phash_distance": p, "dhash_distance": d,
                         "rules": rules, "rgb_comparison": comparison}
            candidates.append(candidate)
            if rules:
                connect(i, j, rules, phash_distance=p, dhash_distance=d, rgb_comparison=comparison)
    components = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    groups, membership = [], {}
    for members in sorted(components.values(), key=lambda m: int(rows[m[0]]["image_id"])):
        ids = sorted((rows[i]["image_id"] for i in members), key=int)
        gid = "family_" + hashlib.sha256(",".join(ids).encode()).hexdigest()[:20]
        labels = {rows[i]["class_id"] for i in members}
        if len(labels) != 1:
            raise ValueError("Cross-class exact component needs manual review; refusing to relabel or split it")
        group = {"group_id": gid, "member_indices": members, "image_ids": ids,
                 "class_id": next(iter(labels)), "size": len(members),
                 "family_identity": "UNVERIFIED", "review_status": "UNREVIEWED",
                 "kind": "INFERRED_GROUP" if len(members) > 1 else "NO_LINK_DETECTED",
                 "evidence_rule_counts": Counter()}
        for i in members:
            membership[i] = gid
        groups.append(group)
    by_id = {g["group_id"]: g for g in groups}
    for edge in edges:
        by_id[membership[edge["first_index"]]]["evidence_rule_counts"].update(edge["rules"])
    for group in groups:
        group["uncertainty_flags"] = []
        if group["size"] > 1:
            group["uncertainty_flags"].append("no_authoritative_family_identifier")
        if group["size"] > 2:
            group["uncertainty_flags"].append("transitive_component_not_all_pairs_verified")
        if group["size"] > 25:
            group["uncertainty_flags"].append("large_component_over25")
        if "qualified_identical_name_max12" in group["evidence_rule_counts"]:
            group["uncertainty_flags"].append("contains_metadata_only_links")
    return groups, membership, edges, candidates, metadata_reviews, text, colors, stop


def assign_splits(groups, seed=SEED):
    """Largest groups first; minimize normalized class-count deviation from 80/10/10."""
    assignments = {}
    for label in sorted({g["class_id"] for g in groups}):
        current = sorted((g for g in groups if g["class_id"] == label), key=lambda g: g["group_id"])
        if len(current) < 3:
            raise ValueError(f"Class {label} has fewer than three groups; cannot preserve it in every split")
        rng = np.random.default_rng(np.random.SeedSequence([seed, label]))
        tie_order = {g["group_id"]: float(rng.random()) for g in current}
        current.sort(key=lambda g: (-g["size"], tie_order[g["group_id"]], g["group_id"]))
        target = sum(g["size"] for g in current) * RATIOS
        counts = np.zeros(3, dtype=int)
        group_counts = np.zeros(3, dtype=int)
        for position, group in enumerate(current):
            empty = np.flatnonzero(group_counts == 0)
            allowed = empty if len(current) - position == len(empty) else np.arange(3)
            scores = []
            for split in allowed:
                trial = counts.copy()
                trial[split] += group["size"]
                scores.append(float(np.sum((trial - target) ** 2 / target)))
            best = np.flatnonzero(np.isclose(scores, min(scores), rtol=0, atol=1e-12))
            chosen = int(allowed[int(rng.choice(best))])
            assignments[group["group_id"]] = SPLITS[chosen]
            counts[chosen] += group["size"]
            group_counts[chosen] += 1
    return assignments


def validate(rows, groups, membership, assignments, edges):
    if len(membership) != len(rows) or sum(g["size"] for g in groups) != len(rows):
        raise ValueError("Lost or duplicated group membership")
    indices = [i for g in groups for i in g["member_indices"]]
    if sorted(indices) != list(range(len(rows))):
        raise ValueError("Groups do not partition the input")
    for edge in edges:
        if membership[edge["first_index"]] != membership[edge["second_index"]]:
            raise ValueError("Accepted evidence crosses groups")
    sets = {s: defaultdict(set) for s in SPLITS}
    counts = {s: Counter() for s in SPLITS}
    for i, row in enumerate(rows):
        split = assignments[membership[i]]
        counts[split][row["class_id"]] += 1
        for key in ("image_id", "file_sha256", "pixel_sha256"):
            sets[split][key].add(row[key])
        sets[split]["group_id"].add(membership[i])
    overlap = {}
    for a, b in itertools.combinations(SPLITS, 2):
        overlap[f"{a}--{b}"] = {key: len(sets[a][key] & sets[b][key])
                                 for key in ("image_id", "file_sha256", "pixel_sha256", "group_id")}
    if any(v for pair in overlap.values() for v in pair.values()):
        raise ValueError("Cross-split identity/hash/group leakage")
    classes = {r["class_id"] for r in rows}
    if any(set(counts[s]) != classes for s in SPLITS):
        raise ValueError("A class is absent from a split")
    return counts, overlap


def load_inputs():
    if digest(SOURCE / "preparation.json") != PREPARATION_SHA256:
        raise ValueError("Fixed Phase 3B preparation changed")
    prep = json.loads((SOURCE / "preparation.json").read_text())
    complete = json.loads((SOURCE / "COMPLETE.json").read_text())
    if complete["status"] != "COMPLETE":
        raise ValueError("Incomplete source preparation")
    for name, expected in complete["files"].items():
        if digest(SOURCE / name) != expected:
            raise ValueError(f"Source artifact changed: {name}")
    root = Path(prep["dataset_root"]).resolve()
    csv_hash = digest(root / "styles.csv")
    if csv_hash != json.loads((SOURCE / "audit.json").read_text())["styles_csv_sha256"]:
        raise ValueError("Original styles.csv changed")
    metadata, anomalies = parse_metadata(root / "styles.csv")
    if any("invalid" in m for m in metadata) or len({m["id"] for m in metadata}) != len(metadata):
        raise ValueError("Invalid or duplicate metadata IDs")
    by_id = {m["id"]: m for m in metadata}
    rows = []
    for split in SPLITS:
        original = [json.loads(line) for line in (SOURCE / f"{split}.jsonl").read_text().splitlines()]
        if len(original) != prep["split_counts"][split]:
            raise ValueError("Source split count changed")
        for row in original:
            if row["split"] != split or row["class_name"] != prep["classes"][row["class_id"]]["class_name"]:
                raise ValueError("Source class/split mapping changed")
            if any(str(row["source_metadata"][key]) != by_id[row["image_id"]][key] for key in FIELDS):
                raise ValueError("CSV/manifest metadata mismatch")
            path = (root / row["relative_path"]).resolve()
            if (not path.is_relative_to(root) or path != Path(row["original_path"]).resolve()
                    or path.stem != row["image_id"] or not path.is_file()):
                raise ValueError("Original image path missing or redirected")
        rows.extend(original)
    rows.sort(key=lambda r: int(r["image_id"]))
    if len(rows) != 25455 or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Expected exactly the 25,455 selected Phase 3B images")
    return rows, prep, complete, csv_hash, anomalies


def prepare():
    rows, prep, source_complete, csv_hash, anomalies = load_inputs()
    images = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, feature in enumerate(pool.map(image_features, rows), 1):
            images.append(feature)
            if index % 5000 == 0:
                print(f"Verified/hashes computed: {index}/{len(rows)}", flush=True)
    print("Building input-only evidence graph across all selected images", flush=True)
    groups, membership, edges, candidates, reviews, text, colors, stop = group_inputs(rows, images)
    assignment = assign_splits(groups)
    if assignment != assign_splits(list(reversed(groups))):
        raise ValueError("Group assignment is not deterministic/order independent")
    counts, overlaps = validate(rows, groups, membership, assignment, edges)
    rule_counts = Counter(rule for e in edges for rule in e["rules"])
    unresolved = [c for c in candidates if assignment[membership[c["first_index"]]] != assignment[membership[c["second_index"]]]]
    if any(c["rules"] for c in unresolved):
        raise ValueError("Accepted visual evidence crosses splits")
    if RUNS.resolve() != RUNS.absolute():
        raise ValueError("Output root must not redirect")
    RUNS.mkdir(parents=True, exist_ok=True)
    run = RUNS / f"family_v1_{uuid.uuid4().hex}"
    run.mkdir(exist_ok=False)
    print(f"NEW_RUN_DIRECTORY={run}", flush=True)

    def external_edge(edge):
        i, j = edge["first_index"], edge["second_index"]
        return {"image_ids": [rows[i]["image_id"], rows[j]["image_id"]],
                "group_ids": [membership[i], membership[j]],
                "splits": [assignment[membership[i]], assignment[membership[j]]],
                **{k: v for k, v in edge.items() if k not in ("first_index", "second_index")}}

    for split in SPLITS:
        write_lines(run / f"{split}.jsonl", (
            {**row, "original_image_level_split": row["split"], "split": split,
             "group_id": membership[i], "group_identity_status": "INFERRED_UNVERIFIED"}
            for i, row in enumerate(rows) if assignment[membership[i]] == split))
    write_lines(run / "groups.jsonl", ({**{k: v for k, v in g.items() if k != "member_indices"},
                                         "split": assignment[g["group_id"]]} for g in groups))
    write_lines(run / "accepted_evidence.jsonl", map(external_edge, edges))
    write_lines(run / "similarity_candidates.jsonl", map(external_edge, candidates))
    write_lines(run / "unresolved_cross_split_candidates.jsonl", map(external_edge, unresolved))
    write_lines(run / "metadata_review.jsonl", (
        {**{k: v for k, v in review.items() if k != "member_indices"},
         "image_ids": [rows[i]["image_id"] for i in review["member_indices"]]}
        for review in reviews))
    write_lines(run / "image_features.jsonl", (
        {"image_id": row["image_id"], "file_sha256": row["file_sha256"],
         "pixel_sha256": row["pixel_sha256"], "phash_hex": f"{images[i]['phash']:016x}",
         "dhash_hex": f"{images[i]['dhash']:016x}", **text[i]}
        for i, row in enumerate(rows)))
    configuration = {
        "schema": SCHEMA, "seed": SEED, "source": str(SOURCE), "dataset_root": prep["dataset_root"],
        "source_manifest_hashes": source_complete["files"], "source_preparation_sha256": PREPARATION_SHA256,
        "styles_csv_sha256": csv_hash, "class_names": [c["class_name"] for c in prep["classes"]],
        "grouping_uses_original_split": False, "model_predictions_or_metrics_used": False,
        "name_normalization": "casefold; ASCII alphanumeric tokens; punctuation becomes whitespace",
        "specific_name_rule": "at least two distinct tokens outside recorded generic/category/color vocabulary",
        "generic_stop_words": stop, "color_phrases": colors, "qualified_exact_name_max_members": 12,
        "compatibility": "same final class, gender, articleType for every non-exact-image edge",
        "rules": {
            "exact_image": "identical byte or decoded-RGB SHA256; mandatory coassignment",
            "qualified_identical_name_max12": "compatible, specific identical normalized names, bucket size 2..12; all coassigned",
            "qualified_identical_non_id_metadata": "additional evidence on qualified identical-name edges; generic metadata alone never merges",
            "specific_colorless_name_plus_strong_visual": "compatible, specific identical color-stripped names AND pHash<=4 AND dHash<=4",
            "full_metadata_plus_very_strong_visual": "compatible, all normalized non-ID CSV fields identical AND pHash<=2 AND dHash<=2",
            "near_identical_rgb_plus_very_strong_visual": "compatible AND pHash<=2 AND dHash<=2 AND foreground RGB RMSE<=0.035 AND mask IoU>=0.95",
        },
        "perceptual_hashes": "Phase 3C.1 method: grayscale Lanczos 32x32, orthonormal DCT-II 8x8, median AC threshold, DC bit zero; horizontal dHash 9x8",
        "candidate_rule": "pHash<=6 AND dHash<=6; all unordered image pairs, independent of old split",
        "rgb_comparison": "RGB Lanczos 32x32 [0,1]; foreground any channel<240/255; RMSE over union of masks; empty union rejected",
        "group_rule": "connected components of accepted evidence; all identities inferred and unverified",
        "split_rule": "per class, largest groups first, seed42 tie breaks; minimize sum((count-target)^2/target); ensure all splits represented; groups indivisible",
        "target_ratios": RATIOS.tolist(), "manual_family_ids_available": False,
        "csv_anomalies_preserved": anomalies,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pillow": pillow_version},
        "script_sha256": digest(__file__),
    }
    write_json(run / "configuration.json", configuration)
    per_class = [{"class_id": c["class_id"], "class_name": c["class_name"],
                  "total": sum(counts[s][c["class_id"]] for s in SPLITS),
                  "groups": sum(g["class_id"] == c["class_id"] for g in groups),
                  **{s: counts[s][c["class_id"]] for s in SPLITS}} for c in prep["classes"]]
    overall_deviation = max(abs(sum(counts[s].values()) / len(rows) - RATIOS[k])
                            for k, s in enumerate(SPLITS))
    per_class_deviation = max(abs(c[s] / c["total"] - RATIOS[k])
                              for c in per_class for k, s in enumerate(SPLITS))
    approximate_ratios = overall_deviation <= 0.02 and per_class_deviation <= 0.03
    report = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "total_images": len(rows), "total_groups": len(groups), "per_class": per_class,
        "split_counts": {s: sum(counts[s].values()) for s in SPLITS},
        "split_group_counts": dict(Counter(assignment.values())),
        "multi_image_inferred_groups": sum(g["size"] > 1 for g in groups),
        "images_in_multi_image_groups": sum(g["size"] for g in groups if g["size"] > 1),
        "singleton_groups": sum(g["size"] == 1 for g in groups),
        "verified_true_families": 0, "largest_group_size": max(g["size"] for g in groups),
        "large_groups_over25": [g["group_id"] for g in groups if g["size"] > 25],
        "metadata_only_link_groups": sum("qualified_identical_name_max12" in g["evidence_rule_counts"] for g in groups),
        "accepted_edges": len(edges), "accepted_rule_counts": dict(rule_counts),
        "metadata_name_buckets_requiring_review": len(reviews),
        "all_perceptual_candidate_pairs": len(candidates),
        "unresolved_cross_split_candidate_pairs": len(unresolved),
        "unresolved_cross_split_candidate_images": len({i for e in unresolved for i in (e["first_index"], e["second_index"])}),
        "unresolved_candidate_interpretation": "weak/unaccepted signals, not confirmed leakage; available for manual review",
        "cross_split_overlap": overlaps, "accepted_evidence_cross_split_edges": 0,
        "class_imbalance_max_to_min": {s: max(counts[s].values()) / min(counts[s].values()) for s in SPLITS},
        "overall_class_imbalance_max_to_min": max(c["total"] for c in per_class) / min(c["total"] for c in per_class),
        "deterministic_assignment_verified": True, "all_source_images_preserved": True,
        "all_ten_classes_in_every_split": True, "training_executed": False,
        "max_overall_ratio_deviation": float(overall_deviation),
        "max_per_class_ratio_deviation": float(per_class_deviation),
        "approximate_ratio_check": "overall within 2 percentage points and each class within 3 points of 80/10/10",
        "ready_for_training": bool(approximate_ratios),
        "readiness_scope": "frozen inferred-group benchmark, not certified independence of unknown true product families",
        "limitations": ["No authoritative product-family IDs; every multi-image group remains UNREVIEWED/UNVERIFIED",
                        "Metadata-only links may overgroup distinct products; transitive links can join larger families",
                        "Strict visual rules may miss different views or unsupported variants; weak candidates remain across splits",
                        "The Phase 3C model was trained on the old split, so cannot be evaluated as an independently trained baseline on this new test split"],
    }
    write_json(run / "audit.json", report)
    write_json(run / "COMPLETE.json", {"schema": SCHEMA, "status": "COMPLETE",
                                       "files": {p.name: digest(p) for p in sorted(run.iterdir()) if p.is_file()}})
    print(json.dumps(report, indent=2), flush=True)
    return run


if __name__ == "__main__":
    prepare()
