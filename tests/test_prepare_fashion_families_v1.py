"""Grouping/split safety on synthetic metadata/features; never modify real sources."""

from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts import prepare_fashion_families_v1 as families


def row(identity, name="Men T-shirt", label=0, color="Black"):
    metadata = dict(
        id=str(identity),
        gender="Men",
        masterCategory="Apparel",
        subCategory="Topwear",
        articleType="Tshirts",
        baseColour=color,
        season="Summer",
        year="2012",
        usage="Casual",
        productDisplayName=name,
    )
    return {
        "image_id": str(identity),
        "class_id": label,
        "class_name": f"class_{label}",
        "source_metadata": metadata,
        "file_sha256": f"bytes{identity}",
        "pixel_sha256": f"pixels{identity}",
        "split": "train",
    }


def feature(value=0, color=0):
    return {
        "phash": value,
        "dhash": value,
        "rgb32": np.full((32, 32, 3), color, dtype=np.uint8),
    }


class FamilyGroupingTests(unittest.TestCase):
    def test_generic_names_and_metadata_alone_do_not_merge(self):
        rows = [row(1), row(2)]
        groups, _, edges, _, reviews, *_ = families.group_inputs(
            rows,
            [feature(), feature(2**64 - 1, 200)],
        )
        self.assertEqual(len(groups), 2)
        self.assertFalse(edges)
        self.assertEqual(
            reviews[0]["reason"], "generic_or_insufficiently_specific_name"
        )

    def test_qualified_identical_names_are_inferred_not_confirmed(self):
        rows = [
            row(1, "Acme Runner Pro Black T-shirt"),
            row(2, "ACME RUNNER PRO BLACK T SHIRT"),
        ]
        # Hyphen-to-space normalization makes these product names equivalent.
        groups, membership, edges, *_ = families.group_inputs(
            rows,
            [feature(), feature(2**64 - 1, 200)],
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(membership[0], membership[1])
        self.assertIn("qualified_identical_name_max12", edges[0]["rules"])
        self.assertEqual(groups[0]["family_identity"], "UNVERIFIED")
        self.assertEqual(groups[0]["review_status"], "UNREVIEWED")

    def test_frequent_names_never_bulk_merge(self):
        rows = [row(i, "Acme Runner Pro Black T-shirt") for i in range(13)]
        # Distinct hashes do not prove distance; check only the metadata edge rule.
        _, _, edges, _, reviews, *_ = families.group_inputs(
            rows,
            [feature(i, i * 17) for i in range(13)],
        )
        self.assertTrue(
            all("qualified_identical_name_max12" not in e["rules"] for e in edges)
        )
        self.assertEqual(reviews[0]["reason"], "frequent_name_over12")

    def test_color_variants_require_specific_metadata_and_strong_visual(self):
        rows = [
            row(1, "Acme Wanderer Pro Black T-shirt"),
            row(2, "Acme Wanderer Pro Blue T-shirt", color="Blue"),
        ]
        text, _, _ = families.make_text_features(rows)
        rules, _ = families.visual_rules(
            *text, 2, 1, feature(color=0)["rgb32"], feature(color=100)["rgb32"]
        )
        self.assertIn("specific_colorless_name_plus_strong_visual", rules)

        rules, _ = families.visual_rules(
            *text, 6, 6, feature()["rgb32"], feature(color=100)["rgb32"]
        )
        self.assertFalse(rules)

    def test_hash_collision_alone_does_not_merge_different_products(self):
        rows = [
            row(1, "Acme Black T-shirt"),
            row(2, "Other Blue T-shirt", color="Blue"),
        ]
        groups, _, edges, candidates, *_ = families.group_inputs(
            rows,
            [feature(color=0), feature(color=120)],
        )
        self.assertEqual(len(candidates), 1)
        self.assertFalse(edges)
        self.assertEqual(len(groups), 2)

    def test_near_identical_pixels_with_compatible_metadata_can_merge(self):
        rows = [row(1, "Acme Black T-shirt"), row(2, "Other Black T-shirt")]
        groups, _, edges, *_ = families.group_inputs(
            rows,
            [feature(color=60), feature(color=61)],
        )
        self.assertEqual(len(groups), 1)
        self.assertIn(
            "near_identical_rgb_plus_very_strong_visual",
            edges[0]["rules"],
        )
        self.assertEqual(
            families.foreground_difference(
                np.full((32, 32, 3), 255, np.uint8),
                np.full((32, 32, 3), 255, np.uint8),
            ),
            (1.0, 0.0),
        )

    def test_grouping_is_independent_of_old_split_and_predictions(self):
        rows = [
            row(1, "Acme Runner Pro Black T-shirt"),
            row(2, "Acme Runner Pro Black T-shirt"),
        ]
        features = [feature(), feature(2**64 - 1, 200)]
        before = deepcopy(rows)
        first = families.group_inputs(rows, features)
        self.assertEqual(rows, before)

        changed = deepcopy(rows)
        changed[0].update(split="test", prediction="ignored", accuracy=0)
        changed[1].update(split="validation", prediction="different", accuracy=1)
        second = families.group_inputs(changed, features)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_transitive_links_and_exact_duplicates_stay_in_one_group(self):
        rows = [row(i, "Acme Runner Pro Black T-shirt") for i in range(3)]
        rows[2]["source_metadata"]["productDisplayName"] = "Other Black T-shirt"
        rows[2]["file_sha256"] = rows[1]["file_sha256"]
        groups, membership, _, *_ = families.group_inputs(
            rows,
            [
                feature(),
                feature(2**64 - 1, 100),
                feature(0xaaaaaaaaaaaaaaaa, 200),
            ],
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(set(membership.values())), 1)


class GroupSplitTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.groups, self.membership = [], [], {}
        for label in range(10):
            for k, size in enumerate([7, 4, 3, 2] + [1] * 34):
                gid = f"group_{label}_{k}"
                members = list(range(len(self.rows), len(self.rows) + size))
                for index in members:
                    self.rows.append(row(index, label=label))
                    self.membership[index] = gid
                self.groups.append({
                    "group_id": gid,
                    "class_id": label,
                    "size": size,
                    "member_indices": members,
                })

    def test_deterministic_group_stratification_counts_and_no_overlap(self):
        first = families.assign_splits(self.groups)
        self.assertEqual(first, families.assign_splits(list(reversed(self.groups))))
        counts, overlap = families.validate(
            self.rows, self.groups, self.membership, first, []
        )
        self.assertTrue(
            all(value == 0 for pair in overlap.values() for value in pair.values())
        )
        for label in range(10):
            self.assertEqual(
                [counts[s][label] for s in families.SPLITS],
                [40, 5, 5],
            )

    def test_duplicate_leakage_is_rejected(self):
        assignment = families.assign_splits(self.groups)
        a = next(
            i for i in self.membership if assignment[self.membership[i]] == "train"
        )
        b = next(
            i for i in self.membership if assignment[self.membership[i]] == "test"
        )
        self.rows[b]["file_sha256"] = self.rows[a]["file_sha256"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            families.validate(
                self.rows, self.groups, self.membership, assignment, []
            )

    def test_insufficient_groups_cannot_silently_remove_a_class(self):
        with self.assertRaisesRegex(ValueError, "fewer than three groups"):
            families.assign_splits(self.groups[:2])

    def test_reports_refuse_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "historical.json"
            families.write_json(path, {"preserve": True})
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):
                families.write_json(path, {"preserve": False})
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()