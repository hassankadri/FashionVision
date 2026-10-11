"""Print a read-only dataset audit as JSON. No automatic report/data writes."""
import argparse
import json
from pathlib import Path

from .pipeline import ClassMapping, scan_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Existing directory containing train, validation and test")
    parser.add_argument("--mapping", help="Existing JSON with class_names and folders fields")
    args = parser.parse_args()
    mapping = None
    if args.mapping:
        config = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
        mapping = ClassMapping(tuple(config["class_names"]), config["folders"])
    audit = scan_dataset(args.root, mapping)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    return 0 if audit["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
