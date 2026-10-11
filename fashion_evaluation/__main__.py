"""Evaluate precomputed predictions; never trains or imports training scripts."""
import argparse
from .predictions import load_precomputed
from .reports import build_report, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="JSON prediction bundle")
    parser.add_argument("--full-test", action="store_true", help="Require verified coverage of all original test rows")
    parser.add_argument("--report-name", help="New JSON filename; existing reports are never overwritten")
    args = parser.parse_args()
    bundle, identity = load_precomputed(args.predictions)
    report = build_report(bundle, source_identity=identity, full=args.full_test)
    print(write_report(report, args.report_name))


if __name__ == "__main__":
    main()
