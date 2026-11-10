"""Check a submission folder before scoring.

    python -m selfdox.validate --sub outputs --inputs data/dev/images
"""
import argparse
import json
import sys
from pathlib import Path

from .schema import gt_sizes, input_sizes, submission_errors


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate a SelfDox submission folder.")
    ap.add_argument("--sub", required=True, help="folder with predictions.json and images/")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", help="folder with the benchmark input images")
    src.add_argument("--gt", help="gt.json (alternative to --inputs)")
    args = ap.parse_args(argv)
    src = Path(args.inputs or args.gt)
    if not src.exists():
        print(f"ERROR   {src} not found. Point --inputs at the photos your agent read (in the "
              "v1.0 package: data/dev/images or data/test/images) or --gt at a gt.json.")
        return 2
    sizes = input_sizes(args.inputs) if args.inputs else \
        gt_sizes(json.loads(Path(args.gt).read_text()))
    errors, warnings = submission_errors(args.sub, sizes)
    for w in warnings:
        print(f"WARNING {w}")
    for e in errors:
        print(f"ERROR   {e}")
    if errors:
        print(f"\n{len(errors)} error(s). Fix them before submitting.")
        return 1
    print(f"OK: submission is valid for {len(sizes)} input image(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
