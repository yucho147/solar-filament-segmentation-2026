"""Strictly verify a completed full inference run against the submitted result.

Usage: python scripts/verify_10model_regression.py OUTPUT_DIR BUNDLE_DIR
Run on the reference GPU/software environment for bitwise comparisons.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify(output, bundle):
    ids = [
        line.strip() for line in (bundle / "test_ids.txt").read_text().splitlines() if line.strip()
    ]
    run_ids = [
        line.strip() for line in (output / "ids.txt").read_text().splitlines() if line.strip()
    ]
    if len(ids) != 180 or len(set(ids)) != 180 or run_ids != ids:
        raise ValueError("run must contain all 180 reference image IDs in the expected order")
    if json.loads((output / "status.json").read_text())["state"] != "done":
        raise ValueError("inference run did not complete")
    identity = json.loads((output / "run_inputs.json").read_text())
    if identity["dry"]:
        raise ValueError("saved-map mode does not verify inference from weights")
    refs = json.loads((bundle / "probs_reference_sha256.json").read_text())
    for group in ("g42", "g43", "ens10"):
        directory = output / f"probs_{group}"
        for image_id in ids:
            path = directory / f"{image_id}.npy"
            if sha256(path) != refs[group][image_id]:
                raise ValueError(f"probability mismatch: {group}/{image_id}")
    if sha256(output / "C_10m.csv") != sha256(bundle / "submission_C_10m.csv"):
        raise ValueError("submitted C CSV mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        verify(args.output, args.bundle)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Regression verification failed: {exc}\n")
    print("PASS: 180 images, three probability maps per image, and submitted C CSV match.")
