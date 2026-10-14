"""Case manifest from the released SynthRAD2025 Task 2 data (<data>/<REGION>/<case>/{cbct,ct,mask}.mha).

    python -m synthrad_eval.manifest --data <Task2 dir> --out cases.csv [--cases <ids or a file of ids>]
        [--ct "<dir>/{case}/ct_def.mha"]

By default the reference is the released ct.mha, which is only rigidly aligned to the CBCT. The
challenge's ground truth was deformed to the CBCT with Elastix; to evaluate the same way, register
the CTs yourself and point --ct at the result ({case} is replaced by the case id).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def find_image(case_dir: Path, name: str) -> Path:
    """Case-insensitive: the data ships as ct.mha / CT.mha / mask.mha in mixed case."""
    for p in case_dir.iterdir():
        if p.name.lower() == f"{name}.mha":
            return p
    raise FileNotFoundError(f"{case_dir}: no {name}.mha")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="Task 2 data root: <REGION>/<case>/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", nargs="*", help="case ids, or one text file of whitespace-separated ids")
    ap.add_argument("--ct", help="reference CT path template with {case} (default: the released ct.mha)")
    args = ap.parse_args()
    names = args.cases
    if names and len(names) == 1 and Path(names[0]).is_file():
        names = Path(names[0]).read_text().split()
    cases = sorted(p for p in args.data.glob("*/2*") if p.is_dir() and (not names or p.name in names))
    rows, missing = [], []
    for c in cases:
        ct = Path(args.ct.format(case=c.name)) if args.ct else find_image(c, "ct")
        if not ct.exists():
            missing.append(c.name)
            continue
        rows.append({"case": c.name, "region": c.name[1:3], "ct": str(ct.resolve()),
                     "mask": str(find_image(c, "mask").resolve())})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "region", "ct", "mask"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cases -> {args.out}" + (f"; no reference CT for {len(missing)}: {' '.join(missing)}"
                                                 if missing else ""))


if __name__ == "__main__":
    main()
