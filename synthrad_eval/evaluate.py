"""SynthRAD2025-style evaluation: all 11 Task 2 challenge metrics on cases with a known CT.

The scoring is the organisers' evaluation container code (vendored in this package, see the file
headers): image metrics, TotalSegmentator 3 mm Dice/HD95 and pyRadPlan dose recalculation. Only the
ground truth the container reads from /opt/ml/input/data/ground_truth is rebuilt here:

    ct/<id>.mha            -> the manifest's reference CT (e.g. the deformed CT used for training)
    mask/<id>.mha          -> the manifest's mask
    segmentation/<id>.mha  -> the container's own TotalSegmentator applied to that CT (prepare)
    dose/<id>/             -> approximate plans from synthrad_eval.plans (prepare)

The manifest is a CSV with columns case,ct,mask (region optional, else from the case id; relative
paths are relative to the manifest). Three steps (prepare and score are resumable and shardable):

    python -m synthrad_eval.evaluate prepare --manifest cases.csv --ref <references>
    python -m synthrad_eval.evaluate score --manifest cases.csv --ref <references> \
        --sct <dir with <id>_sCT.mha, <id>/sCT.mha or <id>.mha> --out <dir> [--no-dose]
    python -m synthrad_eval.evaluate summary --out <dir> [--compare <another metrics dir>]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .plans import PLANS_VERSION

IMAGE = ["mae", "psnr", "ms_ssim"]
SEG = ["DICE", "HD95"]
DOSE = [f"{m}_{p}" for p in ("photon", "proton") for m in ("gamma_pass_rate", "mae", "dvh")]
FIELDS = ["case", "region"] + IMAGE + SEG + DOSE + ["seconds"]
# Report table 7, task 2 columns, for the summary
LOWER_IS_BETTER = {"mae", "HD95", "mae_photon", "dvh_photon", "mae_proton", "dvh_proton"}


def xyz(img: sitk.Image) -> np.ndarray:
    """The container's load_image_file_directly: numpy in (x, y, z) order."""
    return np.transpose(sitk.GetArrayFromImage(img), [2, 1, 0])


def orientation(img: sitk.Image):
    return img.GetSpacing(), img.GetOrigin(), img.GetDirection()


class Case:
    def __init__(self, cid: str, ct: Path, mask: Path, region: str):
        self.id, self.ct, self.mask, self.region = cid, ct, mask, region


def read_manifest(path: Path, names: list[str] | None) -> list[Case]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    base = path.resolve().parent
    cases = [Case(r["case"], base / r["ct"], base / r["mask"], r.get("region") or r["case"][1:3])
             for r in rows if not names or r["case"] in names]
    return sorted(cases, key=lambda c: c.id)


def write_xyz_like(arr_xyz: np.ndarray, ref: sitk.Image, path: Path) -> None:
    img = sitk.GetImageFromArray(np.transpose(arr_xyz, [2, 1, 0]).astype(np.uint8))
    img.CopyInformation(ref)
    tmp = path.with_suffix(".tmp.mha")
    sitk.WriteImage(img, str(tmp), True)
    os.replace(tmp, path)


def prepare_case(case: Case, ref: Path, ts, plans_on: bool) -> str:
    from .plans import plan_case
    cid = case.id
    dst = ref / cid
    dst.mkdir(parents=True, exist_ok=True)
    ct_path = case.ct
    ct = sitk.ReadImage(str(ct_path))
    mask_img = sitk.ReadImage(str(case.mask))
    mask_img.CopyInformation(ct)
    seg_path = dst / "segmentation.mha"
    done = []
    if not seg_path.exists():
        import torch
        t0 = time.time()
        with torch.no_grad():               # as SegmentationMetrics.score_patient calls it
            seg = ts.score_patient(str(ct_path), orientation(ct), xyz(mask_img))
        write_xyz_like(seg, ct, seg_path)
        done.append(f"seg {time.time() - t0:.0f}s")
    if plans_on and not (dst / "dose" / "plan.json").exists():
        t0 = time.time()
        s = plan_case(cid, ct, mask_img, sitk.ReadImage(str(seg_path)), dst / "dose")
        done.append(f"plans {time.time() - t0:.0f}s (PTV {s['ptv_cc']:.0f} cc, {s['side']}, "
                    f"photon x{s['photons']['d95_scale']}, proton x{s['protons']['d95_scale']})")
    return ", ".join(done) or "done before"


def find_sct(sct_dir: Path, cid: str) -> Path:
    for p in (sct_dir / f"{cid}_sCT.mha", sct_dir / cid / "sCT.mha", sct_dir / f"{cid}.mha"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no sCT for {cid} in {sct_dir}")


def plans_version(ref_case: Path) -> int:
    return json.loads((ref_case / "dose" / "plan.json").read_text()).get("plans_version", 1)


def score_case(case: Case, sct_path: Path, ref: Path, evaluators, dose_on: bool) -> dict:
    """The container's process(), with the ground truth of the manifest and --ref."""
    image_ev, seg_ev, dose_ev = evaluators
    cid = case.id
    if dose_on and plans_version(ref / cid) != PLANS_VERSION:
        raise RuntimeError(f"{ref / cid} was built by plans version {plans_version(ref / cid)}, this is "
                           f"{PLANS_VERSION}: rebuild it with 'prepare' into a new --ref")
    gt = sitk.ReadImage(str(case.ct))
    spacing, origin, direction = orientation(gt)
    gt_img = xyz(gt)

    def load(path: Path) -> np.ndarray:
        img = sitk.ReadImage(str(path))
        img.SetSpacing(spacing), img.SetOrigin(origin), img.SetDirection(direction)
        return xyz(img)

    mask_path = case.mask
    sct, gt_seg, mask = load(sct_path), load(ref / cid / "segmentation.mha"), load(mask_path)
    row = dict(image_ev.score_patient(gt_img, sct, mask))
    row.update(seg_ev.score_patient(str(sct_path), mask, gt_seg, cid, orientation=(spacing, origin, direction)))
    if dose_on:
        d = dose_ev.score_patient(str(sct_path), str(mask_path), str(ref / cid / "dose"), cid)
        d.pop("case", None)
        row.update(d)
    return row


def read_rows(out: Path) -> list[dict]:
    rows = []
    for p in sorted(out.glob("metrics*.csv")):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    return rows


def summarize(out: Path, other: Path | None = None) -> None:
    rows = read_rows(out)
    if other:                               # the same cases only, so the two tables compare
        rows_other = read_rows(other)
        common = {r["case"] for r in rows} & {r["case"] for r in rows_other}
        for name, rs in ((out, rows), (other, rows_other)):
            print(f"\n== {name} ({len(common)} common cases)", end="")
            table([r for r in rs if r["case"] in common])
        return
    table(rows)


def table(rows: list[dict]) -> None:
    cols = [k for k in IMAGE + SEG + DOSE if any(r.get(k) for r in rows)]
    print(f"\n{'region':<7}{'n':>4}  " + "  ".join(f"{k:>16}" for k in cols))
    for region in ["AB", "TH", "HN", "all"]:
        rs = [r for r in rows if region == "all" or r["region"] == region]
        if not rs:
            continue
        cells = []
        for k in cols:
            v = np.array([float(r[k]) for r in rs if r.get(k) not in (None, "")])
            p = 4 if k in ("ms_ssim", "DICE") or k.startswith(("mae_", "dvh_")) else 2
            cells.append(f"{v.mean():.{p}f} ± {v.std():.{p}f}".rjust(16) if v.size else " " * 16)
        print(f"{region:<7}{len(rs):>4}  " + "  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "score", "summary"])
    ap.add_argument("--manifest", type=Path, help="CSV: case,ct,mask[,region]")
    ap.add_argument("--ref", type=Path, help="challenge ground truth built by 'prepare'")
    ap.add_argument("--sct", type=Path, help="score: directory of synthetic CTs")
    ap.add_argument("--out", type=Path, help="score/summary: metrics directory")
    ap.add_argument("--compare", type=Path, help="summary: another metrics directory, same cases only")
    ap.add_argument("--cases", nargs="*", help="only these case ids of the manifest")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--no-dose", action="store_true", help="skip plans (prepare) / dose metrics (score)")
    args = ap.parse_args()
    if args.step == "summary":
        summarize(args.out, args.compare)
        return

    k, n = map(int, args.shard.split("/"))
    cases = read_manifest(args.manifest, args.cases)[k::n]
    from .segmentation_metrics import SegmentationMetrics
    seg_ev = SegmentationMetrics()
    failed = []

    if args.step == "prepare":
        for i, case in enumerate(cases, 1):
            try:
                msg = prepare_case(case, args.ref, seg_ev.my_ts, not args.no_dose)
            except Exception as e:
                failed.append(case.id)
                msg = f"FAILED {type(e).__name__}: {e}"
                traceback.print_exc()
            print(f"[{i}/{len(cases)}] {case.id}: {msg}", flush=True)
    else:
        from .dose_metrics import DoseMetrics
        from .image_metrics import ImageMetrics
        evaluators = (ImageMetrics(), seg_ev, DoseMetrics())
        args.out.mkdir(parents=True, exist_ok=True)
        csv_path = args.out / ("metrics.csv" if n == 1 else f"metrics_{k}.csv")
        done = set()
        for p in args.out.glob("metrics*.csv"):
            with open(p) as f:
                done |= {r["case"] for r in csv.DictReader(f)}
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            for i, case in enumerate(cases, 1):
                if case.id in done:
                    continue
                t0 = time.time()
                try:
                    row = score_case(case, find_sct(args.sct, case.id), args.ref, evaluators, not args.no_dose)
                except Exception as e:
                    failed.append(case.id)
                    traceback.print_exc()
                    print(f"[{i}/{len(cases)}] {case.id}: FAILED {type(e).__name__}: {e}", flush=True)
                    continue
                row.update(case=case.id, region=case.region, seconds=round(time.time() - t0, 1))
                w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
                f.flush()
                print(f"[{i}/{len(cases)}] {case.id}: " + "  ".join(
                    f"{k} {row[k]:.4g}" for k in IMAGE + SEG + DOSE if k in row), flush=True)
        summarize(args.out)
    if failed:
        print(f"{len(failed)} failed: {' '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
