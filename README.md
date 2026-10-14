# synthrad2025_eval

Evaluate a SynthRAD2025 Task 2 (CBCT → CT) model on all 11 challenge metrics, on cases whose CT
you have, as close to the challenge's final-test scoring as the public information allows:

* **image:** MAE, PSNR, MS-SSIM
* **segmentation:** Dice and HD95 of TotalSegmentator structures
* **dose:** gamma pass rate (2%/2 mm), dose MAE and DVH metric, for a photon and a proton plan

The segmentation and dose metrics need ground truth that was never released: reference
segmentations and treatment plans. This repository runs the organisers' evaluation code and
rebuilds that ground truth. It is model-agnostic: the inputs are a folder of synthetic CTs and a
list of cases.

## How close to the challenge

The organisers' evaluation container ([SynthRAD2025/evaluation](https://github.com/SynthRAD2025/evaluation)
@ 99e9ef4) is vendored file by file in `synthrad_eval/`, with changes marked `synthrad_eval:`.
`requirements*.txt` pin its environment: Python 3.11, CPU torch, pyRadPlan 0.2.8, and the
TotalSegmentator and nnU-Net commits of its git submodules.

| Metrics | Code | Ground truth |
|---|---|---|
| MAE, PSNR, MS-SSIM | exact | the reference CT you supply (see below) |
| Dice, HD95 | exact: TotalSegmentator `total` 3 mm on CPU, same structures | the organisers' segmentations are not public, so the container's own segmentation of the reference CT stands in |
| gamma, dose MAE, DVH metric (photon, proton) | exact recalculation and metrics: pyRadPlan forward dose on the stitched sCT | **approximate plans**: the clinical PTVs and OARs are private, so `plans.py` builds a synthetic PTV at a typical site per region from TotalSegmentator anatomy |

The plans use everything the challenge report (sec. 2.4.2) fixes: IMRT with 9 equi-angled 6 MV
beams, IMPT with 3 beams mirrored to the near side, 30×2 Gy (HN, AB) or 35×2 Gy (TH) prescribed to
the PTV D95, and the OAR limits of report tables 1–3. What had to be guessed (target sites, proton
angles, dose grid, optimiser) is listed at the top of `plans.py`.

`prepare` writes per case exactly what the container reads from its ground-truth folder
(`segmentation.mha`, `dose/<id>.mat`, `<id>_{photons,protons}_plan.mat`, `..._dose.npy.gz`), so the
container's `DoseMetrics.score_patient` runs unchanged. Scoring the reference CT against itself gives
perfect values on all 11 metrics.

The image metrics are the container's: it clips to [−1024, 3071] HU, while the metrics repository
and the report say [−1024, 3000]. The difference only touches very dense bone and metal.

### Reference CT

The released training CTs are only rigidly aligned to the CBCT; the challenge's validation and
test ground truth were deformed to the CBCT with Elastix, using the organisers' preprocessing
parameter files. Both work as reference here. A deformed CT is closer to the challenge, and gives
better image metrics since less residual misalignment counts as error.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate     # Python 3.11, as the container
scripts/setup_env.sh ~/ts_results                          # pinned packages + TotalSegmentator weights (~160 MB)
export SYNTHRAD_EVAL_TS_RESULTS=~/ts_results
```

Or `docker build -t synthrad-eval .`, which rebuilds the container's environment from
`python:3.11-slim`. `setup_env.sh` passes extra arguments to `pip install`, e.g.
`--target <dir> --upgrade` for a package directory next to a read-only image.

## Use

```bash
# 1. the cases: CSV with case,ct,mask[,region], from the released data layout <REGION>/<case>/
python -m synthrad_eval.manifest --data <Task2 dir> --cases val_ids.txt --out cases.csv \
    [--ct "<registered dir>/{case}/ct_def.mha"]

# 2. ground truth, once per case set: TotalSegmentator + photon and proton plans
python -m synthrad_eval.evaluate prepare --manifest cases.csv --ref refs/

# 3. score a model's sCTs (<id>_sCT.mha, <id>/sCT.mha or <id>.mha)
python -m synthrad_eval.evaluate score --manifest cases.csv --ref refs/ --sct <sCT dir> --out results/model_a/
python -m synthrad_eval.evaluate summary --out results/model_a/ [--compare results/model_b/]
```

`prepare` takes ~2–3 min and ~5 GB per case on 16 CPU cores (TotalSegmentator plus two plan
optimisations); `score` ~40 s per case. Both are resumable, and `--shard K/N` runs every N-th
case starting at K, so N parallel processes cover the case set. `score` writes one CSV row per
case; `summary` prints mean ± std per region, and with `--compare` restricts both runs to their
common cases.

References record the version of `plans.py` that built them (`plans_version`), and `score` refuses
references from another version, so plan changes never mix silently with old results.

On a read-only home (containers), set `MPLCONFIGDIR` and `NUMBA_CACHE_DIR` to writable directories.

## Calibration

Scored here: fold 0 (193 cases) of BreizhCT's public 5-fold split, each case predicted by their
public fold-0 model ([weights](https://huggingface.co/VBoussot/Synthrad2025), flip TTA), which
never saw it, against an Elastix-deformed reference CT. Next to it, their final-test-set row of
the report (Table 7).

| | MAE ↓ | PSNR ↑ | MS-SSIM ↑ | Dice ↑ | HD95 ↓ | γ ph ↑ | DVH ph ↓ | MAE ph ↓ | γ pr ↑ | DVH pr ↓ | MAE pr ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| here, fold 0 | 62.3 | 30.2 | 0.952 | 0.805 | 6.29 | 97.9 | 0.033 | 0.0085 | 80.1 | 0.179 | 0.0149 |
| report, test set | 53.1 | 32.5 | 0.966 | 0.84 | 5.08 | 99.31 | 0.015 | 0.005 | 86.41 | 0.036 | 0.020 |

The rows are not expected to match: the test cases are hidden, and the submission was a 5-fold
ensemble rather than one fold model. That already shows on the image metrics, whose code is exact.
Given that, photon dose is close. Proton gamma matches for AB (86.4) and HN (87.1), while thorax
(66.9, DVH 0.39 ± 0.49) is the weak spot of the approximation: the synthetic thorax target sits in
the lung, where proton range is most sensitive to HU errors.

**Use the dose metrics to compare models scored here, not as estimates of a leaderboard score.**

## License

GPL-3.0, as the vendored evaluation code.
