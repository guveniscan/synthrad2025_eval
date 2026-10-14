"""Approximate SynthRAD2025 treatment plans, written in the organisers' file format.

The challenge recalculates a photon and a proton plan per test patient on the sCT
(dose_metrics.DoseMetrics.score_patient). Those plans are private. What the report (Rogowski et al.,
MedIA 2026, sec. 2.4.2) and the evaluation code fix, and what is reproduced here exactly:

* planning CT = the reference CT (the challenge used the deformed CT), dose cubes per fraction on its grid
* photons: 9 equi-angled co-planar 6 MV beams (matRad generic linac), IMRT
* protons: 3 co-planar beams (matRad generic machine), IMPT, no robust optimisation,
  angles fixed per site and mirrored with the isocenter side to prefer short ranges
* isocenter = PTV centre; prescription 30x2 Gy (HN, AB) or 35x2 Gy (TH) at the 95% isodose (D95)
* OAR limits of report tables 1-3 (where TotalSegmentator has the organ)
* recalculation: pyRadPlan 0.2.8 forward pencil beam, trace_on_dose_grid, 25 mm photon lateral cutoff

What is NOT known and is guessed here (the approximation):

* the clinical PTV and OAR contours -> a synthetic PTV at a typical site for the region, placed
  from TotalSegmentator anatomy, and OARs from TotalSegmentator
* the proton beam angles per site -> PROTON_ANGLES below
* the dose grid of the organisers' plans -> 3 mm (matRad's default resolution)
* the optimiser (matRad/IPOPT) -> pyRadPlan's scipy solver; the reference dose is the pyRadPlan
  forward dose of the optimised weights on the CT, so sCT == CT scores perfectly, as it must

Files per case (what GROUND_TRUTH/dose/<id>/ holds for the organisers):
    <id>.mat                      ct + cst (matRad)
    <id>_{photons,protons}_plan.mat   pln + stf + weights (matRad)
    <id>_{photons,protons}_dose.npy.gz reference physical dose (z, y, x), Gy per fraction
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import io as sio
from scipy import ndimage

# Bump whenever a change here alters the reference plans: evaluate.py refuses to score against
# references built by another version (plan.json "plans_version"; none recorded = 1).
PLANS_VERSION = 1
RX_PER_FRACTION = 2.0
FRACTIONS = {"HN": 30, "AB": 30, "TH": 35}
DOSE_GRID_MM = 3.0
PHOTON_ANGLES = np.arange(0.0, 360.0, 40.0)          # 9 equi-angled beams, all sites
# Proton angles for a target on the patient's right; mirrored (360 - a) for a target on the left.
# Report: chosen per site, mirrored with the isocenter side to prefer lower-range beams.
PROTON_ANGLES = {"HN": (0.0, 270.0, 210.0), "TH": (270.0, 210.0, 180.0), "AB": (270.0, 210.0, 180.0)}

# OARs per site: name -> (TotalSegmentator 'total' classes, [(kind, limit in Gy total)]).
# kind 'max' = D0.03cc limit, 'mean' = Dmean limit (both entered as squared-overdosing objectives,
# as hard constraints are not available in pyRadPlan). Structures without limits are still in the
# structure set: the DVH metric scores the three OARs that receive the most dose.
OARS = {
    "HN": {
        "SpinalCord": (["spinal_cord"], [("max", 48.0)]),
        "Brain": (["brain"], [("max", 54.0)]),                      # brainstem limit, no brainstem class
        "Esophagus": (["esophagus"], [("mean", 30.0)]),
        "Thyroid": (["thyroid_gland"], [("mean", 40.0)]),
        "Trachea": (["trachea"], [("mean", 40.0)]),                 # larynx limit, no larynx class
        "Skull": (["skull"], []),
    },
    "TH": {
        "SpinalCord": (["spinal_cord"], [("max", 50.5)]),
        "Lungs": (["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                   "lung_middle_lobe_right", "lung_lower_lobe_right"], [("mean", 18.0)]),
        "Esophagus": (["esophagus"], [("max", 73.5), ("mean", 30.6)]),
        "Heart": (["heart"], []),
        "Trachea": (["trachea"], []),
        "Liver": (["liver"], []),
        "Stomach": (["stomach"], []),
    },
    "AB": {
        "SpinalCord": (["spinal_cord"], [("max", 50.5)]),
        "Liver": (["liver"], [("mean", 30.0)]),
        "KidneyL": (["kidney_left"], [("mean", 18.0)]),
        "KidneyR": (["kidney_right"], [("mean", 18.0)]),
        "Stomach": (["stomach"], [("max", 45.0)]),
        "Bladder": (["urinary_bladder"], [("max", 65.0)]),
        "SmallBowel": (["small_bowel", "duodenum"], []),
        "Colon": (["colon"], []),
        "Spleen": (["spleen"], []),
    },
}
PTV_RADII_MM = {"HN": (25.0, 25.0, 35.0), "TH": (25.0, 25.0, 25.0), "AB": (30.0, 30.0, 30.0)}  # x, y, z


def _labels(seg: np.ndarray, names: list[str]) -> np.ndarray:
    from totalsegmentator.map_to_binary import class_map
    ids = [i for i, n in class_map["total"].items() if n in names]
    return np.isin(seg, ids)


def _side(case_id: str) -> int:
    """Deterministic patient side (+1 left, -1 right) for sites without an anatomical preference."""
    return 1 if int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 2 else -1


def ptv_center(region: str, seg: np.ndarray, body: np.ndarray, spacing_zyx, case_id: str) -> np.ndarray:
    """Voxel (z, y, x) centre of the synthetic PTV at a typical treatment site for the region.

    HN: oropharynx, 30 mm anterior to the C2-C3 vertebral bodies, 10 mm off midline.
    TH: centre of one lung (side from the case id). AB: pancreas, else liver, else body centre.
    """
    sz, sy, sx = spacing_zyx
    com = lambda m: np.array(ndimage.center_of_mass(m))
    fallback = com(body)
    if region == "HN":
        v = _labels(seg, ["vertebrae_C2", "vertebrae_C3"])
        if v.sum() < 50:
            v = _labels(seg, ["vertebrae_C1", "vertebrae_C4", "vertebrae_C5"])
        if v.sum() < 50:
            return fallback
        c = com(v)
        c[1] -= 30.0 / sy                                   # anterior = -y (LPS, identity direction)
        c[2] += _side(case_id) * 10.0 / sx
        return c
    if region == "TH":
        left = _labels(seg, OARS["TH"]["Lungs"][0][:2])
        right = _labels(seg, OARS["TH"]["Lungs"][0][2:])
        side = left if _side(case_id) > 0 else right
        if side.sum() < 1000:
            side = left if left.sum() > right.sum() else right
        return com(side) if side.sum() >= 1000 else fallback
    for organs in (["pancreas"], ["liver"]):
        m = _labels(seg, organs)
        if m.sum() >= 500:
            return com(m)
    return fallback


def make_ptv(region: str, center_zyx, body: np.ndarray, spacing_zyx) -> np.ndarray:
    """Ellipsoid PTV inside the body (3 mm from the skin) and >= 10 mm from the scan's z edges."""
    rx, ry, rz = PTV_RADII_MM[region]
    sz, sy, sx = spacing_zyx
    nz = body.shape[0]
    cz = float(np.clip(center_zyx[0], (rz + 10.0) / sz, nz - 1 - (rz + 10.0) / sz)) if nz * sz > 2 * (rz + 10) \
        else (nz - 1) / 2.0
    z, y, x = np.ogrid[:body.shape[0], :body.shape[1], :body.shape[2]]
    ell = (((z - cz) * sz / rz) ** 2 + ((y - center_zyx[1]) * sy / ry) ** 2
           + ((x - center_zyx[2]) * sx / rx) ** 2) <= 1.0
    inner = ndimage.binary_erosion(body, iterations=3, structure=np.ones((1, 3, 3), bool))
    return ell & inner


def _mask_image(arr: np.ndarray, ref: sitk.Image) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(ref)
    return img


def build_structures(ct, ct_img: sitk.Image, region: str, seg: np.ndarray, body: np.ndarray,
                     ptv: np.ndarray, fx: int, min_oar_cc: float = 1.0):
    from pyRadPlan.cst import StructureSet, create_voi
    from pyRadPlan.optimization.objectives import MeanDose, SquaredDeviation, SquaredOverdosing

    rx = RX_PER_FRACTION
    vois = [create_voi(name="PTV", voi_type="TARGET", ct_image=ct, mask=_mask_image(ptv, ct_img),
                       objectives=[SquaredDeviation(priority=1000.0, d_ref=rx)])]
    for name, (classes, limits) in OARS[region].items():
        m = _labels(seg, classes) & body
        if m.sum() * np.prod(ct_img.GetSpacing()) < min_oar_cc * 1000:   # slivers vanish on the 3 mm dose grid
            continue
        objs = [SquaredOverdosing(priority=300.0 if kind == "max" else 50.0, d_max=lim / fx)
                for kind, lim in limits]
        vois.append(create_voi(name=name, voi_type="OAR", ct_image=ct, mask=_mask_image(m, ct_img),
                               objectives=objs))
    # normal tissue: keep low doses low and cap hot spots outside the PTV (matRad's usual BODY terms)
    vois.append(create_voi(name="BODY", voi_type="EXTERNAL", ct_image=ct, mask=_mask_image(body, ct_img),
                           objectives=[MeanDose(priority=1.0, d_ref=0.0),
                                       SquaredOverdosing(priority=100.0, d_max=1.05 * rx)]))
    return StructureSet(vois=vois, ct_image=ct)


def _plan(radiation_mode: str, region: str, fx: int, side: int):
    from pyRadPlan import IonPlan, PhotonPlan
    if radiation_mode == "photons":
        angles = PHOTON_ANGLES
        pln = PhotonPlan(machine="Generic")
    else:
        angles = np.array([(360.0 - a) % 360.0 if side > 0 else a for a in PROTON_ANGLES[region]])
        pln = IonPlan(radiation_mode="protons", machine="Generic")
    pln.num_of_fractions = fx
    pln.prescribed_dose = RX_PER_FRACTION * fx
    pln.prop_stf = {"gantry_angles": np.asarray(angles, float), "couch_angles": np.zeros(len(angles))}
    pln.prop_dose_calc = {"dose_grid": {"resolution": {"x": DOSE_GRID_MM, "y": DOSE_GRID_MM, "z": DOSE_GRID_MM}}}
    pln.prop_opt = {"solver": "scipy"}
    return pln


def _recalc_settings(pln, radiation_mode: str):
    """The dose-engine settings DoseMetrics.calculate_dose applies before recalculating on the sCT."""
    from pyRadPlan import validate_pln
    pln = validate_pln(pln)
    pln.prop_dose_calc["trace_on_dose_grid"] = True
    if radiation_mode == "photons":
        pln.prop_dose_calc["geometric_lateral_cutoff"] = 25
    return pln


def _matrad_pln(pln) -> dict:
    """pln.to_matrad() without empty sub-structs: savemat writes {} as [], which reads back as None and
    then fails pyRadPlan's validation (a MATLAB-saved pln simply lacks those fields)."""
    return {k: v for k, v in pln.to_matrad().items() if v is not None and not (isinstance(v, dict) and not v)}


def d95(dose: np.ndarray, mask: np.ndarray) -> float:
    return float(np.percentile(dose[mask], 5))


def plan_case(case_id: str, ct_img: sitk.Image, body_img: sitk.Image, seg_img: sitk.Image, out: Path,
              modes=("photons", "protons")) -> dict:
    """Write the plan files for one case into out/ and return a summary (PTV, D95 scaling, timings)."""
    from pyRadPlan import (calc_dose_forward, calc_dose_influence, fluence_optimization, generate_stf,
                           validate_ct)
    region = case_id[1:3]
    fx = FRACTIONS[region]
    out.mkdir(parents=True, exist_ok=True)
    ct_img = sitk.Cast(ct_img, sitk.sitkFloat32)
    ct = validate_ct({"cube_hu": ct_img})
    body = sitk.GetArrayFromImage(body_img) > 0
    seg = sitk.GetArrayFromImage(seg_img)
    spacing_zyx = ct_img.GetSpacing()[::-1]
    center = ptv_center(region, seg, body, spacing_zyx, case_id)
    ptv = make_ptv(region, center, body, spacing_zyx)
    side = 1 if center[2] > ndimage.center_of_mass(body)[2] else -1   # +x = patient left (LPS, identity direction)
    cst = build_structures(ct, ct_img, region, seg, body, ptv, fx)
    summary = {"plans_version": PLANS_VERSION, "case": case_id, "region": region, "fractions": fx, "ptv_cc": float(ptv.sum() * np.prod(spacing_zyx) / 1000),
               "ptv_center_zyx": [round(float(c), 1) for c in center], "side": "left" if side > 0 else "right",
               "oars": [v.name for v in cst.vois if v.voi_type == "OAR"]}
    sio.savemat(out / f"{case_id}.mat", {"ct": ct.to_matrad(), "cst": cst.to_matrad()}, do_compression=True)

    for mode in modes:
        t0 = time.time()
        pln = _plan(mode, region, fx, side)
        stf = generate_stf(ct, cst, pln)
        dij = calc_dose_influence(ct, cst, stf, pln)
        w = np.asarray(fluence_optimization(ct, cst, stf, dij, pln), dtype=np.float64)
        t_opt = time.time() - t0
        # reference dose: the recalculation engine on the CT itself, with the official settings
        res = calc_dose_forward(ct, cst, stf, _recalc_settings(pln, mode), weights=w)
        dose = sitk.GetArrayFromImage(res["physical_dose"]).astype(np.float64)
        scale = RX_PER_FRACTION / max(d95(dose, ptv), 1e-6)       # prescription at the 95% isodose
        w, dose = w * scale, dose * scale
        sio.savemat(out / f"{case_id}_{mode}_plan.mat",
                    {"pln": _matrad_pln(pln), "stf": stf.to_matrad(), "weights": w}, do_compression=True)
        with gzip.open(out / f"{case_id}_{mode}_dose.npy.gz", "wb") as f:
            np.save(f, dose.astype(np.float32))
        summary[mode] = {"bixels": int(w.size), "d95_scale": round(float(scale), 4),
                         "dmax_rel": round(float(dose.max() / RX_PER_FRACTION), 3),
                         "seconds_opt": round(t_opt, 1), "seconds_total": round(time.time() - t0, 1)}
    (out / "plan.json").write_text(json.dumps(summary, indent=1))
    return summary
