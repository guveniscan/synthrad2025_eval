# Vendored from https://github.com/SynthRAD2025/evaluation @ 99e9ef4 (dose_metrics.py, GPL-3.0).
# Changes are marked "synthrad_eval:"; everything else is the organisers' code, byte for byte.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import SimpleITK as sitk
import pymatreader
import pymedphys
from numpy.typing import NDArray
from .stitch_utils import stitch_image, stitch_image_noresample
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # synthrad_eval: headless
import gzip
import gc
import math

from pyRadPlan import load_patient, validate_stf, validate_pln, calc_dose_forward, validate_ct, CT, StructureSet, DVHCollection, DVH, plot_slice
from pyRadPlan.core import Grid
import contextlib  # synthrad_eval


# synthrad_eval: DVH.get_dy as fixed upstream in pyRadPlan 0.5.0 (pyRadPlan/analysis/_dvh.py), copied
# verbatim. BSD 3-Clause License, Copyright (c) 2024, DKFZ. In 0.2.8, the version the challenge used,
# get_dy interpolates a fraction (y / 100) against a cumulative volume held in percent, with np.interp
# on a decreasing curve, so every D2/D5/D98 is the top bin edge (2.4975 Gy). See the 0.5.0 changelog.
def _get_dy_pyradplan_050(self, y):
    y = np.asarray(y, dtype=np.float64)
    if np.any(y < 0) or np.any(y > 100):
        raise ValueError("Volume percentage must be between 0 and 100.")

    # Dy is the largest quantity value still covering at least y percent of the
    # volume, matching the "at least" reading of the cumulative DVH documented
    # on :attr:`cumulative`. cum_volume decreases with dose, so the search runs
    # over the reversed (increasing) array and the index is mapped back.
    #
    # This is deliberately not an interpolation across the cumulative curve: a
    # uniformly irradiated region is a plateau in the DVH, and interpolating
    # over it reports the dose at the wrong end of that plateau.
    reversed_volume = self.cum_volume[::-1]
    last = self.cum_volume.size - 1
    index = last - np.searchsorted(reversed_volume, y, side="left")
    index = np.clip(index, 0, last)
    values = self.bins[index]
    return float(values) if values.ndim == 0 else values


@contextlib.contextmanager
def fixed_get_dy():
    """synthrad_eval: run the organisers' DVH metric with the pyRadPlan 0.5.0 get_dy, then restore 0.2.8's."""
    original = DVH.get_dy
    DVH.get_dy = _get_dy_pyradplan_050
    try:
        yield
    finally:
        DVH.get_dy = original

class QualityIndicator():
    def __init__(self, dosecube, voi, prescribed_dose, num_points = 1000, max_value=2.5, D_values = [2, 5, 98], V_values=None):
        self.name = voi.name
        self.mask = voi.mask
        self.voi = voi
        self.prescribed_dose = int(prescribed_dose)
        self.dvh = DVH.compute(dosecube.astype(float), voi.mask, num_points = num_points, max_value = max_value, name=voi.name)

        V_values = V_values if V_values is not None else (np.floor(np.linspace(0, np.max(dosecube), 6)*10)//10).astype(int)

        self.dose_stats = self.compute_dose_stats(dosecube, self.mask)

        self.D_values = D_values
        self.V_values = V_values
        self.dosecube = dosecube

        self.Dvals = self.compute_d_vals(self.D_values)
        # self.Vvals = self.compute_v_vals(self.V_values)
        self.QI = self.compute_QI(self.dosecube, self.mask)

    def compute_QI(self, dosecube, mask):
        if self.dose_stats['type'].lower() != 'target':
            return None
        else:
            Vtarget95 = np.sum(dosecube[sitk.GetArrayViewFromImage(mask).astype(bool)] >= 0.95 * self.prescribed_dose)
            Vtreated95 = np.sum(dosecube >= 0.95 * self.prescribed_dose)

            return {f'CI': (Vtarget95**2)/(self.dose_stats['size']*Vtreated95)}

    def compute_dose_stats(self, dosecube, mask):
        return {
            'mean': np.mean(dosecube[sitk.GetArrayViewFromImage(mask).astype(bool)]),
            'size': self.voi.indices.size,
            'type': self.voi.voi_type
        }

    def compute_d_vals(self, D_values):
        retarr = {}
        for d in D_values:
            retarr[f"D{d}"] = self.dvh.get_dy(d)
        return retarr
    def compute_v_vals(self, V_values):
        retarr = {}
        for v in V_values:
            retarr[f"V{v}"] = self.dvh.get_vx(v)
        return retarr



class DoseMetrics():
    def __init__(self, debug=False):
        # Use fixed wide dynamic range
        self.debug = debug


    # Find the largest target with priority for the PTV
    def get_largest_target(self, qi_ref):
        all_targets = [x for x in qi_ref if x.dose_stats['type'].lower()=='target' and len(x.voi.objectives) > 0]
        if len(all_targets) == 0:
            assert False, "No targets in this radiation plan! Exiting..."

        ptv_targets = [x for x in all_targets if 'ptv' in x.name.lower()]
        if len(ptv_targets) == 0:
            ptv_targets = [x for x in all_targets if 'itv' in x.name.lower()]
        if len(ptv_targets) == 0:
            ptv_targets = [x for x in all_targets if 'ctv' in x.name.lower()]
        if len(ptv_targets) == 0:
            ptv_targets = [x for x in all_targets if 'gtv' in x.name.lower()]
        if len(ptv_targets) == 0:
            assert False, "No PTV, ITV, CTV or GTV found! Exiting..."

        return max(ptv_targets, key=lambda x: x.dose_stats['size'])

    def get_dvh_metric(self, dose_ref: NDArray, dose_recalc: NDArray, vois, reference_dose, num_points=1000, max_value=2.5, eps=1e-12):


        qi_ref = [QualityIndicator(dose_ref, voi, reference_dose) for voi in vois]
        qi_sct = [QualityIndicator(dose_recalc.astype(np.float64), voi, reference_dose) for voi in vois]


        # Pick the largest PTV
        ptv_ref = self.get_largest_target(qi_ref) #max([x for x in qi_ref if x.dose_stats['type'].lower()=='target' and len(x.voi.objectives) > 0], key=lambda x: x.dose_stats['size'])
        ptv_sct = next((x for x in qi_sct if x.name == ptv_ref.name), None)
        assert ptv_sct is not None

        D98_target = np.abs(ptv_ref.Dvals['D98'] - ptv_sct.Dvals['D98'] + eps) / (ptv_ref.Dvals['D98'] + eps)
        V95_target = np.abs(ptv_ref.QI['CI'] - ptv_sct.QI['CI'] + eps) / (ptv_ref.QI['CI'] + eps)

        target_term = D98_target + V95_target


        # Sort all OARs by received dose
        oars_by_dose = sorted((
            (oar, (oar.Dvals['D5'] + oar.dose_stats['mean'])/2) for oar in qi_ref 
            if oar.dose_stats['type'].lower() == 'oar' and all(term not in oar.name.lower() for term in ['itv', 'ptv', 'ctv', 'gtv'])
            ), key=lambda x: x[1], reverse=True)
        
        # use the three OARs that receive the most dose (and are closest to the target)
        OARs_used = [(oar, next(oar_sct for oar_sct in qi_sct if oar_sct.name == oar.name)) for (oar, dose) in oars_by_dose[:min(3, len(oars_by_dose))]]


        D2_oar, Dmean_oar = [], []
        for (oar_ref, oar_sct) in OARs_used:
            D2_oar.append(np.abs(oar_ref.Dvals['D2'] - oar_sct.Dvals['D2'] + eps) / (oar_ref.Dvals['D2'] + eps))
            Dmean_oar.append(np.abs(oar_ref.dose_stats['mean'] - oar_sct.dose_stats['mean'] + eps) / (oar_ref.dose_stats['mean'] + eps))

        OAR_term = np.mean(D2_oar) + np.mean(Dmean_oar)
        return target_term + OAR_term
    def get_gamma(
        self,
        dose_ref: NDArray,
        dose_recalc: NDArray,
        dose_grid: Grid,
    ) -> NDArray:
        """
        Calculate the gamma index for the reference and recalculated doses.

        Parameters
        ----------
        physical_dose_reference : NDArray
            The reference physical dose.
        physical_dose_recalc : NDArray
            The recalculated physical dose.
        dose_grid : Grid
            The dose grid.
        dose_threshold : float, optional
            The dose threshold for gamma calculation, by default 0.01.
        distance_threshold : float, optional
            The distance threshold for gamma calculation in mm, by default 1.0.

        Returns
        -------
        NDArray
            The gamma index array.
        """
        
        grid_vectors = (dose_grid.z, dose_grid.y, dose_grid.x)
        gamma = pymedphys.gamma(
            grid_vectors,
            dose_ref,
            grid_vectors,
            dose_recalc,
            distance_mm_threshold=2, 
            dose_percent_threshold=2,
            lower_percent_dose_cutoff=20, 
            interp_fraction=1, 
            max_gamma=2, 
            local_gamma=False,
        )
        gamma = gamma[~np.isnan(gamma)]
        gamma_pass_rate = np.sum(gamma < 1) / gamma.size * 100
        return gamma_pass_rate

    def get_mae(
        self,
        dose_ref: NDArray,
        dose_recalc: NDArray,
        prescribed_dose: float = 2.0
    ) -> float:
        """
        Calculate the Mean Absolute Error (MAE) between the reference and recalculated doses.
        Uses only the values above 10% of the prescribed dose within both arrays.

        Parameters
        ----------
        physical_dose_reference : NDArray
            The reference physical dose.
        physical_dose_recalc : NDArray
            The recalculated physical dose.
        prescribed_dose : float, optional
            The prescribed dose for normalization, by default 2.0.

        Returns
        -------
        float
            The Mean Absolute Error (MAE).
        """

        mask = (dose_ref > 0.9 * prescribed_dose) & (dose_recalc > 0.9 * prescribed_dose)
            
        return np.mean(np.abs(dose_ref[mask] - dose_recalc[mask])) / prescribed_dose

    def compute_metrics(self, dose_ref, dose_sct, dose_grid, vois, particle, reference_dose):

        gamma_pass_rate = self.get_gamma(dose_ref, dose_sct, dose_grid)
        gc.collect()
        dose_mae = self.get_mae(dose_ref, dose_sct, prescribed_dose = reference_dose)
        gc.collect()

        dvh_metric = self.get_dvh_metric(dose_ref, dose_sct, vois, reference_dose)
        gc.collect()
        with fixed_get_dy():  # synthrad_eval: same formula and doses, upstream-fixed D-value lookup
            dvh_metric_fixed = self.get_dvh_metric(dose_ref, dose_sct, vois, reference_dose)

        return {
            f'gamma_pass_rate_{particle}': gamma_pass_rate,
            f'mae_{particle}': dose_mae,
            f'dvh_{particle}': dvh_metric,
            f'dvh_fixed_{particle}': dvh_metric_fixed,  # synthrad_eval
        }

    def calculate_dose(self, ct, sct, cst, plan_path, refdose_path, particle):
        plan_structures = pymatreader.read_mat(plan_path)

        plan_structures["pln"]["multScen"] = "nomScen"
        pln = validate_pln(plan_structures["pln"])
        stf = validate_stf(plan_structures["stf"])
        plan_result = plan_structures.get("resultGUI", None)

        if plan_result is None and refdose_path is None:
            raise ValueError("No reference dose provided. Please provide a reference dose file or a plan result.")
        # If we have a npy.gz file, we will load it, otherwise we will use the dose from matRad's plan result
        if refdose_path is not None:
            with gzip.open(refdose_path, 'rb') as f_in:
                physical_dose_reference_np = np.load(f_in)
        else:
            physical_dose_reference_np = np.transpose(plan_result["physicalDose"],axes=(2,0,1))


        reference_dose = plan_structures["pln"]["prescribedDose"] / plan_structures["pln"]["numOfFractions"]
        weights = plan_structures.get("weights", None)
        if weights is None:
            weights = plan_result["w"]    



        # dose calcualtion settings
        pln.prop_dose_calc["trace_on_dose_grid"] = True
        if pln.radiation_mode == "photons":
            pln.prop_dose_calc["geometric_lateral_cutoff"] = 25

        result = calc_dose_forward(sct,cst,stf,pln, weights=weights)
        gc.collect()
        physical_dose_recalc_np = sitk.GetArrayFromImage(result["physical_dose"])
        # physical_dose_reference_np = np.transpose(result_matrad["physicalDose"],axes=(2,0,1))

        dose_grid_reference = Grid.from_sitk_image(result["physical_dose"])
        dose_grid_recalc = dose_grid_reference

        gc.collect()

        return self.compute_metrics(physical_dose_reference_np, physical_dose_recalc_np, dose_grid_reference, cst.vois, particle, reference_dose)


    def score_patient(self, sct_path, mask_path, dose_base_folder, patient_id, set_orientation=None):     


        if self.debug:
            print(f"Computing dose for patient {patient_id}")
        ct, cst = load_patient(f"{dose_base_folder}/{patient_id}.mat")
        if self.debug:
            print(f"Loaded CT and CST for {patient_id}")

        mask = sitk.Cast(sitk.ReadImage(mask_path), sitk.sitkUInt8)
        sct = sitk.Cast(sitk.ReadImage(sct_path), ct.cube_hu.GetPixelID())

        # sitk.WriteImage(sct, '/Users/maartenterpstra/SynthRAD2025/preres_sct.nii.gz')
        # sitk.WriteImage(mask, '/Users/maartenterpstra/SynthRAD2025/preres_mask.nii.gz')
        mask_postres = sitk.Resample(mask,
                                    ct.cube_hu, # Use imageB directly as the reference
                                    sitk.Transform(), # Identity transform
                                    sitk.sitkNearestNeighbor, # Or sitk.sitkNearestNeighbor if it's a label map
                                    0, # Default pixel value (adjust if 0 is a valid data value)
                                    mask.GetPixelID()) # Maintain original pixel type

        sct_resample = sitk.Resample(sct,
                                    ct.cube_hu, # Use imageB directly as the reference
                                    sitk.Transform(), # Identity transform
                                    sitk.sitkLinear, # Or sitk.sitkNearestNeighbor if it's a label map
                                    -1024.0, # Default pixel value (adjust if 0 is a valid data value)
                                    sct.GetPixelID()) # Maintain original pixel type

        stitched_sct = stitch_image_noresample(sct_resample, ct.cube_hu, mask_postres)


        stitched_sct = validate_ct({'cube_hu': stitched_sct}, grid=ct.grid) # Need to check this
        cst.ct_image = stitched_sct
        for v in cst.vois:
            v.ct_image = stitched_sct

        dose_metrics = []   
        particles = ['photon', 'proton']
        for particle in particles:
            gc.collect()
            plan_path = f"{dose_base_folder}/{patient_id}_{particle}s_plan.mat"
            refdose_path = f"{dose_base_folder}/{patient_id}_{particle}s_dose.npy.gz"
            dose_metrics.append(self.calculate_dose(ct, stitched_sct, cst, plan_path, refdose_path, particle))
            gc.collect()
        all_dose_metrics = {k: v for d in dose_metrics for k, v in d.items()}
        all_dose_metrics['case'] = patient_id
        # print(all_dose_metrics)
        return all_dose_metrics     

if __name__ == '__main__':
    dm = DoseMetrics(debug=True)
    patient_id = '1THA208'
    sct_path = f'/Users/maartenterpstra/SynthRAD2025/input_data/{patient_id}.mha'
    mask_path = f'/Users/maartenterpstra/SynthRAD2025/test_data/mask/{patient_id}.mha'
    dose_base_folder = f'/Users/maartenterpstra/SynthRAD2025/test_data/dose/{patient_id}'
    print(dm.score_patient(sct_path, mask_path, dose_base_folder, patient_id, set_orientation=None))

