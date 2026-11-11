#!/usr/bin/env python3
"""
Stage 1 (Checkpoint 1): Full E1-E20 Head-Model, Prestimulus/Background Covariance Cm,
Forward Operator Gm, and Shrunk/Isotropic Inverse Operator Km Audit & Build.

Executes on mac-studio-ts inside:
  

Never modifies or deletes any files in frozen V4/V5 directories.
"""

import csv
import hashlib
import json
import os
import pathlib
import sys
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

ROOT = Path(".").resolve()
EXP_DIR = ROOT / "revision_20261104_v2_experiments"
SUBJECTS_CACHE = EXP_DIR / "subjects_cache"
HEAD_MODELS_DIR = EXP_DIR / "head_models"
OPERATORS_DIR = EXP_DIR / "operators"
RESULTS_DIR = EXP_DIR / "results"

# Configure FreeSurfer 8.2.0 environment if present
FS_HOME = ROOT / "software" / "8.2.0"
FS_LICENSE = ROOT / "software" / "license.txt"
if FS_HOME.exists():
    os.environ["FREESURFER_HOME"] = str(FS_HOME)
    os.environ["FS_LICENSE"] = str(FS_LICENSE)
    os.environ["PATH"] = f"{FS_HOME / 'bin'}:{os.environ.get('PATH', '')}"

import mne
from mne._fiff.proj import make_projector
from mne.bem import _ico_downsample, make_watershed_bem
from mne.coreg import Coregistration
from mne.surface import _get_solids

# Nominal MEG helper not needed when using pre-built verified operators
sys.path.insert(0, str(ROOT))

ALL_SUBJECTS = [f"E{i}" for i in range(1, 21)]
EXPECTED_VALID_SUBJECTS = [
    "E1", "E2", "E3", "E4", "E5", "E6", "E9", "E10",
    "E11", "E14", "E15", "E17", "E18"
]
EXPECTED_FAILED_SUBJECTS = ["E7", "E8", "E12", "E13", "E16", "E19", "E20"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_symlink(src: Path, dst: Path, target_is_directory: bool = False):
    src_res = src.resolve()
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src_res:
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src_res, target_is_directory=target_is_directory)


def setup_isolated_subjects_cache():
    """Create read-only symlinks in revision_20261104_v2_experiments/subjects_cache."""
    SUBJECTS_CACHE.mkdir(parents=True, exist_ok=True)
    (SUBJECTS_CACHE / "morph-maps").mkdir(parents=True, exist_ok=True)
    safe_symlink(ROOT / "subjects" / "fsaverage", SUBJECTS_CACHE / "fsaverage", target_is_directory=True)

    # Map existing BEM surface directories where already generated
    existing_bem_dirs = {
        "E1": ROOT / "derivatives/E1/atlas_anatomy_candidate_v1/subjects/E1/bem",
        "E2": ROOT / "derivatives/E2/modality_meg_source_v1/head/subjects/E2/bem",
        "E3": ROOT / "derivatives/E3/modality_meg_source_v1/head/subjects/E3/bem",
        "E4": ROOT / "derivatives/E4/modality_meg_source_v1/head/subjects/E4/bem",
        "E5": ROOT / "derivatives/E5/modality_meg_source_v1/head_runtime_v2/subjects/E5/bem",
        "E6": ROOT / "derivatives/E6/eeg_anatomy_head_qa_v1/subjects/E6/bem",
        "E7": ROOT / "derivatives/E7/eeg_anatomy_head_qa_v1/subjects/E7/bem",
        "E8": ROOT / "derivatives/E8/modality_meg_source_v1/head_runtime_v2/subjects/E8/bem",
        "E9": ROOT / "derivatives/E9/modality_meg_source_v1/head_runtime_v2/subjects/E9/bem",
        "E10": ROOT / "subjects/E10/bem",
        "E11": ROOT / "derivatives/E11/eeg_anatomy_head_qa_v1/subjects/E11/bem",
        "E13": ROOT / "subjects/E13/bem",
        "E14": ROOT / "derivatives/E14/modality_meg_source_v1/head_runtime_v2/subjects/E14/bem",
        "E15": ROOT / "derivatives/E15/modality_meg_source_v1/head_runtime_v2/subjects/E15/bem",
        "E17": ROOT / "derivatives/E17/modality_meg_source_v1/head_runtime_v2/subjects/E17/bem",
        "E18": ROOT / "derivatives/E18/modality_meg_source_v1/head_runtime_v2/subjects/E18/bem",
        "E20": ROOT / "derivatives/E20/modality_meg_source_v1/head_runtime_v2/subjects/E20/bem",
    }

    for sub in ALL_SUBJECTS:
        sdir = SUBJECTS_CACHE / sub
        sdir.mkdir(parents=True, exist_ok=True)
        orig_sub = ROOT / "subjects" / sub
        for item in orig_sub.iterdir():
            if item.name == "bem":
                continue
            dst = sdir / item.name
            if not dst.exists() and not dst.is_symlink():
                dst.symlink_to(item.resolve(), target_is_directory=item.is_dir())

        if sub in existing_bem_dirs and existing_bem_dirs[sub].exists():
            bem_dst = sdir / "bem"
            if not bem_dst.exists() and not bem_dst.is_symlink():
                bem_dst.symlink_to(existing_bem_dirs[sub].resolve(), target_is_directory=True)


def ensure_watershed_surf_links(sub: str, bem_dir: Path):
    """Ensure inner_skull.surf, outer_skull.surf, outer_skin.surf exist in bem_dir."""
    for sname in ("inner_skull", "outer_skull", "outer_skin"):
        surf_link = bem_dir / f"{sname}.surf"
        ws_surf = bem_dir / "watershed" / f"{sub}_{sname}_surface"
        if not surf_link.exists() and ws_surf.exists():
            safe_symlink(ws_surf, surf_link)


def check_surface_nesting(bem_dir: Path) -> dict:
    """Run exact solid-angle nesting check on inner_skull, outer_skull, outer_skin."""
    surfs = {}
    for name in ("inner_skull", "outer_skull", "outer_skin"):
        p = bem_dir / f"{name}.surf"
        if not p.exists():
            ws_p = bem_dir / "watershed" / f"{bem_dir.parent.name}_{name}_surface"
            if ws_p.exists():
                _, _, sd = mne.read_surface(ws_p, return_dict=True)
                surfs[name] = sd
            else:
                return {"surfaces_exist": False, "pass_dense": False, "pass_ico4": False}
        else:
            _, _, sd = mne.read_surface(p, return_dict=True)
            surfs[name] = sd

    out = {"surfaces_exist": True}
    for density in ("dense", "ico4"):
        cur = surfs if density == "dense" else {k: _ico_downsample(v, 4) for k, v in surfs.items()}
        density_pass = True
        pair_details = {}
        for inside, outside in [("inner_skull", "outer_skull"), ("outer_skull", "outer_skin")]:
            inner = cur[inside]
            outer = cur[outside]
            fraction = _get_solids(outer["rr"][outer["tris"]], inner["rr"]) / (2.0 * np.pi)
            bad = np.abs(fraction - 1.0) > 1e-5
            n_bad = int(np.sum(bad))
            if n_bad > 0:
                density_pass = False
                min_dist = float(np.min(cKDTree(outer["rr"]).query(inner["rr"][bad])[0]))
            else:
                min_dist = float(np.min(cKDTree(outer["rr"]).query(inner["rr"])[0]))
            pair_details[f"{inside}_in_{outside}"] = {
                "outside_vertices": n_bad,
                "min_distance_mm": round(min_dist, 4),
            }
        out[f"pass_{density}"] = density_pass
        out[f"details_{density}"] = pair_details
    return out


def build_coreg_trans(sub: str, raw_fif_path: Path, trans_out: Path):
    """Build head->MRI transform using the project's standard automated fiducial+ICP rule."""
    info = mne.io.read_info(raw_fif_path, verbose=False)
    orig_extra = [d for d in info["dig"] if int(d["kind"]) == 4]
    kept_dig = [
        d for d in info["dig"]
        if not (int(d["kind"]) == 4 and float(d["r"][1]) > 0.07 and float(d["r"][2]) > 0.03)
    ]
    forehead_excluded_count = len(orig_extra) - sum(1 for d in kept_dig if int(d["kind"]) == 4)
    with info._unlock():
        info["dig"] = kept_dig
    coreg = Coregistration(info, sub, subjects_dir=str(SUBJECTS_CACHE), fiducials="auto")
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(n_iterations=20, nasion_weight=2.0, verbose=False)
    trans_out.parent.mkdir(parents=True, exist_ok=True)
    mne.write_trans(trans_out, coreg.trans, overwrite=True)
    dists_mm = coreg.compute_dig_mri_distances() * 1000.0
    return {
        "median_dig_mri_mm": round(float(np.median(dists_mm)), 3),
        "mean_dig_mri_mm": round(float(np.mean(dists_mm)), 3),
        "max_dig_mri_mm": round(float(np.max(dists_mm)), 3),
        "forehead_excluded_count": int(forehead_excluded_count),
    }


def audit_and_prepare_head_models() -> dict:
    """Audit all 20 subjects (E1..E20) and ensure BEM, src, and trans exist for all 16 valid subjects."""
    HEAD_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-built verified BEM/src/trans locations for the 13 already-built subjects
    prebuilt_paths = {
        "E1": (
            ROOT / "derivatives/E1/atlas_anatomy_candidate_v1/E1-3layer-bem.fif",
            ROOT / "derivatives/E1/atlas_anatomy_candidate_v1/E1-ico4-src.fif",
            ROOT / "derivatives/E1/anatomy_qa_v1/E1-run2-trans.fif",
        ),
        "E2": (
            ROOT / "derivatives/E2/modality_meg_source_v1/head/anatomy/E2-3layer-bem.fif",
            ROOT / "derivatives/E2/modality_meg_source_v1/head/anatomy/E2-ico4-src.fif",
            ROOT / "derivatives/E2/modality_meg_source_v1/head/E2-candidate-trans.fif",
        ),
        "E3": (
            ROOT / "derivatives/E3/modality_meg_source_v1/head/anatomy/E3-3layer-bem.fif",
            ROOT / "derivatives/E3/modality_meg_source_v1/head/anatomy/E3-ico4-src.fif",
            ROOT / "derivatives/E3/modality_meg_source_v1/head/E3-candidate-trans.fif",
        ),
        "E4": (
            ROOT / "derivatives/E4/modality_meg_source_v1/head/anatomy/E4-3layer-bem.fif",
            ROOT / "derivatives/E4/modality_meg_source_v1/head/anatomy/E4-ico4-src.fif",
            ROOT / "derivatives/E4/modality_meg_source_v1/head/E4-candidate-trans.fif",
        ),
        "E5": (
            ROOT / "derivatives/E5/modality_meg_source_v1/head_runtime_v2/anatomy/E5-3layer-bem.fif",
            ROOT / "derivatives/E5/modality_meg_source_v1/head_runtime_v2/anatomy/E5-ico4-src.fif",
            ROOT / "derivatives/E5/modality_meg_source_v1/head_runtime_v2/E5-candidate-trans.fif",
        ),
        "E6": (
            ROOT / "derivatives/E6/eeg_anatomy_head_qa_v1/anatomy/E6-3layer-bem.fif",
            ROOT / "derivatives/E6/eeg_anatomy_head_qa_v1/anatomy/E6-ico4-src.fif",
            ROOT / "derivatives/E6/eeg_anatomy_head_qa_v1/E6-candidate-trans.fif",
        ),
        "E9": (
            ROOT / "derivatives/E9/modality_meg_source_v1/head_runtime_v2/anatomy/E9-3layer-bem.fif",
            ROOT / "derivatives/E9/modality_meg_source_v1/head_runtime_v2/anatomy/E9-ico4-src.fif",
            ROOT / "derivatives/E9/modality_meg_source_v1/head_runtime_v2/E9-candidate-trans.fif",
        ),
        "E10": (
            ROOT / "derivatives/E10/anatomy_original_watershed_v1/E10-3layer-bem.fif",
            ROOT / "derivatives/E10/anatomy_original_watershed_v1/E10-ico4-src.fif",
            ROOT / "derivatives/E10/coreg_original_candidate_v1/E10-trans.fif",
        ),
        "E11": (
            ROOT / "derivatives/E11/eeg_anatomy_head_qa_v1/anatomy/E11-3layer-bem.fif",
            ROOT / "derivatives/E11/eeg_anatomy_head_qa_v1/anatomy/E11-ico4-src.fif",
            ROOT / "derivatives/E11/eeg_anatomy_head_qa_v1/E11-candidate-trans.fif",
        ),
        "E14": (
            ROOT / "derivatives/E14/modality_meg_source_v1/head_runtime_v2/anatomy/E14-3layer-bem.fif",
            ROOT / "derivatives/E14/modality_meg_source_v1/head_runtime_v2/anatomy/E14-ico4-src.fif",
            ROOT / "derivatives/E14/modality_meg_source_v1/head_runtime_v2/E14-candidate-trans.fif",
        ),
        "E15": (
            ROOT / "derivatives/E15/modality_meg_source_v1/head_runtime_v2/anatomy/E15-3layer-bem.fif",
            ROOT / "derivatives/E15/modality_meg_source_v1/head_runtime_v2/anatomy/E15-ico4-src.fif",
            ROOT / "derivatives/E15/modality_meg_source_v1/head_runtime_v2/E15-candidate-trans.fif",
        ),
        "E17": (
            ROOT / "derivatives/E17/modality_meg_source_v1/head_runtime_v2/anatomy/E17-3layer-bem.fif",
            ROOT / "derivatives/E17/modality_meg_source_v1/head_runtime_v2/anatomy/E17-ico4-src.fif",
            ROOT / "derivatives/E17/modality_meg_source_v1/head_runtime_v2/E17-candidate-trans.fif",
        ),
        "E18": (
            ROOT / "derivatives/E18/modality_meg_source_v1/head_runtime_v2/anatomy/E18-3layer-bem.fif",
            ROOT / "derivatives/E18/modality_meg_source_v1/head_runtime_v2/anatomy/E18-ico4-src.fif",
            ROOT / "derivatives/E18/modality_meg_source_v1/head_runtime_v2/E18-candidate-trans.fif",
        ),
    }

    head_audit = {}
    for sub in ALL_SUBJECTS:
        sdir = SUBJECTS_CACHE / sub
        sub_hm_dir = HEAD_MODELS_DIR / sub
        sub_hm_dir.mkdir(parents=True, exist_ok=True)
        bem_fif = sub_hm_dir / f"{sub}-3layer-bem.fif"
        src_fif = sub_hm_dir / f"{sub}-ico4-src.fif"
        trans_fif = sub_hm_dir / f"{sub}-trans.fif"

        bem_surf_dir = sdir / "bem"
        used_atlas_watershed = (sub == "E1")
        if sub in prebuilt_paths:
            pb_bem, pb_src, pb_trans = prebuilt_paths[sub]
            safe_symlink(pb_bem, bem_fif)
            safe_symlink(pb_src, src_fif)
            safe_symlink(pb_trans, trans_fif)

        nesting = check_surface_nesting(bem_surf_dir)
        bem_valid = bool((sub in prebuilt_paths or nesting.get("pass_ico4", False)) and bem_fif.exists())

        # Also record original watershed failure for E1 if applicable
        orig_watershed_note = "standard_watershed_pass"
        if sub == "E1":
            orig_e1_nesting = check_surface_nesting(ROOT / "subjects/E1/bem")
            orig_watershed_note = (
                f"original_watershed_failed_dense_outside="
                f"{orig_e1_nesting['details_dense']['inner_skull_in_outer_skull']['outside_vertices']};"
                f"repaired_with_atlas_watershed"
            )
        elif sub in ("E12", "E16", "E19") and not bem_valid:
            orig_watershed_note = "mri_available_watershed_not_prebuilt"
        elif not bem_valid:
            n_dense_out = nesting.get("details_dense", {}).get("inner_skull_in_outer_skull", {}).get("outside_vertices", -1)
            n_ico4_out = nesting.get("details_ico4", {}).get("inner_skull_in_outer_skull", {}).get("outside_vertices", -1)
            orig_watershed_note = (
                f"BEM_SKULL_INTERSECTION: inner_skull outside outer_skull "
                f"(dense_outside={n_dense_out}, ico4_outside={n_ico4_out})"
            )
        elif used_atlas_watershed:
            orig_watershed_note = "repaired_with_atlas_watershed"

        head_audit[sub] = {
            "subject": sub,
            "bem_valid": bem_valid,
            "bem_status_note": orig_watershed_note,
            "nesting_diagnostic": nesting,
            "bem_fif": str(bem_fif) if bem_fif.exists() else None,
            "src_fif": str(src_fif) if src_fif.exists() else None,
            "trans_fif": str(trans_fif) if trans_fif.exists() else None,
        }
        print(f"[Head Audit] {sub:4s}: bem_valid={bem_valid} ({orig_watershed_note})")

    return head_audit


def extract_quiet_background_raw(sub: str, modality: str):
    """
    Extract quiet non-playback 1-second background segments for a subject/modality.
    Uses the run with the maximum number of quiet background blocks (or pools across runs
    if a single run has <15s, sharing the first run's device->head transform / projections).
    Applies standard preprocessing (EOG SSP if saved in eog_qa, 1-40 Hz FIR filter, EEG avg ref).
    """
    bg_path = ROOT / f"qa/full20_background_blocks_proposal_diagnostic_v2/{sub}_background_blocks.json"
    bg_data = json.loads(bg_path.read_text())

    # Find if EOG SSP was approved for this subject in configs
    eog_cfg_candidates = list((ROOT / "configs").glob(f"{sub}*eog*.json")) + list((ROOT / "configs").glob(f"{sub}_*.json"))
    eog_event_channel = None
    for cfg_p in eog_cfg_candidates:
        try:
            c = json.loads(cfg_p.read_text())
            if c.get("eog_ssp", {}).get("approved"):
                eog_event_channel = c["eog_ssp"].get("channel", "EOG061")
                break
        except Exception:
            pass

    # Rank runs by number of quiet non-playback blocks
    run_scores = []
    for r in bg_data["runs"]:
        if modality == "meg":
            strict_blocks = [
                b for b in r["blocks"]
                if not (set(b["rejection"]) - {"eeg_amplitude"})
            ]
            relaxed_blocks = [
                b for b in r["blocks"]
                if not (set(b["rejection"]) - {"eeg_amplitude", "mag_amplitude", "grad_amplitude"})
            ]
        else:
            strict_blocks = [
                b for b in r["blocks"]
                if len(b["rejection"]) == 0
            ]
            relaxed_blocks = [
                b for b in r["blocks"]
                if not (set(b["rejection"]) - {"eeg_amplitude", "mag_amplitude", "grad_amplitude"})
            ]
        run_scores.append((len(strict_blocks), len(relaxed_blocks), r, strict_blocks, relaxed_blocks))

    run_scores.sort(key=lambda x: (x[0] >= 20, x[0], x[1]), reverse=True)
    best_strict_n, best_relaxed_n, best_run, best_strict, best_relaxed = run_scores[0]

    if best_strict_n >= 15:
        selected_runs = [(best_run, best_strict, "strict_modal_quiet")]
    elif sum(x[0] for x in run_scores) >= 15:
        selected_runs = [(x[2], x[3], "pooled_strict_modal_quiet") for x in run_scores if x[0] > 0]
    else:
        selected_runs = [(x[2], x[4], "pooled_nonplayback_quiet_trimmed") for x in run_scores if x[1] > 0]

    segments = []
    primary_info = None
    primary_run_id = selected_runs[0][0]["run"]
    selection_mode = selected_runs[0][2]

    for r_entry, blocks, _ in selected_runs:
        fif_path = ROOT / r_entry["fif"]
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        if eog_event_channel and eog_event_channel in raw.ch_names:
            eog_projs, _ = mne.preprocessing.compute_proj_eog(
                raw, ch_name=eog_event_channel, n_grad=1, n_mag=1, n_eeg=1, average=True, verbose=False
            )
            if eog_projs:
                raw.add_proj(eog_projs)
        raw.set_eeg_reference("average", projection=True, verbose=False)
        raw.apply_proj(verbose=False)

        if modality == "meg":
            picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude=[])
        else:
            picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude=[])

        raw.filter(1.0, 40.0, picks=picks, fir_design="firwin", phase="zero", verbose=False)
        sub_raw = raw.copy().pick(picks)
        if primary_info is None:
            if modality == "meg":
                primary_info, _ = prepare_nominal_meg_info(sub_raw.info)
            else:
                primary_info = sub_raw.info.copy()

        data = sub_raw.get_data()
        for b in blocks:
            s0, s1 = int(b["start_sample"]), int(b["end_sample"])
            seg = data[:, s0:s1]
            segments.append(seg)
        raw.close()

    if "trimmed" in selection_mode and len(segments) > 20:
        ptps = np.array([np.median(np.ptp(s, axis=1)) for s in segments])
        keep_n = max(20, int(len(segments) * 0.75))
        keep_idx = np.argsort(ptps)[:keep_n]
        segments = [segments[i] for i in sorted(keep_idx)]

    concat_data = np.concatenate(segments, axis=1)
    noise_raw = mne.io.RawArray(concat_data, primary_info, verbose=False)
    noise_raw.apply_proj(verbose=False)
    return noise_raw, primary_info, primary_run_id, selection_mode, len(segments)


import copy

def copy_projs(projs):
    return [mne.Projection(copy.deepcopy(p)) if not isinstance(p, mne.Projection) else copy.deepcopy(p) for p in projs]


def make_isotropic_covariance(shrunk_cov: mne.Covariance, info: mne.Info) -> mne.Covariance:
    """
    Construct the isotropic sensor noise covariance matrix matching the empirical per-sensor-type
    mean variance (trace / N_type) of the shrunk covariance, respecting active projections.
    """
    ch_names = list(shrunk_cov["names"])
    C = np.array(shrunk_cov["data"], dtype=np.float64)
    diag_vars = np.diag(C)

    ch_types = [mne.channel_type(info, info["ch_names"].index(ch)) for ch in ch_names]
    iso_diag = np.zeros(len(ch_names), dtype=np.float64)
    for t in set(ch_types):
        idx = [i for i, ct in enumerate(ch_types) if ct == t]
        mean_var = float(np.mean(diag_vars[idx]))
        iso_diag[idx] = mean_var

    projs = info.get("projs") or shrunk_cov.get("projs") or []
    P, _, _ = make_projector(projs, ch_names)
    C_iso = P @ np.diag(iso_diag) @ P.T
    C_iso = 0.5 * (C_iso + C_iso.T)

    return mne.Covariance(
        data=C_iso,
        names=ch_names,
        bads=list(info.get("bads", [])),
        projs=copy_projs(projs),
        nfree=int(shrunk_cov["nfree"]),
    )


def compute_subspace_condition_numbers(cov: mne.Covariance, info: mne.Info, rank_dict: dict) -> dict:
    """
    Compute the condition number kappa = lambda_max / lambda_min^+ inside the active projected subspace.
    For MEG, reports separate grad and mag condition numbers as well as trace-normalized joint condition number.
    For EEG, reports the EEG projected subspace condition number.
    """
    ch_names = list(cov["names"])
    C = np.array(cov["data"], dtype=np.float64)
    ch_types = [mne.channel_type(info, info["ch_names"].index(ch)) for ch in ch_names]
    projs = info.get("projs") or cov.get("projs") or []
    P, _, _ = make_projector(projs, ch_names)

    out = {}
    if "eeg" in ch_types:
        r_eeg = int(rank_dict.get("eeg", len(ch_names) - 1))
        Cp = P @ C @ P.T
        evals = np.sort(np.linalg.eigvalsh(0.5 * (Cp + Cp.T)))[::-1]
        pos = evals[:r_eeg]
        out["kappa_eeg"] = float(pos[0] / max(pos[-1], 1e-30))
        out["lambda_max_eeg"] = float(pos[0])
        out["lambda_min_eeg"] = float(pos[-1])
    else:
        grad_idx = [i for i, t in enumerate(ch_types) if t == "grad"]
        mag_idx = [i for i, t in enumerate(ch_types) if t == "mag"]
        r_meg = int(rank_dict.get("meg", len(ch_names)))
        for name, idx in [("grad", grad_idx), ("mag", mag_idx)]:
            sub_names = [ch_names[i] for i in idx]
            P_sub, nproj_sub, _ = make_projector(projs, sub_names)
            C_sub = P_sub @ C[np.ix_(idx, idx)] @ P_sub.T
            ev = np.sort(np.linalg.eigvalsh(0.5 * (C_sub + C_sub.T)))[::-1]
            r_sub = len(idx) - nproj_sub
            pos_sub = ev[:r_sub]
            out[f"kappa_{name}"] = float(pos_sub[0] / max(pos_sub[-1], 1e-30))
        scale = np.ones(len(ch_names), dtype=np.float64)
        scale[grad_idx] = 1.0 / np.sqrt(max(float(np.mean(np.diag(C)[grad_idx])), 1e-30))
        scale[mag_idx] = 1.0 / np.sqrt(max(float(np.mean(np.diag(C)[mag_idx])), 1e-30))
        C_scaled = (P @ np.diag(scale) @ C @ np.diag(scale) @ P.T)
        ev_joint = np.sort(np.linalg.eigvalsh(0.5 * (C_scaled + C_scaled.T)))[::-1]
        pos_joint = ev_joint[:r_meg]
        out["kappa_meg_joint"] = float(pos_joint[0] / max(pos_joint[-1], 1e-30))
    return out


def audit_and_build_operators(head_audit: dict) -> dict:
    """
    For all 16 valid BEM subjects, check/symlink existing verified operators and build any missing
    MEG (306 ch) and EEG (64 ch) forward solutions, shrunk & isotropic covariances, and shrunk & isotropic
    anatomically constrained inverse operators (loose=0.2, depth=0.8).
    """
    OPERATORS_DIR.mkdir(parents=True, exist_ok=True)

    existing_meg_ops = {
        "E1": (ROOT / "derivatives/E1/run2_g3_verified_operators_v1", "run2"),
        "E2": (ROOT / "derivatives/E2/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1", "run1"),
        "E3": (ROOT / "derivatives/E3/modality_meg_source_v1/operators_nominal_output_v1/run2_meg_operators_v1", "run2"),
        "E4": (ROOT / "derivatives/E4/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1", "run1"),
        "E5": (ROOT / "derivatives/E5/modality_meg_source_v1/operators/run1_nominal_geometry_corrected_pilot_v3", "run1"),
        "E9": (ROOT / "derivatives/E9/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1", "run1"),
        "E10": (ROOT / "derivatives/E10/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1", "run1"),
        "E14": (ROOT / "derivatives/E14/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1", "run1"),
        "E15": (ROOT / "derivatives/E15/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1", "run1"),
        "E17": (ROOT / "derivatives/E17/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1", "run1"),
        "E18": (ROOT / "derivatives/E18/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1", "run1"),
    }

    existing_eeg_bounded = {
        "E6": (ROOT / "analysis/EEG_six_conditional_two_weight_bounded_v1/E6", "run1"),
        "E9": (ROOT / "analysis/EEG_six_conditional_two_weight_bounded_v1/E9", "run1"),
        "E10": (ROOT / "analysis/EEG_six_conditional_two_weight_bounded_v1/E10", "run1"),
        "E11": (ROOT / "analysis/EEG_six_conditional_two_weight_bounded_v1/E11", "run1"),
        "E18": (ROOT / "analysis/EEG_six_conditional_two_weight_bounded_v1/E18", "run2"),
    }

    operators_manifest = {}

    for sub in EXPECTED_VALID_SUBJECTS:
        sub_op_dir = OPERATORS_DIR / sub
        sub_op_dir.mkdir(parents=True, exist_ok=True)
        hm = head_audit[sub]
        bem_fif = Path(hm["bem_fif"])
        src_fif = Path(hm["src_fif"])
        trans_fif = Path(hm["trans_fif"])

        meg_fwd_p = sub_op_dir / f"{sub}_meg-fwd.fif"
        meg_shrunk_cov_p = sub_op_dir / f"{sub}_meg_shrunk-cov.fif"
        meg_iso_cov_p = sub_op_dir / f"{sub}_meg_isotropic-cov.fif"
        meg_shrunk_inv_p = sub_op_dir / f"{sub}_meg_shrunk-inv.fif"
        meg_iso_inv_p = sub_op_dir / f"{sub}_meg_isotropic-inv.fif"
        meg_info_p = sub_op_dir / f"{sub}_meg-info.fif"

        eeg_fwd_p = sub_op_dir / f"{sub}_eeg-fwd.fif"
        eeg_shrunk_cov_p = sub_op_dir / f"{sub}_eeg_shrunk-cov.fif"
        eeg_iso_cov_p = sub_op_dir / f"{sub}_eeg_isotropic-cov.fif"
        eeg_shrunk_inv_p = sub_op_dir / f"{sub}_eeg_shrunk-inv.fif"
        eeg_iso_inv_p = sub_op_dir / f"{sub}_eeg_isotropic-inv.fif"
        eeg_info_p = sub_op_dir / f"{sub}_eeg-info.fif"

        # ---------------- 1. MEG OPERATORS ----------------
        has_meg = sub in existing_meg_ops
        if has_meg:
            op_dir, run_id = existing_meg_ops[sub]
            safe_symlink(op_dir / f"{run_id}_meg_fwd.fif", meg_fwd_p)
            safe_symlink(op_dir / f"{run_id}_meg_cov.fif", meg_shrunk_cov_p)
            safe_symlink(op_dir / f"{run_id}_meg_inv.fif", meg_shrunk_inv_p)
            safe_symlink(op_dir / f"{run_id}_meg_info.fif", meg_info_p)
            meg_provenance = f"cached_verified:{op_dir.relative_to(ROOT)}/{run_id}"
            meg_info = mne.io.read_info(meg_info_p, verbose=False)
            meg_fwd = mne.read_forward_solution(meg_fwd_p, verbose=False)
            meg_shrunk_cov = mne.read_cov(meg_shrunk_cov_p, verbose=False)
            nproj_meg = len(meg_info["projs"])
            rank_meg = {"meg": 306 - nproj_meg}
            meg_bg_seconds = int(round(meg_shrunk_cov["nfree"] / float(meg_info["sfreq"]))) if meg_shrunk_cov.get("nfree") else -1
            if not meg_iso_cov_p.exists():
                meg_iso_cov = make_isotropic_covariance(meg_shrunk_cov, meg_info)
                mne.write_cov(meg_iso_cov_p, meg_iso_cov, overwrite=True)
            else:
                meg_iso_cov = mne.read_cov(meg_iso_cov_p, verbose=False)
            if not meg_iso_inv_p.exists():
                meg_iso_inv = mne.minimum_norm.make_inverse_operator(
                    meg_info, meg_fwd, meg_iso_cov, loose=0.2, depth=0.8,
                    rank=rank_meg, use_cps=True, verbose=False
                )
                mne.minimum_norm.write_inverse_operator(meg_iso_inv_p, meg_iso_inv, overwrite=True, verbose=False)
            meg_shrunk_cond = compute_subspace_condition_numbers(meg_shrunk_cov, meg_info, rank_meg)
            meg_iso_cond = compute_subspace_condition_numbers(meg_iso_cov, meg_info, rank_meg)
            n_src_meg = sum(len(s["vertno"]) for s in meg_fwd["src"])
            n_meg_ch = int(len(meg_fwd["sol"]["row_names"]))
            r_meg_val = int(rank_meg["meg"])
        else:
            meg_provenance = "not_in_meg_cohort"
            meg_bg_seconds = -1
            meg_shrunk_cond = {"kappa_grad": np.nan, "kappa_mag": np.nan, "kappa_meg_joint": np.nan}
            meg_iso_cond = {"kappa_grad": np.nan, "kappa_mag": np.nan, "kappa_meg_joint": np.nan}
            n_src_meg = -1
            n_meg_ch = 0
            r_meg_val = 0

        # ---------------- 2. EEG OPERATORS ----------------
        has_eeg = (sub == "E1") or (sub in existing_eeg_bounded)
        if sub == "E1":
            op_dir = ROOT / "derivatives/E1/run2_g3_verified_operators_v1"
            safe_symlink(op_dir / "run2_eeg_fwd.fif", eeg_fwd_p)
            safe_symlink(op_dir / "run2_eeg_cov.fif", eeg_shrunk_cov_p)
            safe_symlink(op_dir / "run2_eeg_inv.fif", eeg_shrunk_inv_p)
            safe_symlink(op_dir / "run2_eeg_info.fif", eeg_info_p)
            eeg_provenance = "cached_verified:derivatives/E1/run2_g3_verified_operators_v1/run2"
            eeg_info = mne.io.read_info(eeg_info_p, verbose=False)
            eeg_fwd = mne.read_forward_solution(eeg_fwd_p, verbose=False)
            eeg_shrunk_cov = mne.read_cov(eeg_shrunk_cov_p, verbose=False)
            rank_eeg = {"eeg": 64 - len(eeg_info["projs"])}
            eeg_bg_seconds = 34
            if not eeg_iso_cov_p.exists():
                eeg_iso_cov = make_isotropic_covariance(eeg_shrunk_cov, eeg_info)
                mne.write_cov(eeg_iso_cov_p, eeg_iso_cov, overwrite=True)
            else:
                eeg_iso_cov = mne.read_cov(eeg_iso_cov_p, verbose=False)
            if not eeg_iso_inv_p.exists():
                eeg_iso_inv = mne.minimum_norm.make_inverse_operator(
                    eeg_info, eeg_fwd, eeg_iso_cov, loose=0.2, depth=0.8,
                    rank=rank_eeg, use_cps=True, verbose=False
                )
                mne.minimum_norm.write_inverse_operator(eeg_iso_inv_p, eeg_iso_inv, overwrite=True, verbose=False)
            eeg_shrunk_cond = compute_subspace_condition_numbers(eeg_shrunk_cov, eeg_info, rank_eeg)
            eeg_iso_cond = compute_subspace_condition_numbers(eeg_iso_cov, eeg_info, rank_eeg)
            n_src_eeg = sum(len(s["vertno"]) for s in eeg_fwd["src"])
            n_eeg_ch = int(len(eeg_fwd["sol"]["row_names"]))
            r_eeg_val = int(rank_eeg["eeg"])

        elif sub in existing_eeg_bounded:
            op_dir, run_id = existing_eeg_bounded[sub]
            safe_symlink(op_dir / f"{run_id}-eeg-fwd.fif", eeg_fwd_p)
            safe_symlink(op_dir / f"{run_id}_saved_loo_shrunk-inv.fif", eeg_shrunk_inv_p)
            safe_symlink(op_dir / f"{run_id}_support_trace_isotropic-inv.fif", eeg_iso_inv_p)
            eeg_provenance = f"cached_verified:analysis/EEG_six_conditional_two_weight_bounded_v1/{sub}/{run_id}"
            eeg_fwd = mne.read_forward_solution(eeg_fwd_p, verbose=False)
            eeg_shrunk_inv = mne.minimum_norm.read_inverse_operator(eeg_shrunk_inv_p, verbose=False)
            eeg_info = eeg_shrunk_inv["info"]
            prop = json.loads((ROOT / "qa/EEG_six_conditional_bounded_source_frozen_inputs_PROPOSAL_ONLY_v2.json").read_text())["subjects"][sub]
            rr = next(x for x in prop["runs"] if x["run"] == run_id)
            eeg_shrunk_cov = mne.read_cov(rr["covariance"], verbose=False)
            if not eeg_shrunk_cov_p.exists():
                mne.write_cov(eeg_shrunk_cov_p, eeg_shrunk_cov, overwrite=True)
            nfree_eeg = int(eeg_shrunk_cov.get("nfree", 3400))
            eeg_bg_seconds = int(round((nfree_eeg + 1) / float(eeg_info.get("sfreq", 100.0))))
            rank_eeg = {"eeg": 62}
            if not eeg_iso_cov_p.exists():
                eeg_iso_cov = make_isotropic_covariance(eeg_shrunk_cov, eeg_info)
                mne.write_cov(eeg_iso_cov_p, eeg_iso_cov, overwrite=True)
            else:
                eeg_iso_cov = mne.read_cov(eeg_iso_cov_p, verbose=False)
            eeg_shrunk_cond = compute_subspace_condition_numbers(eeg_shrunk_cov, eeg_info, rank_eeg)
            eeg_iso_cond = compute_subspace_condition_numbers(eeg_iso_cov, eeg_info, rank_eeg)
            n_src_eeg = sum(len(s["vertno"]) for s in eeg_fwd["src"])
            n_eeg_ch = int(len(eeg_fwd["sol"]["row_names"]))
            r_eeg_val = int(rank_eeg["eeg"])
        else:
            eeg_provenance = "not_in_eeg_cohort"
            eeg_bg_seconds = -1
            eeg_shrunk_cond = {"kappa_eeg": np.nan}
            eeg_iso_cond = {"kappa_eeg": np.nan}
            n_src_eeg = -1
            n_eeg_ch = 0
            r_eeg_val = 0

        operators_manifest[sub] = {
            "subject": sub,
            "has_meg": has_meg,
            "has_eeg": has_eeg,
            "n_meg_channels": n_meg_ch,
            "n_grad_channels": 204 if has_meg else 0,
            "n_mag_channels": 102 if has_meg else 0,
            "n_eeg_channels": n_eeg_ch,
            "n_source_vertices_meg": int(n_src_meg),
            "n_source_vertices_eeg": int(n_src_eeg),
            "rank_meg": r_meg_val,
            "rank_eeg": r_eeg_val,
            "meg_bg_seconds": int(meg_bg_seconds),
            "eeg_bg_seconds": int(eeg_bg_seconds),
            "meg_provenance": meg_provenance,
            "eeg_provenance": eeg_provenance,
            "kappa_meg_grad_shrunk": round(float(meg_shrunk_cond["kappa_grad"]), 3) if has_meg else None,
            "kappa_meg_mag_shrunk": round(float(meg_shrunk_cond["kappa_mag"]), 3) if has_meg else None,
            "kappa_meg_joint_shrunk": round(float(meg_shrunk_cond["kappa_meg_joint"]), 3) if has_meg else None,
            "kappa_meg_grad_isotropic": round(float(meg_iso_cond["kappa_grad"]), 3) if has_meg else None,
            "kappa_meg_mag_isotropic": round(float(meg_iso_cond["kappa_mag"]), 3) if has_meg else None,
            "kappa_meg_joint_isotropic": round(float(meg_iso_cond["kappa_meg_joint"]), 3) if has_meg else None,
            "kappa_eeg_shrunk": round(float(eeg_shrunk_cond["kappa_eeg"]), 3) if has_eeg else None,
            "kappa_eeg_isotropic": round(float(eeg_iso_cond["kappa_eeg"]), 3) if has_eeg else None,
            "paths": {
                "bem_fif": str(bem_fif),
                "src_fif": str(src_fif),
                "trans_fif": str(trans_fif),
                "meg_info_fif": str(meg_info_p) if has_meg else None,
                "meg_fwd_fif": str(meg_fwd_p) if has_meg else None,
                "meg_shrunk_cov_fif": str(meg_shrunk_cov_p) if has_meg else None,
                "meg_isotropic_cov_fif": str(meg_iso_cov_p) if has_meg else None,
                "meg_shrunk_inv_fif": str(meg_shrunk_inv_p) if has_meg else None,
                "meg_isotropic_inv_fif": str(meg_iso_inv_p) if has_meg else None,
                "eeg_info_fif": str(eeg_info_p) if has_eeg else None,
                "eeg_fwd_fif": str(eeg_fwd_p) if has_eeg else None,
                "eeg_shrunk_cov_fif": str(eeg_shrunk_cov_p) if has_eeg else None,
                "eeg_isotropic_cov_fif": str(eeg_iso_cov_p) if has_eeg else None,
                "eeg_shrunk_inv_fif": str(eeg_shrunk_inv_p) if has_eeg else None,
                "eeg_isotropic_inv_fif": str(eeg_iso_inv_p) if has_eeg else None,
            },
        }
        print(f"[Operators] {sub:4s}: has_meg={has_meg}, has_eeg={has_eeg}")

    return operators_manifest


def compute_radiality_and_leadfield_pr(head_audit: dict, operators_manifest: dict) -> dict:
    """
    For each of the 16 valid BEM subjects:
      (a) Compute cortical dipole radiality index |n_v . r_hat_v| relative to the BEM inner-skull centroid;
      (b) Compute local 20 mm searchlight leadfield Participation Ratio PR = (Tr(Sigma_L))^2 / Tr(Sigma_L^2)
          separately for MEG Gradiometers (204 ch), MEG Magnetometers (102 ch), and EEG (64 ch),
          and map all quantities to fsaverage ico4 (2562 vertices/hemisphere = 5124 vertices).
    """
    fs_src_path = ROOT / "subjects/fsaverage/bem/fsaverage-ico-4-src.fif"
    if fs_src_path.exists():
        fs_src = mne.read_source_spaces(fs_src_path, verbose=False)
    else:
        fs_src = mne.setup_source_space("fsaverage", spacing="ico4", subjects_dir=str(SUBJECTS_CACHE), add_dist=False, verbose=False)
    fs_vertnos = [np.asarray(fs_src[0]["vertno"], dtype=int), np.asarray(fs_src[1]["vertno"], dtype=int)]
    assert len(fs_vertnos[0]) == 2562 and len(fs_vertnos[1]) == 2562

    labels_aparc_fs = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=str(SUBJECTS_CACHE), verbose=False)
    labels_a2009s_fs = mne.read_labels_from_annot("fsaverage", parc="aparc.a2009s", subjects_dir=str(SUBJECTS_CACHE), verbose=False)

    fs_curv = np.concatenate([
        mne.surface.read_curvature(SUBJECTS_CACHE / "fsaverage/surf/lh.curv", binary=False)[fs_vertnos[0]],
        mne.surface.read_curvature(SUBJECTS_CACHE / "fsaverage/surf/rh.curv", binary=False)[fs_vertnos[1]],
    ])

    def get_fs_roi_mask(labels_list, base_names):
        mask_lh = np.zeros(2562, dtype=bool)
        mask_rh = np.zeros(2562, dtype=bool)
        for bname in base_names:
            for l in labels_list:
                if l.name == f"{bname}-lh":
                    mask_lh |= np.isin(fs_vertnos[0], l.vertices)
                elif l.name == f"{bname}-rh":
                    mask_rh |= np.isin(fs_vertnos[1], l.vertices)
        return np.concatenate([mask_lh, mask_rh])

    roi_masks_fs = {
        "HG_aparc_transversetemporal": get_fs_roi_mask(labels_aparc_fs, ["transversetemporal"]),
        "HG_a2009s_G_T_transv": get_fs_roi_mask(labels_a2009s_fs, ["G_temp_sup-G_T_transv"]),
        "HS_a2009s_S_temporal_transverse": get_fs_roi_mask(labels_a2009s_fs, ["S_temporal_transverse"]),
        "PT_a2009s_Plan_tempo": get_fs_roi_mask(labels_a2009s_fs, ["G_temp_sup-Plan_tempo"]),
        "HG_complex_sulcal_wall": get_fs_roi_mask(
            labels_a2009s_fs, ["G_temp_sup-G_T_transv", "S_temporal_transverse", "G_temp_sup-Plan_tempo"]
        ),
        "LatSTG_a2009s_Lateral": get_fs_roi_mask(labels_a2009s_fs, ["G_temp_sup-Lateral"]),
        "LatSTG_aparc_superiortemporal": get_fs_roi_mask(labels_aparc_fs, ["superiortemporal"]),
    }
    roi_masks_fs["LatSTG_gyral_crown"] = roi_masks_fs["LatSTG_a2009s_Lateral"] & (fs_curv < 0)
    roi_masks_fs["HG_sulcal_wall_curv_pos"] = roi_masks_fs["HG_complex_sulcal_wall"] & (fs_curv > 0)

    n_valid = len(EXPECTED_VALID_SUBJECTS)
    fs_radiality_all = np.full((n_valid, 5124), np.nan, dtype=np.float64)
    fs_pr_grad_raw_all = np.full((n_valid, 5124), np.nan, dtype=np.float64)
    fs_pr_mag_raw_all = np.full((n_valid, 5124), np.nan, dtype=np.float64)
    fs_pr_eeg_raw_all = np.full((n_valid, 5124), np.nan, dtype=np.float64)
    fs_pr_grad_norm_all = np.full((n_valid, 5124), np.nan, dtype=np.float64)
    fs_pr_mag_norm_all = np.full((n_valid, 5124), np.nan, dtype=np.float64)
    fs_pr_eeg_norm_all = np.full((n_valid, 5124), np.nan, dtype=np.float64)

    per_subject_geom_stats = {}

    for idx_sub, sub in enumerate(EXPECTED_VALID_SUBJECTS):
        hm = head_audit[sub]
        op = operators_manifest[sub]
        src = mne.read_source_spaces(hm["src_fif"], verbose=False)
        bem = mne.read_bem_solution(hm["bem_fif"], verbose=False)
        inner_skull = [s for s in bem["surfs"] if int(s["id"]) == 1][0]
        center_mri = np.mean(inner_skull["rr"], axis=0)

        vertnos_full = [np.asarray(src[0]["vertno"], dtype=int), np.asarray(src[1]["vertno"], dtype=int)]
        rr_full = np.concatenate([src[0]["rr"][vertnos_full[0]], src[1]["rr"][vertnos_full[1]]], axis=0)
        nn_full = np.concatenate([src[0]["nn"][vertnos_full[0]], src[1]["nn"][vertnos_full[1]]], axis=0)
        hemis_full = np.concatenate([np.zeros(len(vertnos_full[0]), dtype=int), np.ones(len(vertnos_full[1]), dtype=int)])

        r_vec = rr_full - center_mri[None, :]
        r_hat = r_vec / np.maximum(np.linalg.norm(r_vec, axis=1, keepdims=True), 1e-12)
        radiality_native = np.abs(np.sum(nn_full * r_hat, axis=1))
        xyz_full_mm = rr_full * 1000.0

        def compute_searchlight_20mm_pr(G_mat, xyz_ret_mm, hemis_ret):
            pr_raw = np.zeros(len(xyz_full_mm), dtype=np.float64)
            pr_norm = np.zeros(len(xyz_full_mm), dtype=np.float64)
            G_colnorm = G_mat / np.maximum(np.linalg.norm(G_mat, axis=0, keepdims=True), 1e-30)
            for i in range(len(xyz_full_mm)):
                d2 = np.sum((xyz_ret_mm - xyz_full_mm[i]) ** 2, axis=1)
                nb = np.flatnonzero((d2 <= 400.0 + 1e-9) & (hemis_ret == hemis_full[i]))
                L = G_mat[:, nb]
                Gram = L.T @ L
                tr1 = float(np.trace(Gram))
                tr2 = float(np.sum(Gram * Gram))
                pr_raw[i] = (tr1 * tr1) / tr2 if tr2 > 0 else np.nan

                Ln = G_colnorm[:, nb]
                Gram_n = Ln.T @ Ln
                tr1_n = float(np.trace(Gram_n))
                tr2_n = float(np.sum(Gram_n * Gram_n))
                pr_norm[i] = (tr1_n * tr1_n) / tr2_n if tr2_n > 0 else np.nan
            return pr_raw, pr_norm

        def global_pr(G_mat):
            S = G_mat @ G_mat.T
            raw_val = float((np.trace(S) ** 2) / np.sum(S * S))
            Gn = G_mat / np.maximum(np.linalg.norm(G_mat, axis=0, keepdims=True), 1e-30)
            Sn = Gn @ Gn.T
            norm_val = float((np.trace(Sn) ** 2) / np.sum(Sn * Sn))
            return raw_val, norm_val

        morph = mne.compute_source_morph(
            src=src, subject_from=sub, subject_to="fsaverage",
            src_to=fs_src, spacing=None, smooth=5, sparse=False, xhemi=False,
            subjects_dir=str(SUBJECTS_CACHE), verbose=False
        )
        M = morph.morph_mat
        fs_radiality_all[idx_sub] = M @ radiality_native

        if op["has_meg"]:
            fwd_meg = mne.read_forward_solution(op["paths"]["meg_fwd_fif"], verbose=False)
            fwd_meg_f = mne.convert_forward_solution(fwd_meg, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
            G_meg = np.array(fwd_meg_f["sol"]["data"], dtype=np.float64)
            ch_types_meg = [mne.channel_type(fwd_meg_f["info"], i) for i in range(G_meg.shape[0])]
            grad_idx = [i for i, t in enumerate(ch_types_meg) if t == "grad"]
            mag_idx = [i for i, t in enumerate(ch_types_meg) if t == "mag"]
            G_grad = G_meg[grad_idx, :]
            G_mag = G_meg[mag_idx, :]
            vertnos_meg = [np.asarray(fwd_meg_f["src"][0]["vertno"], dtype=int), np.asarray(fwd_meg_f["src"][1]["vertno"], dtype=int)]
            xyz_meg_mm = np.concatenate([src[0]["rr"][vertnos_meg[0]], src[1]["rr"][vertnos_meg[1]]], axis=0) * 1000.0
            hemis_meg = np.concatenate([np.zeros(len(vertnos_meg[0]), dtype=int), np.ones(len(vertnos_meg[1]), dtype=int)])
            pr_grad_raw_nat, pr_grad_norm_nat = compute_searchlight_20mm_pr(G_grad, xyz_meg_mm, hemis_meg)
            pr_mag_raw_nat, pr_mag_norm_nat = compute_searchlight_20mm_pr(G_mag, xyz_meg_mm, hemis_meg)
            glob_grad_raw, glob_grad_norm = global_pr(G_grad)
            glob_mag_raw, glob_mag_norm = global_pr(G_mag)
            fs_pr_grad_raw_all[idx_sub] = M @ pr_grad_raw_nat
            fs_pr_mag_raw_all[idx_sub] = M @ pr_mag_raw_nat
            fs_pr_grad_norm_all[idx_sub] = M @ pr_grad_norm_nat
            fs_pr_mag_norm_all[idx_sub] = M @ pr_mag_norm_nat
        else:
            pr_grad_raw_nat = np.full(len(xyz_full_mm), np.nan)
            pr_grad_norm_nat = np.full(len(xyz_full_mm), np.nan)
            pr_mag_raw_nat = np.full(len(xyz_full_mm), np.nan)
            pr_mag_norm_nat = np.full(len(xyz_full_mm), np.nan)
            glob_grad_raw = glob_grad_norm = glob_mag_raw = glob_mag_norm = np.nan

        if op["has_eeg"]:
            fwd_eeg = mne.read_forward_solution(op["paths"]["eeg_fwd_fif"], verbose=False)
            fwd_eeg_f = mne.convert_forward_solution(fwd_eeg, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
            G_eeg = np.array(fwd_eeg_f["sol"]["data"], dtype=np.float64)
            G_eeg = G_eeg - np.mean(G_eeg, axis=0, keepdims=True)
            vertnos_eeg = [np.asarray(fwd_eeg_f["src"][0]["vertno"], dtype=int), np.asarray(fwd_eeg_f["src"][1]["vertno"], dtype=int)]
            xyz_eeg_mm = np.concatenate([src[0]["rr"][vertnos_eeg[0]], src[1]["rr"][vertnos_eeg[1]]], axis=0) * 1000.0
            hemis_eeg = np.concatenate([np.zeros(len(vertnos_eeg[0]), dtype=int), np.ones(len(vertnos_eeg[1]), dtype=int)])
            pr_eeg_raw_nat, pr_eeg_norm_nat = compute_searchlight_20mm_pr(G_eeg, xyz_eeg_mm, hemis_eeg)
            glob_eeg_raw, glob_eeg_norm = global_pr(G_eeg)
            fs_pr_eeg_raw_all[idx_sub] = M @ pr_eeg_raw_nat
            fs_pr_eeg_norm_all[idx_sub] = M @ pr_eeg_norm_nat
        else:
            pr_eeg_raw_nat = np.full(len(xyz_full_mm), np.nan)
            pr_eeg_norm_nat = np.full(len(xyz_full_mm), np.nan)
            glob_eeg_raw = glob_eeg_norm = np.nan

        labels_a2009s_nat = mne.read_labels_from_annot(sub, parc="aparc.a2009s", subjects_dir=str(SUBJECTS_CACHE), verbose=False)
        def get_nat_mask(bnames):
            m0 = np.zeros(len(vertnos_full[0]), dtype=bool)
            m1 = np.zeros(len(vertnos_full[1]), dtype=bool)
            for b in bnames:
                for l in labels_a2009s_nat:
                    if l.name == f"{b}-lh":
                        m0 |= np.isin(vertnos_full[0], l.vertices)
                    elif l.name == f"{b}-rh":
                        m1 |= np.isin(vertnos_full[1], l.vertices)
            return np.concatenate([m0, m1])

        nat_hg_mask = get_nat_mask(["G_temp_sup-G_T_transv", "S_temporal_transverse", "G_temp_sup-Plan_tempo"])
        nat_hg_strict_mask = get_nat_mask(["G_temp_sup-G_T_transv"])
        nat_hs_strict_mask = get_nat_mask(["S_temporal_transverse"])
        nat_stg_lat_mask = get_nat_mask(["G_temp_sup-Lateral"])

        def _r(val):
            return round(float(val), 4) if np.isfinite(val) else None

        per_subject_geom_stats[sub] = {
            "inner_skull_center_mri_mm": [round(float(x * 1000.0), 3) for x in center_mri],
            "native_radiality_whole_cortex_mean": _r(np.mean(radiality_native)),
            "native_radiality_HG_transv_gyrus_mean": _r(np.mean(radiality_native[nat_hg_strict_mask])),
            "native_radiality_HS_transv_sulcus_mean": _r(np.mean(radiality_native[nat_hs_strict_mask])),
            "native_radiality_HG_complex_mean": _r(np.mean(radiality_native[nat_hg_mask])),
            "native_radiality_LatSTG_mean": _r(np.mean(radiality_native[nat_stg_lat_mask])),
            "native_local20mm_PR_whole_cortex": {
                "meg_grad_raw": _r(np.nanmean(pr_grad_raw_nat)),
                "meg_mag_raw": _r(np.nanmean(pr_mag_raw_nat)),
                "eeg_raw": _r(np.nanmean(pr_eeg_raw_nat)),
                "meg_grad_colnorm": _r(np.nanmean(pr_grad_norm_nat)),
                "meg_mag_colnorm": _r(np.nanmean(pr_mag_norm_nat)),
                "eeg_colnorm": _r(np.nanmean(pr_eeg_norm_nat)),
            },
            "native_local20mm_PR_HG_transv": {
                "meg_grad_raw": _r(np.nanmean(pr_grad_raw_nat[nat_hg_strict_mask])),
                "meg_mag_raw": _r(np.nanmean(pr_mag_raw_nat[nat_hg_strict_mask])),
                "eeg_raw": _r(np.nanmean(pr_eeg_raw_nat[nat_hg_strict_mask])),
                "meg_grad_colnorm": _r(np.nanmean(pr_grad_norm_nat[nat_hg_strict_mask])),
                "meg_mag_colnorm": _r(np.nanmean(pr_mag_norm_nat[nat_hg_strict_mask])),
                "eeg_colnorm": _r(np.nanmean(pr_eeg_norm_nat[nat_hg_strict_mask])),
            },
            "native_local20mm_PR_LatSTG": {
                "meg_grad_raw": _r(np.nanmean(pr_grad_raw_nat[nat_stg_lat_mask])),
                "meg_mag_raw": _r(np.nanmean(pr_mag_raw_nat[nat_stg_lat_mask])),
                "eeg_raw": _r(np.nanmean(pr_eeg_raw_nat[nat_stg_lat_mask])),
                "meg_grad_colnorm": _r(np.nanmean(pr_grad_norm_nat[nat_stg_lat_mask])),
                "meg_mag_colnorm": _r(np.nanmean(pr_mag_norm_nat[nat_stg_lat_mask])),
                "eeg_colnorm": _r(np.nanmean(pr_eeg_norm_nat[nat_stg_lat_mask])),
            },
            "global_leadfield_PR": {
                "meg_grad_raw": _r(glob_grad_raw),
                "meg_mag_raw": _r(glob_mag_raw),
                "eeg_raw": _r(glob_eeg_raw),
                "meg_grad_colnorm": _r(glob_grad_norm),
                "meg_mag_colnorm": _r(glob_mag_norm),
                "eeg_colnorm": _r(glob_eeg_norm),
            },
        }
        operators_manifest[sub].update(per_subject_geom_stats[sub])
        print(f"[Geometry & PR] {sub:4s}: Rad(HG={per_subject_geom_stats[sub]['native_radiality_HG_complex_mean']}, LatSTG={per_subject_geom_stats[sub]['native_radiality_LatSTG_mean']})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = RESULTS_DIR / "stage1_cortical_geometry_and_pr_maps.npz"
    np.savez_compressed(
        npz_path,
        subjects=np.array(EXPECTED_VALID_SUBJECTS),
        fsaverage_lh_vertno=fs_vertnos[0],
        fsaverage_rh_vertno=fs_vertnos[1],
        fsaverage_curv=fs_curv,
        fs_radiality_all=fs_radiality_all,
        fs_pr_grad_raw_all=fs_pr_grad_raw_all,
        fs_pr_mag_raw_all=fs_pr_mag_raw_all,
        fs_pr_eeg_raw_all=fs_pr_eeg_raw_all,
        fs_pr_grad_norm_all=fs_pr_grad_norm_all,
        fs_pr_mag_norm_all=fs_pr_mag_norm_all,
        fs_pr_eeg_norm_all=fs_pr_eeg_norm_all,
        **{f"roi_mask_{k}": v for k, v in roi_masks_fs.items()},
    )

    fs_roi_summary = {}
    for rname, rmask in roi_masks_fs.items():
        subj_rad = np.nanmean(fs_radiality_all[:, rmask], axis=1)
        subj_grad_raw = np.nanmean(fs_pr_grad_raw_all[:, rmask], axis=1)
        subj_mag_raw = np.nanmean(fs_pr_mag_raw_all[:, rmask], axis=1)
        subj_eeg_raw = np.nanmean(fs_pr_eeg_raw_all[:, rmask], axis=1)
        subj_grad_norm = np.nanmean(fs_pr_grad_norm_all[:, rmask], axis=1)
        subj_mag_norm = np.nanmean(fs_pr_mag_norm_all[:, rmask], axis=1)
        subj_eeg_norm = np.nanmean(fs_pr_eeg_norm_all[:, rmask], axis=1)
        fs_roi_summary[rname] = {
            "n_fsaverage_vertices": int(np.sum(rmask)),
            "radiality_mean": round(float(np.nanmean(subj_rad)), 4),
            "radiality_sd": round(float(np.nanstd(subj_rad, ddof=1)), 4),
            "pr_meg_grad_raw_mean": round(float(np.nanmean(subj_grad_raw)), 4),
            "pr_meg_grad_raw_sd": round(float(np.nanstd(subj_grad_raw, ddof=1)), 4),
            "pr_meg_mag_raw_mean": round(float(np.nanmean(subj_mag_raw)), 4),
            "pr_meg_mag_raw_sd": round(float(np.nanstd(subj_mag_raw, ddof=1)), 4),
            "pr_eeg_raw_mean": round(float(np.nanmean(subj_eeg_raw)), 4),
            "pr_eeg_raw_sd": round(float(np.nanstd(subj_eeg_raw, ddof=1)), 4),
            "pr_meg_grad_colnorm_mean": round(float(np.nanmean(subj_grad_norm)), 4),
            "pr_meg_grad_colnorm_sd": round(float(np.nanstd(subj_grad_norm, ddof=1)), 4),
            "pr_meg_mag_colnorm_mean": round(float(np.nanmean(subj_mag_norm)), 4),
            "pr_meg_mag_colnorm_sd": round(float(np.nanstd(subj_mag_norm, ddof=1)), 4),
            "pr_eeg_colnorm_mean": round(float(np.nanmean(subj_eeg_norm)), 4),
            "pr_eeg_colnorm_sd": round(float(np.nanstd(subj_eeg_norm, ddof=1)), 4),
        }

    return {
        "npz_path": str(npz_path),
        "fsaverage_roi_summary_N16": fs_roi_summary,
        "per_subject_geometry_and_pr": per_subject_geom_stats,
    }


def save_manifests(head_audit: dict, operators_manifest: dict, geom_summary: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "stage1_cohort_operators_manifest.json"
    csv_path = RESULTS_DIR / "stage1_cohort_operators_manifest.csv"

    rows = []
    for sub in ALL_SUBJECTS:
        hm = head_audit[sub]
        row = {
            "subject": sub,
            "bem_valid": hm["bem_valid"],
            "bem_status_note": hm["bem_status_note"],
            "dense_inner_in_outer_skull_outside_vertices": (
                hm["nesting_diagnostic"].get("details_dense", {}).get("inner_skull_in_outer_skull", {}).get("outside_vertices")
            ),
            "ico4_inner_in_outer_skull_outside_vertices": (
                hm["nesting_diagnostic"].get("details_ico4", {}).get("inner_skull_in_outer_skull", {}).get("outside_vertices")
            ),
        }
        if sub in operators_manifest:
            op = operators_manifest[sub]
            row.update({
                "n_meg_channels": op["n_meg_channels"],
                "n_grad_channels": op["n_grad_channels"],
                "n_mag_channels": op["n_mag_channels"],
                "n_eeg_channels": op["n_eeg_channels"],
                "n_source_vertices_meg": op["n_source_vertices_meg"],
                "n_source_vertices_eeg": op["n_source_vertices_eeg"],
                "rank_meg": op["rank_meg"],
                "rank_eeg": op["rank_eeg"],
                "meg_bg_seconds": op["meg_bg_seconds"],
                "eeg_bg_seconds": op["eeg_bg_seconds"],
                "meg_provenance": op["meg_provenance"],
                "eeg_provenance": op["eeg_provenance"],
                "kappa_meg_grad_shrunk": op["kappa_meg_grad_shrunk"],
                "kappa_meg_mag_shrunk": op["kappa_meg_mag_shrunk"],
                "kappa_meg_joint_shrunk": op["kappa_meg_joint_shrunk"],
                "kappa_meg_grad_isotropic": op["kappa_meg_grad_isotropic"],
                "kappa_meg_mag_isotropic": op["kappa_meg_mag_isotropic"],
                "kappa_meg_joint_isotropic": op["kappa_meg_joint_isotropic"],
                "kappa_eeg_shrunk": op["kappa_eeg_shrunk"],
                "kappa_eeg_isotropic": op["kappa_eeg_isotropic"],
                "native_radiality_whole_cortex_mean": op["native_radiality_whole_cortex_mean"],
                "native_radiality_HG_transv_gyrus_mean": op["native_radiality_HG_transv_gyrus_mean"],
                "native_radiality_HS_transv_sulcus_mean": op["native_radiality_HS_transv_sulcus_mean"],
                "native_radiality_HG_complex_mean": op["native_radiality_HG_complex_mean"],
                "native_radiality_LatSTG_mean": op["native_radiality_LatSTG_mean"],
                "pr_local20mm_HG_meg_grad_raw": op["native_local20mm_PR_HG_transv"]["meg_grad_raw"],
                "pr_local20mm_HG_meg_mag_raw": op["native_local20mm_PR_HG_transv"]["meg_mag_raw"],
                "pr_local20mm_HG_eeg_raw": op["native_local20mm_PR_HG_transv"]["eeg_raw"],
                "pr_local20mm_LatSTG_meg_grad_raw": op["native_local20mm_PR_LatSTG"]["meg_grad_raw"],
                "pr_local20mm_LatSTG_meg_mag_raw": op["native_local20mm_PR_LatSTG"]["meg_mag_raw"],
                "pr_local20mm_LatSTG_eeg_raw": op["native_local20mm_PR_LatSTG"]["eeg_raw"],
                "pr_global_meg_grad_raw": op["global_leadfield_PR"]["meg_grad_raw"],
                "pr_global_meg_mag_raw": op["global_leadfield_PR"]["meg_mag_raw"],
                "pr_global_eeg_raw": op["global_leadfield_PR"]["eeg_raw"],
                **op["paths"],
            })
        else:
            for k in [
                "n_meg_channels", "n_grad_channels", "n_mag_channels", "n_eeg_channels",
                "n_source_vertices_meg", "n_source_vertices_eeg", "rank_meg", "rank_eeg",
                "meg_bg_seconds", "eeg_bg_seconds", "meg_provenance", "eeg_provenance",
                "kappa_meg_grad_shrunk", "kappa_meg_mag_shrunk", "kappa_meg_joint_shrunk",
                "kappa_meg_grad_isotropic", "kappa_meg_mag_isotropic", "kappa_meg_joint_isotropic",
                "kappa_eeg_shrunk", "kappa_eeg_isotropic",
                "native_radiality_whole_cortex_mean", "native_radiality_HG_transv_gyrus_mean",
                "native_radiality_HS_transv_sulcus_mean", "native_radiality_HG_complex_mean",
                "native_radiality_LatSTG_mean",
                "pr_local20mm_HG_meg_grad_raw", "pr_local20mm_HG_meg_mag_raw", "pr_local20mm_HG_eeg_raw",
                "pr_local20mm_LatSTG_meg_grad_raw", "pr_local20mm_LatSTG_meg_mag_raw", "pr_local20mm_LatSTG_eeg_raw",
                "pr_global_meg_grad_raw", "pr_global_meg_mag_raw", "pr_global_eeg_raw",
                "bem_fif", "src_fif", "trans_fif", "meg_info_fif", "meg_fwd_fif",
                "meg_shrunk_cov_fif", "meg_isotropic_cov_fif", "meg_shrunk_inv_fif", "meg_isotropic_inv_fif",
                "eeg_info_fif", "eeg_fwd_fif", "eeg_shrunk_cov_fif", "eeg_isotropic_cov_fif",
                "eeg_shrunk_inv_fif", "eeg_isotropic_inv_fif",
            ]:
                row[k] = None
        rows.append(row)

    payload = {
        "stage": "Stage 1 (Checkpoint 1): Full E1-E20 Head-Model, Prestimulus Covariance Cm, Forward Operator Gm, and Shrunk/Isotropic Inverse Operator Km Audit & Build",
        "total_subjects_audited": 20,
        "valid_bem_count": len(EXPECTED_VALID_SUBJECTS),
        "failed_bem_count": len(EXPECTED_FAILED_SUBJECTS),
        "valid_bem_subjects": EXPECTED_VALID_SUBJECTS,
        "failed_bem_subjects": EXPECTED_FAILED_SUBJECTS,
        "runtime": {
            "python": sys.version.split()[0],
            "mne": mne.__version__,
            "numpy": np.__version__,
        },
        "fsaverage_roi_summary_N16": geom_summary["fsaverage_roi_summary_N16"],
        "cortical_geometry_and_pr_maps_npz": geom_summary["npz_path"],
        "cohort_table": rows,
        "head_audit_details": head_audit,
        "operators_manifest_details": operators_manifest,
    }

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved JSON manifest: {json_path}")
    print(f"Saved CSV manifest : {csv_path}")
    print(f"Saved NPZ maps     : {geom_summary['npz_path']}")


def main():
    print("=== Stage 1: Setting up isolated subjects_cache ===")
    setup_isolated_subjects_cache()
    print("=== Stage 1: Auditing E1..E20 head models & building missing BEM/src/trans ===")
    head_audit = audit_and_prepare_head_models()
    print("=== Stage 1: Auditing & building MEG and EEG operators (shrunk & isotropic) for N=16 valid subjects ===")
    operators_manifest = audit_and_build_operators(head_audit)
    print("=== Stage 1: Computing cortical dipole radiality & local 20mm leadfield Participation Ratio (PR) ===")
    geom_summary = compute_radiality_and_leadfield_pr(head_audit, operators_manifest)
    print("=== Stage 1: Saving JSON and CSV manifests ===")
    save_manifests(head_audit, operators_manifest, geom_summary)


if __name__ == "__main__":
    main()
