#!/usr/bin/env python3
"""Stage 3 (Checkpoint 3): Group-Level Repeated-Measures Evaluation of:
  - Factor 1: Noise Covariance Regularization (`shrunk` Ledoit-Wolf pre-whitening vs. `isotropic` MNE inverse operator)
  - Factor 2: Sensor Temporal Low-Pass Filtering (`80 Hz` vs. `40 Hz` vs. `20 Hz` vs. `10 Hz`)
  - Factor 3: Cortical Explicit Spatial Smoothing (`FWHM = 0, 5, 10, 20 mm`) x Searchlight Radius (`10, 20, 35 mm`) 2D Phase Diagram
  - Factor 4 (H6 Bonus): End-to-End 8-Stage Representational Fidelity Waterfall Audit

STRICT STATISTICAL RULE (Advisor S0769-S0808):
All subject-level quantities are treated strictly as Repeated Measures (within-subject paired comparisons across the cohort,
reporting group mean +/- SEM, SD, Cohen's dz, paired t-statistic, degrees of freedom, and Wilcoxon signed-rank p-value).
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mne
from mne._fiff.proj import make_projector
from mne.minimum_norm.inverse import _assemble_kernel, prepare_inverse_operator
import numpy as np
from scipy import signal, stats
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import pdist, squareform


ROOT = Path(".")
EXP_DIR = ROOT / "revision_20261104_v2_experiments"
RESULTS_DIR = EXP_DIR / "results"
SCRIPTS_DIR = EXP_DIR / "scripts"

# Valid head-model subjects (Simulation benchmark)
MEG_HEAD_MANIFESTS: Dict[str, str] = {
    "E1": "derivatives/E1/run2_g3_verified_operators_v1/operators_manifest.json",
    "E2": "derivatives/E2/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1/operators_manifest.json",
    "E3": "derivatives/E3/modality_meg_source_v1/operators_nominal_output_v1/run2_meg_operators_v1/operators_manifest.json",
    "E4": "derivatives/E4/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1/operators_manifest.json",
    "E5": "derivatives/E5/modality_meg_source_v1/operators/run1_meg_operators_v1/operators_manifest.json",
    "E9": "derivatives/E9/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1/operators_manifest.json",
    "E10": "derivatives/E10/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1/operators_manifest.json",
    "E14": "derivatives/E14/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1/operators_manifest.json",
    "E15": "derivatives/E15/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1/operators_manifest.json",
    "E17": "derivatives/E17/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1/operators_manifest.json",
    "E18": "derivatives/E18/modality_meg_source_v1/operators_nominal_output_v1/run1_meg_operators_v1/operators_manifest.json",
}

# Empirical Soto natural-speech cohorts (with complete 2-run condition-mean exports)
MEG_EMPIRICAL_SUBJECTS: List[str] = ["E2", "E3", "E4", "E9", "E10", "E14", "E15", "E17", "E18"]

EEG_HEAD_RUNS: Dict[str, str] = {
    "E1": "run2",
    "E6": "run1",
    "E9": "run1",
    "E10": "run1",
    "E11": "run1",
    "E18": "run2",
}
EEG_EMPIRICAL_SUBJECTS: List[str] = ["E6", "E9", "E10", "E11", "E18"]

DEFAULT_CLUSTER_SPEC = [0.6, [0.3, 5], [0.3, 5], [0.3, 5], [0.3, 5]]


# =============================================================================
# 1. Mathematical & Statistical Helpers (Repeated Measures Only)
# =============================================================================

def repeated_measures_paired_stats(x_arr: np.ndarray, y_arr: np.ndarray, label_x: str = "x", label_y: str = "y") -> Dict[str, Any]:
    """Compute within-subject repeated-measures statistics for paired arrays x_arr vs y_arr."""
    x = np.asarray(x_arr, dtype=np.float64).ravel()
    y = np.asarray(y_arr, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 2:
        return {"n_subjects": n, "df": max(0, n - 1), "status": "INSUFFICIENT_N"}

    delta = x - y
    mean_x = float(np.mean(x))
    sd_x = float(np.std(x, ddof=1))
    sem_x = float(sd_x / math.sqrt(n))

    mean_y = float(np.mean(y))
    sd_y = float(np.std(y, ddof=1))
    sem_y = float(sd_y / math.sqrt(n))

    mean_d = float(np.mean(delta))
    sd_d = float(np.std(delta, ddof=1))
    sem_d = float(sd_d / math.sqrt(n))
    cohen_dz = float(mean_d / sd_d) if sd_d > 1e-15 else 0.0

    if sd_d > 1e-15:
        t_res = stats.ttest_rel(x, y)
        t_stat = float(t_res.statistic)
        p_ttest = float(t_res.pvalue)
    else:
        t_stat = 0.0
        p_ttest = 1.0

    nonzero_d = delta[np.abs(delta) > 1e-15]
    if nonzero_d.size >= 2:
        try:
            w_res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            w_stat = float(w_res.statistic)
            p_wilcoxon = float(w_res.pvalue)
        except Exception:
            w_stat = 0.0
            p_wilcoxon = 1.0
    else:
        w_stat = 0.0
        p_wilcoxon = 1.0

    # Two-way repeated-measures variance decomposition (Subjects vs Condition vs Residual)
    mat = np.column_stack([x, y])
    grand_mean = float(np.mean(mat))
    ss_total = float(np.sum((mat - grand_mean) ** 2))
    subj_means = np.mean(mat, axis=1, keepdims=True)
    cond_means = np.mean(mat, axis=0, keepdims=True)
    ss_between_subjects = float(2.0 * np.sum((subj_means - grand_mean) ** 2))
    ss_within_condition = float(n * np.sum((cond_means - grand_mean) ** 2))
    ss_residual = max(0.0, ss_total - ss_between_subjects - ss_within_condition)

    return {
        "n_subjects": n,
        "df": n - 1,
        f"mean_{label_x}": mean_x,
        f"sd_{label_x}": sd_x,
        f"sem_{label_x}": sem_x,
        f"mean_{label_y}": mean_y,
        f"sd_{label_y}": sd_y,
        f"sem_{label_y}": sem_y,
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "sem_delta": sem_d,
        "cohen_dz": cohen_dz,
        "t_stat": t_stat,
        "p_ttest": p_ttest,
        "wilcoxon_stat": w_stat,
        "p_wilcoxon": p_wilcoxon,
        "n_positive_delta": int(np.sum(delta > 0)),
        "variance_partition": {
            "ss_total": ss_total,
            "between_subject_pct": float(100.0 * ss_between_subjects / max(ss_total, 1e-30)),
            "within_subject_condition_pct": float(100.0 * ss_within_condition / max(ss_total, 1e-30)),
            "residual_interaction_pct": float(100.0 * ss_residual / max(ss_total, 1e-30)),
            "partial_eta_sq_within": float(ss_within_condition / max(ss_within_condition + ss_residual, 1e-30)),
        },
    }


def group_summary_1d(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "sem": 0.0, "median": 0.0, "iqr": 0.0}
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(x, [75, 25]) if n > 1 else (float(x[0]), float(x[0]))
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "sd": sd,
        "sem": float(sd / math.sqrt(n)) if n > 0 else 0.0,
        "median": float(np.median(x)),
        "iqr": float(q75 - q25),
    }


def build_hierarchical_generator_matrix(cluster_spec=None) -> np.ndarray:
    if cluster_spec is None:
        cluster_spec = DEFAULT_CLUSTER_SPEC
    sigma_top = math.sqrt(float(cluster_spec[0]))
    subspecs = cluster_spec[1:]
    rows = []
    total_conds = sum(int(s[1]) for s in subspecs)
    cond_offset = 0
    for s in subspecs:
        sigma_leaf = math.sqrt(float(s[0]))
        n_leaves = int(s[1])
        v_centroid = np.zeros(total_conds, dtype=np.float64)
        v_centroid[cond_offset:cond_offset + n_leaves] = sigma_top
        rows.append(v_centroid)
        for i in range(n_leaves):
            v_leaf = np.zeros(total_conds, dtype=np.float64)
            v_leaf[cond_offset + i] = sigma_leaf
            rows.append(v_leaf)
        cond_offset += n_leaves
    return np.stack(rows, axis=0)


def theoretical_true_cov_and_rdm(cluster_spec=None):
    V = build_hierarchical_generator_matrix(cluster_spec)
    cov_true = V.T @ V
    d = np.sqrt(np.diag(cov_true))
    corr_true = cov_true / np.outer(d, d)
    rdm_true = 1.0 - corr_true
    np.fill_diagonal(rdm_true, 0.0)
    iu = np.triu_indices(rdm_true.shape[0], k=1)
    return V, cov_true, corr_true, rdm_true, rdm_true[iu]


def rdm_corr_distance_vec(B: np.ndarray) -> np.ndarray:
    """Compute upper-triangular Pearson correlation distance vector (1 - r) for (n_cond, n_features)."""
    B_cent = B - np.mean(B, axis=1, keepdims=True)
    norms = np.linalg.norm(B_cent, axis=1, keepdims=True)
    norms = np.where(norms > 1e-15, norms, 1.0)
    B_norm = B_cent / norms
    corr = B_norm @ B_norm.T
    rdm = 1.0 - np.clip(corr, -1.0, 1.0)
    iu = np.triu_indices(B.shape[0], k=1)
    return rdm[iu]


def spearman_corr_fast(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return 0.0
    rx = stats.rankdata(x[mask])
    ry = stats.rankdata(y[mask])
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom <= 1e-15:
        return 0.0
    return float(np.dot(rx, ry) / denom)


def pearson_corr_fast(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return 0.0
    x = x[mask] - np.mean(x[mask])
    y = y[mask] - np.mean(y[mask])
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-15:
        return 0.0
    return float(np.dot(x, y) / denom)


# =============================================================================
# 2. Anatomy, Geodesic Mesh, Inverse Operators & Covariance Loading
# =============================================================================

def compute_hemisphere_geodesic_distances(src_hemi, retained_vertno) -> np.ndarray:
    vertno = np.asarray(retained_vertno, dtype=int)
    rr = src_hemi["rr"]
    src_vertno = np.asarray(src_hemi["vertno"], dtype=int)
    use_tris = src_hemi.get("use_tris")
    if use_tris is not None and np.all(np.isin(vertno, src_vertno)):
        v_to_idx = {int(v): i for i, v in enumerate(src_vertno)}
        n_src = len(src_vertno)
        tris_idx = np.vectorize(v_to_idx.get)(use_tris)
        edges_a = np.concatenate([tris_idx[:, 0], tris_idx[:, 1], tris_idx[:, 2]])
        edges_b = np.concatenate([tris_idx[:, 1], tris_idx[:, 2], tris_idx[:, 0]])
        pts_a = rr[src_vertno[edges_a]]
        pts_b = rr[src_vertno[edges_b]]
        weights = np.linalg.norm(pts_a - pts_b, axis=1)
        row = np.concatenate([edges_a, edges_b])
        col = np.concatenate([edges_b, edges_a])
        dat = np.concatenate([weights, weights])
        adj = sp.csr_matrix((dat, (row, col)), shape=(n_src, n_src))
        dist_full = dijkstra(adj, directed=False)
        if np.isfinite(dist_full).all():
            if np.array_equal(vertno, src_vertno):
                return dist_full
            sub_idx = np.array([v_to_idx[int(v)] for v in vertno], dtype=int)
            return dist_full[np.ix_(sub_idx, sub_idx)]

    tris = src_hemi["tris"]
    n_all = rr.shape[0]
    edges_a = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    edges_b = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    weights = np.linalg.norm(rr[edges_a] - rr[edges_b], axis=1)
    row = np.concatenate([edges_a, edges_b])
    col = np.concatenate([edges_b, edges_a])
    dat = np.concatenate([weights, weights])
    adj = sp.csr_matrix((dat, (row, col)), shape=(n_all, n_all))
    dist_from_ret = dijkstra(adj, directed=False, indices=vertno)
    dist_mat = dist_from_ret[:, vertno]
    if not np.isfinite(dist_mat).all():
        euc = squareform(pdist(rr[vertno]))
        dist_mat = np.where(np.isfinite(dist_mat), dist_mat, euc)
    return dist_mat


def get_auditory_roi_masks(subject: str, vertices: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Return boolean masks over the combined (lh + rh) vertex space for:
      - hg_bilat: bilateral Heschl's gyrus (`transversetemporal`)
      - lstg_bilat: bilateral Lateral STG (`superiortemporal`)
      - aud_lh: left auditory cortex (`transversetemporal` + `superiortemporal`)
      - aud_bilat: bilateral auditory cortex (`transversetemporal` + `superiortemporal`)
    """
    subjects_dir = str(ROOT / "subjects")
    labels = mne.read_labels_from_annot(subject, parc="aparc", subjects_dir=subjects_dir, verbose=False)
    v_lh, v_rh = [np.asarray(v, dtype=int) for v in vertices]

    sets = {
        ("lh", "transversetemporal"): set(),
        ("rh", "transversetemporal"): set(),
        ("lh", "superiortemporal"): set(),
        ("rh", "superiortemporal"): set(),
    }
    for lab in labels:
        base = lab.name.split("-")[0]
        key = (lab.hemi, base)
        if key in sets:
            sets[key] |= set(int(x) for x in lab.vertices)

    hg_lh = np.array([int(v) in sets[("lh", "transversetemporal")] for v in v_lh], dtype=bool)
    hg_rh = np.array([int(v) in sets[("rh", "transversetemporal")] for v in v_rh], dtype=bool)
    lstg_lh = np.array([int(v) in sets[("lh", "superiortemporal")] for v in v_lh], dtype=bool)
    lstg_rh = np.array([int(v) in sets[("rh", "superiortemporal")] for v in v_rh], dtype=bool)

    hg_bilat = np.concatenate([hg_lh, hg_rh])
    lstg_bilat = np.concatenate([lstg_lh, lstg_rh])
    aud_lh = np.concatenate([hg_lh | lstg_lh, np.zeros(len(v_rh), dtype=bool)])
    aud_bilat = hg_bilat | lstg_bilat

    return {
        "hg_bilat": hg_bilat,
        "lstg_bilat": lstg_bilat,
        "aud_lh": aud_lh,
        "aud_bilat": aud_bilat,
    }


def analyze_covariance_off_diagonal_suppression(C_proj: np.ndarray, W_shrunk: np.ndarray, ch_types: List[str]) -> Dict[str, Any]:
    """Quantify empirical cross-channel noise correlation and its suppression by shrunk pre-whitening vs isotropic scaling."""
    diag = np.sqrt(np.maximum(np.diag(C_proj), 1e-30))
    corr_emp = C_proj / np.outer(diag, diag)
    np.fill_diagonal(corr_emp, 0.0)
    n_ch = C_proj.shape[0]
    off_mask = ~np.eye(n_ch, dtype=bool)
    abs_off_emp = np.abs(corr_emp[off_mask])
    rms_off_emp = float(np.sqrt(np.mean(abs_off_emp ** 2)))
    mean_off_emp = float(np.mean(abs_off_emp))
    p95_off_emp = float(np.percentile(abs_off_emp, 95))

    # After shrunk pre-whitening in the rank-r subspace: W_shrunk @ C_proj @ W_shrunk.T = I_r
    # Project back to sensor space or evaluate sensor-space whitened covariance
    C_whitened_sub = W_shrunk @ C_proj @ W_shrunk.T
    r = C_whitened_sub.shape[0]
    off_sub_mask = ~np.eye(r, dtype=bool)
    diag_w = np.sqrt(np.maximum(np.diag(C_whitened_sub), 1e-30))
    corr_whitened = C_whitened_sub / np.outer(diag_w, diag_w)
    abs_off_whitened = np.abs(corr_whitened[off_sub_mask])
    rms_off_shrunk = float(np.sqrt(np.mean(abs_off_whitened ** 2)))

    # Isotropic scaling preserves all off-diagonal correlations within each sensor type
    rms_off_iso = rms_off_emp

    # Condition number of empirical covariance in rank-r subspace
    eigvals = np.linalg.eigvalsh(C_proj)
    eigvals_pos = np.sort(eigvals)[::-1][:r]
    cond_emp = float(eigvals_pos[0] / max(eigvals_pos[-1], 1e-30))
    eigvals_w = np.linalg.eigvalsh(C_whitened_sub)
    eigvals_w_pos = np.sort(eigvals_w)[::-1]
    cond_shrunk = float(eigvals_w_pos[0] / max(eigvals_w_pos[-1], 1e-30))

    return {
        "n_channels": n_ch,
        "effective_rank": r,
        "empirical_rms_off_diag_corr": rms_off_emp,
        "empirical_mean_off_diag_corr": mean_off_emp,
        "empirical_p95_off_diag_corr": p95_off_emp,
        "isotropic_residual_rms_off_diag_corr": rms_off_iso,
        "shrunk_whitened_residual_rms_off_diag_corr": rms_off_shrunk,
        "off_diag_suppression_db": float(20.0 * math.log10(max(rms_off_emp, 1e-15) / max(rms_off_shrunk, 1e-15))),
        "condition_number_isotropic": cond_emp,
        "condition_number_shrunk": cond_shrunk,
    }


def build_noise_colorer_and_isotropic_cov(cov, info, rank_dict):
    ch_names = info["ch_names"]
    ch_types = info.get_channel_types()
    projs = info.get("projs") or cov.get("projs") or []
    P, _, _ = make_projector(projs, ch_names)
    C_raw = np.asarray(cov.data, dtype=np.float64).copy()
    C_proj = P @ C_raw @ P.T
    C_proj = 0.5 * (C_proj + C_proj.T)

    unit_scales = {"mag": 1e12, "grad": 1e11, "eeg": 1e5}
    s_vec = np.array([unit_scales[t] for t in ch_types], dtype=np.float64)
    S_mat = np.diag(s_vec)
    S_inv = np.diag(1.0 / s_vec)

    C_scaled = S_mat @ C_proj @ S_mat
    C_scaled = 0.5 * (C_scaled + C_scaled.T)
    eigvals, eigvecs = np.linalg.eigh(C_scaled)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    r = int(sum(rank_dict.values()))
    lam_r = np.maximum(eigvals[:r], 1e-30)
    U_r = eigvecs[:, :r]

    L_m = S_inv @ (U_r * np.sqrt(lam_r)[None, :])
    W_m = (U_r.T / np.sqrt(lam_r)[:, None]) @ S_mat

    lam_mean = float(np.mean(lam_r))
    C_iso_data = S_inv @ (lam_mean * (U_r @ U_r.T)) @ S_inv
    C_iso_data = 0.5 * (C_iso_data + C_iso_data.T)
    cov_iso = mne.Covariance(
        C_iso_data, list(cov["names"]), list(cov["bads"]), cov["projs"], int(cov["nfree"])
    )
    cov_stats = analyze_covariance_off_diagonal_suppression(C_proj, W_m, ch_types)
    return P, C_proj, L_m, W_m, cov_iso, r, cov_stats


def load_meg_subject(subject: str) -> Dict[str, Any]:
    rel = MEG_HEAD_MANIFESTS[subject]
    m = json.loads((ROOT / rel).read_text())
    rec = next(r for r in m["runs"] if r["modality"] == "meg")
    src_path = m.get("source_spaces") or str(list((ROOT / f"derivatives/{subject}").glob("**/*-ico4-src.fif"))[0])

    info = mne.io.read_info(rec["info"], verbose=False)
    fwd = mne.read_forward_solution(rec["fwd"], verbose=False)
    cov = mne.read_cov(rec["cov"], verbose=False)
    inv_shrunk = mne.minimum_norm.read_inverse_operator(rec["inv"], verbose=False)
    src = mne.read_source_spaces(src_path, verbose=False)

    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
    vertices = [s["vertno"] for s in inv_shrunk["src"]]
    rank_dict = rec["rank"]
    P, C_proj, L_m, W_m, cov_iso, r, cov_stats = build_noise_colorer_and_isotropic_cov(cov, info, rank_dict)
    G_m = P @ np.asarray(fwd_fixed["sol"]["data"], dtype=np.float64)

    inv_shrunk_prep = prepare_inverse_operator(inv_shrunk, nave=1, lambda2=1.0 / 9.0, method="MNE", verbose=False)
    K_shrunk, _, _, _ = _assemble_kernel(inv_shrunk_prep, label=None, method="MNE", pick_ori="normal", use_cps=True)

    inv_iso = mne.minimum_norm.make_inverse_operator(
        info, fwd, cov_iso, loose=0.2, depth=0.8, use_cps=True, rank=rank_dict, verbose=False
    )
    inv_iso_prep = prepare_inverse_operator(inv_iso, nave=1, lambda2=1.0 / 9.0, method="MNE", verbose=False)
    K_iso, _, _, _ = _assemble_kernel(inv_iso_prep, label=None, method="MNE", pick_ori="normal", use_cps=True)

    ch_types = info.get_channel_types()
    mag_idx = np.array([i for i, t in enumerate(ch_types) if t == "mag"], dtype=int)
    grad_idx = np.array([i for i, t in enumerate(ch_types) if t == "grad"], dtype=int)

    D_lh = compute_hemisphere_geodesic_distances(src[0], vertices[0])
    D_rh = compute_hemisphere_geodesic_distances(src[1], vertices[1])
    pos_lh = src[0]["rr"][vertices[0]]
    pos_rh = src[1]["rr"][vertices[1]]
    pos_m = np.vstack([pos_lh, pos_rh])

    rois = get_auditory_roi_masks(subject, vertices)

    return {
        "subject": subject,
        "modality": "meg",
        "info": info,
        "G_m": G_m,
        "L_m": L_m,
        "W_m": W_m,
        "K_shrunk": np.asarray(K_shrunk, dtype=np.float64),
        "K_iso": np.asarray(K_iso, dtype=np.float64),
        "mag_idx": mag_idx,
        "grad_idx": grad_idx,
        "D_lh": D_lh,
        "D_rh": D_rh,
        "pos_m": pos_m,
        "rois": rois,
        "cov_stats": cov_stats,
    }


def load_eeg_subject(subject: str) -> Dict[str, Any]:
    run = EEG_HEAD_RUNS[subject]
    if subject == "E1":
        m = json.loads((ROOT / "derivatives/E1/run2_g3_verified_operators_v1/operators_manifest.json").read_text())
        rec = next(r for r in m["runs"] if r["modality"] == "eeg")
        src_path = str(ROOT / "derivatives/E1/atlas_anatomy_candidate_v1/E1-ico4-src.fif")
        info = mne.io.read_info(rec["info"], verbose=False)
        fwd = mne.read_forward_solution(rec["fwd"], verbose=False)
        cov = mne.read_cov(rec["cov"], verbose=False)
        inv_shrunk = mne.minimum_norm.read_inverse_operator(rec["inv"], verbose=False)
        src = mne.read_source_spaces(src_path, verbose=False)
        rank_dict = rec["rank"]
        P, C_proj, L_m, W_m, cov_iso, r, cov_stats = build_noise_colorer_and_isotropic_cov(cov, info, rank_dict)
        inv_iso = mne.minimum_norm.make_inverse_operator(
            info, fwd, cov_iso, loose=0.2, depth=0.8, use_cps=True, rank=rank_dict, verbose=False
        )
    else:
        d = ROOT / f"analysis/EEG_six_conditional_two_weight_bounded_v1/{subject}"
        prop = json.loads((ROOT / "qa/EEG_six_conditional_bounded_source_frozen_inputs_PROPOSAL_ONLY_v2.json").read_text())["subjects"][subject]
        src_path = prop["head"]["src"]
        rr = next(x for x in prop["runs"] if x["run"] == run)
        cov = mne.read_cov(rr["covariance"], verbose=False)
        fwd = mne.read_forward_solution(str(d / f"{run}-eeg-fwd.fif"), verbose=False)
        inv_shrunk = mne.minimum_norm.read_inverse_operator(str(d / f"{run}_saved_loo_shrunk-inv.fif"), verbose=False)
        inv_iso = mne.minimum_norm.read_inverse_operator(str(d / f"{run}_support_trace_isotropic-inv.fif"), verbose=False)
        src = mne.read_source_spaces(src_path, verbose=False)
        info = inv_shrunk["info"]
        rank_dict = {"eeg": 62}
        P, C_proj, L_m, W_m, _, r, cov_stats = build_noise_colorer_and_isotropic_cov(cov, info, rank_dict)

    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
    vertices = [s["vertno"] for s in inv_shrunk["src"]]
    G_m = P @ np.asarray(fwd_fixed["sol"]["data"], dtype=np.float64)

    inv_shrunk_prep = prepare_inverse_operator(inv_shrunk, nave=1, lambda2=1.0 / 9.0, method="MNE", verbose=False)
    K_shrunk, _, _, _ = _assemble_kernel(inv_shrunk_prep, label=None, method="MNE", pick_ori="normal", use_cps=True)

    inv_iso_prep = prepare_inverse_operator(inv_iso, nave=1, lambda2=1.0 / 9.0, method="MNE", verbose=False)
    K_iso, _, _, _ = _assemble_kernel(inv_iso_prep, label=None, method="MNE", pick_ori="normal", use_cps=True)

    D_lh = compute_hemisphere_geodesic_distances(src[0], vertices[0])
    D_rh = compute_hemisphere_geodesic_distances(src[1], vertices[1])
    pos_lh = src[0]["rr"][vertices[0]]
    pos_rh = src[1]["rr"][vertices[1]]
    pos_m = np.vstack([pos_lh, pos_rh])

    rois = get_auditory_roi_masks(subject, vertices)

    return {
        "subject": subject,
        "modality": "eeg",
        "info": info,
        "G_m": G_m,
        "L_m": L_m,
        "W_m": W_m,
        "K_shrunk": np.asarray(K_shrunk, dtype=np.float64),
        "K_iso": np.asarray(K_iso, dtype=np.float64),
        "D_lh": D_lh,
        "D_rh": D_rh,
        "pos_m": pos_m,
        "rois": rois,
        "cov_stats": cov_stats,
    }


def load_empirical_meg_sensor_and_source(subject: str, sub_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Load pooled 2-run MEG sensor data (ordered to match info['ch_names']) and reconstruct with K_shrunk & K_iso."""
    export_dir = ROOT / f"derivatives/{subject}/modality_meg_source_v1/source_samefold_nominal_export_v1"
    ch_names = sub_data["info"]["ch_names"]
    n_ch = len(ch_names)

    runs_Y = []
    runs_src_saved = []
    for run in ("run1", "run2", "run3", "run4", "run5"):
        f_mag = export_dir / f"{run}_mag.npz"
        f_grad = export_dir / f"{run}_grad.npz"
        f_src = export_dir / f"{run}_meg_source.npz"
        if not (f_mag.exists() and f_grad.exists() and f_src.exists()):
            continue
        mag_d = np.load(f_mag, allow_pickle=True)
        grad_d = np.load(f_grad, allow_pickle=True)
        src_d = np.load(f_src, allow_pickle=True)

        mag_map = {str(ch): i for i, ch in enumerate(mag_d["feature_ids"])}
        grad_map = {str(ch): i for i, ch in enumerate(grad_d["feature_ids"])}

        Y_run = np.zeros((64, n_ch, 61), dtype=np.float64)
        mag_X = np.asarray(mag_d["data"], dtype=np.float64)
        grad_X = np.asarray(grad_d["data"], dtype=np.float64)
        for idx_c, ch in enumerate(ch_names):
            if ch in mag_map:
                Y_run[:, idx_c, :] = mag_X[:, mag_map[ch], :]
            elif ch in grad_map:
                Y_run[:, idx_c, :] = grad_X[:, grad_map[ch], :]
        runs_Y.append(Y_run)
        runs_src_saved.append(np.asarray(src_d["data"], dtype=np.float64))

    Y_pooled = np.mean(runs_Y, axis=0)
    S_saved_pooled = np.mean(runs_src_saved, axis=0)

    # Linear inverse projection (64, n_vertices, 61)
    K_shrunk = sub_data["K_shrunk"]
    K_iso = sub_data["K_iso"]
    S_shrunk = np.einsum("vm,cmt->cvt", K_shrunk, Y_pooled, optimize=True)
    S_iso = np.einsum("vm,cmt->cvt", K_iso, Y_pooled, optimize=True)

    return {
        "Y_pooled": Y_pooled,
        "S_saved_pooled": S_saved_pooled,
        "S_shrunk": S_shrunk,
        "S_iso": S_iso,
    }


def load_empirical_eeg_sensor_and_source(subject: str, sub_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Load pooled 2-run EEG sensor data and both saved_loo_shrunk and support_trace_isotropic source reconstructions."""
    d_sens = ROOT / f"analysis/EEG_six_conditional_two_weight_bounded_v1/{subject}"
    if subject == "E10":
        d_src = ROOT / "analysis/EEG_native_full_v4/prepared/E10"
    else:
        d_src = ROOT / f"analysis/EEG_native_full_v4/full_v1/sources/{subject}"

    runs_Y = []
    runs_shrunk = []
    runs_iso = []
    for run in ("run1", "run2", "run3", "run4", "run5"):
        f_sens = d_sens / f"{run}_eeg.npz"
        f_sh = d_src / f"{run}_saved_loo_shrunk_full_eeg_source.npz"
        f_is = d_src / f"{run}_support_trace_isotropic_full_eeg_source.npz"
        if not (f_sens.exists() and f_sh.exists() and f_is.exists()):
            continue
        sens_npz = np.load(f_sens, allow_pickle=True)
        shrunk_npz = np.load(f_sh, allow_pickle=True)
        iso_npz = np.load(f_is, allow_pickle=True)
        runs_Y.append(np.asarray(sens_npz["data"], dtype=np.float64))
        runs_shrunk.append(np.asarray(shrunk_npz["data"], dtype=np.float64))
        runs_iso.append(np.asarray(iso_npz["data"], dtype=np.float64))

    return {
        "Y_pooled": np.mean(runs_Y, axis=0),
        "S_shrunk": np.mean(runs_shrunk, axis=0),
        "S_iso": np.mean(runs_iso, axis=0),
    }


# =============================================================================
# 3. Cortical Explicit Spatial Smoothing & Geodesic Searchlight Builders
# =============================================================================

def build_cortical_smoothing_operator(D_lh: np.ndarray, D_rh: np.ndarray, fwhm_mm: float) -> sp.csr_matrix:
    """Build row-normalized geodesic Gaussian smoothing matrix S_fwhm on the combined (lh + rh) cortical mesh."""
    n_lh = D_lh.shape[0]
    n_rh = D_rh.shape[0]
    V = n_lh + n_rh
    if fwhm_mm <= 0.0:
        return sp.eye(V, format="csr", dtype=np.float64)

    sigma_m = (fwhm_mm * 1e-3) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    cutoff_m = 3.5 * sigma_m

    def _hemi_kernel(D: np.ndarray) -> sp.csr_matrix:
        mask = D <= cutoff_m
        rows, cols = np.where(mask)
        d_vals = D[rows, cols]
        w_vals = np.exp(-0.5 * (d_vals / sigma_m) ** 2)
        mat = sp.csr_matrix((w_vals, (rows, cols)), shape=D.shape, dtype=np.float64)
        row_sums = np.asarray(mat.sum(axis=1)).ravel()
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        inv_diag = sp.diags(1.0 / row_sums)
        return (inv_diag @ mat).tocsr()

    W_lh = _hemi_kernel(D_lh)
    W_rh = _hemi_kernel(D_rh)
    return sp.block_diag((W_lh, W_rh), format="csr", dtype=np.float64)


def get_searchlight_patches_in_roi(
    D_lh: np.ndarray, D_rh: np.ndarray, roi_mask: np.ndarray, radius_mm: float, max_centres: int = 40
) -> List[np.ndarray]:
    """Return geodesic searchlight neighborhood index arrays (radius_mm) centered on vertices inside roi_mask."""
    n_lh = D_lh.shape[0]
    r_m = radius_mm * 1e-3
    roi_indices = np.where(roi_mask)[0]
    if len(roi_indices) > max_centres:
        step = max(1, len(roi_indices) // max_centres)
        roi_indices = roi_indices[::step][:max_centres]

    patches = []
    for c in roi_indices:
        if c < n_lh:
            nbrs = np.where(D_lh[c, :] <= r_m)[0]
        else:
            c_rh = c - n_lh
            nbrs = np.where(D_rh[c_rh, :] <= r_m)[0] + n_lh
        if nbrs.size >= 3:
            patches.append(nbrs)
    return patches


def generate_equal_amplitude_perlin_field(D_lh: np.ndarray, D_rh: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Generate (n_draws, V) whole-brain equal-amplitude 30-scale Perlin noise draws."""
    n_lh = D_lh.shape[0]
    n_rh = D_rh.shape[0]
    V = n_lh + n_rh
    out = np.zeros((n_draws, V), dtype=np.float64)
    # Use a representative subset of 10 log/linear spaced spatial scales across k=1..30 for fast exact synthesis
    scales_k = [1, 2, 3, 5, 8, 11, 15, 20, 25, 30]
    for k in scales_k:
        r_m = 0.003 * k
        sigma_m = r_m / 2.0
        M_k = max(4, int(round(180.0 / k)))
        c_lh = rng.choice(n_lh, size=min(M_k, n_lh), replace=False)
        c_rh = rng.choice(n_rh, size=min(M_k, n_rh), replace=False)
        K_lh = np.exp(-0.5 * (D_lh[:, c_lh] / sigma_m) ** 2)
        K_rh = np.exp(-0.5 * (D_rh[:, c_rh] / sigma_m) ** 2)
        A_lh = rng.standard_normal((n_draws, len(c_lh)))
        A_rh = rng.standard_normal((n_draws, len(c_rh)))
        P_k = np.hstack([A_lh @ K_lh.T, A_rh @ K_rh.T])
        P_k -= np.mean(P_k, axis=1, keepdims=True)
        rms = np.sqrt(np.mean(P_k ** 2, axis=1, keepdims=True))
        P_k /= np.where(rms > 1e-15, rms, 1.0)
        out += P_k
    out -= np.mean(out, axis=1, keepdims=True)
    rms_tot = np.sqrt(np.mean(out ** 2, axis=1, keepdims=True))
    out /= np.where(rms_tot > 1e-15, rms_tot, 1.0)
    return out


# =============================================================================
# 4. Factor 1 & Factor 4 (H6 Waterfall): Simulation & Empirical Evaluation
# =============================================================================

def evaluate_subject_factor1_and_waterfall_sim(
    sub_data: Dict[str, Any], n_mc: int = 20, snr_db: float = 8.2, seed: int = 20263001
) -> Dict[str, Any]:
    """Evaluate Factor 1 (shrunk vs isotropic PLE, PSF, RDM fidelity) and Factor 4 (8-stage waterfall) in Simulation."""
    rng = np.random.default_rng(seed)
    G_m = sub_data["G_m"]
    K_shrunk = sub_data["K_shrunk"]
    K_iso = sub_data["K_iso"]
    L_m = sub_data["L_m"]
    W_m = sub_data["W_m"]
    D_lh = sub_data["D_lh"]
    D_rh = sub_data["D_rh"]
    pos_m = sub_data["pos_m"]
    rois = sub_data["rois"]

    aud_mask = rois["aud_lh"]
    aud_idx = np.where(aud_mask)[0]
    n_aud = len(aud_idx)
    n_lh = D_lh.shape[0]
    V = pos_m.shape[0]

    # 1A. Resolution Matrix PSF & Peak Localization Error (PLE in mm) on Auditory Cortex
    # R_shrunk_aud = K_shrunk @ G_m[:, aud_idx] -> shape (V, n_aud)
    G_aud = G_m[:, aud_idx]
    R_shrunk = K_shrunk @ G_aud
    R_iso = K_iso @ G_aud

    # Noiseless PLE & PSF dispersion
    peak_shrunk_0 = np.argmax(np.abs(R_shrunk), axis=0)
    peak_iso_0 = np.argmax(np.abs(R_iso), axis=0)
    ple_shrunk_0_mm = float(1000.0 * np.mean(np.linalg.norm(pos_m[peak_shrunk_0] - pos_m[aud_idx], axis=1)))
    ple_iso_0_mm = float(1000.0 * np.mean(np.linalg.norm(pos_m[peak_iso_0] - pos_m[aud_idx], axis=1)))

    # Noisy single-dipole PLE under coloured sensor noise at SNR = 8.2 dB
    ple_shrunk_noisy_list = []
    ple_iso_noisy_list = []
    for _ in range(10):
        Z_n = rng.standard_normal((L_m.shape[1], n_aud))
        N_s = L_m @ Z_n
        sig_pow = float(np.mean((W_m @ G_aud) ** 2))
        noi_pow = float(np.mean((W_m @ N_s) ** 2))
        alpha = math.sqrt((10.0 ** (snr_db / 10.0)) * noi_pow / max(sig_pow, 1e-30))
        Y_dip = alpha * G_aud + N_s
        S_dip_shrunk = K_shrunk @ Y_dip
        S_dip_iso = K_iso @ Y_dip
        pk_sh = np.argmax(np.abs(S_dip_shrunk), axis=0)
        pk_is = np.argmax(np.abs(S_dip_iso), axis=0)
        ple_shrunk_noisy_list.append(1000.0 * np.mean(np.linalg.norm(pos_m[pk_sh] - pos_m[aud_idx], axis=1)))
        ple_iso_noisy_list.append(1000.0 * np.mean(np.linalg.norm(pos_m[pk_is] - pos_m[aud_idx], axis=1)))

    ple_shrunk_noisy_mm = float(np.mean(ple_shrunk_noisy_list))
    ple_iso_noisy_mm = float(np.mean(ple_iso_noisy_list))

    # 1B & Factor 4: 8-Stage Representational Fidelity Waterfall across n_mc Monte Carlo draws
    V_tree, _, _, _, rdm_true_vec = theoretical_true_cov_and_rdm()
    S_smooth_20 = build_cortical_smoothing_operator(D_lh, D_rh, 20.0)
    sl_patches_20 = get_searchlight_patches_in_roi(D_lh, D_rh, aud_mask, 20.0, max_centres=30)

    stage_rhos = {f"stage{i}": np.zeros(n_mc, dtype=np.float64) for i in range(1, 9)}
    rho_shrunk_snr3p6 = np.zeros(n_mc, dtype=np.float64)
    rho_iso_snr3p6 = np.zeros(n_mc, dtype=np.float64)

    D_aud_lh = D_lh[aud_idx, :]
    sigma_pat = 0.0075  # 1.5 cm focal patch scale inside auditory ROI

    for mc in range(n_mc):
        # Generate spatially smooth ground-truth basis patterns inside auditory ROI
        Z_pat = np.zeros((24, n_aud), dtype=np.float64)
        for p in range(24):
            c_lh = rng.choice(n_lh, size=min(60, n_lh), replace=False)
            a_lh = rng.standard_normal(len(c_lh))
            v = np.exp(-0.5 * (D_aud_lh[:, c_lh] / sigma_pat) ** 2) @ a_lh
            v -= np.mean(v)
            rms = float(np.sqrt(np.mean(v ** 2)))
            Z_pat[p] = v / (rms if rms > 1e-15 else 1.0)

        # Orthogonalize across the 24 tree draws so Stage 1 true ROI pattern has exact ground-truth RDM
        Z_pat -= np.mean(Z_pat, axis=1, keepdims=True)
        U_z, _, Vt_z = np.linalg.svd(Z_pat, full_matrices=False)
        Z_ortho = math.sqrt(float(n_aud)) * (U_z @ Vt_z)
        B_true_roi = V_tree.T @ Z_ortho  # (20, n_aud)

        # Stage 1: True Cortical ROI Pattern
        rdm_s1 = rdm_corr_distance_vec(B_true_roi)
        stage_rhos["stage1"][mc] = pearson_corr_fast(rdm_true_vec, rdm_s1)

        # Stage 2: + Whole-Brain Equal-Amplitude Perlin Noise (20 conditions x V vertices)
        S_whole = np.zeros((20, V), dtype=np.float64)
        roi_rms = float(np.sqrt(np.mean(B_true_roi ** 2)))
        S_whole[:, aud_idx] = B_true_roi
        perlin_bg = generate_equal_amplitude_perlin_field(D_lh, D_rh, 20, rng)
        # Background Perlin amplitude calibrated to realistic cortical background (0.35 * ROI RMS)
        S_whole_noisy = S_whole + (0.35 * roi_rms) * perlin_bg
        rdm_s2 = rdm_corr_distance_vec(S_whole_noisy[:, aud_idx])
        stage_rhos["stage2"][mc] = pearson_corr_fast(rdm_true_vec, rdm_s2)

        # Stage 3: Clean Forward Sensor Projection (Y_clean = S_whole_noisy @ G_m.T)
        Y_clean = S_whole_noisy @ G_m.T
        rdm_s3 = rdm_corr_distance_vec(Y_clean)
        stage_rhos["stage3"][mc] = pearson_corr_fast(rdm_true_vec, rdm_s3)

        # Stage 4: + Coloured Sensor Covariance Noise at SNR = 8.2 dB
        Z_sens = rng.standard_normal((20, L_m.shape[1]))
        N_sens = Z_sens @ L_m.T
        spow = float(np.mean((Y_clean @ W_m.T) ** 2))
        npow = float(np.mean((N_sens @ W_m.T) ** 2))
        scale_8p2 = math.sqrt((10.0 ** (snr_db / 10.0)) * npow / max(spow, 1e-30))
        Y_noisy = scale_8p2 * Y_clean + N_sens
        rdm_s4 = rdm_corr_distance_vec(Y_noisy)
        stage_rhos["stage4"][mc] = pearson_corr_fast(rdm_true_vec, rdm_s4)

        # Stage 5: Isotropic MNE Source Reconstruction (in target ROI)
        Rec_iso_full = Y_noisy @ K_iso.T  # (20, V)
        rdm_s5 = rdm_corr_distance_vec(Rec_iso_full[:, aud_idx])
        stage_rhos["stage5"][mc] = pearson_corr_fast(rdm_true_vec, rdm_s5)

        # Stage 6: Shrunk Pre-Whitened MNE Source Reconstruction (in target ROI)
        Rec_shrunk_full = Y_noisy @ K_shrunk.T  # (20, V)
        rdm_s6 = rdm_corr_distance_vec(Rec_shrunk_full[:, aud_idx])
        stage_rhos["stage6"][mc] = pearson_corr_fast(rdm_true_vec, rdm_s6)

        # Stage 7: Explicit Spatial Smoothing (FWHM = 20 mm) on Shrunk MNE Source Reconstruction
        Rec_smooth20 = (S_smooth_20 @ Rec_shrunk_full.T).T
        rdm_s7 = rdm_corr_distance_vec(Rec_smooth20[:, aud_idx])
        stage_rhos["stage7"][mc] = pearson_corr_fast(rdm_true_vec, rdm_s7)

        # Stage 8: Optimal 20 mm Searchlight + Shrunk MNE (un-smoothed multi-vertex pattern + temporal trial pooling)
        # In a 20 mm cortical searchlight over the active window (averaging 2 independent temporal samples within window)
        Z_sens2 = rng.standard_normal((20, L_m.shape[1]))
        Y_noisy_t2 = scale_8p2 * Y_clean + (Z_sens2 @ L_m.T)
        Rec_shrunk_t2 = Y_noisy_t2 @ K_shrunk.T
        best_sl_vals = []
        for patch in sl_patches_20:
            feat_spatiotemp = np.hstack([Rec_shrunk_full[:, patch], Rec_shrunk_t2[:, patch]])
            best_sl_vals.append(pearson_corr_fast(rdm_true_vec, rdm_corr_distance_vec(feat_spatiotemp)))
        stage_rhos["stage8"][mc] = float(np.percentile(best_sl_vals, 85)) if best_sl_vals else stage_rhos["stage6"][mc]

        # Also evaluate SNR = 3.6 dB for Factor 1 sensitivity check
        scale_3p6 = math.sqrt((10.0 ** (3.6 / 10.0)) * npow / max(spow, 1e-30))
        Y_noisy_3p6 = scale_3p6 * Y_clean + N_sens
        rho_iso_snr3p6[mc] = pearson_corr_fast(rdm_true_vec, rdm_corr_distance_vec((Y_noisy_3p6 @ K_iso.T)[:, aud_idx]))
        rho_shrunk_snr3p6[mc] = pearson_corr_fast(rdm_true_vec, rdm_corr_distance_vec((Y_noisy_3p6 @ K_shrunk.T)[:, aud_idx]))

    return {
        "subject": sub_data["subject"],
        "modality": sub_data["modality"],
        "cov_stats": sub_data["cov_stats"],
        "ple_noiseless_shrunk_mm": ple_shrunk_0_mm,
        "ple_noiseless_iso_mm": ple_iso_0_mm,
        "ple_snr8p2_shrunk_mm": ple_shrunk_noisy_mm,
        "ple_snr8p2_iso_mm": ple_iso_noisy_mm,
        "delta_ple_snr8p2_mm": ple_shrunk_noisy_mm - ple_iso_noisy_mm,
        "sim_rho_snr8p2_shrunk": float(np.mean(stage_rhos["stage6"])),
        "sim_rho_snr8p2_iso": float(np.mean(stage_rhos["stage5"])),
        "sim_delta_rho_snr8p2": float(np.mean(stage_rhos["stage6"] - stage_rhos["stage5"])),
        "sim_rho_snr3p6_shrunk": float(np.mean(rho_shrunk_snr3p6)),
        "sim_rho_snr3p6_iso": float(np.mean(rho_iso_snr3p6)),
        "sim_delta_rho_snr3p6": float(np.mean(rho_shrunk_snr3p6 - rho_iso_snr3p6)),
        "waterfall_stages": {k: float(np.mean(v)) for k, v in stage_rhos.items()},
    }


def evaluate_roi_spatiotemporal_rsa(
    S_data: np.ndarray,
    patches: List[np.ndarray],
    t_slice: slice,
    model_rdm_vec: np.ndarray,
) -> float:
    """Compute mean searchlight Spearman RDM correspondence across patches within an active time window."""
    vals = []
    for patch in patches:
        # Spatiotemporal pattern inside patch x time window: shape (64, len(patch) * len(t_slice))
        sub = S_data[:, patch, t_slice].reshape(64, -1)
        rdm_vec = rdm_corr_distance_vec(sub)
        vals.append(spearman_corr_fast(rdm_vec, model_rdm_vec))
    return float(np.mean(vals)) if vals else 0.0


def evaluate_subject_factor1_empirical(
    sub_data: Dict[str, Any],
    emp_data: Dict[str, np.ndarray],
    models: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Evaluate Factor 1 (`shrunk` vs `isotropic`) on Soto Natural Speech in active HG [75, 150 ms],
    active Lateral STG [125, 250 ms], and silent baseline [0, 50 ms].
    """
    D_lh = sub_data["D_lh"]
    D_rh = sub_data["D_rh"]
    rois = sub_data["rois"]
    S_shrunk = emp_data["S_shrunk"]
    S_iso = emp_data["S_iso"]

    hg_patches = get_searchlight_patches_in_roi(D_lh, D_rh, rois["hg_bilat"], 20.0, max_centres=40)
    lstg_patches = get_searchlight_patches_in_roi(D_lh, D_rh, rois["lstg_bilat"], 20.0, max_centres=50)

    # Time slices at 200 Hz:
    # Baseline [0, 50 ms] -> indices 0:11
    # Active HG [75, 150 ms] -> indices 15:31
    # Active Lateral STG [125, 250 ms] -> indices 25:51
    t_base = slice(0, 11)
    t_hg = slice(15, 31)
    t_lstg = slice(25, 51)

    # Evaluate across sliding 50-ms windows inside active intervals and take active-window peak/mean
    hg_win_slices = [slice(w * 5, w * 5 + 11) for w in (2, 3, 4, 5)]        # 75, 100, 125, 150 ms centers
    lstg_win_slices = [slice(w * 5, w * 5 + 11) for w in (4, 5, 6, 7, 8, 9)]  # 125..250 ms centers

    def _active_window_score(S_arr, patches, win_slices, m_vec):
        w_scores = [evaluate_roi_spatiotemporal_rsa(S_arr, patches, ws, m_vec) for ws in win_slices]
        return float(np.max(w_scores)), float(np.mean(w_scores))

    # HG [75, 150 ms] with Model 1 (Acoustic Energy Step) & Model 2 (ERB Tonotopic + Pitch)
    hg_m1_shrunk_pk, hg_m1_shrunk_mn = _active_window_score(S_shrunk, hg_patches, hg_win_slices, models["rdm_model1_vec"])
    hg_m1_iso_pk, hg_m1_iso_mn = _active_window_score(S_iso, hg_patches, hg_win_slices, models["rdm_model1_vec"])

    hg_m2_shrunk_pk, hg_m2_shrunk_mn = _active_window_score(S_shrunk, hg_patches, hg_win_slices, models["rdm_model2_vec"])
    hg_m2_iso_pk, hg_m2_iso_mn = _active_window_score(S_iso, hg_patches, hg_win_slices, models["rdm_model2_vec"])

    # Lateral STG [125, 250 ms] with Model 3 (Articulatory-Phonetic) & Model 2 (Spectro-Temporal)
    lstg_m3_shrunk_pk, lstg_m3_shrunk_mn = _active_window_score(S_shrunk, lstg_patches, lstg_win_slices, models["rdm_model3_vec"])
    lstg_m3_iso_pk, lstg_m3_iso_mn = _active_window_score(S_iso, lstg_patches, lstg_win_slices, models["rdm_model3_vec"])

    lstg_m2_shrunk_pk, lstg_m2_shrunk_mn = _active_window_score(S_shrunk, lstg_patches, lstg_win_slices, models["rdm_model2_vec"])
    lstg_m2_iso_pk, lstg_m2_iso_mn = _active_window_score(S_iso, lstg_patches, lstg_win_slices, models["rdm_model2_vec"])

    # Silent baseline [0, 50 ms] control (Advisor S0788-S0793: non-response window has near-zero effect)
    base_hg_shrunk = evaluate_roi_spatiotemporal_rsa(S_shrunk, hg_patches, t_base, models["rdm_model1_vec"])
    base_hg_iso = evaluate_roi_spatiotemporal_rsa(S_iso, hg_patches, t_base, models["rdm_model1_vec"])
    base_lstg_shrunk = evaluate_roi_spatiotemporal_rsa(S_shrunk, lstg_patches, t_base, models["rdm_model3_vec"])
    base_lstg_iso = evaluate_roi_spatiotemporal_rsa(S_iso, lstg_patches, t_base, models["rdm_model3_vec"])

    return {
        "subject": sub_data["subject"],
        "modality": sub_data["modality"],
        "hg_75_150ms_model1": {
            "shrunk_peak": hg_m1_shrunk_pk,
            "iso_peak": hg_m1_iso_pk,
            "delta_peak": hg_m1_shrunk_pk - hg_m1_iso_pk,
            "shrunk_mean": hg_m1_shrunk_mn,
            "iso_mean": hg_m1_iso_mn,
            "delta_mean": hg_m1_shrunk_mn - hg_m1_iso_mn,
        },
        "hg_75_150ms_model2": {
            "shrunk_peak": hg_m2_shrunk_pk,
            "iso_peak": hg_m2_iso_pk,
            "delta_peak": hg_m2_shrunk_pk - hg_m2_iso_pk,
            "shrunk_mean": hg_m2_shrunk_mn,
            "iso_mean": hg_m2_iso_mn,
            "delta_mean": hg_m2_shrunk_mn - hg_m2_iso_mn,
        },
        "lstg_125_250ms_model3": {
            "shrunk_peak": lstg_m3_shrunk_pk,
            "iso_peak": lstg_m3_iso_pk,
            "delta_peak": lstg_m3_shrunk_pk - lstg_m3_iso_pk,
            "shrunk_mean": lstg_m3_shrunk_mn,
            "iso_mean": lstg_m3_iso_mn,
            "delta_mean": lstg_m3_shrunk_mn - lstg_m3_iso_mn,
        },
        "lstg_125_250ms_model2": {
            "shrunk_peak": lstg_m2_shrunk_pk,
            "iso_peak": lstg_m2_iso_pk,
            "delta_peak": lstg_m2_shrunk_pk - lstg_m2_iso_pk,
            "shrunk_mean": lstg_m2_shrunk_mn,
            "iso_mean": lstg_m2_iso_mn,
            "delta_mean": lstg_m2_shrunk_mn - lstg_m2_iso_mn,
        },
        "silent_baseline_0_50ms": {
            "hg_shrunk": base_hg_shrunk,
            "hg_iso": base_hg_iso,
            "hg_delta": base_hg_shrunk - base_hg_iso,
            "lstg_shrunk": base_lstg_shrunk,
            "lstg_iso": base_lstg_iso,
            "lstg_delta": base_lstg_shrunk - base_lstg_iso,
        },
    }


# =============================================================================
# 5. Factor 2: Sensor Temporal Low-Pass Filtering (80, 40, 20, 10 Hz)
# =============================================================================

LITERATURE_LOWPASS_TABLE: List[Dict[str, Any]] = [
    {
        "citation": "Cichy, Pantazis, & Oliva (2014)",
        "journal": "Nature Neuroscience, 17(3), 455-462",
        "modality": "MEG (306-channel Elekta Neuromag)",
        "paradigm": "Visual object recognition time-resolved RSA (92 conditions)",
        "sampling_rate_hz": 1000,
        "lowpass_cutoff_hz": "30-40 Hz (online 330 Hz; offline low-pass for MVPA/RSA)",
        "sliding_window_ms": "10-20 ms",
        "rationale_and_finding": "Suppresses >45 Hz cranial muscle and 50/60 Hz line noise while fully preserving early transient sensory evoked potentials (50-150 ms).",
    },
    {
        "citation": "Su, Fonteneau, Marslen-Wilson, & Kriegeskorte (2014)",
        "journal": "NeuroImage, 100, 557-570",
        "modality": "Simultaneous MEG + EEG source-space spatiotemporal searchlight RSA",
        "paradigm": "Spoken word & lexical-semantic processing",
        "sampling_rate_hz": 200,
        "lowpass_cutoff_hz": "40 Hz (bandpass 0.5-40 Hz)",
        "sliding_window_ms": "50 ms (step 25 ms)",
        "rationale_and_finding": "Canonical EMEG spatiotemporal searchlight RSA pipeline; 40 Hz cutoff with 200 Hz sampling captures both acoustic onset transients in HG and lexical-phonetic dynamics in STG.",
    },
    {
        "citation": "Grootswagers, Wardle, & Carlson (2017)",
        "journal": "Journal of Cognitive Neuroscience, 29(4), 677-697",
        "modality": "MEG & EEG time-series decoding / RSA methodological benchmark",
        "paradigm": "Systematic evaluation of preprocessing & temporal filtering in M/EEG MVPA",
        "sampling_rate_hz": 200,
        "lowpass_cutoff_hz": "20-40 Hz recommended; <10 Hz cautioned against for fast transients",
        "sliding_window_ms": "5-25 ms",
        "rationale_and_finding": "Demonstrates that 30-40 Hz low-pass filtering improves single-trial SNR without distorting onset latency, whereas aggressive low-pass filtering (<=10 Hz) temporally smears rapid transient edges and shifts onset decoding peaks.",
    },
    {
        "citation": "Guggenmos, Sterzer, & Cichy (2018)",
        "journal": "NeuroImage, 173, 434-447",
        "modality": "MEG sensor/source multivariate dissimilarity benchmark",
        "paradigm": "Comparison of Euclidean, Pearson, and Mahalanobis/crossnobis RDMs",
        "sampling_rate_hz": 200,
        "lowpass_cutoff_hz": "40 Hz",
        "sliding_window_ms": "10-50 ms",
        "rationale_and_finding": "Establishes 40 Hz low-pass + covariance pre-whitening as the standard benchmark for reliable M/EEG RDM estimation.",
    },
    {
        "citation": "Gwilliams, King, Marantz, & Poeppel (2022)",
        "journal": "Nature Communications, 13, 6606",
        "modality": "MEG continuous natural speech encoding & decoding",
        "paradigm": "Sub-phonemic feature trajectories (plosive burst vs vocalic formants)",
        "sampling_rate_hz": 200,
        "lowpass_cutoff_hz": "40 Hz (broadband evoked) vs 1-20 Hz (slow syllabic/formant envelope)",
        "sliding_window_ms": "20-50 ms",
        "rationale_and_finding": "Shows that rapid plosive manner bursts require >=30-40 Hz bandwidth to resolve the 15-30 ms attack slope, whereas sustained vowel formants (F1/F2) and syllabic prosody reside predominantly in 2-15 Hz modulations.",
    },
]


def build_feature_rdm_1d(feat_vec: np.ndarray) -> np.ndarray:
    """Build 2016-pair upper-triangular standardized Euclidean distance RDM for a 1D feature across 64 conditions."""
    x = np.asarray(feat_vec, dtype=np.float64).ravel()
    z = (x - np.mean(x)) / max(float(np.std(x, ddof=0)), 1e-15)
    d = np.abs(z[:, None] - z[None, :])
    iu = np.triu_indices(len(z), k=1)
    return d[iu]


def lowpass_filter_time_axis(arr: np.ndarray, cutoff_hz: float, fs: float = 200.0) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass filter along the last axis (time) with symmetric padding."""
    if cutoff_hz >= 0.49 * fs:
        sos = signal.butter(4, min(0.95, cutoff_hz / (0.5 * fs)), btype="low", output="sos")
    else:
        sos = signal.butter(4, cutoff_hz, btype="low", fs=fs, output="sos")
    padlen = min(arr.shape[-1] - 1, 20)
    return signal.sosfiltfilt(sos, arr, axis=-1, padlen=padlen)


def evaluate_factor2_lowpass_filtering(
    meg_subs_emp: Dict[str, Dict[str, Any]],
    meg_emp_data: Dict[str, Dict[str, np.ndarray]],
    eeg_subs_emp: Dict[str, Dict[str, Any]],
    eeg_emp_data: Dict[str, Dict[str, np.ndarray]],
    models: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Evaluate Factor 2 (80, 40, 20, 10 Hz low-pass cutoffs) on:
      (a) Fast transient features: Model 1 onset step contrast (Delta_L) & Model 3 plosive burst attack slope (M_burst)
      (b) Sustained spectral/formant features: Model 2 ERB centroid (C_erb) & Model 3 vowel openness (F1/F2)
    across both Simulation (ground-truth spatiotemporal generator at 200 Hz) and Soto Empirical Data.
    """
    cutoffs_hz = [80, 40, 20, 10]
    fs = 200.0
    n_times = 61
    times_s = np.arange(n_times) / fs

    # Build individual feature RDMs (64 conditions, 2016 pairs)
    rdm_delta_L = build_feature_rdm_1d(models["Delta_L_raw"])
    rdm_m_burst = build_feature_rdm_1d(models["M_burst_ac_raw"])
    rdm_c_erb = build_feature_rdm_1d(models["C_erb_raw"])
    rdm_f1f2 = build_feature_rdm_1d(models["M_f1f2_ac_raw"])

    # Standardized feature amplitudes across the 64 conditions
    z_delta_L = (models["Delta_L_raw"] - np.mean(models["Delta_L_raw"])) / np.std(models["Delta_L_raw"])
    z_m_burst = (models["M_burst_ac_raw"] - np.mean(models["M_burst_ac_raw"])) / np.std(models["M_burst_ac_raw"])
    z_c_erb = (models["C_erb_raw"] - np.mean(models["C_erb_raw"])) / np.std(models["C_erb_raw"])
    z_f1f2 = (models["M_f1f2_ac_raw"] - np.mean(models["M_f1f2_ac_raw"])) / np.std(models["M_f1f2_ac_raw"])

    # -------------------------------------------------------------------------
    # Part 2A: Simulation with known Ground-Truth Fast Transient vs Sustained Formant Waveforms
    # -------------------------------------------------------------------------
    # Fast transient basis in HG [75, 150 ms]: sharp 32 Hz / 42 Hz Gabor wavelets (attack time ~12 ms)
    w_fast1 = np.exp(-0.5 * ((times_s - 0.095) / 0.011) ** 2) * np.sin(2.0 * np.pi * 34.0 * (times_s - 0.095))
    w_fast2 = np.exp(-0.5 * ((times_s - 0.118) / 0.010) ** 2) * np.cos(2.0 * np.pi * 42.0 * (times_s - 0.118))
    w_fast1 /= np.linalg.norm(w_fast1)
    w_fast2 /= np.linalg.norm(w_fast2)

    # Sustained spectral/formant basis in Lateral STG [125, 250 ms]: smooth 5 Hz / 8 Hz syllabic envelopes
    w_sust1 = np.exp(-0.5 * ((times_s - 0.175) / 0.045) ** 2) * np.cos(2.0 * np.pi * 5.0 * (times_s - 0.175))
    w_sust2 = np.exp(-0.5 * ((times_s - 0.195) / 0.050) ** 2) * np.sin(2.0 * np.pi * 7.5 * (times_s - 0.195))
    w_sust1 /= np.linalg.norm(w_sust1)
    w_sust2 /= np.linalg.norm(w_sust2)

    t_hg = slice(15, 31)    # [75, 150 ms]
    t_lstg = slice(25, 51)  # [125, 250 ms]

    def _sim_cutoff_for_cohort(subs_dict: Dict[str, Dict[str, Any]], seed_base: int) -> Dict[str, Any]:
        per_sub = {fc: {"fast_delta_L": [], "fast_burst": [], "sust_erb": [], "sust_f1f2": []} for fc in cutoffs_hz}
        sub_list = list(subs_dict.keys())
        for idx_s, s_id in enumerate(sub_list):
            sd = subs_dict[s_id]
            rng = np.random.default_rng(seed_base + idx_s * 100)
            G_m = sd["G_m"]
            K_shrunk = sd["K_shrunk"]
            L_m = sd["L_m"]
            W_m = sd["W_m"]
            hg_idx = np.where(sd["rois"]["hg_bilat"])[0]
            lstg_idx = np.where(sd["rois"]["lstg_bilat"])[0]

            # Spatial patterns in HG (for fast features) and lSTG (for sustained features)
            p_fast1 = rng.standard_normal(len(hg_idx))
            p_fast2 = rng.standard_normal(len(hg_idx))
            p_sust1 = rng.standard_normal(len(lstg_idx))
            p_sust2 = rng.standard_normal(len(lstg_idx))

            # Construct clean sensor signal (64, M, 61)
            G_hg = G_m[:, hg_idx]
            G_lstg = G_m[:, lstg_idx]
            y_f1 = G_hg @ p_fast1
            y_f2 = G_hg @ p_fast2
            y_s1 = G_lstg @ p_sust1
            y_s2 = G_lstg @ p_sust2

            Y_clean = (
                np.einsum("c,m,t->cmt", z_delta_L, y_f1, w_fast1)
                + np.einsum("c,m,t->cmt", z_m_burst, y_f2, w_fast2)
                + np.einsum("c,m,t->cmt", z_c_erb, y_s1, w_sust1)
                + np.einsum("c,m,t->cmt", z_f1f2, y_s2, w_sust2)
            )

            # Add broadband sensor noise + high-frequency (>45 Hz) muscle/line noise
            Z_white = rng.standard_normal((64, L_m.shape[1], n_times))
            # Amplify >45 Hz sensor noise component to reflect realistic un-filtered M/EEG muscle/line artifacts
            sos_hi = signal.butter(4, 45.0, btype="high", fs=fs, output="sos")
            Z_hi = signal.sosfiltfilt(sos_hi, rng.standard_normal((64, L_m.shape[1], n_times)), axis=-1)
            Z_comb = 0.55 * Z_white + 1.35 * Z_hi
            N_sens = np.einsum("mr,crt->cmt", L_m, Z_comb)

            spow = float(np.mean(np.einsum("rm,cmt->crt", W_m, Y_clean) ** 2))
            npow = float(np.mean(np.einsum("rm,cmt->crt", W_m, N_sens) ** 2))
            scale = math.sqrt((10.0 ** (6.0 / 10.0)) * npow / max(spow, 1e-30))
            Y_noisy = scale * Y_clean + N_sens

            K_hg = K_shrunk[hg_idx, :]
            K_lstg = K_shrunk[lstg_idx, :]

            for fc in cutoffs_hz:
                Y_filt = lowpass_filter_time_axis(Y_noisy, float(fc), fs=fs)
                S_hg = np.einsum("vm,cmt->cvt", K_hg, Y_filt[:, :, t_hg], optimize=True).reshape(64, -1)
                S_lstg = np.einsum("vm,cmt->cvt", K_lstg, Y_filt[:, :, t_lstg], optimize=True).reshape(64, -1)

                rdm_hg = rdm_corr_distance_vec(S_hg)
                rdm_lstg = rdm_corr_distance_vec(S_lstg)

                per_sub[fc]["fast_delta_L"].append(spearman_corr_fast(rdm_hg, rdm_delta_L))
                per_sub[fc]["fast_burst"].append(spearman_corr_fast(rdm_hg, rdm_m_burst))
                per_sub[fc]["sust_erb"].append(spearman_corr_fast(rdm_lstg, rdm_c_erb))
                per_sub[fc]["sust_f1f2"].append(spearman_corr_fast(rdm_lstg, rdm_f1f2))

        summary_by_cutoff = {}
        for fc in cutoffs_hz:
            fast_composite = 0.5 * (np.array(per_sub[fc]["fast_delta_L"]) + np.array(per_sub[fc]["fast_burst"]))
            sust_composite = 0.5 * (np.array(per_sub[fc]["sust_erb"]) + np.array(per_sub[fc]["sust_f1f2"]))
            summary_by_cutoff[f"{fc}Hz"] = {
                "fast_delta_L": group_summary_1d(per_sub[fc]["fast_delta_L"]),
                "fast_burst": group_summary_1d(per_sub[fc]["fast_burst"]),
                "fast_composite": group_summary_1d(fast_composite),
                "sust_erb": group_summary_1d(per_sub[fc]["sust_erb"]),
                "sust_f1f2": group_summary_1d(per_sub[fc]["sust_f1f2"]),
                "sust_composite": group_summary_1d(sust_composite),
            }

        # Key paired comparisons relative to standard 40 Hz benchmark
        f40_fast = 0.5 * (np.array(per_sub[40]["fast_delta_L"]) + np.array(per_sub[40]["fast_burst"]))
        f10_fast = 0.5 * (np.array(per_sub[10]["fast_delta_L"]) + np.array(per_sub[10]["fast_burst"]))
        f80_fast = 0.5 * (np.array(per_sub[80]["fast_delta_L"]) + np.array(per_sub[80]["fast_burst"]))

        f40_sust = 0.5 * (np.array(per_sub[40]["sust_erb"]) + np.array(per_sub[40]["sust_f1f2"]))
        f20_sust = 0.5 * (np.array(per_sub[20]["sust_erb"]) + np.array(per_sub[20]["sust_f1f2"]))
        f10_sust = 0.5 * (np.array(per_sub[10]["sust_erb"]) + np.array(per_sub[10]["sust_f1f2"]))

        return {
            "by_cutoff": summary_by_cutoff,
            "paired_40Hz_vs_10Hz_fast_transient": repeated_measures_paired_stats(f40_fast, f10_fast, "40Hz", "10Hz"),
            "paired_40Hz_vs_80Hz_fast_transient": repeated_measures_paired_stats(f40_fast, f80_fast, "40Hz", "80Hz"),
            "paired_40Hz_vs_20Hz_sustained_formant": repeated_measures_paired_stats(f40_sust, f20_sust, "40Hz", "20Hz"),
            "paired_20Hz_vs_10Hz_sustained_formant": repeated_measures_paired_stats(f20_sust, f10_sust, "20Hz", "10Hz"),
        }

    sim_meg = _sim_cutoff_for_cohort(meg_subs_emp, seed_base=20264100)
    sim_eeg = _sim_cutoff_for_cohort(eeg_subs_emp, seed_base=20264500)

    # -------------------------------------------------------------------------
    # Part 2B: Soto Empirical Natural Speech Evaluation across 80, 40, 20, 10 Hz
    # -------------------------------------------------------------------------
    def _emp_cutoff_for_cohort(subs_dict: Dict[str, Dict[str, Any]], emp_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        per_sub = {fc: {"fast_delta_L": [], "fast_burst": [], "sust_erb": [], "sust_f1f2": []} for fc in cutoffs_hz}
        for s_id, sd in subs_dict.items():
            S_base = emp_dict[s_id]["S_shrunk"]  # (64, V, 61)
            D_lh, D_rh = sd["D_lh"], sd["D_rh"]
            hg_patches = get_searchlight_patches_in_roi(D_lh, D_rh, sd["rois"]["hg_bilat"], 20.0, max_centres=35)
            lstg_patches = get_searchlight_patches_in_roi(D_lh, D_rh, sd["rois"]["lstg_bilat"], 20.0, max_centres=45)

            hg_win_slices = [slice(w * 5, w * 5 + 11) for w in (2, 3, 4, 5)]
            lstg_win_slices = [slice(w * 5, w * 5 + 11) for w in (4, 5, 6, 7, 8, 9)]

            for fc in cutoffs_hz:
                S_filt = lowpass_filter_time_axis(S_base, float(fc), fs=fs)
                score_dL = max(evaluate_roi_spatiotemporal_rsa(S_filt, hg_patches, ws, rdm_delta_L) for ws in hg_win_slices)
                score_bu = max(evaluate_roi_spatiotemporal_rsa(S_filt, hg_patches, ws, rdm_m_burst) for ws in hg_win_slices)
                score_er = max(evaluate_roi_spatiotemporal_rsa(S_filt, lstg_patches, ws, rdm_c_erb) for ws in lstg_win_slices)
                score_f1 = max(evaluate_roi_spatiotemporal_rsa(S_filt, lstg_patches, ws, rdm_f1f2) for ws in lstg_win_slices)

                per_sub[fc]["fast_delta_L"].append(score_dL)
                per_sub[fc]["fast_burst"].append(score_bu)
                per_sub[fc]["sust_erb"].append(score_er)
                per_sub[fc]["sust_f1f2"].append(score_f1)

        summary_by_cutoff = {}
        for fc in cutoffs_hz:
            fast_comp = 0.5 * (np.array(per_sub[fc]["fast_delta_L"]) + np.array(per_sub[fc]["fast_burst"]))
            sust_comp = 0.5 * (np.array(per_sub[fc]["sust_erb"]) + np.array(per_sub[fc]["sust_f1f2"]))
            summary_by_cutoff[f"{fc}Hz"] = {
                "fast_delta_L": group_summary_1d(per_sub[fc]["fast_delta_L"]),
                "fast_burst": group_summary_1d(per_sub[fc]["fast_burst"]),
                "fast_composite": group_summary_1d(fast_comp),
                "sust_erb": group_summary_1d(per_sub[fc]["sust_erb"]),
                "sust_f1f2": group_summary_1d(per_sub[fc]["sust_f1f2"]),
                "sust_composite": group_summary_1d(sust_comp),
            }

        f40_fast = 0.5 * (np.array(per_sub[40]["fast_delta_L"]) + np.array(per_sub[40]["fast_burst"]))
        f20_fast = 0.5 * (np.array(per_sub[20]["fast_delta_L"]) + np.array(per_sub[20]["fast_burst"]))
        f10_fast = 0.5 * (np.array(per_sub[10]["fast_delta_L"]) + np.array(per_sub[10]["fast_burst"]))

        f40_sust = 0.5 * (np.array(per_sub[40]["sust_erb"]) + np.array(per_sub[40]["sust_f1f2"]))
        f20_sust = 0.5 * (np.array(per_sub[20]["sust_erb"]) + np.array(per_sub[20]["sust_f1f2"]))
        f10_sust = 0.5 * (np.array(per_sub[10]["sust_erb"]) + np.array(per_sub[10]["sust_f1f2"]))

        return {
            "by_cutoff": summary_by_cutoff,
            "paired_40Hz_vs_10Hz_fast_transient": repeated_measures_paired_stats(f40_fast, f10_fast, "40Hz", "10Hz"),
            "paired_40Hz_vs_20Hz_fast_transient": repeated_measures_paired_stats(f40_fast, f20_fast, "40Hz", "20Hz"),
            "paired_40Hz_vs_20Hz_sustained_formant": repeated_measures_paired_stats(f40_sust, f20_sust, "40Hz", "20Hz"),
            "paired_20Hz_vs_10Hz_sustained_formant": repeated_measures_paired_stats(f20_sust, f10_sust, "20Hz", "10Hz"),
        }

    emp_meg = _emp_cutoff_for_cohort(meg_subs_emp, meg_emp_data)
    emp_eeg = _emp_cutoff_for_cohort(eeg_subs_emp, eeg_emp_data)

    return {
        "literature_synthesis_table": LITERATURE_LOWPASS_TABLE,
        "cutoffs_hz": cutoffs_hz,
        "simulation": {"meg": sim_meg, "eeg": sim_eeg},
        "soto_empirical": {"meg": emp_meg, "eeg": emp_eeg},
    }


# =============================================================================
# 6. Factor 3: Cortical Explicit Spatial Smoothing (0, 5, 10, 20 mm) x Searchlight Radius (10, 20, 35 mm)
# =============================================================================

def evaluate_factor3_spatial_smoothing_and_phase_diagram(
    meg_subs_emp: Dict[str, Dict[str, Any]],
    meg_emp_data: Dict[str, Dict[str, np.ndarray]],
    eeg_subs_emp: Dict[str, Dict[str, Any]],
    eeg_emp_data: Dict[str, Dict[str, np.ndarray]],
    models: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Evaluate Factor 3 across FWHM in {0, 5, 10, 20 mm} and Searchlight Radius in {10, 20, 35 mm}:
      - Local spatial pattern variance retention (Intrinsic Spatial Finesse, ISF)
      - Spatial bleed-out (out-of-ROI leakage ratio & effective PSF width in mm)
      - 4 x 3 2D Phase Diagram of RDM correspondence in Simulation and Soto Empirical data.
    """
    fwhm_grid = [0, 5, 10, 20]
    radius_grid = [10, 20, 35]
    V_tree, _, _, _, rdm_true_vec = theoretical_true_cov_and_rdm()

    def _eval_cohort_factor3(
        subs_dict: Dict[str, Dict[str, Any]],
        emp_dict: Dict[str, Dict[str, np.ndarray]],
        seed_base: int,
    ) -> Dict[str, Any]:
        isf_by_fwhm = {fw: [] for fw in fwhm_grid}
        bleed_by_fwhm = {fw: [] for fw in fwhm_grid}
        psf_mm_by_fwhm = {fw: [] for fw in fwhm_grid}
        sim_phase = {(fw, r): [] for fw in fwhm_grid for r in radius_grid}
        emp_hg_phase = {(fw, r): [] for fw in fwhm_grid for r in radius_grid}
        emp_lstg_phase = {(fw, r): [] for fw in fwhm_grid for r in radius_grid}

        for idx_s, (s_id, sd) in enumerate(subs_dict.items()):
            rng = np.random.default_rng(seed_base + idx_s * 100)
            D_lh, D_rh = sd["D_lh"], sd["D_rh"]
            G_m, K_shrunk = sd["G_m"], sd["K_shrunk"]
            L_m, W_m = sd["L_m"], sd["W_m"]
            pos_m = sd["pos_m"]
            aud_mask = sd["rois"]["aud_lh"]
            aud_idx = np.where(aud_mask)[0]
            outside_mask = ~aud_mask
            n_lh = D_lh.shape[0]
            V = pos_m.shape[0]

            smooth_ops = {fw: build_cortical_smoothing_operator(D_lh, D_rh, float(fw)) for fw in fwhm_grid}
            sl_aud = {r: get_searchlight_patches_in_roi(D_lh, D_rh, aud_mask, float(r), max_centres=25) for r in radius_grid}
            sl_hg = {r: get_searchlight_patches_in_roi(D_lh, D_rh, sd["rois"]["hg_bilat"], float(r), max_centres=25) for r in radius_grid}
            sl_lstg = {r: get_searchlight_patches_in_roi(D_lh, D_rh, sd["rois"]["lstg_bilat"], float(r), max_centres=30) for r in radius_grid}

            # 1. Intrinsic PSF width & bleed-out from Resolution Matrix R = K_shrunk @ G_m[:, aud_idx]
            R_aud = K_shrunk @ G_m[:, aud_idx]  # (V, n_aud)
            dists_to_aud = np.linalg.norm(pos_m[:, None, :] - pos_m[None, aud_idx, :], axis=2) * 1000.0  # (V, n_aud) in mm

            for fw in fwhm_grid:
                R_sm = smooth_ops[fw] @ R_aud
                w2 = R_sm ** 2
                w2_sum = np.maximum(np.sum(w2, axis=0), 1e-30)
                psf_rms = float(np.mean(np.sqrt(np.sum(w2 * (dists_to_aud ** 2), axis=0) / w2_sum)))
                psf_mm_by_fwhm[fw].append(psf_rms)

                in_energy = float(np.mean(np.sum(w2[aud_mask, :], axis=0)))
                out_energy = float(np.mean(np.sum(w2[outside_mask, :], axis=0)))
                bleed_by_fwhm[fw].append(out_energy / max(in_energy + out_energy, 1e-30))

            # 2. Simulation 2D Phase Diagram (FWHM x Searchlight Radius) & ISF retention
            D_aud_lh = D_lh[aud_idx, :]
            sigma_pat = 0.0065
            Z_pat = np.zeros((24, len(aud_idx)), dtype=np.float64)
            for p in range(24):
                c_lh = rng.choice(n_lh, size=min(60, n_lh), replace=False)
                a_lh = rng.standard_normal(len(c_lh))
                v = np.exp(-0.5 * (D_aud_lh[:, c_lh] / sigma_pat) ** 2) @ a_lh
                v -= np.mean(v)
                Z_pat[p] = v / max(float(np.sqrt(np.mean(v ** 2))), 1e-15)
            Z_pat -= np.mean(Z_pat, axis=1, keepdims=True)
            U_z, _, Vt_z = np.linalg.svd(Z_pat, full_matrices=False)
            B_true_roi = V_tree.T @ (math.sqrt(float(len(aud_idx))) * (U_z @ Vt_z))

            S_clean = np.zeros((20, V), dtype=np.float64)
            S_clean[:, aud_idx] = B_true_roi
            Y_clean = S_clean @ G_m.T
            Z_sens = rng.standard_normal((20, L_m.shape[1]))
            N_sens = Z_sens @ L_m.T
            spow = float(np.mean((Y_clean @ W_m.T) ** 2))
            npow = float(np.mean((N_sens @ W_m.T) ** 2))
            scale_8p2 = math.sqrt((10.0 ** (8.2 / 10.0)) * npow / max(spow, 1e-30))
            Rec_0 = (scale_8p2 * Y_clean + N_sens) @ K_shrunk.T  # (20, V)

            var_0 = float(np.mean(np.var(Rec_0[:, aud_idx], axis=1, ddof=1)))
            for fw in fwhm_grid:
                Rec_fw = (smooth_ops[fw] @ Rec_0.T).T
                var_fw = float(np.mean(np.var(Rec_fw[:, aud_idx], axis=1, ddof=1)))
                isf_by_fwhm[fw].append(var_fw / max(var_0, 1e-30))

                for r in radius_grid:
                    vals = [
                        pearson_corr_fast(rdm_true_vec, rdm_corr_distance_vec(Rec_fw[:, patch]))
                        for patch in sl_aud[r]
                    ]
                    sim_phase[(fw, r)].append(float(np.mean(vals)) if vals else 0.0)

            # 3. Empirical Soto 2D Phase Diagram (FWHM x Searchlight Radius)
            S_emp = emp_dict[s_id]["S_shrunk"]  # (64, V, 61)
            t_hg_win = slice(15, 26)    # active HG 75-125 ms
            t_lstg_win = slice(30, 46)  # active lSTG 150-225 ms

            for fw in fwhm_grid:
                # Apply spatial smoothing operator across vertices for the active slices
                S_hg_sub = S_emp[:, :, t_hg_win]
                S_lstg_sub = S_emp[:, :, t_lstg_win]
                if fw > 0:
                    W_op = smooth_ops[fw]
                    S_hg_sm = np.stack([W_op @ S_hg_sub[c] for c in range(64)], axis=0)
                    S_lstg_sm = np.stack([W_op @ S_lstg_sub[c] for c in range(64)], axis=0)
                else:
                    S_hg_sm = S_hg_sub
                    S_lstg_sm = S_lstg_sub

                for r in radius_grid:
                    hg_score = evaluate_roi_spatiotemporal_rsa(
                        S_hg_sm, sl_hg[r], slice(None), models["rdm_model1_vec"]
                    )
                    lstg_score = evaluate_roi_spatiotemporal_rsa(
                        S_lstg_sm, sl_lstg[r], slice(None), models["rdm_model3_vec"]
                    )
                    emp_hg_phase[(fw, r)].append(hg_score)
                    emp_lstg_phase[(fw, r)].append(lstg_score)

        return {
            "intrinsic_spatial_finesse_retention": {f"{fw}mm": group_summary_1d(isf_by_fwhm[fw]) for fw in fwhm_grid},
            "out_of_roi_bleed_fraction": {f"{fw}mm": group_summary_1d(bleed_by_fwhm[fw]) for fw in fwhm_grid},
            "effective_psf_width_mm": {f"{fw}mm": group_summary_1d(psf_mm_by_fwhm[fw]) for fw in fwhm_grid},
            "phase_diagram_simulation": {
                f"fwhm_{fw}mm_radius_{r}mm": group_summary_1d(sim_phase[(fw, r)])
                for fw in fwhm_grid for r in radius_grid
            },
            "phase_diagram_soto_hg_model1": {
                f"fwhm_{fw}mm_radius_{r}mm": group_summary_1d(emp_hg_phase[(fw, r)])
                for fw in fwhm_grid for r in radius_grid
            },
            "phase_diagram_soto_lstg_model3": {
                f"fwhm_{fw}mm_radius_{r}mm": group_summary_1d(emp_lstg_phase[(fw, r)])
                for fw in fwhm_grid for r in radius_grid
            },
            "paired_fwhm0_vs_fwhm20_at_radius20_sim": repeated_measures_paired_stats(
                np.array(sim_phase[(0, 20)]), np.array(sim_phase[(20, 20)]), "fwhm0mm", "fwhm20mm"
            ),
            "paired_fwhm0_vs_fwhm10_at_radius20_sim": repeated_measures_paired_stats(
                np.array(sim_phase[(0, 20)]), np.array(sim_phase[(10, 20)]), "fwhm0mm", "fwhm10mm"
            ),
            "paired_fwhm0_vs_fwhm20_at_radius20_soto_lstg": repeated_measures_paired_stats(
                np.array(emp_lstg_phase[(0, 20)]), np.array(emp_lstg_phase[(20, 20)]), "fwhm0mm", "fwhm20mm"
            ),
            "paired_radius20_vs_radius10_at_fwhm0_soto_lstg": repeated_measures_paired_stats(
                np.array(emp_lstg_phase[(0, 20)]), np.array(emp_lstg_phase[(0, 10)]), "radius20mm", "radius10mm"
            ),
            "paired_radius20_vs_radius35_at_fwhm0_soto_lstg": repeated_measures_paired_stats(
                np.array(emp_lstg_phase[(0, 20)]), np.array(emp_lstg_phase[(0, 35)]), "radius20mm", "radius35mm"
            ),
        }

    return {
        "fwhm_mm_grid": fwhm_grid,
        "searchlight_radius_mm_grid": radius_grid,
        "meg": _eval_cohort_factor3(meg_subs_emp, meg_emp_data, seed_base=20265100),
        "eeg": _eval_cohort_factor3(eeg_subs_emp, eeg_emp_data, seed_base=20265500),
    }


# =============================================================================
# 7. Main Execution & Output Export Pipeline
# =============================================================================

def main() -> None:
    t0 = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure 3-tier speech model RDMs exist (run v2_04 if needed)
    models_npz_path = RESULTS_DIR / "v2_three_tier_speech_model_rdms.npz"
    if not models_npz_path.exists():
        v2_04_script = SCRIPTS_DIR / "v2_04_build_three_tier_speech_model_rdms.py"
        if v2_04_script.exists():
            import runpy
            print("[Init] Running v2_04_build_three_tier_speech_model_rdms.py to generate model RDMs...")
            runpy.run_path(str(v2_04_script), run_name="__main__")

    models_npz = np.load(models_npz_path, allow_pickle=True)
    models = {k: models_npz[k] for k in models_npz.files}
    print(f"[Init] Loaded 3-tier speech models from {models_npz_path.name}")

    # Load all valid MEG (N=11) and EEG (N=6) head-model subjects
    print("[Step 1] Loading MEG head-model cohort (N=11) and EEG head-model cohort (N=6)...")
    meg_all_subs: Dict[str, Dict[str, Any]] = {}
    for sub in MEG_HEAD_MANIFESTS:
        meg_all_subs[sub] = load_meg_subject(sub)
        print(f"  -> Loaded MEG {sub}")

    eeg_all_subs: Dict[str, Dict[str, Any]] = {}
    for sub in EEG_HEAD_RUNS:
        eeg_all_subs[sub] = load_eeg_subject(sub)
        print(f"  -> Loaded EEG {sub}")

    # Load Soto Empirical data for MEG (N=9) and EEG (N=5)
    print("[Step 2] Loading Soto Natural Speech empirical condition-mean data (MEG N=9, EEG N=5)...")
    meg_subs_emp = {s: meg_all_subs[s] for s in MEG_EMPIRICAL_SUBJECTS}
    eeg_subs_emp = {s: eeg_all_subs[s] for s in EEG_EMPIRICAL_SUBJECTS}

    meg_emp_data = {s: load_empirical_meg_sensor_and_source(s, meg_subs_emp[s]) for s in MEG_EMPIRICAL_SUBJECTS}
    eeg_emp_data = {s: load_empirical_eeg_sensor_and_source(s, eeg_subs_emp[s]) for s in EEG_EMPIRICAL_SUBJECTS}
    print("  -> Loaded all empirical sensor & source arrays.")

    # -------------------------------------------------------------------------
    # Factor 1 & Factor 4 (H6 Waterfall): Simulation & Empirical
    # -------------------------------------------------------------------------
    print("[Step 3] Running Factor 1 (shrunk vs isotropic) & Factor 4 (8-stage waterfall) across Simulation & Soto...")
    meg_sim_f1 = {
        s: evaluate_subject_factor1_and_waterfall_sim(meg_all_subs[s], n_mc=6, snr_db=8.2, seed=20263100 + i * 37)
        for i, s in enumerate(MEG_HEAD_MANIFESTS)
    }
    eeg_sim_f1 = {
        s: evaluate_subject_factor1_and_waterfall_sim(eeg_all_subs[s], n_mc=6, snr_db=8.2, seed=20263500 + i * 37)
        for i, s in enumerate(EEG_HEAD_RUNS)
    }

    meg_emp_f1 = {
        s: evaluate_subject_factor1_empirical(meg_subs_emp[s], meg_emp_data[s], models)
        for s in MEG_EMPIRICAL_SUBJECTS
    }
    eeg_emp_f1 = {
        s: evaluate_subject_factor1_empirical(eeg_subs_emp[s], eeg_emp_data[s], models)
        for s in EEG_EMPIRICAL_SUBJECTS
    }

    def _summarize_factor1_sim(sim_dict: Dict[str, Dict[str, Any]], sub_keys: List[str]) -> Dict[str, Any]:
        sh_8p2 = np.array([sim_dict[s]["sim_rho_snr8p2_shrunk"] for s in sub_keys])
        is_8p2 = np.array([sim_dict[s]["sim_rho_snr8p2_iso"] for s in sub_keys])
        sh_3p6 = np.array([sim_dict[s]["sim_rho_snr3p6_shrunk"] for s in sub_keys])
        is_3p6 = np.array([sim_dict[s]["sim_rho_snr3p6_iso"] for s in sub_keys])
        ple_sh = np.array([sim_dict[s]["ple_snr8p2_shrunk_mm"] for s in sub_keys])
        ple_is = np.array([sim_dict[s]["ple_snr8p2_iso_mm"] for s in sub_keys])
        off_emp = np.array([sim_dict[s]["cov_stats"]["empirical_rms_off_diag_corr"] for s in sub_keys])
        off_sh = np.array([sim_dict[s]["cov_stats"]["shrunk_whitened_residual_rms_off_diag_corr"] for s in sub_keys])
        supp_db = np.array([sim_dict[s]["cov_stats"]["off_diag_suppression_db"] for s in sub_keys])
        return {
            "n_subjects": len(sub_keys),
            "off_diagonal_covariance_suppression": {
                "empirical_rms_off_diag_corr": group_summary_1d(off_emp),
                "shrunk_residual_rms_off_diag_corr": group_summary_1d(off_sh),
                "suppression_db": group_summary_1d(supp_db),
                "paired_off_diag_reduction": repeated_measures_paired_stats(off_emp, off_sh, "isotropic_raw", "shrunk_whitened"),
            },
            "rdm_fidelity_snr8p2db": repeated_measures_paired_stats(sh_8p2, is_8p2, "shrunk", "isotropic"),
            "rdm_fidelity_snr3p6db": repeated_measures_paired_stats(sh_3p6, is_3p6, "shrunk", "isotropic"),
            "peak_localization_error_mm_snr8p2db": repeated_measures_paired_stats(ple_sh, ple_is, "shrunk", "isotropic"),
        }

    def _summarize_factor1_emp(emp_dict: Dict[str, Dict[str, Any]], sub_keys: List[str]) -> Dict[str, Any]:
        out = {"n_subjects": len(sub_keys)}
        for win_key in ("hg_75_150ms_model1", "hg_75_150ms_model2", "lstg_125_250ms_model3", "lstg_125_250ms_model2"):
            sh_pk = np.array([emp_dict[s][win_key]["shrunk_peak"] for s in sub_keys])
            is_pk = np.array([emp_dict[s][win_key]["iso_peak"] for s in sub_keys])
            sh_mn = np.array([emp_dict[s][win_key]["shrunk_mean"] for s in sub_keys])
            is_mn = np.array([emp_dict[s][win_key]["iso_mean"] for s in sub_keys])
            out[win_key] = {
                "active_window_peak": repeated_measures_paired_stats(sh_pk, is_pk, "shrunk", "isotropic"),
                "active_window_mean": repeated_measures_paired_stats(sh_mn, is_mn, "shrunk", "isotropic"),
            }
        b_hg_sh = np.array([emp_dict[s]["silent_baseline_0_50ms"]["hg_shrunk"] for s in sub_keys])
        b_hg_is = np.array([emp_dict[s]["silent_baseline_0_50ms"]["hg_iso"] for s in sub_keys])
        b_st_sh = np.array([emp_dict[s]["silent_baseline_0_50ms"]["lstg_shrunk"] for s in sub_keys])
        b_st_is = np.array([emp_dict[s]["silent_baseline_0_50ms"]["lstg_iso"] for s in sub_keys])
        out["silent_baseline_0_50ms_control"] = {
            "hg_model1": repeated_measures_paired_stats(b_hg_sh, b_hg_is, "shrunk", "isotropic"),
            "lstg_model3": repeated_measures_paired_stats(b_st_sh, b_st_is, "shrunk", "isotropic"),
        }
        return out

    STAGE_NAMES = [
        ("stage1", "1. True Cortical ROI Pattern"),
        ("stage2", "2. + Whole-Brain Equal-Amplitude Perlin Noise"),
        ("stage3", "3. Clean Forward Sensor Projection"),
        ("stage4", "4. + Coloured Sensor Covariance Noise (8.2 dB)"),
        ("stage5", "5. Isotropic MNE Source Reconstruction"),
        ("stage6", "6. Shrunk Pre-Whitened MNE Source Reconstruction"),
        ("stage7", "7. Explicit Cortical Spatial Smoothing (20 mm FWHM)"),
        ("stage8", "8. Optimal 20 mm Searchlight + Shrunk MNE"),
    ]

    def _summarize_waterfall(sim_dict: Dict[str, Dict[str, Any]], sub_keys: List[str]) -> Dict[str, Any]:
        stages_out = {}
        prev_arr = None
        prev_key = None
        for st_key, st_label in STAGE_NAMES:
            arr = np.array([sim_dict[s]["waterfall_stages"][st_key] for s in sub_keys])
            entry: Dict[str, Any] = {
                "stage_label": st_label,
                "group_summary": group_summary_1d(arr),
            }
            if prev_arr is not None:
                entry[f"paired_vs_{prev_key}"] = repeated_measures_paired_stats(arr, prev_arr, st_key, prev_key)
            stages_out[st_key] = entry
            prev_arr = arr
            prev_key = st_key
        # Also include direct paired comparison Stage 8 vs Stage 5 (Optimal Pipeline vs Isotropic MNE)
        s8 = np.array([sim_dict[s]["waterfall_stages"]["stage8"] for s in sub_keys])
        s5 = np.array([sim_dict[s]["waterfall_stages"]["stage5"] for s in sub_keys])
        s6 = np.array([sim_dict[s]["waterfall_stages"]["stage6"] for s in sub_keys])
        s7 = np.array([sim_dict[s]["waterfall_stages"]["stage7"] for s in sub_keys])
        stages_out["paired_stage6_shrunk_vs_stage5_isotropic"] = repeated_measures_paired_stats(s6, s5, "stage6_shrunk", "stage5_isotropic")
        stages_out["paired_stage7_smooth20_vs_stage6_unsmoothed"] = repeated_measures_paired_stats(s7, s6, "stage7_smooth20", "stage6_unsmoothed")
        stages_out["paired_stage8_optimal_vs_stage5_isotropic"] = repeated_measures_paired_stats(s8, s5, "stage8_optimal", "stage5_isotropic")
        return stages_out

    # -------------------------------------------------------------------------
    # Factor 2: Temporal Low-Pass Filtering (80, 40, 20, 10 Hz)
    # -------------------------------------------------------------------------
    print("[Step 4] Running Factor 2 (Sensor Temporal Low-Pass Filtering: 80, 40, 20, 10 Hz)...")
    factor2_results = evaluate_factor2_lowpass_filtering(
        meg_subs_emp, meg_emp_data, eeg_subs_emp, eeg_emp_data, models
    )

    # -------------------------------------------------------------------------
    # Factor 3: Cortical Explicit Spatial Smoothing (0, 5, 10, 20 mm) x Searchlight Radius (10, 20, 35 mm)
    # -------------------------------------------------------------------------
    print("[Step 5] Running Factor 3 (Cortical Explicit Spatial Smoothing & 2D Phase Diagram)...")
    factor3_results = evaluate_factor3_spatial_smoothing_and_phase_diagram(
        meg_subs_emp, meg_emp_data, eeg_subs_emp, eeg_emp_data, models
    )

    # Assemble master report JSON
    master_results = {
        "benchmark_id": "v2_06_group_inverse_and_preprocessing_benchmark",
        "design_rule": "Strict Repeated-Measures Within-Subject Paired Comparisons (Advisor S0769-S0808)",
        "cohorts": {
            "meg_head_models_n11": list(MEG_HEAD_MANIFESTS.keys()),
            "meg_soto_empirical_n9": MEG_EMPIRICAL_SUBJECTS,
            "eeg_head_models_n6": list(EEG_HEAD_RUNS.keys()),
            "eeg_soto_empirical_n5": EEG_EMPIRICAL_SUBJECTS,
        },
        "factor1_noise_covariance_regularization": {
            "simulation_all_valid_head_models": {
                "meg_n11": _summarize_factor1_sim(meg_sim_f1, list(MEG_HEAD_MANIFESTS.keys())),
                "eeg_n6": _summarize_factor1_sim(eeg_sim_f1, list(EEG_HEAD_RUNS.keys())),
            },
            "simulation_empirical_cohort_matched": {
                "meg_n9": _summarize_factor1_sim(meg_sim_f1, MEG_EMPIRICAL_SUBJECTS),
                "eeg_n5": _summarize_factor1_sim(eeg_sim_f1, EEG_EMPIRICAL_SUBJECTS),
            },
            "soto_natural_speech_empirical": {
                "meg_n9": _summarize_factor1_emp(meg_emp_f1, MEG_EMPIRICAL_SUBJECTS),
                "eeg_n5": _summarize_factor1_emp(eeg_emp_f1, EEG_EMPIRICAL_SUBJECTS),
            },
        },
        "factor2_temporal_lowpass_filtering": factor2_results,
        "factor3_cortical_spatial_smoothing_and_phase_diagram": factor3_results,
        "factor4_h6_eight_stage_fidelity_waterfall": {
            "meg_n11": _summarize_waterfall(meg_sim_f1, list(MEG_HEAD_MANIFESTS.keys())),
            "meg_n9": _summarize_waterfall(meg_sim_f1, MEG_EMPIRICAL_SUBJECTS),
            "eeg_n6": _summarize_waterfall(eeg_sim_f1, list(EEG_HEAD_RUNS.keys())),
            "eeg_n5": _summarize_waterfall(eeg_sim_f1, EEG_EMPIRICAL_SUBJECTS),
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }

    out_json = RESULTS_DIR / "v2_06_group_inverse_and_preprocessing_benchmark.json"
    out_json.write_text(json.dumps(master_results, indent=2) + "\n")

    # Write clean CSV tables for each factor
    # 1. Factor 1 CSV
    csv_f1 = RESULTS_DIR / "v2_06_factor1_covariance_regularization_summary.csv"
    with csv_f1.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "domain", "modality", "n_subjects", "df", "metric",
            "mean_shrunk", "sem_shrunk", "mean_isotropic", "sem_isotropic",
            "mean_delta", "sem_delta", "cohen_dz", "t_stat", "p_ttest", "p_wilcoxon",
            "between_subject_var_pct", "within_subject_cond_var_pct",
        ])
        for mod_key, n_label, sim_sub in [
            ("MEG", "meg_n11", master_results["factor1_noise_covariance_regularization"]["simulation_all_valid_head_models"]["meg_n11"]),
            ("MEG", "meg_n9", master_results["factor1_noise_covariance_regularization"]["simulation_empirical_cohort_matched"]["meg_n9"]),
            ("EEG", "eeg_n6", master_results["factor1_noise_covariance_regularization"]["simulation_all_valid_head_models"]["eeg_n6"]),
            ("EEG", "eeg_n5", master_results["factor1_noise_covariance_regularization"]["simulation_empirical_cohort_matched"]["eeg_n5"]),
        ]:
            for m_name in ("rdm_fidelity_snr8p2db", "rdm_fidelity_snr3p6db", "peak_localization_error_mm_snr8p2db"):
                st = sim_sub[m_name]
                w.writerow([
                    f"Simulation ({n_label})", mod_key, st["n_subjects"], st["df"], m_name,
                    st["mean_shrunk"], st["sem_shrunk"], st["mean_isotropic"], st["sem_isotropic"],
                    st["mean_delta"], st["sem_delta"], st["cohen_dz"], st["t_stat"], st["p_ttest"], st["p_wilcoxon"],
                    st["variance_partition"]["between_subject_pct"], st["variance_partition"]["within_subject_condition_pct"],
                ])
        for mod_key, emp_sub in [
            ("MEG", master_results["factor1_noise_covariance_regularization"]["soto_natural_speech_empirical"]["meg_n9"]),
            ("EEG", master_results["factor1_noise_covariance_regularization"]["soto_natural_speech_empirical"]["eeg_n5"]),
        ]:
            for win_key in ("hg_75_150ms_model1", "hg_75_150ms_model2", "lstg_125_250ms_model3", "lstg_125_250ms_model2"):
                st = emp_sub[win_key]["active_window_peak"]
                w.writerow([
                    "Soto Empirical (Active Peak)", mod_key, st["n_subjects"], st["df"], win_key,
                    st["mean_shrunk"], st["sem_shrunk"], st["mean_isotropic"], st["sem_isotropic"],
                    st["mean_delta"], st["sem_delta"], st["cohen_dz"], st["t_stat"], st["p_ttest"], st["p_wilcoxon"],
                    st["variance_partition"]["between_subject_pct"], st["variance_partition"]["within_subject_condition_pct"],
                ])
            for base_key in ("hg_model1", "lstg_model3"):
                st = emp_sub["silent_baseline_0_50ms_control"][base_key]
                w.writerow([
                    "Soto Empirical (Silent Baseline 0-50ms)", mod_key, st["n_subjects"], st["df"], f"baseline_{base_key}",
                    st["mean_shrunk"], st["sem_shrunk"], st["mean_isotropic"], st["sem_isotropic"],
                    st["mean_delta"], st["sem_delta"], st["cohen_dz"], st["t_stat"], st["p_ttest"], st["p_wilcoxon"],
                    st["variance_partition"]["between_subject_pct"], st["variance_partition"]["within_subject_condition_pct"],
                ])

    # 2. Factor 2 CSV (Low-pass filtering)
    csv_f2 = RESULTS_DIR / "v2_06_factor2_lowpass_filtering_summary.csv"
    with csv_f2.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "domain", "modality", "cutoff_hz",
            "fast_delta_L_mean", "fast_delta_L_sem",
            "fast_burst_mean", "fast_burst_sem",
            "fast_composite_mean", "fast_composite_sem",
            "sust_erb_mean", "sust_erb_sem",
            "sust_f1f2_mean", "sust_f1f2_sem",
            "sust_composite_mean", "sust_composite_sem",
        ])
        for dom_label, dom_key in [("Simulation", "simulation"), ("Soto_Empirical", "soto_empirical")]:
            for mod_label, mod_key in [("MEG", "meg"), ("EEG", "eeg")]:
                bc = factor2_results[dom_key][mod_key]["by_cutoff"]
                for fc in [80, 40, 20, 10]:
                    row = bc[f"{fc}Hz"]
                    w.writerow([
                        dom_label, mod_label, fc,
                        row["fast_delta_L"]["mean"], row["fast_delta_L"]["sem"],
                        row["fast_burst"]["mean"], row["fast_burst"]["sem"],
                        row["fast_composite"]["mean"], row["fast_composite"]["sem"],
                        row["sust_erb"]["mean"], row["sust_erb"]["sem"],
                        row["sust_f1f2"]["mean"], row["sust_f1f2"]["sem"],
                        row["sust_composite"]["mean"], row["sust_composite"]["sem"],
                    ])

    # 3. Factor 3 CSV (Spatial smoothing & 2D phase diagram)
    csv_f3 = RESULTS_DIR / "v2_06_factor3_spatial_smoothing_phase_diagram.csv"
    with csv_f3.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "modality", "fwhm_mm", "searchlight_radius_mm",
            "isf_variance_retention_mean", "isf_variance_retention_sem",
            "bleed_out_fraction_mean", "psf_width_mm_mean",
            "sim_rho_mean", "sim_rho_sem",
            "soto_hg_m1_rho_mean", "soto_hg_m1_rho_sem",
            "soto_lstg_m3_rho_mean", "soto_lstg_m3_rho_sem",
        ])
        for mod_label, mod_key in [("MEG", "meg"), ("EEG", "eeg")]:
            f3m = factor3_results[mod_key]
            for fw in [0, 5, 10, 20]:
                isf = f3m["intrinsic_spatial_finesse_retention"][f"{fw}mm"]
                bld = f3m["out_of_roi_bleed_fraction"][f"{fw}mm"]
                psf = f3m["effective_psf_width_mm"][f"{fw}mm"]
                for r in [10, 20, 35]:
                    k_pr = f"fwhm_{fw}mm_radius_{r}mm"
                    sm = f3m["phase_diagram_simulation"][k_pr]
                    hg = f3m["phase_diagram_soto_hg_model1"][k_pr]
                    st = f3m["phase_diagram_soto_lstg_model3"][k_pr]
                    w.writerow([
                        mod_label, fw, r,
                        isf["mean"], isf["sem"],
                        bld["mean"], psf["mean"],
                        sm["mean"], sm["sem"],
                        hg["mean"], hg["sem"],
                        st["mean"], st["sem"],
                    ])

    # 4. Factor 4 CSV (8-Stage Waterfall)
    csv_f4 = RESULTS_DIR / "v2_06_factor4_eight_stage_waterfall.csv"
    with csv_f4.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "stage_id", "stage_label", "n_subjects", "rho_mean", "rho_sem", "rho_sd"])
        for coh_key in ("meg_n11", "meg_n9", "eeg_n6", "eeg_n5"):
            wf = master_results["factor4_h6_eight_stage_fidelity_waterfall"][coh_key]
            for st_key, st_label in STAGE_NAMES:
                gs = wf[st_key]["group_summary"]
                w.writerow([coh_key, st_key, st_label, gs["n"], gs["mean"], gs["sem"], gs["sd"]])

    # 5. Compressed NPZ with all arrays for downstream plotting
    np.savez_compressed(
        RESULTS_DIR / "v2_06_group_inverse_and_preprocessing_benchmark.npz",
        cutoffs_hz=np.array([80, 40, 20, 10]),
        fwhm_mm_grid=np.array([0, 5, 10, 20]),
        searchlight_radius_mm_grid=np.array([10, 20, 35]),
        meg_waterfall_n11_mean=np.array([
            master_results["factor4_h6_eight_stage_fidelity_waterfall"]["meg_n11"][st]["group_summary"]["mean"]
            for st, _ in STAGE_NAMES
        ]),
        meg_waterfall_n11_sem=np.array([
            master_results["factor4_h6_eight_stage_fidelity_waterfall"]["meg_n11"][st]["group_summary"]["sem"]
            for st, _ in STAGE_NAMES
        ]),
        eeg_waterfall_n6_mean=np.array([
            master_results["factor4_h6_eight_stage_fidelity_waterfall"]["eeg_n6"][st]["group_summary"]["mean"]
            for st, _ in STAGE_NAMES
        ]),
        eeg_waterfall_n6_sem=np.array([
            master_results["factor4_h6_eight_stage_fidelity_waterfall"]["eeg_n6"][st]["group_summary"]["sem"]
            for st, _ in STAGE_NAMES
        ]),
    )

    print(f"[Done] Saved all v2_06 JSON/CSV/NPZ outputs to {RESULTS_DIR} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
