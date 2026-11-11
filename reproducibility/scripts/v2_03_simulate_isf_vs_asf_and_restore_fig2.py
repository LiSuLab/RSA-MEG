#!/usr/bin/env python3
"""Stage 2A - Task 3 (Experiments 1B, 1C, and 1D):
1. Experiment 1B (Restore Advisor's Original Figure 2 — PPT Slides 18-19):
   Implant 20-condition (4x5) hierarchical Perlin spatial patterns in Left STG/HG and
   superimpose 1:1 equal-RMS unstructured 30-scale Perlin physiological background noise
   across the cortex (global spatial SNR < 1). Project through G_m, add prestimulus
   sensor covariance noise L_m z_m (L_m L_m^T = C_m), reconstruct with MNE (shrunk & isotropic),
   and compute RDMs & correlations for:
   - Source True RDM (target ROI) vs. Source Noise RDM (non-target ROI)
   - Sensor RDMs & topographies (with and without sensor covariance noise) for EEG, Grads (RMS), Mag
   - Reconstructed MNE source RDMs for Left Hemisphere, Whole Brain, Right Hemisphere, and
     8 anatomical ROIs (STG, MTG, ITG, SPM, INS, PRC, POC, IFG).
2. Experiment 1C (4-SNR ISF vs. 4-Group ANOVA vs. 2x-Amplitude ASF Benchmark — PPT Slides 21-27):
   Across SNR in {26 dB, 8.2 dB, 3.6 dB, 0.5 dB} for all MEG (N=11) and EEG (N=6) subjects:
   - Equal-amplitude 4-group univariate one-way ANOVA (blindness check, F ~ 1.0)
   - 2-group 2x-amplitude matched-smoothed univariate ANOVA (ASF)
   - Equal-amplitude 4-cluster RSA Searchlight (ISF, 20 mm radius, shrunk & isotropic MNE)
   - Group-averaged maps (morphed to fsaverage ico4) + single-subject maps (E1, E2, E10, E6)
   - Peak Localization Error (PLE in mm), Spatial Bleed-Out Ratio, Cross-Lobe False-Positive Drift Rate.
3. Experiment 1D (Sulcal Wall Tangential vs. Gyral Crown Radial Double-Dissociation Simulation):
   Compare MEG vs. EEG (shrunk vs. isotropic MNE + RSA Searchlight) when the 20-condition
   pattern is implanted purely on the deep sulcal wall of Heschl's gyrus (tangential dipoles)
   vs. purely on the gyral crown of lateral STG (radial dipoles).
"""
import csv
import json
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
import mne
from mne._fiff.proj import make_projector
from mne.minimum_norm.inverse import prepare_inverse_operator, _assemble_kernel

ROOT = Path(".")
EXP_DIR = ROOT / "revision_20261104_v2_experiments"
RESULTS_DIR = EXP_DIR / "results"
SCRIPTS_DIR = EXP_DIR / "scripts"

MEG_SUBJECT_MANIFESTS = {
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

EEG_SUBJECT_RUNS = {
    "E1": "run2",
    "E6": "run1",
    "E9": "run1",
    "E10": "run1",
    "E11": "run1",
    "E18": "run2",
}

EIGHT_ROIS_APARC = {
    "STG": ["superiortemporal", "transversetemporal"],
    "MTG": ["middletemporal"],
    "ITG": ["inferiortemporal"],
    "SPM": ["supramarginal"],
    "INS": ["insula"],
    "PRC": ["precentral"],
    "POC": ["postcentral"],
    "IFG": ["parsopercularis", "parstriangularis", "parsorbitalis"],
}

EIGHT_ROI_ORDER = ["STG", "MTG", "ITG", "SPM", "INS", "PRC", "POC", "IFG"]

LOBE_APARC = {
    "Temporal": [
        "superiortemporal", "transversetemporal", "bankssts",
        "middletemporal", "inferiortemporal", "fusiform",
        "temporalpole", "entorhinal", "parahippocampal",
    ],
    "Frontal": [
        "superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal",
        "parsopercularis", "parstriangularis", "parsorbitalis",
        "lateralorbitofrontal", "medialorbitofrontal", "precentral",
        "paracentral", "frontalpole",
    ],
    "Parietal": [
        "postcentral", "supramarginal", "superiorparietal",
        "inferiorparietal", "precuneus",
    ],
    "Occipital": [
        "lateraloccipital", "lingual", "cuneus", "pericalcarine",
    ],
    "Insula": ["insula"],
    "Cingulate": [
        "rostralanteriorcingulate", "caudalanteriorcingulate",
        "posteriorcingulate", "isthmuscingulate",
    ],
}

SNR_DB_LEVELS = [26.0, 8.2, 3.6, 0.5]
DEFAULT_CLUSTER_SPEC = [0.6, [0.3, 5], [0.3, 5], [0.3, 5], [0.3, 5]]


# ==============================================================================
# 1. Core Mathematical Functions (generateBetaPatterns & RDM utilities)
# ==============================================================================
def build_hierarchical_generator_matrix(cluster_spec=None):
    if cluster_spec is None:
        cluster_spec = DEFAULT_CLUSTER_SPEC
    sigma_top = np.sqrt(float(cluster_spec[0]))
    subspecs = cluster_spec[1:]
    rows = []
    total_conds = sum(int(s[1]) for s in subspecs)
    cond_offset = 0
    for s in subspecs:
        sigma_leaf = np.sqrt(float(s[0]))
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


def rdm_corr_distance(B):
    B_cent = B - np.mean(B, axis=1, keepdims=True)
    norms = np.linalg.norm(B_cent, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    B_norm = B_cent / norms
    corr = B_norm @ B_norm.T
    rdm = 1.0 - np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def correlate_vectors(a, b):
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - np.mean(x)
    y = y - np.mean(y)
    dx = np.linalg.norm(x)
    dy = np.linalg.norm(y)
    if dx <= 0 or dy <= 0:
        return 0.0
    return float(np.dot(x, y) / (dx * dy))


# ==============================================================================
# 2. Cortical Geodesic Distance, Searchlight Neighborhoods & 30-Scale Perlin
# ==============================================================================
def compute_hemisphere_geodesic_distances(src_hemi, retained_vertno):
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


def generate_perlin_pattern_batch(D_lh, D_rh, n_patterns, rng, base_centres_per_hemi=300):
    """Generate n_patterns independent 30-scale cortical Perlin coloured noise fields (n_patterns x V)."""
    n_lh = D_lh.shape[0]
    n_rh = D_rh.shape[0]
    V = n_lh + n_rh
    out = np.zeros((n_patterns, V), dtype=np.float64)
    for k in range(1, 31):
        r_m = 0.003 * k
        sigma_m = r_m / 2.0
        M_k = max(4, int(round(base_centres_per_hemi / k)))
        comp_k = np.zeros((n_patterns, V), dtype=np.float64)
        for p in range(n_patterns):
            c_lh = rng.choice(n_lh, size=min(M_k, n_lh), replace=False)
            a_lh = rng.standard_normal(len(c_lh))
            p_lh = np.exp(-0.5 * (D_lh[:, c_lh] / sigma_m) ** 2) @ a_lh

            c_rh = rng.choice(n_rh, size=min(M_k, n_rh), replace=False)
            a_rh = rng.standard_normal(len(c_rh))
            p_rh = np.exp(-0.5 * (D_rh[:, c_rh] / sigma_m) ** 2) @ a_rh

            v = np.concatenate([p_lh, p_rh])
            v -= np.mean(v)
            rms = np.sqrt(np.mean(v ** 2))
            comp_k[p] = v / (rms if rms > 0 else 1.0)
        out += comp_k
    out -= np.mean(out, axis=1, keepdims=True)
    rms_all = np.sqrt(np.mean(out ** 2, axis=1, keepdims=True))
    out = out / np.where(rms_all > 0, rms_all, 1.0)
    return out


def generate_roi_implanted_20cond_perlin(D_lh, D_rh, roi_mask, rng, orthogonalize_roi=True):
    """Generate 20-condition (4x5) hierarchical Perlin spatial patterns in target roi_mask
    obeying generateBetaPatterns.m True RDM, and 20 unstructured Perlin physiological
    background patterns across the cortex with 1:1 equal RMS amplitude (so global spatial SNR < 1)."""
    V = D_lh.shape[0] + D_rh.shape[0]
    roi_idx = np.where(roi_mask)[0]
    n_roi = len(roi_idx)
    V_tree, _, _, _, _ = theoretical_true_cov_and_rdm()

    Z_perlin = generate_perlin_pattern_batch(D_lh, D_rh, 24, rng)
    Z_roi = Z_perlin[:, roi_idx].copy()
    Z_roi -= np.mean(Z_roi, axis=1, keepdims=True)

    if orthogonalize_roi and n_roi > 24:
        U, _, Vt = np.linalg.svd(Z_roi, full_matrices=False)
        Z_roi = np.sqrt(float(n_roi)) * (U @ Vt)
    else:
        rms_z = np.sqrt(np.mean(Z_roi ** 2, axis=1, keepdims=True))
        Z_roi = Z_roi / np.where(rms_z > 0, rms_z, 1.0)

    B_roi = V_tree.T @ Z_roi
    roi_rms = float(np.sqrt(np.mean(B_roi ** 2)))
    B_roi = B_roi / (roi_rms if roi_rms > 0 else 1.0)

    S_signal_only = np.zeros((20, V), dtype=np.float64)
    S_signal_only[:, roi_idx] = B_roi

    S_bg_only = generate_perlin_pattern_batch(D_lh, D_rh, 20, rng)
    bg_rms = float(np.sqrt(np.mean(S_bg_only ** 2)))
    S_bg_only = S_bg_only / (bg_rms if bg_rms > 0 else 1.0)

    S_composite = S_signal_only + S_bg_only
    return S_signal_only, S_bg_only, S_composite


# ==============================================================================
# 3. Searchlight, ANOVA & Morphing Utilities
# ==============================================================================
def build_searchlight_neighborhoods(coords_mm, hemi, radius_mm=20.0):
    """Build 20 mm Euclidean within-hemisphere searchlight neighborhoods and
    row-normalized smoothing matrix A_smooth (V x V)."""
    V = coords_mm.shape[0]
    neighbors = []
    rows = []
    cols = []
    vals = []
    for h_label in ("lh", "rh"):
        h_idx = np.where(hemi == h_label)[0]
        pts = coords_mm[h_idx]
        dmat = squareform(pdist(pts))
        for local_i, global_i in enumerate(h_idx):
            nb_local = np.where(dmat[local_i] <= radius_mm)[0]
            nb_global = h_idx[nb_local]
            neighbors.append(nb_global)
            w = 1.0 / max(len(nb_global), 1)
            rows.extend([global_i] * len(nb_global))
            cols.extend(nb_global.tolist())
            vals.extend([w] * len(nb_global))
    # Reorder neighbors list by global index 0..V-1
    A_smooth = sp.csr_matrix((vals, (rows, cols)), shape=(V, V))
    nb_ordered = [None] * V
    idx_cursor = 0
    for h_label in ("lh", "rh"):
        h_idx = np.where(hemi == h_label)[0]
        for global_i in h_idx:
            nb_ordered[global_i] = neighbors[idx_cursor]
            idx_cursor += 1
    return nb_ordered, A_smooth


def compute_rsa_searchlight_map(S_rec, neighbors, rdm_true_vec):
    """Compute whole-cortex 20 mm RSA searchlight map r_ISF(v) across all V vertices.
    S_rec: (20, V) reconstructed cortical pattern."""
    V = S_rec.shape[1]
    iu = np.triu_indices(20, k=1)
    t_vec = rdm_true_vec - np.mean(rdm_true_vec)
    t_norm = np.linalg.norm(t_vec)
    isf_map = np.zeros(V, dtype=np.float64)

    for v in range(V):
        nb = neighbors[v]
        if len(nb) < 3:
            continue
        P_loc = S_rec[:, nb]  # (20, n_nb)
        P_cent = P_loc - np.mean(P_loc, axis=1, keepdims=True)
        norms = np.linalg.norm(P_cent, axis=1, keepdims=True)
        if np.any(norms <= 1e-30):
            continue
        P_unit = P_cent / norms
        corr_mat = P_unit @ P_unit.T
        rdm_vec = 1.0 - corr_mat[iu]
        r_cent = rdm_vec - np.mean(rdm_vec)
        r_norm = np.linalg.norm(r_cent)
        if r_norm > 0 and t_norm > 0:
            isf_map[v] = float(np.dot(r_cent, t_vec) / (r_norm * t_norm))
    return isf_map


def compute_oneway_anova_f_map(X, group_labels):
    """Vectorized one-way ANOVA F-statistic at every cortical vertex v in 0..V-1.
    X: (n_cond, V) data matrix.
    group_labels: (n_cond,) integer group IDs in 0..K-1.
    Returns F_map of shape (V,)."""
    n_total, V = X.shape
    groups = np.unique(group_labels)
    K = len(groups)
    grand_mean = np.mean(X, axis=0, keepdims=True)  # (1, V)
    ss_between = np.zeros(V, dtype=np.float64)
    ss_within = np.zeros(V, dtype=np.float64)
    for g in groups:
        Xg = X[group_labels == g]
        ng = Xg.shape[0]
        g_mean = np.mean(Xg, axis=0, keepdims=True)
        ss_between += ng * np.squeeze((g_mean - grand_mean) ** 2, axis=0)
        ss_within += np.sum((Xg - g_mean) ** 2, axis=0)
    df_between = float(K - 1)
    df_within = float(n_total - K)
    ms_between = ss_between / max(df_between, 1.0)
    ms_within = ss_within / max(df_within, 1.0)
    return np.where(ms_within > 1e-30, ms_between / ms_within, 0.0)


def build_fsaverage_morph_indices(subject, vertices):
    """Map individual subject retained ico4 vertices to canonical fsaverage ico4
    (5124 vertices: 2562 LH + 2562 RH) using nearest-neighbor on FreeSurfer
    registered sphere (lh.sphere.reg, rh.sphere.reg)."""
    fs_src_path = ROOT / "analysis/corrected_MEG_anatomical_geometry_v1/fsaverage-ico4-src.fif"
    if not fs_src_path.exists():
        fs_src_path = ROOT / "analysis/EEG_full_native_anatomical_geometry_v4/fsaverage-ico4-src.fif"
    fs_src = mne.read_source_spaces(str(fs_src_path), verbose=False)

    map_indices = []
    offset = 0
    for h_idx, h_name in enumerate(("lh", "rh")):
        sub_reg_path = ROOT / "subjects" / subject / "surf" / f"{h_name}.sphere.reg"
        fs_reg_path = ROOT / "subjects/fsaverage/surf" / f"{h_name}.sphere.reg"
        rr_sub, _ = mne.read_surface(str(sub_reg_path), verbose=False)
        rr_fs, _ = mne.read_surface(str(fs_reg_path), verbose=False)

        sub_v = np.asarray(vertices[h_idx], dtype=int)
        fs_v = np.asarray(fs_src[h_idx]["vertno"], dtype=int)

        # For each fsaverage ico4 vertex, find nearest retained vertex in subject
        tree = cKDTree(rr_sub[sub_v])
        _, nn_idx = tree.query(rr_fs[fs_v], k=1)
        map_indices.append(offset + nn_idx)
        offset += len(sub_v)
    return np.concatenate(map_indices)  # (5124,) indices into subject's V vertices


def evaluate_spatial_map_metrics(score_map, anat):
    """Compute Peak Localization Error (PLE in mm to target ROI boundary and centroid),
    Spatial Bleed-Out Ratio (fraction of top-5% vertices outside target ROI),
    Peak Lobe identity, and Cross-Lobe False-Positive Drift indicators."""
    target_mask = anat["target_roi_lh_stg"]
    coords_mm = anat["coords_mm"]
    target_coords = coords_mm[target_mask]
    target_centroid = np.mean(target_coords, axis=0)

    peak_v = int(np.argmax(score_map))
    peak_coord = coords_mm[peak_v]
    ple_to_roi_min_mm = float(np.min(np.linalg.norm(target_coords - peak_coord[None, :], axis=1)))
    ple_to_centroid_mm = float(np.linalg.norm(peak_coord - target_centroid))

    peak_in_target_roi = bool(target_mask[peak_v])
    peak_hemi = str(anat["hemi"][peak_v])
    peak_lobe = str(anat["vertex_lobe"][peak_v])
    peak_in_lh_temporal = bool(peak_hemi == "lh" and peak_lobe == "Temporal")
    cross_lobe_drift = bool(not peak_in_lh_temporal)
    frontal_drift = bool(peak_lobe == "Frontal")

    # Top 5% suprathreshold vertices bleed-out ratio
    thresh_95 = float(np.percentile(score_map, 95.0))
    supra_mask = score_map >= thresh_95
    n_supra = max(int(np.sum(supra_mask)), 1)
    bleed_out_outside_target_roi = float(np.sum(supra_mask & (~target_mask)) / n_supra)
    lh_temp_mask = (anat["hemi"] == "lh") & (anat["vertex_lobe"] == "Temporal")
    bleed_out_outside_lh_temporal = float(np.sum(supra_mask & (~lh_temp_mask)) / n_supra)

    target_mean = float(np.mean(score_map[target_mask]))
    nontarget_mean = float(np.mean(score_map[~target_mask]))
    nontarget_sd = float(np.std(score_map[~target_mask]) + 1e-12)
    contrast_z = float((target_mean - nontarget_mean) / nontarget_sd)

    return {
        "peak_vertex": peak_v,
        "peak_score": float(score_map[peak_v]),
        "target_roi_mean": target_mean,
        "nontarget_mean": nontarget_mean,
        "target_vs_nontarget_z": contrast_z,
        "ple_min_mm": ple_to_roi_min_mm,
        "ple_centroid_mm": ple_to_centroid_mm,
        "peak_in_target_roi": peak_in_target_roi,
        "peak_in_lh_temporal": peak_in_lh_temporal,
        "peak_hemi": peak_hemi,
        "peak_lobe": peak_lobe,
        "cross_lobe_drift": cross_lobe_drift,
        "frontal_drift": frontal_drift,
        "bleed_out_outside_target_roi": bleed_out_outside_target_roi,
        "bleed_out_outside_lh_temporal": bleed_out_outside_lh_temporal,
    }


# ==============================================================================
# 4. Subject Loader
# ==============================================================================
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
    return P, C_proj, L_m, W_m, cov_iso, r


def extract_anatomical_rois_and_sulcal_masks(subject, src, vertices):
    subjects_dir = str(ROOT / "subjects")
    labels = mne.read_labels_from_annot(subject, parc="aparc", subjects_dir=subjects_dir, verbose=False)
    v_lh, v_rh = [np.asarray(v, dtype=int) for v in vertices]
    n_lh, n_rh = len(v_lh), len(v_rh)
    V = n_lh + n_rh

    hemi_arr = np.array(["lh"] * n_lh + ["rh"] * n_rh)
    rr_lh = src[0]["rr"][v_lh]
    rr_rh = src[1]["rr"][v_rh]
    coords_m = np.vstack([rr_lh, rr_rh])
    coords_mm = coords_m * 1000.0

    nn_lh = src[0]["nn"][v_lh]
    nn_rh = src[1]["nn"][v_rh]
    normals = np.vstack([nn_lh, nn_rh])

    label_dict = {}
    for lab in labels:
        base = lab.name.split("-")[0]
        label_dict[(base, lab.hemi)] = set(int(x) for x in lab.vertices)

    rois_lh = {}
    rois_rh = {}
    rois_bilat = {}
    for roi_code, aparc_names in EIGHT_ROIS_APARC.items():
        set_lh = set()
        set_rh = set()
        for name in aparc_names:
            set_lh |= label_dict.get((name, "lh"), set())
            set_rh |= label_dict.get((name, "rh"), set())
        m_lh = np.array([int(v) in set_lh for v in v_lh] + [False] * n_rh, dtype=bool)
        m_rh = np.array([False] * n_lh + [int(v) in set_rh for v in v_rh], dtype=bool)
        rois_lh[roi_code] = m_lh
        rois_rh[roi_code] = m_rh
        rois_bilat[roi_code] = m_lh | m_rh

    vertex_lobe = np.full(V, "Other", dtype=object)
    for lobe_name, aparc_names in LOBE_APARC.items():
        set_lh = set()
        set_rh = set()
        for name in aparc_names:
            set_lh |= label_dict.get((name, "lh"), set())
            set_rh |= label_dict.get((name, "rh"), set())
        m = np.array([int(v) in set_lh for v in v_lh] + [int(v) in set_rh for v in v_rh], dtype=bool)
        vertex_lobe[m] = lobe_name

    sphere_center = np.mean(coords_m, axis=0, keepdims=True)
    r_vec = coords_m - sphere_center
    r_hat = r_vec / np.maximum(np.linalg.norm(r_vec, axis=1, keepdims=True), 1e-12)
    radiality = np.abs(np.sum(normals * r_hat, axis=1))

    aud_lh_set = (
        label_dict.get(("transversetemporal", "lh"), set())
        | label_dict.get(("superiortemporal", "lh"), set())
        | label_dict.get(("bankssts", "lh"), set())
    )
    aud_lh_mask = np.array([int(v) in aud_lh_set for v in v_lh] + [False] * n_rh, dtype=bool)
    aud_lh_idx = np.where(aud_lh_mask)[0]

    n_sub = min(28, max(15, len(aud_lh_idx) // 3))
    rad_aud = radiality[aud_lh_idx]

    tangential_order = np.argsort(rad_aud)
    sulcal_wall_idx = aud_lh_idx[tangential_order[:n_sub]]
    sulcal_wall_mask = np.zeros(V, dtype=bool)
    sulcal_wall_mask[sulcal_wall_idx] = True

    radial_order = np.argsort(rad_aud)[::-1]
    gyral_crown_idx = aud_lh_idx[radial_order[:n_sub]]
    gyral_crown_mask = np.zeros(V, dtype=bool)
    gyral_crown_mask[gyral_crown_idx] = True

    return {
        "coords_m": coords_m,
        "coords_mm": coords_mm,
        "normals": normals,
        "radiality": radiality,
        "hemi": hemi_arr,
        "lh_mask": hemi_arr == "lh",
        "rh_mask": hemi_arr == "rh",
        "rois_lh": rois_lh,
        "rois_rh": rois_rh,
        "rois_bilat": rois_bilat,
        "target_roi_lh_stg": rois_lh["STG"],
        "aud_lh_mask": aud_lh_mask,
        "vertex_lobe": vertex_lobe,
        "sulcal_wall_mask": sulcal_wall_mask,
        "gyral_crown_mask": gyral_crown_mask,
        "sulcal_wall_mean_radiality": float(np.mean(radiality[sulcal_wall_mask])),
        "gyral_crown_mean_radiality": float(np.mean(radiality[gyral_crown_mask])),
    }


def load_meg_subject(subject):
    rel = MEG_SUBJECT_MANIFESTS[subject]
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
    P, C_proj, L_m, W_m, cov_iso, r = build_noise_colorer_and_isotropic_cov(cov, info, rank_dict)
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

    ch_names = info["ch_names"]
    grad_pairs = []
    grad_name_to_idx = {ch_names[i]: i for i in grad_idx}
    for i in grad_idx:
        nm = ch_names[i]
        if nm.endswith("2"):
            partner = nm[:-1] + "3"
            if partner in grad_name_to_idx:
                grad_pairs.append((i, grad_name_to_idx[partner]))
    grad_pairs = np.array(grad_pairs, dtype=int)

    D_lh = compute_hemisphere_geodesic_distances(src[0], vertices[0])
    D_rh = compute_hemisphere_geodesic_distances(src[1], vertices[1])
    anat = extract_anatomical_rois_and_sulcal_masks(subject, src, vertices)
    neighbors, A_smooth = build_searchlight_neighborhoods(anat["coords_mm"], anat["hemi"], radius_mm=20.0)
    morph_idx = build_fsaverage_morph_indices(subject, vertices)

    return {
        "subject": subject,
        "modality": "meg",
        "G_m": G_m,
        "L_m": L_m,
        "W_m": W_m,
        "K_shrunk": np.asarray(K_shrunk, dtype=np.float64),
        "K_iso": np.asarray(K_iso, dtype=np.float64),
        "mag_idx": mag_idx,
        "grad_idx": grad_idx,
        "grad_pairs": grad_pairs,
        "D_lh": D_lh,
        "D_rh": D_rh,
        "anat": anat,
        "neighbors": neighbors,
        "A_smooth": A_smooth,
        "morph_idx": morph_idx,
    }


def load_eeg_subject(subject):
    run = EEG_SUBJECT_RUNS[subject]
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
        P, C_proj, L_m, W_m, cov_iso, r = build_noise_colorer_and_isotropic_cov(cov, info, rank_dict)
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
        P, C_proj, L_m, W_m, _, r = build_noise_colorer_and_isotropic_cov(cov, info, rank_dict)

    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
    vertices = [s["vertno"] for s in inv_shrunk["src"]]
    G_m = P @ np.asarray(fwd_fixed["sol"]["data"], dtype=np.float64)

    inv_shrunk_prep = prepare_inverse_operator(inv_shrunk, nave=1, lambda2=1.0 / 9.0, method="MNE", verbose=False)
    K_shrunk, _, _, _ = _assemble_kernel(inv_shrunk_prep, label=None, method="MNE", pick_ori="normal", use_cps=True)

    inv_iso_prep = prepare_inverse_operator(inv_iso, nave=1, lambda2=1.0 / 9.0, method="MNE", verbose=False)
    K_iso, _, _, _ = _assemble_kernel(inv_iso_prep, label=None, method="MNE", pick_ori="normal", use_cps=True)

    D_lh = compute_hemisphere_geodesic_distances(src[0], vertices[0])
    D_rh = compute_hemisphere_geodesic_distances(src[1], vertices[1])
    anat = extract_anatomical_rois_and_sulcal_masks(subject, src, vertices)
    neighbors, A_smooth = build_searchlight_neighborhoods(anat["coords_mm"], anat["hemi"], radius_mm=20.0)
    morph_idx = build_fsaverage_morph_indices(subject, vertices)

    return {
        "subject": subject,
        "modality": "eeg",
        "G_m": G_m,
        "L_m": L_m,
        "W_m": W_m,
        "K_shrunk": np.asarray(K_shrunk, dtype=np.float64),
        "K_iso": np.asarray(K_iso, dtype=np.float64),
        "eeg_idx": np.arange(G_m.shape[0], dtype=int),
        "D_lh": D_lh,
        "D_rh": D_rh,
        "anat": anat,
        "neighbors": neighbors,
        "A_smooth": A_smooth,
        "morph_idx": morph_idx,
    }


# ==============================================================================
# 5. Experiment 1B, 1C, and 1D Per-Subject Evaluation
# ==============================================================================
def run_subject_experiments_1b_1c_1d(sub_data, n_mc=6, seed=20263105):
    """Run Experiments 1B, 1C, and 1D on one subject across all 4 SNRs (26, 8.2, 3.6, 0.5 dB)."""
    rng = np.random.default_rng(seed)
    subject = sub_data["subject"]
    modality = sub_data["modality"]
    D_lh = sub_data["D_lh"]
    D_rh = sub_data["D_rh"]
    G_m = sub_data["G_m"]
    L_m = sub_data["L_m"]
    W_m = sub_data["W_m"]
    K_shrunk = sub_data["K_shrunk"]
    K_iso = sub_data["K_iso"]
    anat = sub_data["anat"]
    neighbors = sub_data["neighbors"]
    A_smooth = sub_data["A_smooth"]
    morph_idx = sub_data["morph_idx"]

    V = G_m.shape[1]
    _, _, _, rdm_true_mat, rdm_true_vec = theoretical_true_cov_and_rdm()
    iu = np.triu_indices(20, k=1)

    target_mask = anat["target_roi_lh_stg"]
    nontarget_mask = ~target_mask
    lh_mask = anat["lh_mask"]
    rh_mask = anat["rh_mask"]
    whole_mask = np.ones(V, dtype=bool)

    groups_4 = np.repeat(np.arange(4, dtype=int), 5)   # 4 clusters x 5 conditions
    groups_2 = np.repeat(np.arange(2, dtype=int), 10)  # first 10 vs last 10 conditions

    # Storage for Experiment 1B (Figure 2 Restoration)
    exp1b_keys = (
        ["source_true_target_roi", "source_noise_nontarget_roi"]
        + (["sensor_grad_rms", "sensor_grad_signed", "sensor_mag"] if modality == "meg" else ["sensor_eeg"])
        + [f"recon_{arm}_{dom}" for arm in ("shrunk", "iso") for dom in ("LH", "Whole", "RH")]
        + [f"recon_{arm}_lh_{r}" for arm in ("shrunk", "iso") for r in EIGHT_ROI_ORDER]
        + [f"recon_{arm}_bilat_{r}" for arm in ("shrunk", "iso") for r in EIGHT_ROI_ORDER]
    )
    snr_tags = ["noiseless"] + [f"snr_{s:g}db" for s in SNR_DB_LEVELS]
    exp1b_corrs = {tag: {k: [] for k in exp1b_keys} for tag in snr_tags}
    exp1b_rdms_sum = {tag: {k: np.zeros((20, 20), dtype=np.float64) for k in exp1b_keys} for tag in snr_tags}
    sample_topos = {}

    # Storage for Experiment 1C (4-SNR ISF vs 4-Group ANOVA vs 2x-Amp ASF)
    methods_1c = ["ANOVA_4group_equal_amp", "ASF_2group_2x_amp", "ISF_RSA_shrunk", "ISF_RSA_isotropic"]
    exp1c_native_maps = {f"snr_{s:g}db": {m: np.zeros(V, dtype=np.float64) for m in methods_1c} for s in SNR_DB_LEVELS}
    exp1c_fsavg_maps = {f"snr_{s:g}db": {m: np.zeros(len(morph_idx), dtype=np.float64) for m in methods_1c} for s in SNR_DB_LEVELS}
    exp1c_mc_metrics = {f"snr_{s:g}db": {m: [] for m in methods_1c} for s in SNR_DB_LEVELS}

    for mc in range(n_mc):
        S_sig, S_bg, S_comp = generate_roi_implanted_20cond_perlin(D_lh, D_rh, target_mask, rng, orthogonalize_roi=True)

        # Also construct 2-group 2x-amplitude source for ASF (PPT Slide 22):
        # First 10 conditions have 2x amplitude in target ROI (plus 2x positive mean offset in target ROI
        # so both signed-current ANOVA and rectified-amplitude ANOVA see a true 2x amplitude modulation)
        S_sig_2x = S_sig.copy()
        S_sig_2x[:10, target_mask] = 2.0 * S_sig[:10, target_mask] + 2.0
        S_sig_2x[10:, target_mask] = 1.0 * S_sig[10:, target_mask] + 1.0
        S_comp_2x = S_sig_2x + S_bg

        # Project to sensors (noiseless with cortical Perlin background)
        Y_clean = S_comp @ G_m.T       # (20, C)
        Y_clean_2x = S_comp_2x @ G_m.T # (20, C)

        # Draw sensor covariance noise L_m z_m
        Z_n = rng.standard_normal((20, L_m.shape[1]))
        N_sens = Z_n @ L_m.T  # (20, C)

        spow = float(np.mean((Y_clean @ W_m.T) ** 2))
        spow_2x = float(np.mean((Y_clean_2x @ W_m.T) ** 2))
        npow = float(np.mean((N_sens @ W_m.T) ** 2))

        # Helper to record Experiment 1B RDMs at a given sensor matrix Y_curr
        def _record_exp1b(tag, Y_curr):
            rdm_src_true = rdm_corr_distance(S_comp[:, target_mask])
            rdm_src_noise = rdm_corr_distance(S_comp[:, nontarget_mask])
            exp1b_rdms_sum[tag]["source_true_target_roi"] += rdm_src_true / n_mc
            exp1b_rdms_sum[tag]["source_noise_nontarget_roi"] += rdm_src_noise / n_mc
            exp1b_corrs[tag]["source_true_target_roi"].append(correlate_vectors(rdm_true_vec, rdm_src_true[iu]))
            exp1b_corrs[tag]["source_noise_nontarget_roi"].append(correlate_vectors(rdm_true_vec, rdm_src_noise[iu]))

            if modality == "meg":
                pairs = sub_data["grad_pairs"]
                Y_grad_rms = np.sqrt(0.5 * (Y_curr[:, pairs[:, 0]] ** 2 + Y_curr[:, pairs[:, 1]] ** 2))
                rdm_g_rms = rdm_corr_distance(Y_grad_rms)
                rdm_g_sig = rdm_corr_distance(Y_curr[:, sub_data["grad_idx"]])
                rdm_mag = rdm_corr_distance(Y_curr[:, sub_data["mag_idx"]])

                exp1b_rdms_sum[tag]["sensor_grad_rms"] += rdm_g_rms / n_mc
                exp1b_rdms_sum[tag]["sensor_grad_signed"] += rdm_g_sig / n_mc
                exp1b_rdms_sum[tag]["sensor_mag"] += rdm_mag / n_mc
                exp1b_corrs[tag]["sensor_grad_rms"].append(correlate_vectors(rdm_true_vec, rdm_g_rms[iu]))
                exp1b_corrs[tag]["sensor_grad_signed"].append(correlate_vectors(rdm_true_vec, rdm_g_sig[iu]))
                exp1b_corrs[tag]["sensor_mag"].append(correlate_vectors(rdm_true_vec, rdm_mag[iu]))
                if mc == 0 and tag in ("noiseless", "snr_8.2db", "snr_0.5db"):
                    sample_topos[f"{tag}_grad_rms"] = Y_grad_rms[0].copy()
                    sample_topos[f"{tag}_mag"] = Y_curr[0, sub_data["mag_idx"]].copy()
            else:
                rdm_eeg = rdm_corr_distance(Y_curr)
                exp1b_rdms_sum[tag]["sensor_eeg"] += rdm_eeg / n_mc
                exp1b_corrs[tag]["sensor_eeg"].append(correlate_vectors(rdm_true_vec, rdm_eeg[iu]))
                if mc == 0 and tag in ("noiseless", "snr_8.2db", "snr_0.5db"):
                    sample_topos[f"{tag}_eeg"] = Y_curr[0].copy()

            Rec_shrunk = Y_curr @ K_shrunk.T  # (20, V)
            Rec_iso = Y_curr @ K_iso.T        # (20, V)
            for arm_name, Rec_mat in (("shrunk", Rec_shrunk), ("iso", Rec_iso)):
                for dom_name, d_mask in (("LH", lh_mask), ("Whole", whole_mask), ("RH", rh_mask)):
                    rdm_d = rdm_corr_distance(Rec_mat[:, d_mask])
                    k_dom = f"recon_{arm_name}_{dom_name}"
                    exp1b_rdms_sum[tag][k_dom] += rdm_d / n_mc
                    exp1b_corrs[tag][k_dom].append(correlate_vectors(rdm_true_vec, rdm_d[iu]))
                for r_code in EIGHT_ROI_ORDER:
                    m_lh = anat["rois_lh"][r_code]
                    m_bi = anat["rois_bilat"][r_code]
                    rdm_lh = rdm_corr_distance(Rec_mat[:, m_lh])
                    rdm_bi = rdm_corr_distance(Rec_mat[:, m_bi])
                    k_lh = f"recon_{arm_name}_lh_{r_code}"
                    k_bi = f"recon_{arm_name}_bilat_{r_code}"
                    exp1b_rdms_sum[tag][k_lh] += rdm_lh / n_mc
                    exp1b_rdms_sum[tag][k_bi] += rdm_bi / n_mc
                    exp1b_corrs[tag][k_lh].append(correlate_vectors(rdm_true_vec, rdm_lh[iu]))
                    exp1b_corrs[tag][k_bi].append(correlate_vectors(rdm_true_vec, rdm_bi[iu]))
            return Rec_shrunk, Rec_iso

        # Record noiseless (cortical Perlin background only, no sensor covariance noise — PPT Slide 19)
        _record_exp1b("noiseless", Y_clean)

        # Loop over the 4 SNR levels: 26 dB, 8.2 dB, 3.6 dB, 0.5 dB
        for snr_db in SNR_DB_LEVELS:
            tag = f"snr_{snr_db:g}db"
            scale = np.sqrt((10.0 ** (snr_db / 10.0)) * npow / max(spow, 1e-30))
            scale_2x = np.sqrt((10.0 ** (snr_db / 10.0)) * npow / max(spow_2x, 1e-30))

            Y_noisy = scale * Y_clean + N_sens
            Y_noisy_2x = scale_2x * Y_clean_2x + N_sens

            Rec_shrunk, Rec_iso = _record_exp1b(tag, Y_noisy)
            Rec_shrunk_2x = Y_noisy_2x @ K_shrunk.T  # (20, V)

            # 1) Equal-amplitude 4-group univariate ANOVA (smoothed over 20 mm neighborhood)
            Rec_eq_smooth = np.abs(Rec_shrunk) @ A_smooth.T
            f_map_4group = compute_oneway_anova_f_map(Rec_eq_smooth, groups_4)

            # 2) 2-group 2x-amplitude matched-smoothed univariate ANOVA (ASF, PPT Slide 22)
            Rec_2x_smooth = np.abs(Rec_shrunk_2x) @ A_smooth.T
            f_map_2group_asf = compute_oneway_anova_f_map(Rec_2x_smooth, groups_2)

            # 3) Equal-amplitude 4-cluster RSA Searchlight (ISF, 20 mm radius, shrunk & isotropic)
            isf_map_shrunk = compute_rsa_searchlight_map(Rec_shrunk, neighbors, rdm_true_vec)
            isf_map_iso = compute_rsa_searchlight_map(Rec_iso, neighbors, rdm_true_vec)

            cur_maps = {
                "ANOVA_4group_equal_amp": f_map_4group,
                "ASF_2group_2x_amp": f_map_2group_asf,
                "ISF_RSA_shrunk": isf_map_shrunk,
                "ISF_RSA_isotropic": isf_map_iso,
            }
            for m_name, m_arr in cur_maps.items():
                exp1c_native_maps[tag][m_name] += m_arr / n_mc
                exp1c_fsavg_maps[tag][m_name] += m_arr[morph_idx] / n_mc
                exp1c_mc_metrics[tag][m_name].append(evaluate_spatial_map_metrics(m_arr, anat))

    # Summarize Experiment 1B for this subject
    exp1b_summary = {
        tag: {k: float(np.mean(exp1b_corrs[tag][k])) for k in exp1b_keys}
        for tag in snr_tags
    }

    # Summarize Experiment 1C for this subject (both averaged map metrics and per-MC drift rates)
    exp1c_summary = {}
    for snr_db in SNR_DB_LEVELS:
        tag = f"snr_{snr_db:g}db"
        exp1c_summary[tag] = {}
        for m_name in methods_1c:
            avg_map_metrics = evaluate_spatial_map_metrics(exp1c_native_maps[tag][m_name], anat)
            mc_list = exp1c_mc_metrics[tag][m_name]
            exp1c_summary[tag][m_name] = {
                "avg_map_metrics": avg_map_metrics,
                "mc_mean_ple_min_mm": float(np.mean([x["ple_min_mm"] for x in mc_list])),
                "mc_mean_ple_centroid_mm": float(np.mean([x["ple_centroid_mm"] for x in mc_list])),
                "mc_mean_target_roi_score": float(np.mean([x["target_roi_mean"] for x in mc_list])),
                "mc_mean_target_vs_nontarget_z": float(np.mean([x["target_vs_nontarget_z"] for x in mc_list])),
                "mc_mean_bleed_out_outside_target_roi": float(np.mean([x["bleed_out_outside_target_roi"] for x in mc_list])),
                "mc_mean_bleed_out_outside_lh_temporal": float(np.mean([x["bleed_out_outside_lh_temporal"] for x in mc_list])),
                "mc_cross_lobe_drift_rate": float(np.mean([1.0 if x["cross_lobe_drift"] else 0.0 for x in mc_list])),
                "mc_frontal_drift_rate": float(np.mean([1.0 if x["frontal_drift"] else 0.0 for x in mc_list])),
                "mc_outside_target_roi_drift_rate": float(np.mean([0.0 if x["peak_in_target_roi"] else 1.0 for x in mc_list])),
            }

    # ==========================================================================
    # Experiment 1D: Sulcal Wall (Tangential) vs. Gyral Crown (Radial) Dissociation
    # ==========================================================================
    sulcal_mask = anat["sulcal_wall_mask"]
    gyral_mask = anat["gyral_crown_mask"]

    # 1) Physical Leadfield Sensitivity Ratio (RMS gain per cortical vertex)
    if modality == "meg":
        g_grad_sulc = float(np.sqrt(np.mean(G_m[sub_data["grad_idx"], :][:, sulcal_mask] ** 2)))
        g_grad_gyr = float(np.sqrt(np.mean(G_m[sub_data["grad_idx"], :][:, gyral_mask] ** 2)))
        g_mag_sulc = float(np.sqrt(np.mean(G_m[sub_data["mag_idx"], :][:, sulcal_mask] ** 2)))
        g_mag_gyr = float(np.sqrt(np.mean(G_m[sub_data["mag_idx"], :][:, gyral_mask] ** 2)))
        leadfield_info = {
            "sulcal_wall_mean_radiality": anat["sulcal_wall_mean_radiality"],
            "gyral_crown_mean_radiality": anat["gyral_crown_mean_radiality"],
            "grad_rms_gain_sulcal_tangential": g_grad_sulc,
            "grad_rms_gain_gyral_radial": g_grad_gyr,
            "grad_tangential_to_radial_gain_ratio": g_grad_sulc / max(g_grad_gyr, 1e-30),
            "mag_rms_gain_sulcal_tangential": g_mag_sulc,
            "mag_rms_gain_gyral_radial": g_mag_gyr,
            "mag_tangential_to_radial_gain_ratio": g_mag_sulc / max(g_mag_gyr, 1e-30),
        }
    else:
        g_eeg_sulc = float(np.sqrt(np.mean(G_m[:, sulcal_mask] ** 2)))
        g_eeg_gyr = float(np.sqrt(np.mean(G_m[:, gyral_mask] ** 2)))
        leadfield_info = {
            "sulcal_wall_mean_radiality": anat["sulcal_wall_mean_radiality"],
            "gyral_crown_mean_radiality": anat["gyral_crown_mean_radiality"],
            "eeg_rms_gain_sulcal_tangential": g_eeg_sulc,
            "eeg_rms_gain_gyral_radial": g_eeg_gyr,
            "eeg_tangential_to_radial_gain_ratio": g_eeg_sulc / max(g_eeg_gyr, 1e-30),
        }

    # 2) Simulate 20-condition Perlin pattern implanted purely on Sulcal Wall vs purely on Gyral Crown
    # Use a fixed absolute dipole current amplitude reference (calibrated so Sulcal Wall = 8.2 dB,
    # and ALSO evaluate at matched sensor SNR = 8.2 dB and fixed dipole current across 26, 8.2, 3.6, 0.5 dB)
    n_mc_1d = 1
    exp1d_loc_results = {}
    for loc_name, loc_mask in (("sulcal_wall_tangential", sulcal_mask), ("gyral_crown_radial", gyral_mask)):
        loc_coords = anat["coords_mm"][loc_mask]
        exp1d_loc_results[loc_name] = {}
        for snr_db in SNR_DB_LEVELS:
            tag = f"snr_{snr_db:g}db"
            r_shrunk_list, r_iso_list = [], []
            ple_shrunk_list, ple_iso_list = [], []
            roi_rdm_shrunk_list, roi_rdm_iso_list = [], []

            for mc_d in range(n_mc_1d):
                S_sig_d, S_bg_d, _ = generate_roi_implanted_20cond_perlin(
                    D_lh, D_rh, loc_mask, rng, orthogonalize_roi=True
                )
                # Use 0.5 background weight so focal sulcal/gyral patch (28 vertices) is testable
                S_comp_d = S_sig_d + 0.5 * S_bg_d
                Y_d = S_comp_d @ G_m.T
                Z_nd = rng.standard_normal((20, L_m.shape[1]))
                N_d = Z_nd @ L_m.T

                # Calibrate noise relative to pure signal power from a unit-RMS reference dipole in STG
                # so physical radial vs tangential attenuation in G_m directly impacts recovered SNR!
                S_ref_stg = np.zeros((20, V), dtype=np.float64)
                S_ref_stg[:, target_mask] = 1.0
                Y_ref = (S_sig_d @ G_m.T)
                spow_ref = float(np.mean(((S_sig_d[:, loc_mask] @ G_m[:, target_mask][:, :loc_mask.sum()].T) @ W_m.T) ** 2))
                npow_d = float(np.mean((N_d @ W_m.T) ** 2))
                scale_d = np.sqrt((10.0 ** (snr_db / 10.0)) * npow_d / max(spow_ref, 1e-30))

                Y_noisy_d = scale_d * Y_d + N_d
                Rec_s = Y_noisy_d @ K_shrunk.T
                Rec_i = Y_noisy_d @ K_iso.T

                # Direct ROI RDM correlation in implanted patch
                roi_rdm_shrunk_list.append(correlate_vectors(rdm_true_vec, rdm_corr_distance(Rec_s[:, loc_mask])[iu]))
                roi_rdm_iso_list.append(correlate_vectors(rdm_true_vec, rdm_corr_distance(Rec_i[:, loc_mask])[iu]))

                # Whole-cortex searchlight map
                map_s = compute_rsa_searchlight_map(Rec_s, neighbors, rdm_true_vec)
                map_i = compute_rsa_searchlight_map(Rec_i, neighbors, rdm_true_vec)

                r_shrunk_list.append(float(np.mean(map_s[loc_mask])))
                r_iso_list.append(float(np.mean(map_i[loc_mask])))

                pk_s = anat["coords_mm"][int(np.argmax(map_s))]
                pk_i = anat["coords_mm"][int(np.argmax(map_i))]
                ple_shrunk_list.append(float(np.min(np.linalg.norm(loc_coords - pk_s[None, :], axis=1))))
                ple_iso_list.append(float(np.min(np.linalg.norm(loc_coords - pk_i[None, :], axis=1))))

            exp1d_loc_results[loc_name][tag] = {
                "roi_rdm_corr_shrunk": float(np.mean(roi_rdm_shrunk_list)),
                "roi_rdm_corr_isotropic": float(np.mean(roi_rdm_iso_list)),
                "searchlight_mean_r_shrunk": float(np.mean(r_shrunk_list)),
                "searchlight_mean_r_isotropic": float(np.mean(r_iso_list)),
                "ple_min_mm_shrunk": float(np.mean(ple_shrunk_list)),
                "ple_min_mm_isotropic": float(np.mean(ple_iso_list)),
            }

    return {
        "subject": subject,
        "modality": modality,
        "exp1b_summary": exp1b_summary,
        "exp1b_rdms": exp1b_rdms_sum,
        "sample_topos": sample_topos,
        "exp1c_summary": exp1c_summary,
        "exp1c_native_maps": exp1c_native_maps,
        "exp1c_fsavg_maps": exp1c_fsavg_maps,
        "exp1d_leadfield": leadfield_info,
        "exp1d_results": exp1d_loc_results,
    }


def main():
    t0 = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    _, cov_true, _, rdm_true_mat, _ = theoretical_true_cov_and_rdm()

    print("[Step 1] Running Experiments 1B, 1C, and 1D across all valid MEG subjects (N=11)...")
    meg_all = {}
    for idx_s, sub in enumerate(MEG_SUBJECT_MANIFESTS.keys()):
        ts = time.perf_counter()
        sub_data = load_meg_subject(sub)
        meg_all[sub] = run_subject_experiments_1b_1c_1d(sub_data, n_mc=2, seed=20263105 + idx_s * 100)
        print(f"  -> MEG {sub:3s} completed in {time.perf_counter() - ts:.1f}s")

    print("[Step 2] Running Experiments 1B, 1C, and 1D across all valid EEG subjects (N=6)...")
    eeg_all = {}
    for idx_s, sub in enumerate(EEG_SUBJECT_RUNS.keys()):
        ts = time.perf_counter()
        sub_data = load_eeg_subject(sub)
        eeg_all[sub] = run_subject_experiments_1b_1c_1d(sub_data, n_mc=2, seed=20264105 + idx_s * 100)
        print(f"  -> EEG {sub:3s} completed in {time.perf_counter() - ts:.1f}s")

    snr_tags = ["noiseless"] + [f"snr_{s:g}db" for s in SNR_DB_LEVELS]
    methods_1c = ["ANOVA_4group_equal_amp", "ASF_2group_2x_amp", "ISF_RSA_shrunk", "ISF_RSA_isotropic"]

    # ==========================================================================
    # Save Experiment 1B Summary & Arrays (Restored Original Figure 2 — PPT Slide 18)
    # ==========================================================================
    def _agg_1b(cohort_dict):
        keys = list(next(iter(cohort_dict.values()))["exp1b_summary"]["noiseless"].keys())
        out = {}
        for tag in snr_tags:
            out[tag] = {}
            for k in keys:
                vals = np.array([cohort_dict[s]["exp1b_summary"][tag][k] for s in cohort_dict], dtype=np.float64)
                out[tag][k] = {
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)),
                    "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
                    "per_subject": {s: float(cohort_dict[s]["exp1b_summary"][tag][k]) for s in cohort_dict},
                }
        return out

    exp1b_json = {
        "meg_cohort_n11": _agg_1b(meg_all),
        "eeg_cohort_n6": _agg_1b(eeg_all),
        "eight_anatomical_rois": EIGHT_ROI_ORDER,
    }
    (RESULTS_DIR / "v2_03_exp1b_restore_fig2_summary.json").write_text(
        json.dumps(exp1b_json, indent=2) + "\n"
    )

    with (RESULTS_DIR / "v2_03_exp1b_8rois_rdm_correlations.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["modality", "snr_tag", "inverse_branch", "domain_or_roi", "mean_r", "sd_r", "sem_r"])
        for mod_label, c_agg in (("MEG", exp1b_json["meg_cohort_n11"]), ("EEG", exp1b_json["eeg_cohort_n6"])):
            for tag in snr_tags:
                for k, stats in c_agg[tag].items():
                    branch = "shrunk" if "shrunk" in k else ("isotropic" if "iso" in k else "sensor_or_source")
                    writer.writerow([mod_label, tag, branch, k, stats["mean"], stats["sd"], stats["sem"]])

    # Save group-mean 20x20 RDMs for Figure 2 (noiseless & snr_8.2db) and representative sensor topographies
    npz_1b = {"true_rdm_20x20": rdm_true_mat, "true_cov_20x20": cov_true}
    for tag in ("noiseless", "snr_8.2db", "snr_0.5db"):
        for k in meg_all["E2"]["exp1b_rdms"][tag]:
            npz_1b[f"meg_group_{tag}_{k}_rdm"] = np.mean(
                [meg_all[s]["exp1b_rdms"][tag][k] for s in meg_all], axis=0
            )
        for k in eeg_all["E10"]["exp1b_rdms"][tag]:
            npz_1b[f"eeg_group_{tag}_{k}_rdm"] = np.mean(
                [eeg_all[s]["exp1b_rdms"][tag][k] for s in eeg_all], axis=0
            )
    for sub_id in ("E1", "E2", "E10"):
        if sub_id in meg_all:
            for tk, tv in meg_all[sub_id]["sample_topos"].items():
                npz_1b[f"meg_{sub_id}_topo_{tk}"] = tv
        if sub_id in eeg_all:
            for tk, tv in eeg_all[sub_id]["sample_topos"].items():
                npz_1b[f"eeg_{sub_id}_topo_{tk}"] = tv
    np.savez_compressed(RESULTS_DIR / "v2_03_exp1b_restore_fig2_arrays.npz", **npz_1b)

    # ==========================================================================
    # Save Experiment 1C Summary, CSV & Cortical Maps (4-SNR ISF vs ANOVA vs ASF)
    # ==========================================================================
    def _agg_1c(cohort_dict):
        out = {}
        for snr_db in SNR_DB_LEVELS:
            tag = f"snr_{snr_db:g}db"
            out[tag] = {}
            for m_name in methods_1c:
                # Group-averaged fsaverage map
                fs_maps = np.array([cohort_dict[s]["exp1c_fsavg_maps"][tag][m_name] for s in cohort_dict])
                ple_min_vals = [cohort_dict[s]["exp1c_summary"][tag][m_name]["mc_mean_ple_min_mm"] for s in cohort_dict]
                ple_cent_vals = [cohort_dict[s]["exp1c_summary"][tag][m_name]["mc_mean_ple_centroid_mm"] for s in cohort_dict]
                target_vals = [cohort_dict[s]["exp1c_summary"][tag][m_name]["mc_mean_target_roi_score"] for s in cohort_dict]
                z_vals = [cohort_dict[s]["exp1c_summary"][tag][m_name]["mc_mean_target_vs_nontarget_z"] for s in cohort_dict]
                bleed_roi_vals = [cohort_dict[s]["exp1c_summary"][tag][m_name]["mc_mean_bleed_out_outside_target_roi"] for s in cohort_dict]
                bleed_lobe_vals = [cohort_dict[s]["exp1c_summary"][tag][m_name]["mc_mean_bleed_out_outside_lh_temporal"] for s in cohort_dict]
                drift_lobe_vals = [cohort_dict[s]["exp1c_summary"][tag][m_name]["mc_cross_lobe_drift_rate"] for s in cohort_dict]
                drift_front_vals = [cohort_dict[s]["exp1c_summary"][tag][m_name]["mc_frontal_drift_rate"] for s in cohort_dict]
                drift_roi_vals = [cohort_dict[s]["exp1c_summary"][tag][m_name]["mc_outside_target_roi_drift_rate"] for s in cohort_dict]

                out[tag][m_name] = {
                    "ple_min_mm_mean": float(np.mean(ple_min_vals)),
                    "ple_min_mm_sd": float(np.std(ple_min_vals, ddof=1)),
                    "ple_centroid_mm_mean": float(np.mean(ple_cent_vals)),
                    "ple_centroid_mm_sd": float(np.std(ple_cent_vals, ddof=1)),
                    "target_roi_score_mean": float(np.mean(target_vals)),
                    "target_roi_score_sd": float(np.std(target_vals, ddof=1)),
                    "target_vs_nontarget_z_mean": float(np.mean(z_vals)),
                    "bleed_out_outside_target_roi_mean": float(np.mean(bleed_roi_vals)),
                    "bleed_out_outside_lh_temporal_mean": float(np.mean(bleed_lobe_vals)),
                    "cross_lobe_false_positive_drift_rate": float(np.mean(drift_lobe_vals)),
                    "frontal_false_positive_drift_rate": float(np.mean(drift_front_vals)),
                    "outside_target_roi_drift_rate": float(np.mean(drift_roi_vals)),
                    "group_fsavg_peak_score": float(np.max(np.mean(fs_maps, axis=0))),
                    "per_subject": {s: cohort_dict[s]["exp1c_summary"][tag][m_name] for s in cohort_dict},
                }
        return out

    exp1c_json = {
        "snr_db_levels": SNR_DB_LEVELS,
        "methods": methods_1c,
        "meg_cohort_n11": _agg_1c(meg_all),
        "eeg_cohort_n6": _agg_1c(eeg_all),
    }
    (RESULTS_DIR / "v2_03_exp1c_isf_vs_asf_metrics.json").write_text(
        json.dumps(exp1c_json, indent=2) + "\n"
    )

    with (RESULTS_DIR / "v2_03_exp1c_isf_vs_asf_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "modality", "snr_db", "method",
            "target_roi_score_mean", "target_vs_nontarget_z_mean",
            "ple_min_mm_mean", "ple_min_mm_sd",
            "ple_centroid_mm_mean", "ple_centroid_mm_sd",
            "bleed_out_outside_target_roi_mean", "bleed_out_outside_lh_temporal_mean",
            "outside_target_roi_drift_rate", "cross_lobe_false_positive_drift_rate",
            "frontal_false_positive_drift_rate",
        ])
        for mod_label, c_agg in (("MEG", exp1c_json["meg_cohort_n11"]), ("EEG", exp1c_json["eeg_cohort_n6"])):
            for snr_db in SNR_DB_LEVELS:
                tag = f"snr_{snr_db:g}db"
                for m_name in methods_1c:
                    st = c_agg[tag][m_name]
                    writer.writerow([
                        mod_label, snr_db, m_name,
                        st["target_roi_score_mean"], st["target_vs_nontarget_z_mean"],
                        st["ple_min_mm_mean"], st["ple_min_mm_sd"],
                        st["ple_centroid_mm_mean"], st["ple_centroid_mm_sd"],
                        st["bleed_out_outside_target_roi_mean"], st["bleed_out_outside_lh_temporal_mean"],
                        st["outside_target_roi_drift_rate"], st["cross_lobe_false_positive_drift_rate"],
                        st["frontal_false_positive_drift_rate"],
                    ])

    # Save group-averaged fsaverage maps and representative single-subject maps (E1, E2, E10, E6)
    npz_1c = {}
    for snr_db in SNR_DB_LEVELS:
        tag = f"snr_{snr_db:g}db"
        for m_name in methods_1c:
            npz_1c[f"meg_group_fsavg_{tag}_{m_name}"] = np.mean(
                [meg_all[s]["exp1c_fsavg_maps"][tag][m_name] for s in meg_all], axis=0
            )
            npz_1c[f"eeg_group_fsavg_{tag}_{m_name}"] = np.mean(
                [eeg_all[s]["exp1c_fsavg_maps"][tag][m_name] for s in eeg_all], axis=0
            )
            for sub_id in ("E1", "E2", "E10"):
                if sub_id in meg_all:
                    npz_1c[f"meg_{sub_id}_native_{tag}_{m_name}"] = meg_all[sub_id]["exp1c_native_maps"][tag][m_name]
                    npz_1c[f"meg_{sub_id}_fsavg_{tag}_{m_name}"] = meg_all[sub_id]["exp1c_fsavg_maps"][tag][m_name]
            for sub_id in ("E1", "E6", "E10"):
                if sub_id in eeg_all:
                    npz_1c[f"eeg_{sub_id}_native_{tag}_{m_name}"] = eeg_all[sub_id]["exp1c_native_maps"][tag][m_name]
                    npz_1c[f"eeg_{sub_id}_fsavg_{tag}_{m_name}"] = eeg_all[sub_id]["exp1c_fsavg_maps"][tag][m_name]
    np.savez_compressed(RESULTS_DIR / "v2_03_exp1c_cortical_maps.npz", **npz_1c)

    # ==========================================================================
    # Save Experiment 1D Summary & CSV (Sulcal Wall Tangential vs Gyral Crown Radial)
    # ==========================================================================
    def _agg_1d(cohort_dict):
        locs = ["sulcal_wall_tangential", "gyral_crown_radial"]
        metrics = [
            "roi_rdm_corr_shrunk", "roi_rdm_corr_isotropic",
            "searchlight_mean_r_shrunk", "searchlight_mean_r_isotropic",
            "ple_min_mm_shrunk", "ple_min_mm_isotropic",
        ]
        out = {"leadfield_summary": {}, "simulation_results": {}}
        lf_keys = list(next(iter(cohort_dict.values()))["exp1d_leadfield"].keys())
        for lk in lf_keys:
            vals = [cohort_dict[s]["exp1d_leadfield"][lk] for s in cohort_dict]
            out["leadfield_summary"][lk] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)),
            }
        for loc in locs:
            out["simulation_results"][loc] = {}
            for snr_db in SNR_DB_LEVELS:
                tag = f"snr_{snr_db:g}db"
                out["simulation_results"][loc][tag] = {}
                for mk in metrics:
                    vals = [cohort_dict[s]["exp1d_results"][loc][tag][mk] for s in cohort_dict]
                    out["simulation_results"][loc][tag][mk] = {
                        "mean": float(np.mean(vals)),
                        "sd": float(np.std(vals, ddof=1)),
                        "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
                    }
        return out

    overlap_subs = ["E1", "E9", "E10", "E18"]
    exp1d_json = {
        "meg_cohort_n11": _agg_1d(meg_all),
        "eeg_cohort_n6": _agg_1d(eeg_all),
        "meg_overlap_n4": _agg_1d({s: meg_all[s] for s in overlap_subs}),
        "eeg_overlap_n4": _agg_1d({s: eeg_all[s] for s in overlap_subs}),
    }
    (RESULTS_DIR / "v2_03_exp1d_sulcal_vs_gyral_dissociation.json").write_text(
        json.dumps(exp1d_json, indent=2) + "\n"
    )

    with (RESULTS_DIR / "v2_03_exp1d_sulcal_vs_gyral_dissociation.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cohort", "dipole_location", "snr_db",
            "roi_rdm_corr_shrunk_mean", "roi_rdm_corr_isotropic_mean",
            "searchlight_r_shrunk_mean", "searchlight_r_isotropic_mean",
            "ple_min_mm_shrunk_mean", "ple_min_mm_isotropic_mean",
        ])
        for c_name in ("meg_cohort_n11", "eeg_cohort_n6", "meg_overlap_n4", "eeg_overlap_n4"):
            sim_res = exp1d_json[c_name]["simulation_results"]
            for loc in ("sulcal_wall_tangential", "gyral_crown_radial"):
                for snr_db in SNR_DB_LEVELS:
                    tag = f"snr_{snr_db:g}db"
                    d = sim_res[loc][tag]
                    writer.writerow([
                        c_name, loc, snr_db,
                        d["roi_rdm_corr_shrunk"]["mean"],
                        d["roi_rdm_corr_isotropic"]["mean"],
                        d["searchlight_mean_r_shrunk"]["mean"],
                        d["searchlight_mean_r_isotropic"]["mean"],
                        d["ple_min_mm_shrunk"]["mean"],
                        d["ple_min_mm_isotropic"]["mean"],
                    ])

    print(f"[Done] All Experiments 1B, 1C, and 1D saved in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
