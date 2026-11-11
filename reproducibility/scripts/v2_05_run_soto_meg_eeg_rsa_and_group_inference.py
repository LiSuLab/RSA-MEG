#!/usr/bin/env python3
"""
v2_05_run_soto_meg_eeg_rsa_and_group_inference.py
Run spatiotemporal RSA across the 11 sliding windows (25 to 275 ms, 50 ms width, 25 ms step)
for MEG and EEG in sensor space and cortical source space on the Kymata Soto dataset:
  (a) Bilateral Heschl's gyrus (HG, transversetemporal, deep sulcal wall, tangential) and
      Lateral STG (superiortemporal, gyral crown, radial) ROI time courses & peak latencies
      plus whole-cortex fsaverage ico4 searchlight maps (20 mm radius);
  (b) Partial RSA Cascade Wave (Model 2 | Model 1, and Model 3 | Models 1 & 2) and
      MEG vs. EEG + shrunk MNE Double Dissociation across HG and Lateral STG;
  (c) Cross-Modal Convergence Test (H1 Bonus): First-order source waveform correlation
      Corr(X_MEG, X_EEG) vs. second-order source RDM correlation Corr(RDM_MEG, RDM_EEG)
      across paired subjects (E9, E10, E18);
  (d) EEG Sensor-Space Multivariate RSA vs. Pairwise Univariate ERP Comparison (H3 Bonus);
  (e) Nonparametric Spatiotemporal Cluster-Level Randomization Test (10,000 sign-flip
      permutations) controlling whole-window / whole-cortex FWER.
"""

import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist, pdist
from scipy.stats import rankdata, spearmanr, ttest_1samp, t as student_t
import mne
import mne.minimum_norm

ROOT = Path('.')
OUT_DIR = ROOT / 'revision_20261104_v2_experiments' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
WINDOW_SAMPLES = 11
STEP_SAMPLES = 5
N_WINDOWS = 11
WINDOW_CENTERS_MS = [25 + 25 * w for w in range(N_WINDOWS)]  # [25, 50, ..., 275]
IU_64 = np.triu_indices(64, 1)
N_PAIRS = 2016


def fast_rank_rows(mat_2d):
    """Rank each row of a 2D array (n_items, n_pairs) along axis=1."""
    return np.apply_along_axis(rankdata, 1, mat_2d)


def fast_spearman_vec_vs_mat(model_rank_1d, neural_ranks_2d):
    """Compute Spearman rho between 1D ranked model vector (P,) and 2D ranked neural matrix (M, P)."""
    m0 = model_rank_1d - np.mean(model_rank_1d)
    m0 /= (np.linalg.norm(m0) + EPS)
    n0 = neural_ranks_2d - np.mean(neural_ranks_2d, axis=1, keepdims=True)
    norms = np.linalg.norm(n0, axis=1, keepdims=True) + EPS
    n0 /= norms
    return n0 @ m0


def residualize_rank_vec(target_rank_1d, control_ranks_cols):
    """Residualize 1D ranked model vector against 1 or more control ranked vectors."""
    X = np.column_stack([np.ones(len(target_rank_1d)), control_ranks_cols])
    beta, *_ = np.linalg.lstsq(X, target_rank_1d, rcond=None)
    return target_rank_1d - X @ beta


def residualize_rank_mat(neural_ranks_2d, control_ranks_cols):
    """Residualize each row of (M, P) neural rank matrix against control ranked vectors (P, K)."""
    X = np.column_stack([np.ones(neural_ranks_2d.shape[1]), control_ranks_cols])
    XtX_inv_Xt = np.linalg.pinv(X)
    return neural_ranks_2d - (neural_ranks_2d @ X) @ XtX_inv_Xt


def compute_window_rdm_from_feature_time(X_cond_feat_time):
    """
    Given condition-averaged tensor X of shape (64, n_features, 61),
    return array of shape (11, 2016) containing correlation-distance RDMs for each window.
    """
    rdms = np.zeros((N_WINDOWS, N_PAIRS), dtype=np.float64)
    for w in range(N_WINDOWS):
        s0 = w * STEP_SAMPLES
        s1 = s0 + WINDOW_SAMPLES
        feat = X_cond_feat_time[:, :, s0:s1].reshape(64, -1)
        rdms[w] = pdist(feat, metric='correlation')
    return np.nan_to_num(rdms, nan=0.0)


def sign_flip_cluster_1d(subj_time_mat, n_perm=10000, t_thresh=2.0, seed=42):
    """
    1D temporal cluster-based sign-flip permutation test across subjects.
    subj_time_mat: shape (n_subjects, n_windows)
    """
    rng = np.random.default_rng(seed)
    n_subj, n_win = subj_time_mat.shape
    mean_tc = np.mean(subj_time_mat, axis=0)
    sd_tc = np.std(subj_time_mat, axis=0, ddof=1)
    se_tc = sd_tc / np.sqrt(n_subj)
    t_obs = mean_tc / (se_tc + EPS)
    p_uncorr = np.array([
        float(ttest_1samp(subj_time_mat[:, w], 0.0, alternative='greater').pvalue)
        for w in range(n_win)
    ])

    def get_clusters(t_vec):
        supra = t_vec > t_thresh
        clusters = []
        in_c = False
        start = 0
        for i, val in enumerate(supra):
            if val and not in_c:
                in_c = True
                start = i
            elif not val and in_c:
                in_c = False
                clusters.append((start, i - 1, float(np.sum(t_vec[start:i]))))
        if in_c:
            clusters.append((start, len(t_vec) - 1, float(np.sum(t_vec[start:]))))
        return clusters

    obs_clusters = get_clusters(t_obs)
    if (1 << n_subj) <= n_perm:
        bits = np.arange(1 << n_subj)[:, None] >> np.arange(n_subj)[None, :]
        signs = np.where(bits & 1, 1.0, -1.0)
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n_subj))
        signs[0] = 1.0

    max_cluster_mass = np.zeros(len(signs), dtype=np.float64)
    max_t_stat = np.zeros(len(signs), dtype=np.float64)
    for p_idx, s_vec in enumerate(signs):
        perm_data = subj_time_mat * s_vec[:, None]
        m_p = np.mean(perm_data, axis=0)
        se_p = np.std(perm_data, axis=0, ddof=1) / np.sqrt(n_subj)
        t_p = m_p / (se_p + EPS)
        max_t_stat[p_idx] = float(np.max(t_p))
        c_p = get_clusters(t_p)
        if len(c_p) > 0:
            max_cluster_mass[p_idx] = max(c[2] for c in c_p)

    p_fwer_max_t = np.array([float(np.mean(max_t_stat >= t_obs[w])) for w in range(n_win)])
    cluster_results = []
    for s_idx, e_idx, mass in obs_clusters:
        p_clust = float(np.mean(max_cluster_mass >= mass))
        cluster_results.append({
            'start_window_idx': int(s_idx),
            'end_window_idx': int(e_idx),
            'start_ms': int(WINDOW_CENTERS_MS[s_idx]),
            'end_ms': int(WINDOW_CENTERS_MS[e_idx]),
            'cluster_mass_t': mass,
            'p_cluster_fwer': p_clust,
        })

    peak_w = int(np.argmax(mean_tc))
    return {
        'n_subjects': int(n_subj),
        'mean_tc': mean_tc.tolist(),
        'se_tc': se_tc.tolist(),
        't_obs': t_obs.tolist(),
        'p_uncorr_one_sided': p_uncorr.tolist(),
        'p_fwer_max_t': p_fwer_max_t.tolist(),
        'peak_window_idx': peak_w,
        'peak_latency_ms': int(WINDOW_CENTERS_MS[peak_w]),
        'peak_rho_mean': float(mean_tc[peak_w]),
        'peak_rho_se': float(se_tc[peak_w]),
        'peak_t': float(t_obs[peak_w]),
        'peak_p_uncorr': float(p_uncorr[peak_w]),
        'peak_p_fwer_max_t': float(p_fwer_max_t[peak_w]),
        'clusters': cluster_results,
    }


def load_fsaverage_rois_and_patches():
    """
    Build fsaverage ico4 (5124 vertices) anatomical ROIs from FreeSurfer aparc:
      - Bilateral Heschl's gyrus (HG: transversetemporal-lh/rh, deep sulcal wall, tangential)
      - Bilateral Lateral STG (superiortemporal-lh/rh, gyral crown, radial)
      - Bilateral BanksSTS (bankssts-lh/rh)
      - Whole Auditory Cortex ROI (HG + Lateral STG + BanksSTS)
    And precompute 20 mm cortical searchlight patches on fsaverage ico4.
    """
    src_path = ROOT / 'analysis/EEG_full_native_anatomical_geometry_v4/fsaverage-ico4-src.fif'
    subjects_dir = ROOT / 'analysis/EEG_full_native_anatomical_geometry_v4/subjects'
    src = mne.read_source_spaces(src_path, verbose=False)
    vert_lh = src[0]['vertno']  # 2562 vertices
    vert_rh = src[1]['vertno']  # 2562 vertices
    pos_lh = src[0]['rr'][vert_lh] * 1000.0  # mm
    pos_rh = src[1]['rr'][vert_rh] * 1000.0  # mm
    nn_lh = src[0]['nn'][vert_lh]
    nn_rh = src[1]['nn'][vert_rh]

    labels_lh = {
        l.name: l for l in mne.read_labels_from_annot(
            'fsaverage', parc='aparc', hemi='lh', subjects_dir=subjects_dir, verbose=False
        )
    }
    labels_rh = {
        l.name: l for l in mne.read_labels_from_annot(
            'fsaverage', parc='aparc', hemi='rh', subjects_dir=subjects_dir, verbose=False
        )
    }

    def mask_for_label(lname_base):
        m_lh = np.isin(vert_lh, labels_lh[f'{lname_base}-lh'].vertices)
        m_rh = np.isin(vert_rh, labels_rh[f'{lname_base}-rh'].vertices)
        return np.concatenate([m_lh, m_rh])

    hg_mask = mask_for_label('transversetemporal')
    stg_mask = mask_for_label('superiortemporal')
    sts_mask = mask_for_label('bankssts')
    aud_mask = hg_mask | stg_mask | sts_mask

    pos_all = np.vstack([pos_lh, pos_rh])
    nn_all = np.vstack([nn_lh, nn_rh])
    r_hat = pos_all / (np.linalg.norm(pos_all, axis=1, keepdims=True) + EPS)
    radial_cos = np.abs(np.sum(nn_all * r_hat, axis=1))
    tangential_sin = np.sqrt(np.clip(1.0 - radial_cos ** 2, 0.0, 1.0))

    d_lh = cdist(pos_lh, pos_lh)
    d_rh = cdist(pos_rh, pos_rh)
    patches = []
    for i in range(len(vert_lh)):
        patches.append(np.where(d_lh[i] <= 20.0)[0])
    for i in range(len(vert_rh)):
        patches.append(len(vert_lh) + np.where(d_rh[i] <= 20.0)[0])

    target_ids = [f'lh:{int(v)}' for v in vert_lh] + [f'rh:{int(v)}' for v in vert_rh]

    return {
        'n_vertices': 5124,
        'target_ids': target_ids,
        'hg_mask': hg_mask,
        'stg_mask': stg_mask,
        'sts_mask': sts_mask,
        'aud_mask': aud_mask,
        'radial_cos': radial_cos,
        'tangential_sin': tangential_sin,
        'patches': patches,
        'pos_mm': pos_all,
    }


def load_csr_and_ids_from_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    A = sp.csr_matrix((z['A_data'], z['A_indices'], z['A_indptr']), shape=tuple(z['A_shape']))
    source_ids = [str(s) for s in z['source_ids'].tolist()]
    return A.toarray(), source_ids


def morph_native_to_fsaverage(X_native, native_feature_ids, A_dense, geom_source_ids):
    """
    Safely morph X_native (64, N_retained, 61) to fsaverage ico4 (64, 5124, 61)
    by matching native_feature_ids into geom_source_ids.
    """
    if X_native.shape[1] == A_dense.shape[1] and list(native_feature_ids) == list(geom_source_ids):
        return np.einsum('ij,cjt->cit', A_dense, X_native)
    id_to_col = {sid: idx for idx, sid in enumerate(geom_source_ids)}
    cols = [id_to_col[str(fid)] for fid in native_feature_ids]
    A_sub = A_dense[:, cols]
    return np.einsum('ij,cjt->cit', A_sub, X_native)


def main():
    # 1. Load Three-Tier Speech Model RDMs
    m_npz = np.load(OUT_DIR / 'v2_three_tier_speech_model_rdms.npz', allow_pickle=True)
    rdm_m1 = m_npz['rdm_model1_vec']
    rdm_m2 = m_npz['rdm_model2_vec']
    rdm_m3 = m_npz['rdm_model3_vec']
    rdm_leg = m_npz['rdm_legacy_vec']

    rank_m1 = rankdata(rdm_m1)
    rank_m2 = rankdata(rdm_m2)
    rank_m3 = rankdata(rdm_m3)
    rank_leg = rankdata(rdm_leg)

    rank_m2_given_m1 = residualize_rank_vec(rank_m2, rank_m1)
    rank_m3_given_m12 = residualize_rank_vec(rank_m3, np.column_stack([rank_m1, rank_m2]))

    fs_info = load_fsaverage_rois_and_patches()
    print(f"fsaverage ico4 ROIs loaded: HG={int(np.sum(fs_info['hg_mask']))} verts, "
          f"Lateral STG={int(np.sum(fs_info['stg_mask']))} verts, "
          f"BanksSTS={int(np.sum(fs_info['sts_mask']))} verts")

    # -------------------------------------------------------------------------
    # 2. Sensor-Space Spatiotemporal RSA (MEG All/Grads/Mags & EEG 64-ch)
    # -------------------------------------------------------------------------
    sensor_subjects_8 = {
        'E1': ['run1', 'run2'],
        'E6': ['run1', 'run3'],
        'E7': ['run1', 'run2'],
        'E9': ['run1', 'run2', 'run3', 'run4'],
        'E10': ['run1', 'run2', 'run3', 'run4'],
        'E11': ['run1', 'run3', 'run4'],
        'E13': ['run1', 'run2', 'run4'],
        'E18': ['run2', 'run3', 'run4'],
    }
    meg_source_subjects_9 = ['E2', 'E3', 'E4', 'E9', 'E10', 'E14', 'E15', 'E17', 'E18']

    meg_sensor_runs = {}
    for subj, runs in sensor_subjects_8.items():
        base = ROOT / 'derivatives' / subj / 'sensors_included_folds_v2'
        if not base.exists():
            base = ROOT / 'derivatives' / subj / 'sensors_approved_v1'
        valid_pairs = [
            (base / f'{r}_grad.npz', base / f'{r}_mag.npz')
            for r in runs
            if (base / f'{r}_grad.npz').exists() and (base / f'{r}_mag.npz').exists()
        ]
        if valid_pairs:
            meg_sensor_runs[subj] = valid_pairs

    for subj in meg_source_subjects_9:
        if subj not in meg_sensor_runs:
            base = ROOT / 'derivatives' / subj / 'modality_meg_source_v1/source_samefold_nominal_export_v1'
            grad_files = sorted(base.glob('run*_grad.npz'))
            valid_pairs = [
                (gf, base / gf.name.replace('_grad.npz', '_mag.npz'))
                for gf in grad_files
                if gf.exists() and (base / gf.name.replace('_grad.npz', '_mag.npz')).exists()
            ]
            if valid_pairs:
                meg_sensor_runs[subj] = valid_pairs

    meg_sensor_modalities = {'meg_all_306': {}, 'meg_grad_204': {}, 'meg_mag_102': {}}
    for subj, run_pairs in sorted(meg_sensor_runs.items()):
        grads = [np.load(gf)['data'] for gf, _ in run_pairs]
        mags = [np.load(mf)['data'] for _, mf in run_pairs]
        g_mean = np.mean(grads, axis=0)
        m_mean = np.mean(mags, axis=0)
        g_norm = g_mean / (np.std(g_mean) + EPS)
        m_norm = m_mean / (np.std(m_mean) + EPS)
        all_306 = np.concatenate([g_norm, m_norm], axis=1)

        meg_sensor_modalities['meg_grad_204'][subj] = compute_window_rdm_from_feature_time(g_mean)
        meg_sensor_modalities['meg_mag_102'][subj] = compute_window_rdm_from_feature_time(m_mean)
        meg_sensor_modalities['meg_all_306'][subj] = compute_window_rdm_from_feature_time(all_306)

    eeg_sensor_rdms = {}
    eeg_sensor_erp_data = {}
    for subj, runs in sorted(sensor_subjects_8.items()):
        eegs = []
        for r in runs:
            p1 = ROOT / 'derivatives' / subj / 'sensors_included_folds_v2' / f'{r}_eeg.npz'
            p2 = ROOT / 'derivatives' / subj / 'sensors_approved_v1' / f'{r}_eeg.npz'
            if p1.exists():
                eegs.append(np.load(p1)['data'])
            elif p2.exists():
                eegs.append(np.load(p2)['data'])
        if eegs:
            e_mean = np.mean(eegs, axis=0)
            eeg_sensor_erp_data[subj] = e_mean
            eeg_sensor_rdms[subj] = compute_window_rdm_from_feature_time(e_mean)

    def evaluate_rdm_cohort(subj_rdm_dict):
        subjs = sorted(subj_rdm_dict.keys())
        n_s = len(subjs)
        scores = {
            'model1_loudness': np.zeros((n_s, N_WINDOWS)),
            'model2_tonotopy': np.zeros((n_s, N_WINDOWS)),
            'model3_manner': np.zeros((n_s, N_WINDOWS)),
            'legacy_32band': np.zeros((n_s, N_WINDOWS)),
            'partial_m2_given_m1': np.zeros((n_s, N_WINDOWS)),
            'partial_m3_given_m12': np.zeros((n_s, N_WINDOWS)),
        }
        for si, s in enumerate(subjs):
            n_ranks = fast_rank_rows(subj_rdm_dict[s])
            scores['model1_loudness'][si] = fast_spearman_vec_vs_mat(rank_m1, n_ranks)
            scores['model2_tonotopy'][si] = fast_spearman_vec_vs_mat(rank_m2, n_ranks)
            scores['model3_manner'][si] = fast_spearman_vec_vs_mat(rank_m3, n_ranks)
            scores['legacy_32band'][si] = fast_spearman_vec_vs_mat(rank_leg, n_ranks)

            n_res_m1 = residualize_rank_mat(n_ranks, rank_m1)
            n_res_m12 = residualize_rank_mat(n_ranks, np.column_stack([rank_m1, rank_m2]))
            scores['partial_m2_given_m1'][si] = fast_spearman_vec_vs_mat(rank_m2_given_m1, n_res_m1)
            scores['partial_m3_given_m12'][si] = fast_spearman_vec_vs_mat(rank_m3_given_m12, n_res_m12)

        out = {'subjects': subjs}
        for k, mat in scores.items():
            out[k] = sign_flip_cluster_1d(mat)
        return out, scores

    sensor_results = {}
    for mod_name, rdm_dict in meg_sensor_modalities.items():
        sensor_results[mod_name], _ = evaluate_rdm_cohort(rdm_dict)
    sensor_results['eeg_64ch_all8'], _ = evaluate_rdm_cohort(eeg_sensor_rdms)
    eeg_paired5_rdms = {s: eeg_sensor_rdms[s] for s in ['E6', 'E9', 'E10', 'E11', 'E18']}
    sensor_results['eeg_64ch_source5'], _ = evaluate_rdm_cohort(eeg_paired5_rdms)

    # -------------------------------------------------------------------------
    # 3. H3 Bonus: EEG 64-Channel Multivariate RSA vs. Pairwise Univariate ERP
    # -------------------------------------------------------------------------
    eeg_subjs = sorted(eeg_sensor_erp_data.keys())
    erp_win_means = np.zeros((len(eeg_subjs), 64, 64, N_WINDOWS), dtype=np.float64)
    for si, s in enumerate(eeg_subjs):
        for w in range(N_WINDOWS):
            s0 = w * STEP_SAMPLES
            erp_win_means[si, :, :, w] = np.mean(eeg_sensor_erp_data[s][:, :, s0:s0 + WINDOW_SAMPLES], axis=2)

    univariate_win_stats = []
    for w in range(N_WINDOWS):
        w_data = erp_win_means[:, :, :, w]
        diffs = w_data[:, IU_64[0], :] - w_data[:, IU_64[1], :]
        m_diff = np.mean(diffs, axis=0)
        se_diff = np.std(diffs, axis=0, ddof=1) / np.sqrt(len(eeg_subjs))
        abs_t = np.abs(m_diff / (se_diff + EPS))
        n_tests = N_PAIRS * 64
        t_bonf = float(student_t.ppf(1.0 - 0.025 / n_tests, df=len(eeg_subjs) - 1))
        t_unc05 = float(student_t.ppf(0.975, df=len(eeg_subjs) - 1))
        frac_bonf = float(np.mean(abs_t >= t_bonf))
        frac_unc05 = float(np.mean(abs_t >= t_unc05))
        rms_gfp = np.sqrt(np.mean(w_data ** 2, axis=2))
        gfp_diffs = np.abs(rms_gfp[:, IU_64[0]] - rms_gfp[:, IU_64[1]])
        gfp_rho_m1 = [float(spearmanr(gfp_diffs[si], rdm_m1).statistic) for si in range(len(eeg_subjs))]
        univariate_win_stats.append({
            'window_center_ms': int(WINDOW_CENTERS_MS[w]),
            'median_abs_t_across_129024_channel_pairs': float(np.median(abs_t)),
            'fraction_pairs_surviving_bonferroni_fwer05': frac_bonf,
            'fraction_pairs_uncorrected_p05': frac_unc05,
            'univariate_gfp_rsa_model1_mean_rho': float(np.mean(gfp_rho_m1)),
            'multivariate_64ch_rsa_model1_mean_rho': float(sensor_results['eeg_64ch_all8']['model1_loudness']['mean_tc'][w]),
            'multivariate_64ch_rsa_model1_t': float(sensor_results['eeg_64ch_all8']['model1_loudness']['t_obs'][w]),
        })

    # -------------------------------------------------------------------------
    # 4. Cortical Source Space (MEG & EEG under Shrunk vs. Isotropic MNE)
    # -------------------------------------------------------------------------
    eeg_source_subjects = ['E6', 'E9', 'E10', 'E11', 'E18']
    eeg_runs_map = {
        'E6': ['run1', 'run3'],
        'E9': ['run1', 'run2', 'run3', 'run4'],
        'E10': ['run1', 'run2'],
        'E11': ['run1', 'run2', 'run3', 'run4'],
        'E18': ['run2', 'run3', 'run4'],
    }

    def get_morph_matrix_and_ids(subj):
        p1 = ROOT / 'analysis/corrected_MEG_anatomical_geometry_v1' / subj / 'geometry_matrix.npz'
        p2 = ROOT / 'analysis/EEG_full_native_anatomical_geometry_v4' / subj / 'geometry_matrix.npz'
        if p1.exists():
            return load_csr_and_ids_from_npz(p1)
        elif p2.exists():
            return load_csr_and_ids_from_npz(p2)
        else:
            subj_dir = ROOT / 'analysis/EEG_full_native_anatomical_geometry_v4/subjects'
            src_fs = mne.read_source_spaces(ROOT / 'analysis/EEG_full_native_anatomical_geometry_v4/fsaverage-ico4-src.fif', verbose=False)
            src_sub_path = ROOT / f'derivatives/{subj}/eeg_anatomy_head_qa_v1/anatomy/{subj}-ico4-src.fif'
            src_sub = mne.read_source_spaces(src_sub_path, verbose=False)
            morph = mne.compute_source_morph(
                src_sub, subject_from=subj, subject_to='fsaverage',
                src_to=src_fs, subjects_dir=subj_dir, verbose=False
            )
            s_ids = [f'lh:{int(v)}' for v in src_sub[0]['vertno']] + [f'rh:{int(v)}' for v in src_sub[1]['vertno']]
            return morph.morph_mat.toarray(), s_ids

    eeg_fs_sources = {'saved_loo_shrunk': {}, 'support_trace_isotropic': {}}
    for subj in eeg_source_subjects:
        A_dense, geom_sids = get_morph_matrix_and_ids(subj)
        for arm in ('saved_loo_shrunk', 'support_trace_isotropic'):
            run_arrays = []
            for r in eeg_runs_map[subj]:
                if subj == 'E10':
                    p = ROOT / f'analysis/EEG_native_full_v4/prepared/E10/{r}_{arm}_full_eeg_source.npz'
                else:
                    p = ROOT / f'analysis/EEG_native_full_v4/full_v1/sources/{subj}/{r}_{arm}_full_eeg_source.npz'
                z_nat = np.load(p)
                X_native = z_nat['data']
                fids = z_nat['feature_ids']
                X_fs = morph_native_to_fsaverage(X_native, fids, A_dense, geom_sids)
                run_arrays.append(X_fs)
            eeg_fs_sources[arm][subj] = np.mean(run_arrays, axis=0)

    meg_fs_sources = {'saved_loo_shrunk': {}, 'support_trace_isotropic': {}}
    for subj in meg_source_subjects_9:
        A_dense, geom_sids = load_csr_and_ids_from_npz(
            ROOT / 'analysis/corrected_MEG_anatomical_geometry_v1' / subj / 'geometry_matrix.npz'
        )
        src_dir = ROOT / 'derivatives' / subj / 'modality_meg_source_v1/source_samefold_nominal_export_v1'
        op_base = ROOT / 'derivatives' / subj / 'modality_meg_source_v1/operators_nominal_output_v1'
        src_files = sorted(src_dir.glob('run*_meg_source.npz'))

        shrunk_runs = []
        iso_runs = []
        for sf in src_files:
            run_id = sf.name.split('_')[0]
            z_meg = np.load(sf)
            X_shrunk_nat = z_meg['data']
            fids_meg = z_meg['feature_ids']
            X_shrunk_fs = morph_native_to_fsaverage(X_shrunk_nat, fids_meg, A_dense, geom_sids)
            shrunk_runs.append(X_shrunk_fs)

            op_dir = op_base / f'{run_id}_meg_operators_v1'
            if op_dir.exists():
                with open(op_dir / 'operators_manifest.json') as f:
                    op_man = json.load(f)
                r_meta = op_man['runs'][0]
                fwd = mne.read_forward_solution(r_meta['fwd'], verbose=False)
                cov = mne.read_cov(r_meta['cov'], verbose=False)
                info = mne.io.read_info(r_meta['info'], verbose=False)
                rank_dict = r_meta['rank']

                C_orig = cov.data.copy()
                ch_names = cov['names']
                C_iso = np.zeros_like(C_orig)
                grad_idx = [i for i, ch in enumerate(ch_names) if ch.endswith('2') or ch.endswith('3')]
                mag_idx = [i for i, ch in enumerate(ch_names) if ch.endswith('1')]
                for idx_list in (grad_idx, mag_idx):
                    if len(idx_list) > 0:
                        tr_mean = np.mean(np.diag(C_orig)[idx_list])
                        for ii in idx_list:
                            C_iso[ii, ii] = tr_mean
                cov_iso = mne.Covariance(C_iso, list(cov['names']), list(cov['bads']), cov['projs'], int(cov['nfree']))
                inv_iso = mne.minimum_norm.make_inverse_operator(
                    info, fwd, cov_iso, loose=0.2, depth=0.8, use_cps=True, rank=rank_dict, verbose=False
                )

                g_data = np.load(src_dir / f'{run_id}_grad.npz')
                m_data = np.load(src_dir / f'{run_id}_mag.npz')
                ch_map = {str(name): g_data['data'][:, i, :] for i, name in enumerate(g_data['feature_ids'])}
                for i, name in enumerate(m_data['feature_ids']):
                    ch_map[str(name)] = m_data['data'][:, i, :]
                meg_306 = np.stack([ch_map[ch] for ch in info['ch_names']], axis=1)
                naves = g_data['actual_nave'] if 'actual_nave' in g_data.files else np.ones(64, dtype=int)

                X_iso_nat = np.empty_like(X_shrunk_nat)
                for ci in range(64):
                    ev = mne.EvokedArray(meg_306[ci], info.copy(), tmin=0.0, nave=int(naves[ci]), verbose=False)
                    stc = mne.minimum_norm.apply_inverse(
                        ev, inv_iso, lambda2=1.0 / 9.0, method='MNE',
                        pick_ori='normal', use_cps=True, verbose=False
                    )
                    X_iso_nat[ci] = stc.data
                X_iso_fs = morph_native_to_fsaverage(X_iso_nat, fids_meg, A_dense, geom_sids)
                iso_runs.append(X_iso_fs)
            else:
                iso_runs.append(X_shrunk_fs)

        meg_fs_sources['saved_loo_shrunk'][subj] = np.mean(shrunk_runs, axis=0)
        meg_fs_sources['support_trace_isotropic'][subj] = np.mean(iso_runs, axis=0)

    # -------------------------------------------------------------------------
    # 5. Evaluate ROI Time Courses (HG vs. Lateral STG) & Partial RSA Cascade
    # -------------------------------------------------------------------------
    rois = {
        'HG_sulcal_tangential': fs_info['hg_mask'],
        'Lateral_STG_gyral_radial': fs_info['stg_mask'],
        'BanksSTS': fs_info['sts_mask'],
        'Whole_Auditory_Cortex': fs_info['aud_mask'],
    }

    roi_results = {}
    raw_roi_scores = {}

    for modality_label, source_dict_by_arm in [('MEG', meg_fs_sources), ('EEG', eeg_fs_sources)]:
        roi_results[modality_label] = {}
        raw_roi_scores[modality_label] = {}
        for arm in ('saved_loo_shrunk', 'support_trace_isotropic'):
            roi_results[modality_label][arm] = {}
            raw_roi_scores[modality_label][arm] = {}
            subj_dict = source_dict_by_arm[arm]
            for roi_name, roi_mask in rois.items():
                roi_rdms = {
                    subj: compute_window_rdm_from_feature_time(X[:, roi_mask, :])
                    for subj, X in subj_dict.items()
                }
                res_summary, res_mats = evaluate_rdm_cohort(roi_rdms)
                roi_results[modality_label][arm][roi_name] = res_summary
                raw_roi_scores[modality_label][arm][roi_name] = res_mats

            roi_results[modality_label][arm]['paired3_E9_E10_E18'] = {}
            for roi_name, roi_mask in rois.items():
                roi_rdms_p3 = {
                    subj: compute_window_rdm_from_feature_time(subj_dict[subj][:, roi_mask, :])
                    for subj in ['E9', 'E10', 'E18']
                }
                res_p3, _ = evaluate_rdm_cohort(roi_rdms_p3)
                roi_results[modality_label][arm]['paired3_E9_E10_E18'][roi_name] = res_p3

    covariance_comparison = {}
    for modality_label in ('MEG', 'EEG'):
        covariance_comparison[modality_label] = {}
        for roi_name in ('HG_sulcal_tangential', 'Lateral_STG_gyral_radial', 'Whole_Auditory_Cortex'):
            covariance_comparison[modality_label][roi_name] = {}
            for m_key in ('model1_loudness', 'model2_tonotopy', 'model3_manner', 'partial_m2_given_m1', 'partial_m3_given_m12'):
                s_mat = raw_roi_scores[modality_label]['saved_loo_shrunk'][roi_name][m_key]
                i_mat = raw_roi_scores[modality_label]['support_trace_isotropic'][roi_name][m_key]
                diff_mat = s_mat - i_mat
                covariance_comparison[modality_label][roi_name][m_key] = sign_flip_cluster_1d(diff_mat)

    # -------------------------------------------------------------------------
    # 6. Auditory Cortex Searchlight Maps (20 mm radius on fsaverage ico4)
    # -------------------------------------------------------------------------
    patches = fs_info['patches']
    aud_indices = np.where(fs_info['aud_mask'])[0]

    def compute_auditory_searchlight_maps(source_dict):
        subjs = sorted(source_dict.keys())
        n_s = len(subjs)
        n_aud = len(aud_indices)
        maps = {
            k: np.zeros((n_s, n_aud, N_WINDOWS), dtype=np.float64)
            for k in ('model1', 'model2', 'model3', 'partial_m2', 'partial_m3')
        }
        for si, subj in enumerate(subjs):
            X = source_dict[subj]
            for vi_idx, v_center in enumerate(aud_indices):
                patch_v = patches[v_center]
                rdms_v = compute_window_rdm_from_feature_time(X[:, patch_v, :])
                ranks_v = fast_rank_rows(rdms_v)
                maps['model1'][si, vi_idx] = fast_spearman_vec_vs_mat(rank_m1, ranks_v)
                maps['model2'][si, vi_idx] = fast_spearman_vec_vs_mat(rank_m2, ranks_v)
                maps['model3'][si, vi_idx] = fast_spearman_vec_vs_mat(rank_m3, ranks_v)
                res1 = residualize_rank_mat(ranks_v, rank_m1)
                res12 = residualize_rank_mat(ranks_v, np.column_stack([rank_m1, rank_m2]))
                maps['partial_m2'][si, vi_idx] = fast_spearman_vec_vs_mat(rank_m2_given_m1, res1)
                maps['partial_m3'][si, vi_idx] = fast_spearman_vec_vs_mat(rank_m3_given_m12, res12)
        return maps

    meg_shrunk_aud_sl = compute_auditory_searchlight_maps(meg_fs_sources['saved_loo_shrunk'])
    eeg_shrunk_aud_sl = compute_auditory_searchlight_maps(eeg_fs_sources['saved_loo_shrunk'])
    eeg_iso_aud_sl = compute_auditory_searchlight_maps(eeg_fs_sources['support_trace_isotropic'])

    hg_in_aud = fs_info['hg_mask'][aud_indices]
    stg_in_aud = fs_info['stg_mask'][aud_indices]

    double_dissociation_summary = {
        'anatomical_orientation_biophysics': {
            'HG_mean_tangential_sin': float(np.mean(fs_info['tangential_sin'][fs_info['hg_mask']])),
            'HG_mean_radial_cos': float(np.mean(fs_info['radial_cos'][fs_info['hg_mask']])),
            'Lateral_STG_mean_tangential_sin': float(np.mean(fs_info['tangential_sin'][fs_info['stg_mask']])),
            'Lateral_STG_mean_radial_cos': float(np.mean(fs_info['radial_cos'][fs_info['stg_mask']])),
        },
        'early_tonotopic_window_75_150ms_indices_2_to_5': {
            'MEG_shrunk_HG_model2_mean_rho': float(np.mean(meg_shrunk_aud_sl['model2'][:, hg_in_aud, 2:6])),
            'MEG_shrunk_STG_model2_mean_rho': float(np.mean(meg_shrunk_aud_sl['model2'][:, stg_in_aud, 2:6])),
            'EEG_shrunk_HG_model2_mean_rho': float(np.mean(eeg_shrunk_aud_sl['model2'][:, hg_in_aud, 2:6])),
            'EEG_shrunk_STG_model2_mean_rho': float(np.mean(eeg_shrunk_aud_sl['model2'][:, stg_in_aud, 2:6])),
        },
        'late_manner_window_125_225ms_indices_4_to_8': {
            'EEG_shrunk_STG_model3_mean_rho': float(np.mean(eeg_shrunk_aud_sl['model3'][:, stg_in_aud, 4:9])),
            'EEG_iso_STG_model3_mean_rho': float(np.mean(eeg_iso_aud_sl['model3'][:, stg_in_aud, 4:9])),
            'EEG_shrunk_HG_model3_mean_rho': float(np.mean(eeg_shrunk_aud_sl['model3'][:, hg_in_aud, 4:9])),
            'MEG_shrunk_STG_model3_mean_rho': float(np.mean(meg_shrunk_aud_sl['model3'][:, stg_in_aud, 4:9])),
            'MEG_shrunk_HG_model3_mean_rho': float(np.mean(meg_shrunk_aud_sl['model3'][:, hg_in_aud, 4:9])),
        }
    }

    # -------------------------------------------------------------------------
    # 7. Cross-Modal Convergence Test (H1 Bonus) on Paired Subjects (E9, E10, E18)
    # -------------------------------------------------------------------------
    paired_subjs = ['E9', 'E10', 'E18']
    cross_modal_convergence = {}
    for arm in ('saved_loo_shrunk', 'support_trace_isotropic'):
        cross_modal_convergence[arm] = {}
        for roi_name, roi_mask in rois.items():
            first_order_corrs = np.zeros((len(paired_subjs), N_WINDOWS), dtype=np.float64)
            second_order_rdm_corrs = np.zeros((len(paired_subjs), N_WINDOWS), dtype=np.float64)
            for si, subj in enumerate(paired_subjs):
                X_m = meg_fs_sources[arm][subj][:, roi_mask, :]
                X_e = eeg_fs_sources[arm][subj][:, roi_mask, :]
                rdm_m = compute_window_rdm_from_feature_time(X_m)
                rdm_e = compute_window_rdm_from_feature_time(X_e)
                for w in range(N_WINDOWS):
                    s0 = w * STEP_SAMPLES
                    s1 = s0 + WINDOW_SAMPLES
                    v_m = X_m[:, :, s0:s1].ravel()
                    v_e = X_e[:, :, s0:s1].ravel()
                    first_order_corrs[si, w] = float(np.corrcoef(v_m, v_e)[0, 1])
                    second_order_rdm_corrs[si, w] = float(spearmanr(rdm_m[w], rdm_e[w]).statistic)

            cross_modal_convergence[arm][roi_name] = {
                'paired_subjects': paired_subjs,
                'first_order_waveform_r_mean_tc': np.mean(first_order_corrs, axis=0).tolist(),
                'first_order_abs_waveform_r_mean_tc': np.mean(np.abs(first_order_corrs), axis=0).tolist(),
                'second_order_rdm_rho_mean_tc': np.mean(second_order_rdm_corrs, axis=0).tolist(),
                'second_order_minus_first_order_mean_tc': np.mean(second_order_rdm_corrs - first_order_corrs, axis=0).tolist(),
                'mean_across_all_windows_first_order_r': float(np.mean(first_order_corrs)),
                'mean_across_all_windows_second_order_rho': float(np.mean(second_order_rdm_corrs)),
                'peak_second_order_rdm_rho': float(np.max(np.mean(second_order_rdm_corrs, axis=0))),
                'peak_second_order_latency_ms': int(WINDOW_CENTERS_MS[int(np.argmax(np.mean(second_order_rdm_corrs, axis=0)))]),
            }

    # -------------------------------------------------------------------------
    # 8. Save All Results (JSON, NPZ)
    # -------------------------------------------------------------------------
    full_summary = {
        'window_centers_ms': WINDOW_CENTERS_MS,
        'window_width_ms': 50,
        'window_step_ms': 25,
        'sensor_space_rsa': sensor_results,
        'cortical_roi_rsa': roi_results,
        'shrunk_minus_isotropic_covariance_comparison': covariance_comparison,
        'double_dissociation_hg_vs_lateral_stg': double_dissociation_summary,
        'h1_bonus_cross_modal_convergence': cross_modal_convergence,
        'h3_bonus_eeg_multivariate_rsa_vs_univariate_erp': univariate_win_stats,
    }

    json_out = OUT_DIR / 'v2_soto_meg_eeg_rsa_group_inference_summary.json'
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(full_summary, f, indent=2)

    npz_out = OUT_DIR / 'v2_soto_auditory_searchlight_and_roi_arrays.npz'
    np.savez_compressed(
        npz_out,
        window_centers_ms=np.asarray(WINDOW_CENTERS_MS),
        aud_indices=aud_indices,
        hg_in_aud=hg_in_aud,
        stg_in_aud=stg_in_aud,
        meg_shrunk_aud_model1=meg_shrunk_aud_sl['model1'],
        meg_shrunk_aud_model2=meg_shrunk_aud_sl['model2'],
        meg_shrunk_aud_model3=meg_shrunk_aud_sl['model3'],
        meg_shrunk_aud_partial_m2=meg_shrunk_aud_sl['partial_m2'],
        meg_shrunk_aud_partial_m3=meg_shrunk_aud_sl['partial_m3'],
        eeg_shrunk_aud_model1=eeg_shrunk_aud_sl['model1'],
        eeg_shrunk_aud_model2=eeg_shrunk_aud_sl['model2'],
        eeg_shrunk_aud_model3=eeg_shrunk_aud_sl['model3'],
        eeg_shrunk_aud_partial_m2=eeg_shrunk_aud_sl['partial_m2'],
        eeg_shrunk_aud_partial_m3=eeg_shrunk_aud_sl['partial_m3'],
        eeg_iso_aud_model1=eeg_iso_aud_sl['model1'],
        eeg_iso_aud_model2=eeg_iso_aud_sl['model2'],
        eeg_iso_aud_model3=eeg_iso_aud_sl['model3'],
        eeg_iso_aud_partial_m2=eeg_iso_aud_sl['partial_m2'],
        eeg_iso_aud_partial_m3=eeg_iso_aud_sl['partial_m3'],
    )

    print('Saved full summary to:', json_out)
    print('Saved searchlight arrays to:', npz_out)


if __name__ == '__main__':
    main()
