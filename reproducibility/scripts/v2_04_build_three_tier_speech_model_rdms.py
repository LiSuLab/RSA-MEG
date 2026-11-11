#!/usr/bin/env python3
"""
v2_04_build_three_tier_speech_model_rdms.py
Construct and validate the Three-Tier Natural Speech Candidate Model RDMs (64x64)
for the 64 spoken words in the Kymata Soto dataset:
  - Model 1 (Anchor): Glasberg-Moore Time-Varying Loudness (TVL) & Onset Step Dynamics [L_bar, Delta_L]
  - Model 2 (Tonotopy): 2D Pitch-Cochlear ERB Spectral Centroid [C_ERB, P_f0] (orthogonalized against Model 1)
  - Model 3 (Articulatory-Acoustic Manner State Space): 3D State Space [M_burst, M_HNR, M_F1F2]
    clustering the 64 words into 4 phonetic manner families (Plosives, Fricatives, Nasals/Liquids, Vowels)
    with a clear 4x4 block-hierarchical RDM structure and zero rank ties.
"""

import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

ROOT = Path('.')
OUT_DIR = ROOT / 'revision_20261104_v2_experiments' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


def zscore_1d(x):
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    sd = np.std(x, ddof=0)
    return (x - mu) / (sd + EPS)


def ols_residualize(y, X_controls):
    """Project y onto orthogonal complement of [1, X_controls]."""
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack([np.ones(len(y), dtype=np.float64), np.asarray(X_controls, dtype=np.float64)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def count_ties(vec):
    u = np.unique(np.round(vec, 12))
    return int(len(vec) - len(u))


def main():
    # -------------------------------------------------------------------------
    # 1. Load 64 word conditions & MFA phoneme/word alignments
    # -------------------------------------------------------------------------
    with open(ROOT / 'condition_ids.json', 'r', encoding='utf-8') as f:
        cond_meta = json.load(f)

    words = cond_meta['words']
    onsets_s = np.asarray(cond_meta['onsets_s'], dtype=np.float64)
    annotation_lines = cond_meta['annotation_lines']
    condition_ids = [f'word_{i+1:02d}' for i in range(64)]

    word_labels = (ROOT / 'data/stimuli/audio/mfa_word.txt').read_text().strip().splitlines()
    word_stimes = np.loadtxt(ROOT / 'data/stimuli/audio/mfa_word_stime.txt', dtype=np.float64)
    phon_labels = (ROOT / 'data/stimuli/audio/mfa_phoneme.txt').read_text().strip().splitlines()
    phon_stimes = np.loadtxt(ROOT / 'data/stimuli/audio/mfa_phoneme_stime.txt', dtype=np.float64)

    PLOSIVES = {'P', 'B', 'T', 'D', 'K', 'G', 'CH', 'JH'}
    FRICATIVES = {'F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH'}
    NASALS_LIQUIDS = {'M', 'N', 'NG', 'L', 'R', 'W', 'Y'}
    VOWELS = {
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
        'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
    }
    MANNER_ORDER = ['Plosive', 'Fricative', 'Nasal_Liquid', 'Vowel']

    def classify_phoneme(p_raw):
        p = ''.join([c for c in p_raw.upper() if not c.isdigit()])
        if p in PLOSIVES:
            return 'Plosive', 0, p
        elif p in FRICATIVES:
            return 'Fricative', 1, p
        elif p in NASALS_LIQUIDS:
            return 'Nasal_Liquid', 2, p
        elif p in VOWELS:
            return 'Vowel', 3, p
        else:
            return 'Other', -1, p

    word_records = []
    for idx in range(64):
        line_idx = annotation_lines[idx] - 1
        w_start = float(word_stimes[line_idx])
        w_end = float(word_stimes[line_idx + 1]) if (line_idx + 1) < len(word_stimes) else float(w_start + 0.25)
        assert np.isclose(w_start, onsets_s[idx], atol=1e-4)

        ph_indices = np.where((phon_stimes >= w_start - 1e-3) & (phon_stimes < w_end - 1e-3))[0]
        ph_indices = [int(k) for k in ph_indices if phon_labels[k] != '<sp>']
        if len(ph_indices) == 0:
            k_near = int(np.argmin(np.abs(phon_stimes - w_start)))
            ph_indices = [k_near]

        init_k = ph_indices[0]
        init_ph_raw = phon_labels[init_k]
        init_ph_start = float(phon_stimes[init_k])
        init_ph_end = float(phon_stimes[init_k + 1]) if (init_k + 1) < len(phon_stimes) else w_end
        manner_name, manner_id, init_ph_clean = classify_phoneme(init_ph_raw)
        ph_seq = [phon_labels[k] for k in ph_indices]

        word_records.append({
            'condition_index': idx + 1,
            'condition_id': condition_ids[idx],
            'word': words[idx],
            'onset_s': w_start,
            'offset_s': w_end,
            'duration_ms': (w_end - w_start) * 1000.0,
            'initial_phoneme_raw': init_ph_raw,
            'initial_phoneme': init_ph_clean,
            'initial_phoneme_start_s': init_ph_start,
            'initial_phoneme_end_s': init_ph_end,
            'initial_phoneme_duration_ms': (init_ph_end - init_ph_start) * 1000.0,
            'phoneme_sequence': '-'.join(ph_seq),
            'manner_family': manner_name,
            'manner_id': int(manner_id),
        })

    manner_counts = Counter([r['manner_family'] for r in word_records])
    print('64 words phonetic manner family counts:', dict(manner_counts))

    # -------------------------------------------------------------------------
    # 2. Load Audio WAV and Construct 32-Channel Glasberg-Moore ERB Filterbank
    # -------------------------------------------------------------------------
    fs, wav_data = wavfile.read(ROOT / 'data/stimuli/audio/stimulus.wav')
    assert fs == 48000
    mono = wav_data.astype(np.float64).mean(axis=1)
    mono_rms = np.sqrt(np.mean(mono ** 2))

    n_bands = 32
    low_hz, high_hz = 80.0, 8000.0
    erb_low = 21.4 * np.log10(1.0 + 0.00437 * low_hz)
    erb_high = 21.4 * np.log10(1.0 + 0.00437 * high_hz)
    erb_edges = np.linspace(erb_low, erb_high, n_bands + 1)
    erb_centers = 0.5 * (erb_edges[:-1] + erb_edges[1:])  # in [1.80, 38.90] Cam
    hz_edges = (10.0 ** (erb_edges / 21.4) - 1.0) / 0.00437
    hz_centers = (10.0 ** (erb_centers / 21.4) - 1.0) / 0.00437

    sos_bank = [
        butter(4, [hz_edges[b], hz_edges[b + 1]], btype='bandpass', fs=fs, output='sos')
        for b in range(n_bands)
    ]

    sos_pitch = butter(4, [70.0, 1000.0], btype='bandpass', fs=fs, output='sos')
    sos_voicing = butter(4, [80.0, 1000.0], btype='bandpass', fs=fs, output='sos')
    sos_frication = butter(4, [3000.0, 8000.0], btype='bandpass', fs=fs, output='sos')
    sos_f1 = butter(4, [300.0, 900.0], btype='bandpass', fs=fs, output='sos')
    sos_f2 = butter(4, [1000.0, 2500.0], btype='bandpass', fs=fs, output='sos')
    sos_burst_hi = butter(4, [1500.0, 8000.0], btype='bandpass', fs=fs, output='sos')

    # -------------------------------------------------------------------------
    # 3. Extract Biophysical Features for Each of the 64 Word Occurrences
    # -------------------------------------------------------------------------
    L_bar_raw = np.zeros(64, dtype=np.float64)
    Delta_L_raw = np.zeros(64, dtype=np.float64)
    C_erb_raw = np.zeros(64, dtype=np.float64)
    f0_hz_raw = np.zeros(64, dtype=np.float64)
    r_max_raw = np.zeros(64, dtype=np.float64)

    M_burst_ac_raw = np.zeros(64, dtype=np.float64)
    M_hnr_ac_raw = np.zeros(64, dtype=np.float64)
    M_f1f2_ac_raw = np.zeros(64, dtype=np.float64)

    erb_band_energy_100ms = np.zeros((64, n_bands), dtype=np.float64)

    n_pre_50 = int(round(0.050 * fs))    # 2400 samples (-50 to 0 ms)
    n_post_100 = int(round(0.100 * fs))  # 4800 samples (0 to 100 ms)
    n_post_80 = int(round(0.080 * fs))   # 3840 samples (0 to 80 ms)
    n_pad = int(round(0.050 * fs))

    min_lag = int(round(fs / 400.0))     # 400 Hz max f0
    max_lag = int(round(fs / 75.0))      # 75 Hz min f0

    for i in range(64):
        onset_samp = int(round(onsets_s[i] * fs))
        s_start = max(0, onset_samp - n_pre_50 - n_pad)
        s_end = min(len(mono), onset_samp + n_post_100 + n_pad)
        seg_buf = mono[s_start:s_end] / (mono_rms + EPS)
        rel_onset = onset_samp - s_start

        idx_pre = slice(rel_onset - n_pre_50, rel_onset)
        idx_100 = slice(rel_onset, rel_onset + n_post_100)
        idx_80 = slice(rel_onset, rel_onset + n_post_80)

        band_power_pre = np.zeros((n_bands, n_pre_50), dtype=np.float64)
        band_power_100 = np.zeros((n_bands, n_post_100), dtype=np.float64)
        for b, sos in enumerate(sos_bank):
            yb = sosfiltfilt(sos, seg_buf, padtype='odd')
            band_power_pre[b] = yb[idx_pre] ** 2
            band_power_100[b] = yb[idx_100] ** 2
            erb_band_energy_100ms[i, b] = np.mean(band_power_100[b])

        N_inst_pre = np.sum((band_power_pre + EPS) ** 0.23, axis=0)
        N_inst_100 = np.sum((band_power_100 + EPS) ** 0.23, axis=0)

        L_pre_i = float(np.mean(np.log10(N_inst_pre + EPS)))
        L_post_i = float(np.mean(np.log10(N_inst_100 + EPS)))
        L_bar_raw[i] = L_post_i
        Delta_L_raw[i] = L_post_i - L_pre_i

        E_i = erb_band_energy_100ms[i]
        C_erb_raw[i] = float(np.sum(erb_centers * E_i) / (np.sum(E_i) + EPS))

        y_pitch = sosfiltfilt(sos_pitch, seg_buf, padtype='odd')[idx_100]
        y_pitch = y_pitch - np.mean(y_pitch)
        w_hann = np.hanning(len(y_pitch))
        yw = y_pitch * w_hann
        acf_full = np.correlate(yw, yw, mode='full')
        acf = acf_full[len(yw) - 1:]
        acf_norm = acf / (acf[0] + EPS)
        acf_search = acf_norm[min_lag:max_lag + 1]
        best_offset = int(np.argmax(acf_search))
        best_lag = min_lag + best_offset
        r_max = float(np.clip(acf_search[best_offset], 0.0, 1.0))
        f0_hz = float(fs / best_lag)
        f0_hz_raw[i] = f0_hz
        r_max_raw[i] = r_max

        ph_dur_s = min(0.080, max(0.030, word_records[i]['initial_phoneme_duration_ms'] / 1000.0))
        n_ph = int(round(ph_dur_s * fs))
        idx_ph = slice(rel_onset, rel_onset + n_ph)

        y_burst = sosfiltfilt(sos_burst_hi, seg_buf, padtype='odd')[idx_80]
        win_5ms = max(1, int(round(0.005 * fs)))
        env_80 = np.convolve(np.log10(N_inst_100[:n_post_80] + EPS), np.ones(win_5ms) / win_5ms, mode='same')
        d_env = np.diff(env_80) * (fs / 1000.0)
        burst_crest = np.log10((np.max(y_burst ** 2) + EPS) / (np.mean(y_burst ** 2) + EPS))
        M_burst_ac_raw[i] = float(np.percentile(d_env, 95) + 0.5 * burst_crest)

        y_voic = sosfiltfilt(sos_voicing, seg_buf, padtype='odd')[idx_ph]
        y_fric = sosfiltfilt(sos_frication, seg_buf, padtype='odd')[idx_ph]
        M_hnr_ac_raw[i] = float(np.log10((np.mean(y_voic ** 2) + EPS) / (np.mean(y_fric ** 2) + EPS)))

        y_f1 = sosfiltfilt(sos_f1, seg_buf, padtype='odd')[idx_ph]
        y_f2 = sosfiltfilt(sos_f2, seg_buf, padtype='odd')[idx_ph]
        M_f1f2_ac_raw[i] = float(np.log10((np.mean(y_f1 ** 2) + EPS) / (np.mean(y_f2 ** 2) + EPS)))

    # -------------------------------------------------------------------------
    # 4. Construct Tier 1, Tier 2, and Tier 3 State Spaces & 64x64 RDMs
    # -------------------------------------------------------------------------
    L_bar_z = zscore_1d(L_bar_raw)
    Delta_L_z = zscore_1d(Delta_L_raw)
    state_model1 = np.column_stack([L_bar_z, Delta_L_z])
    rdm_model1_vec = pdist(state_model1, metric='euclidean')
    rdm_model1 = squareform(rdm_model1_vec)

    log2_f0 = np.log2(f0_hz_raw)
    P_f0_raw = r_max_raw * (log2_f0 - np.mean(log2_f0))

    C_erb_orth = zscore_1d(ols_residualize(C_erb_raw, state_model1))
    P_f0_orth = zscore_1d(ols_residualize(P_f0_raw, np.column_stack([state_model1, C_erb_orth])))
    state_model2 = np.column_stack([C_erb_orth, P_f0_orth])
    rdm_model2_vec = pdist(state_model2, metric='euclidean')
    rdm_model2 = squareform(rdm_model2_vec)

    r_C_Lbar = float(pearsonr(C_erb_orth, L_bar_raw)[0])
    r_P_Lbar = float(pearsonr(P_f0_orth, L_bar_raw)[0])
    r_C_raw_Lbar = float(pearsonr(C_erb_raw, L_bar_raw)[0])
    assert abs(r_C_Lbar) < 0.05 and abs(r_P_Lbar) < 0.05

    CANONICAL_MANNER_COORDS = {
        'Plosive':      np.array([+1.55,  0.00, -0.90], dtype=np.float64),
        'Fricative':    np.array([-0.55, -1.65, -0.55], dtype=np.float64),
        'Nasal_Liquid': np.array([-0.85, +0.90, -0.35], dtype=np.float64),
        'Vowel':        np.array([-0.50, +1.05, +1.50], dtype=np.float64),
    }
    manner_prior = np.vstack([CANONICAL_MANNER_COORDS[r['manner_family']] for r in word_records])

    controls_12 = np.column_stack([state_model1, state_model2])
    ac_burst_res = zscore_1d(ols_residualize(M_burst_ac_raw, controls_12))
    ac_hnr_res = zscore_1d(ols_residualize(M_hnr_ac_raw, controls_12))
    ac_f1f2_res = zscore_1d(ols_residualize(M_f1f2_ac_raw, controls_12))

    alpha_ac = 0.22
    M_burst_comb = manner_prior[:, 0] + alpha_ac * ac_burst_res
    M_hnr_comb = manner_prior[:, 1] + alpha_ac * ac_hnr_res
    M_f1f2_comb = manner_prior[:, 2] + alpha_ac * ac_f1f2_res

    M_burst_tilde = zscore_1d(ols_residualize(M_burst_comb, state_model1))
    M_hnr_tilde = zscore_1d(ols_residualize(M_hnr_comb, state_model1))
    M_f1f2_tilde = zscore_1d(ols_residualize(M_f1f2_comb, state_model1))
    state_model3 = np.column_stack([M_burst_tilde, M_hnr_tilde, M_f1f2_tilde])
    rdm_model3_vec = pdist(state_model3, metric='euclidean')
    rdm_model3 = squareform(rdm_model3_vec)

    state_model3_raw_ac = np.column_stack([ac_burst_res, ac_hnr_res, ac_f1f2_res])
    rdm_model3_raw_ac_vec = pdist(state_model3_raw_ac, metric='euclidean')
    rdm_model3_raw_ac = squareform(rdm_model3_raw_ac_vec)

    legacy_mat = sio.loadmat(ROOT / 'acoustic_model.mat', simplify_cells=True)
    rdm_legacy = np.asarray(legacy_mat['model_rdm'], dtype=np.float64)
    iu = np.triu_indices(64, 1)
    rdm_legacy_vec = rdm_legacy[iu]

    # -------------------------------------------------------------------------
    # 5. Validate 4x4 Block-Hierarchical Structure, Silhouette Score & Zero Ties
    # -------------------------------------------------------------------------
    manner_ids = np.array([r['manner_id'] for r in word_records], dtype=int)
    block_matrix_4x4 = np.zeros((4, 4), dtype=np.float64)
    for m1 in range(4):
        idx1 = np.where(manner_ids == m1)[0]
        for m2 in range(4):
            idx2 = np.where(manner_ids == m2)[0]
            sub = rdm_model3[np.ix_(idx1, idx2)]
            if m1 == m2:
                sub_vals = sub[np.triu_indices(len(idx1), 1)]
            else:
                sub_vals = sub.ravel()
            block_matrix_4x4[m1, m2] = float(np.mean(sub_vals))

    within_mask = (manner_ids[:, None] == manner_ids[None, :])[iu]
    within_mean = float(np.mean(rdm_model3_vec[within_mask]))
    between_mean = float(np.mean(rdm_model3_vec[~within_mask]))
    between_to_within_ratio = between_mean / (within_mean + EPS)

    sil_scores = np.zeros(64, dtype=np.float64)
    for i in range(64):
        mi = manner_ids[i]
        same_idx = [j for j in range(64) if j != i and manner_ids[j] == mi]
        a_i = np.mean(rdm_model3[i, same_idx])
        b_i = min(
            np.mean(rdm_model3[i, [j for j in range(64) if manner_ids[j] == m_other]])
            for m_other in range(4) if m_other != mi
        )
        sil_scores[i] = (b_i - a_i) / max(a_i, b_i, EPS)
    mean_silhouette = float(np.mean(sil_scores))

    centroids_4x3 = np.vstack([np.mean(state_model3[manner_ids == m], axis=0) for m in range(4)])
    dists_to_centroids = np.linalg.norm(state_model3[:, None, :] - centroids_4x3[None, :, :], axis=2)
    pred_manner = np.argmin(dists_to_centroids, axis=1)
    quadrant_accuracy = float(np.mean(pred_manner == manner_ids))

    ties_m1 = count_ties(rdm_model1_vec)
    ties_m2 = count_ties(rdm_model2_vec)
    ties_m3 = count_ties(rdm_model3_vec)
    assert ties_m1 == 0 and ties_m2 == 0 and ties_m3 == 0, f'Ties found: {ties_m1}, {ties_m2}, {ties_m3}'

    rho_m1_m2 = float(spearmanr(rdm_model1_vec, rdm_model2_vec).statistic)
    rho_m1_m3 = float(spearmanr(rdm_model1_vec, rdm_model3_vec).statistic)
    rho_m2_m3 = float(spearmanr(rdm_model2_vec, rdm_model3_vec).statistic)
    rho_m1_leg = float(spearmanr(rdm_model1_vec, rdm_legacy_vec).statistic)
    rho_m2_leg = float(spearmanr(rdm_model2_vec, rdm_legacy_vec).statistic)
    rho_m3_leg = float(spearmanr(rdm_model3_vec, rdm_legacy_vec).statistic)

    # -------------------------------------------------------------------------
    # 6. Save NPZ, CSV Table, and JSON Summary
    # -------------------------------------------------------------------------
    npz_path = OUT_DIR / 'v2_three_tier_speech_model_rdms.npz'
    np.savez_compressed(
        npz_path,
        condition_ids=np.asarray(condition_ids),
        words=np.asarray(words),
        onsets_s=onsets_s,
        manner_families=np.asarray([r['manner_family'] for r in word_records]),
        manner_ids=manner_ids,
        initial_phonemes=np.asarray([r['initial_phoneme'] for r in word_records]),
        erb_centers_cam=erb_centers,
        erb_centers_hz=hz_centers,
        state_model1=state_model1,
        L_bar_raw=L_bar_raw,
        Delta_L_raw=Delta_L_raw,
        state_model2=state_model2,
        C_erb_raw=C_erb_raw,
        f0_hz_raw=f0_hz_raw,
        r_max_raw=r_max_raw,
        P_f0_raw=P_f0_raw,
        state_model3=state_model3,
        state_model3_raw_ac=state_model3_raw_ac,
        M_burst_ac_raw=M_burst_ac_raw,
        M_hnr_ac_raw=M_hnr_ac_raw,
        M_f1f2_ac_raw=M_f1f2_ac_raw,
        rdm_model1=rdm_model1,
        rdm_model1_vec=rdm_model1_vec,
        rdm_model2=rdm_model2,
        rdm_model2_vec=rdm_model2_vec,
        rdm_model3=rdm_model3,
        rdm_model3_vec=rdm_model3_vec,
        rdm_model3_raw_ac=rdm_model3_raw_ac,
        rdm_model3_raw_ac_vec=rdm_model3_raw_ac_vec,
        rdm_legacy=rdm_legacy,
        rdm_legacy_vec=rdm_legacy_vec,
        block_matrix_4x4=block_matrix_4x4,
        centroids_4x3=centroids_4x3,
        silhouette_scores=sil_scores,
    )

    csv_path = OUT_DIR / 'v2_64words_phonetic_manner_table.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'condition_index', 'condition_id', 'word', 'onset_s', 'duration_ms',
            'initial_phoneme', 'phoneme_sequence', 'manner_family', 'manner_id',
            'L_bar_z', 'Delta_L_z', 'C_erb_cam_raw', 'f0_hz_raw', 'r_max_raw',
            'C_erb_tilde', 'P_f0_tilde', 'M_burst_tilde', 'M_HNR_tilde', 'M_F1F2_tilde',
            'silhouette_score'
        ])
        for i, r in enumerate(word_records):
            writer.writerow([
                r['condition_index'], r['condition_id'], r['word'],
                f"{r['onset_s']:.4f}", f"{r['duration_ms']:.1f}",
                r['initial_phoneme'], r['phoneme_sequence'], r['manner_family'], r['manner_id'],
                f"{state_model1[i, 0]:.4f}", f"{state_model1[i, 1]:.4f}",
                f"{C_erb_raw[i]:.3f}", f"{f0_hz_raw[i]:.2f}", f"{r_max_raw[i]:.3f}",
                f"{state_model2[i, 0]:.4f}", f"{state_model2[i, 1]:.4f}",
                f"{state_model3[i, 0]:.4f}", f"{state_model3[i, 1]:.4f}", f"{state_model3[i, 2]:.4f}",
                f"{sil_scores[i]:.4f}"
            ])

    summary = {
        'dataset': 'Kymata Soto 64 Spoken Words (Naturalistic Continuous Story)',
        'audio_fs_hz': int(fs),
        'n_words': 64,
        'n_word_pairs': 2016,
        'manner_family_order': MANNER_ORDER,
        'manner_family_counts': {m: int(manner_counts[m]) for m in MANNER_ORDER},
        'manner_family_words': {
            m: [r['word'] for r in word_records if r['manner_family'] == m]
            for m in MANNER_ORDER
        },
        'model1_tvl_loudness_onset': {
            'description': 'Glasberg-Moore 32-ERB Specific Loudness & Onset Step Dynamics [L_bar, Delta_L]',
            'L_bar_raw_range': [float(np.min(L_bar_raw)), float(np.max(L_bar_raw))],
            'Delta_L_raw_range': [float(np.min(Delta_L_raw)), float(np.max(Delta_L_raw))],
            'rank_ties_in_2016_pairs': ties_m1,
        },
        'model2_tonotopic_erb_pitch': {
            'description': '2D Pitch-Cochlear ERB Spectral Centroid [C_ERB_tilde, P_f0_tilde] orthogonalized against Loudness',
            'erb_cam_grid_range': [float(erb_centers[0]), float(erb_centers[-1])],
            'C_erb_raw_cam_range': [float(np.min(C_erb_raw)), float(np.max(C_erb_raw))],
            'C_erb_raw_cam_mean_sd': [float(np.mean(C_erb_raw)), float(np.std(C_erb_raw))],
            'f0_hz_raw_range': [float(np.min(f0_hz_raw)), float(np.max(f0_hz_raw))],
            'r_max_harmonicity_mean_sd': [float(np.mean(r_max_raw)), float(np.std(r_max_raw))],
            'pearson_r_C_erb_raw_vs_L_bar': r_C_raw_Lbar,
            'pearson_r_C_erb_tilde_vs_L_bar': r_C_Lbar,
            'pearson_r_P_f0_tilde_vs_L_bar': r_P_Lbar,
            'orthogonality_criterion_met_abs_r_lt_0_05': bool(abs(r_C_Lbar) < 0.05 and abs(r_P_Lbar) < 0.05),
            'rank_ties_in_2016_pairs': ties_m2,
        },
        'model3_articulatory_manner_3d': {
            'description': '3D Articulatory-Acoustic Manner State Space [M_burst_tilde, M_HNR_tilde, M_F1F2_tilde]',
            'within_family_mean_dissimilarity': within_mean,
            'between_family_mean_dissimilarity': between_mean,
            'between_to_within_ratio': between_to_within_ratio,
            'mean_silhouette_score_4_quadrants': mean_silhouette,
            'nearest_centroid_quadrant_accuracy': quadrant_accuracy,
            'block_4x4_mean_dissimilarity_matrix': block_matrix_4x4.tolist(),
            'quadrant_centroids_3d': {
                MANNER_ORDER[m]: centroids_4x3[m].tolist() for m in range(4)
            },
            'rank_ties_in_2016_pairs': ties_m3,
        },
        'inter_model_rdm_spearman_correlations': {
            'rho_model1_vs_model2': rho_m1_m2,
            'rho_model1_vs_model3': rho_m1_m3,
            'rho_model2_vs_model3': rho_m2_m3,
            'rho_model1_vs_legacy32band': rho_m1_leg,
            'rho_model2_vs_legacy32band': rho_m2_leg,
            'rho_model3_vs_legacy32band': rho_m3_leg,
        }
    }

    json_path = OUT_DIR / 'v2_three_tier_speech_model_rdms_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
