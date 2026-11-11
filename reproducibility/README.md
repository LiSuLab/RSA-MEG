# V2 Manuscript Reproducibility & Numerical Figure Sources (`0.2.0-v2`)

This directory contains the reproducible experiment scripts and numerical source data corresponding 1-to-1 with the V2 manuscript (*Representational similarity analysis of MEG and EEG data*, `RSA_MEEG_Manuscript_v2.docx` and `RSA_MEEG_Supplementary_v2.docx`).

## Stage-to-Figure Mapping

| Script | Scientific Stage | Manuscript Figures & Tables |
|---|---|---|
| `scripts/v2_01_audit_and_build_full_cohort_operators.py` | Full-cohort 3-layer BEM forward & MNE/dSPM/sLORETA inverse operator and cortical dipole radiality audit (`N=11` MEG + `N=8` EEG) | Figure 2a–c, Figure S1, Table S1 |
| `scripts/v2_02_cortical_perlin_and_spatial_freq_curve.py` | 30-scale (`2–60 mm`) cortical geodesic Perlin spatial frequency Modulation Transfer Function (MTF) and cross-hemisphere leakage | Figure 2d, Figure S3 |
| `scripts/v2_03_simulate_isf_vs_asf_and_restore_fig2.py` | 64-condition (4x16) Equal-Amplitude ANOVA vs. 2x-Amplitude ASF vs. Shrinkage-Whitened ISF, 8-ROI x 3-Inverse recovery, and sulcal-gyral double dissociation | Figure 3a–c, Figure 4a–d, Figure S2, Figure S4, Figure S5, Table 1, Table S2, Table S3 |
| `scripts/v2_04_build_three_tier_speech_model_rdms.py` | 64-word (2,016-pair) Three-Tier Zero-Tie Speech RDM Hierarchy (Loudness Envelope, 24-Band Gammatone Tonotopy, 6-Channel Articulatory Manner) | Figure 5a–c, Figure S6, Table S4 |
| `scripts/v2_05_run_soto_meg_eeg_rsa_and_group_inference.py` | Kymata Soto Natural Speech MEG (`N=11`) and EEG (`N=8`) sensor & cortical RSA, 129,024-test univariate ERP Bonferroni collapse, and partial Spearman regression | Figure 3d, Figure 5d, Figure 6a–d, Figure S7, Table 1, Table S5 |
| `scripts/v2_06_group_inverse_and_preprocessing_benchmark.py` | Second-order MEG–EEG representational convergence in BanksSTS/AC, inverse regularization $\lambda^2$ landscapes, temporal low-pass cutoff, searchlight radius $\times$ smoothing FWHM, and 8-stage preprocessing waterfall | Figure 8a–d, Figure S9, Figure S10, Table S6 |
| `scripts/v2_08_eeg_spatial_rescue_and_superresolution.py` | Scalp EEG spatial rescue via spatiotemporal searchlight + Ledoit-Wolf shrinkage precision and 64/32/19-channel montage density subsampling | Figure 7a–d, Figure S8, Table 1 |
| `scripts/v2_12_reviewer_robustness_controls.py` | Interdisciplinary robustness controls: word duration / uniqueness-point partial correlation, cross-run split-half reliability, and spatial vs. temporal whitening decomposition | Results & Discussion robustness controls |
| `scripts/v2_09_render_v5_style_publication_figures.py` | Reproduces all 8 Main Figures (`Figures 1–8`) from the numerical figure sources | Figures 1–8 |
| `scripts/v2_10_render_v5_style_supplementary_figures.py` | Reproduces all 10 Supplementary Figures (`Figures S1–S10`) from the numerical figure sources | Figures S1–S10 |
| `scripts/v2_07_supervisor_robustness_and_zero_hallucination_audit.py` | Automated numerical verification across all result manifests | Supplementary Note 6 |
