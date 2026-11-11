# Changelog

## 0.2.0-v2 — 2026-11-10

- **Dual MATLAB (`+rsa_meeg` and `+rsa`) and Python (`python/rsa_meeg`) Toolbox Architecture**:
  - Unified namespace support (`+rsa_meeg` and `+rsa/+sim, +rdm, +mapping, +stats`) in MATLAB alongside a standalone Python package (`python/rsa_meeg`).
- **Biophysical & Multi-Scale Cortical Geodesic Perlin Simulation (`sim.generate_beta_patterns`, `sim.cortical_perlin_noise`)**:
  - Recursive hierarchical condition covariance and multivariate pattern generator supporting the canonical 64-condition (`4 clusters x 16 exemplars`, 2,016 condition pairs) and arbitrary nested cluster trees with exact or Monte Carlo covariance parameterization.
  - 30-scale (`2–60 mm`) cortical geodesic and Euclidean Perlin continuous spatial texture generator with $1/f^\alpha$ spectral weighting and hybrid prestimulus sensor-covariance Cholesky noise synthesis.
- **Analytical Shrinkage Precision & Spatiotemporal Searchlight Fusion (`rdm.shrinkage_precision`, `rdm.searchlight`)**:
  - Analytical Ledoit-Wolf, Oracle Approximating Shrinkage (OAS), fixed-$\lambda$, and diagonal noise precision matrix estimation ($\mathbf{P} = \hat{\boldsymbol{\Sigma}}^{-1}$) across spatial, temporal, and spatiotemporal searchlights to project out zero-lag volume-conducted crosstalk and rescue scalp EEG cortical localization.
- **Three-Tier Zero-Tie Continuous Speech RDM Parameterization & Partial Rank Regression (`mapping.partial_compare`)**:
  - Continuous Hilbert loudness envelope (Model 1), 24-band Gammatone cochlear tonotopy (Model 2), and 6-channel articulatory-manner trajectory (Model 3) RDMs eliminating discrete binary RDM ties (`0.00%` tie fraction across all 2,016 word pairs).
  - Analytical QR rank-residualized partial Spearman and partial Pearson RDM regression controlling for nested acoustic/phonological or lexical covariates.
- **Complete V2 Manuscript Reproducibility Suite (`reproducibility/`)**:
  - Added full Stage 1–3 experiment scripts (`v2_01`–`v2_10`, `v2_12`) and numerical figure sources reproducing Main Figures 1–8, Supplementary Figures S1–S10, Table 1, and Supplementary Tables S1–S6.
  - Removed obsolete pre-V2 `0.1.0-dev` legacy migration tables, 23-epoch toy MNE sample exporters, and deprecated benchmark scripts.
