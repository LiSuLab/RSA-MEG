# RSA-MEEG Toolbox for MATLAB and Python (`v0.2.0-v2`)

Open-source MATLAB (`+rsa_meeg` and `+rsa/+sim, +rdm, +mapping, +stats`) and Python (`python/rsa_meeg`) toolbox accompanying the manuscript:

> **Representational similarity analysis of MEG and EEG data**  
> Jingmin An, Cai Wingfield, Chengxuan Li, Andrew Thwaites, and Li Su (2026).  
> Repository: [https://github.com/LiSuLab/RSA-MEG/tree/rsa-meeg](https://github.com/LiSuLab/RSA-MEG/tree/rsa-meeg)

**Version `0.2.0-v2`.** Tested in MATLAB R2026a and Python 3.11+ on Apple Silicon. The MATLAB analysis core and synthetic examples require only base MATLAB; the Python package requires only `numpy` and `scipy` (with `mne` and `matplotlib` for full-cohort BEM/MNE source reconstruction and publication figure rendering).

---

## Quick Start

### 1. MATLAB (`+rsa_meeg` and `+rsa`)

In MATLAB, change to the repository root and run:

```matlab
setup_rsa_meeg
out = demo_synthetic('eeg', false);       % also 'mag', 'grad', or 'source'
res = demo_eeg_meg_spatial_rescue(false); % V2: 64-condition ISF vs. ASF & EEG spatial rescue
results = run_tests;                      % Executes the full MATLAB unit/integration test suite
```

Example workflow with spatiotemporal Ledoit-Wolf shrinkage precision and partial Spearman RDM regression:

```matlab
% X: observations x features x time (one subject at a time)
ds = rsa_meeg.dataset(X, times_seconds, condition_ids, ...
    'FoldIds', fold_ids, 'FeatureIds', channel_labels, ...
    'FeatureTypes', repmat({'eeg'}, 1, size(X, 2)), ...
    'FeatureUnits', repmat({'V'}, 1, size(X, 2)), ...
    'Positions', electrode_xyz, 'PositionUnit', 'mm', ...
    'Reference', 'average reference');

neighbors = rsa_meeg.geometry.neighborhoods(ds.positions_mm, 30);
P = rsa.rdm.shrinkage_precision(ds, neighbors, ...
    'Method', 'ledoit-wolf', 'Mode', 'spatiotemporal', 'WindowSamples', 9);
R = rsa.rdm.searchlight(ds, neighbors, ...
    'WindowSamples', 9, 'StepSamples', 4, 'Mode', 'spatiotemporal', ...
    'Metric', 'crossnobis', 'Precision', P, 'CenterIds', ds.feature_ids);

fit = rsa.mapping.compare(R, target_model, 'Correlation', 'spearman');
fit_partial = rsa.mapping.partial_compare(R, target_model, control_models, ...
    'Correlation', 'spearman');
```

### 2. Python (`python/rsa_meeg`)

Install the Python package in editable mode and run the unit test suite:

```bash
pip install -e ./python
python3 -m unittest discover -s ./python/tests -v
```

### 3. Reproducing V2 Manuscript Figures & Tables (`reproducibility/`)

All Stage 1–3 experiment scripts (`v2_01` through `v2_10` and `v2_12`) and numerical figure sources for **Figures 1–8**, **Supplementary Figures S1–S10**, **Table 1**, and **Supplementary Tables S1–S6** are provided in [`reproducibility/`](reproducibility/README.md):

```bash
python3 reproducibility/scripts/v2_07_supervisor_robustness_and_zero_hallucination_audit.py
python3 reproducibility/scripts/v2_09_render_v5_style_publication_figures.py
python3 reproducibility/scripts/v2_10_render_v5_style_supplementary_figures.py
```

---

## Architecture Overview

| Module | MATLAB (`+rsa_meeg` / `+rsa`) | Python (`rsa_meeg`) | Core Capabilities |
|---|---|---|---|
| **Input & Geometry** | `dataset`, `io.*`, `geometry.*` | `sim`, `rdm` | Epoched EEGLAB/FieldTrip/MNE-STC adapters, FreeSurfer surfaces/labels, Euclidean/mesh/spacetime neighborhoods |
| **Simulation (`+sim`)** | `sim.generate_beta_patterns`, `sim.cortical_perlin_noise` | `sim.generate_beta_patterns`, `sim.cortical_perlin_noise` | 64-condition (`4x16`) & hierarchical multivariate patterns; 30-scale (`2–60 mm`) cortical geodesic Perlin spatial textures; hybrid prestimulus sensor-covariance noise |
| **RDMs & Whitening (`+rdm`)** | `rdm.shrinkage_precision`, `rdm.searchlight`, `rdm.average` | `rdm.shrinkage_precision`, `rdm.compute_rdm`, `rdm.build_three_tier_speech_rdms` | Ledoit-Wolf / OAS / fixed shrinkage precision $\mathbf{P} = \hat{\boldsymbol{\Sigma}}^{-1}$; spatial, temporal, and spatiotemporal crossnobis/correlation/sqeuclidean searchlights; three-tier zero-tie continuous speech RDMs |
| **Model Mapping (`+mapping`)** | `mapping.compare`, `mapping.partial_compare`, `mapping.glm`, `mapping.dynamic_glm` | `mapping.compare_rdm`, `mapping.partial_compare_rdm` | Condition-aligned Spearman/Pearson/Kendall $\tau_a$; analytical QR rank-residualized partial Spearman/Pearson RDM regression; lagged dynamic GLM |
| **Inference (`+stats` / `+inference`)** | `inference.group`, `inference.condition_permutation`, `inference.bh` | `stats.group_permutation_test`, `stats.bh_fdr` | Subject-level whole-map sign-flip max-$t$, cluster-mass, and TFCE permutation inference; condition-label permutation; Benjamini-Hochberg FDR |

See [API Reference](docs/API.md), [Scientific Methods](docs/METHODS.md), [Validation Record](docs/VALIDATION.md), [Scientific Scope & Operating Guardrails](docs/SCIENTIFIC_SCOPE.md), and [References](docs/REFERENCES.md).

---

## License & Citation

Released under **LGPL-3.0-only**; see [COPYING.LESSER](COPYING.LESSER), [COPYING](COPYING), [NOTICE](NOTICE), and [CITATION.cff](CITATION.cff). Cite **An et al. (2026)**, **Nili et al. (2014)** for the RSA toolbox foundations, and **Su et al. (2014)** for spatiotemporal searchlight RSA.
