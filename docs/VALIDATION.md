# Software and Numerical Validation Record (`v0.2.0-v2`)

## 1. Automated Unit and Integration Suites
- **MATLAB Suite (`run_tests.m`)**: Tested in MATLAB R2026a (`26.1.0`) on Apple Silicon (`macOS`). All 10 test suites (`test_average`, `test_core`, `test_end_to_end`, `test_inference`, `test_io`, `test_mapping`, `test_review_core`, `test_review_inference`, `test_streaming_dynamic_glm`, and `test_v2_new_features`) pass with zero failures, covering:
  - Hand-computed `sqeuclidean`, `correlation`, and `crossnobis` searchlight distances;
  - Analytical Ledoit-Wolf, OAS, fixed-$\lambda$, and diagonal shrinkage precision (`rdm.shrinkage_precision`);
  - 64-condition (`4x16`) and recursive hierarchical pattern generation (`sim.generate_beta_patterns`) and 30-scale (`2–60 mm`) cortical Perlin spatial texture generation (`sim.cortical_perlin_noise`);
  - Analytical QR rank-residualized partial Spearman and partial Pearson RDM regression (`mapping.partial_compare`);
  - Exact ($2^N$) and Monte Carlo subject sign-flip max-$t$, cluster-mass, and TFCE permutation inference (`inference.group`), condition-label permutation (`inference.condition_permutation`), and BH-FDR (`inference.bh`);
  - Both `+rsa_meeg` and `+rsa` (`+sim`, `+rdm`, `+mapping`, `+stats`) namespace interfaces.
- **Python Suite (`python/tests/test_rsa_meeg_python.py`)**: Tested in Python 3.11+ with `unittest`, verifying exact covariance reconstruction, 30-scale Perlin noise generation, Ledoit-Wolf shrinkage precision positive-definiteness, unbiased crossnobis expectation under the null, three-tier zero-tie (`0.00%` tie fraction) speech model RDM construction, partial Spearman rank regression, and exact 256-sign-flip permutation inference.

## 2. Manuscript Numerical Audit (`v2_07_supervisor_robustness_and_zero_hallucination_audit.py`)
- All numerical figure sources (`reproducibility/results/*.json` and `*.csv`) are verified by `reproducibility/scripts/v2_07_supervisor_robustness_and_zero_hallucination_audit.py` for:
  - Exact condition-pair counts (`2,016` pairs for 64 conditions; `703` pairs for 38 subset conditions);
  - Zero RDM tie fraction (`0.0000%` ties across Models 1, 2, and 3);
  - Finite, bounded correlation and localization metrics across all `N = 11` MEG and `N = 8` EEG subjects and all 8 bilateral cortical ROIs.
