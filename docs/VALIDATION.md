# Validation of 0.1.0-dev

Validation was performed on 2026-09-25 in licensed MATLAB **26.1.0.3203278 (R2026a), MACA64**. All runnable MATLAB examples and analysis functions use base MATLAB; the optional retrieval/export of a real FIF uses Python/MNE. No other platform/version is certified by this run.

The final suite passed **54/54 tests**, with no failures or incomplete tests. The versioned result table in `provenance/validation-tests.csv` is the authoritative final test list. Tests cover hand-computed distances, negative crossnobis, independent folds, text and nonconsecutive identifiers, full-window boundaries, explicit times, units, condition reordering, GLM rank, lag semantics, exact enumeration, RNG preservation, positive/negative clusters and TFCE, BH empty rejection sets, input adapters, STC known bytes and synthetic end-to-end analysis. A separate clean-path extraction run verifies the distributable package rather than relying only on the working directory.

## Independent comparisons

`tests/compare_legacy.m` accepts the original Nili 2014 Software S1 directory as an explicit argument. It adds only its Engines directory temporarily and removes it afterward. Across 37 fixed/seeded, tied, missing-value and orientation cases, Kendall tau-a agreed exactly (maximum absolute error **0**). A separate correlation-distance check against base-MATLAB `corrcoef` also had maximum error **0**. The latter is a formula check, not an additional claim of original-toolbox execution. Historical startup scripts and cluster/email functions were not run or included in this release.

Agreement of inherited mathematics is desirable and is not novelty. Fixed indexing, input support, joint-family permutation behavior, explicit lags and numerical boundary handling are checked separately; the entire historical pipeline is not claimed to be reproduced.

## Predeclared null calibration

`tests/calibrate_group.m`: 1,000 independently simulated Gaussian null datasets; 12 subjects, 6 connected graph nodes, 199 Monte Carlo sign flips per test, two-sided alpha .05; fixed generation seed 20260925 and per-replicate permutation seeds. Cluster threshold 2 t units; TFCE E=.5/H=2. Parameters and seeds were fixed before the full run, not tuned to obtain favorable outcomes.

| Method | Any false rejection / 1,000 | Empirical FWER | Wilson 95% interval |
|---|---:|---:|---:|
| max-t | 53 | .053 | [.040747, .068675] |
| cluster mass | 43 | .043 | [.032079, .057419] |
| TFCE | 52 | .052 | [.039873, .067556] |

This checks calibration under this particular sign-symmetric null. It does not prove control under arbitrary nonexchangeable data, all preprocessing schemes, selected lags, all graph resolutions or a population-level biological experiment. A conservative valid procedure need not have an interval centered exactly on .05. Application-specific robustness and power studies remain scientific work.

## Genuine recording

The pinned MNE sample excerpt completed the MATLAB RDM → condition-labeled model pipeline separately for **23 × 59 × 121 EEG**, **23 × 102 × 121 mag**, and **23 × 203 × 121 grad** arrays (observation × feature × time). Each yielded 15 complete windows with finite distances and model similarities. The source data, actual preprocessing, label provenance, download command and redistribution boundary are in [REAL_DATA.md](REAL_DATA.md). `provenance/realdata-results.json` records numerical ranges, dimensions and sampled endpoints.

This validates real-data execution and modality/coordinate handling, not accuracy, a powered population effect or source reconstruction. Native source STC I/O and source searchlights are tested with known binary fixtures and synthetic signals. A complete, registered real source dataset and the historical manuscript's simulations/results have not been reproduced in this development release.

## Remaining release and research validation

An independent user installation, representative larger source datasets, performance/memory characterization on realistic 400-condition cortical meshes, more MATLAB versions and platforms, and the manuscript's matched source/sensor and RSA/univariate experiments are required before a stable paper-linked release. Clinical interpretation is outside scope. All declared behavior changes and deferred historical functions are recorded rather than silently presented as ported.

## Bounded 400-condition benchmark

On a fixed synthetic dataset with 400 conditions, 64 centers, 24 input features and 5 windows, full and BatchSize=1 static mapping agreed exactly (maximum absolute difference 0). The full returned RDM structure occupied 206871717.0 bytes; full calculation took 2.172 s and batched calculation 4.237 s in this run. Batching trades additional setup time for retaining one center RDM batch instead of the full result. These are single-run timings, not a cross-platform speed advantage or a measured peak-memory claim. The actual whole-cortex 400-condition analysis is not benchmarked here. Reproduce with `benchmark_searchlight(400,64)`.
