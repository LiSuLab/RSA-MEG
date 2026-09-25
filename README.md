# RSA-MEEG for MATLAB

An executable development release of the LiSuLab M/EEG representational similarity workflow, with explicit time/condition labels, general sensor and source neighborhoods, model mapping, and permutation inference.

**Version 0.1.0-dev.** Tested in MATLAB R2026a on Apple Silicon. The MATLAB analysis core and synthetic examples require only base MATLAB. Other MATLAB versions and operating systems have not been tested. This is a development branch, not a claim of reproducing the historical manuscript or introducing RSA/ssRSA for the first time.

## Download and run

Download the [rsa-meeg branch ZIP](https://github.com/LiSuLab/RSA-MEG/archive/refs/heads/rsa-meeg.zip), unzip, and in MATLAB change to its root:

```matlab
setup_rsa_meeg
out = demo_synthetic('eeg');     % also 'mag', 'grad', or 'source'
results = run_tests;
```

The example generates known synthetic representations, computes time-resolved local RDMs, compares a condition-labeled model, and tests the complete center × time family using independent subjects. No private files, cluster accounts, optional toolboxes or downloads are needed. Results are examples of computation, not biological observations.

For a real EEG/MEG recording, follow [REAL_DATA.md](docs/REAL_DATA.md). A pinned 12 MB public MNE example is downloaded only on explicit request; Python is needed for that optional FIF-to-MAT conversion, not the MATLAB analysis core. The release does not distribute the underlying recording.

## Analyze your data

```matlab
% X: observations x features x time. One subject at a time.
ds = rsa_meeg.dataset(X, times_seconds, condition_ids, ...
    'FoldIds', fold_ids, 'FeatureIds', channel_labels, ...
    'FeatureTypes', repmat({'eeg'},1,size(X,2)), ...
    'FeatureUnits', repmat({'V'},1,size(X,2)), ...
    'Positions', electrode_xyz, 'PositionUnit', 'mm', ...
    'Reference', 'describe the reference already applied');
neighbors = rsa_meeg.geometry.neighborhoods(ds.positions_mm, 30);
R = rsa_meeg.rdm.searchlight(ds, neighbors, ...
    'WindowSamples', 9, 'StepSamples', 4, 'Metric', 'correlation', ...
    'CenterIds', ds.feature_ids);
model.condition_ids = R.condition_ids;
model.values = my_model_vector; % lexicographic pairs in R.pairs
fit = rsa_meeg.mapping.compare(R, model, 'Correlation', 'kendall-a');
```

Electrode radius is in millimeters in the specified coordinate frame; this example's radius/window are illustrative, not recommended physiological defaults. Supply a single neighborhood for ROI RSA. Analyze EEG, magnetometers and gradiometers separately; incompatible physical units are not silently pooled. Source neighborhoods can be restricted by hemisphere and use nonconsecutive vertex identifiers. Crossnobis requires independent folds and externally estimated precision in the actual analysis pattern space.

## What is included

| Stage | Functions |
|---|---|
| Input | array Dataset; in-memory epoched EEGLAB/FieldTrip adapters; STC, FreeSurfer triangular surface and ASCII label readers; STC writer |
| Geometry | arbitrary coordinate-radius neighborhoods; triangle-mesh adjacency; space × time adjacency |
| RDMs | spatial, temporal and spatiotemporal patterns; ROI/searchlight windows; correlation, normalized squared Euclidean and crossnobis distances; condition-aligned descriptive subject averaging |
| Models | condition-aligned Pearson/Spearman/Kendall tau-a; explicit dynamic-model lags; static/dynamic multiple-model OLS; center-batched static mapping |
| Inference | condition-label FFX permutation; subject sign-flip max-t, signed cluster mass and TFCE; BH adjustment |
| Examples and checks | synthetic EEG/mag/grad/source; optional genuine M/EEG excerpt; numerical/IO/independent review tests; null calibration; legacy tau-a regression |

See [API](docs/API.md), [scientific methods](docs/METHODS.md), [validation](docs/VALIDATION.md), [migration](docs/MIGRATION.md), and [scientific scope](docs/SCIENTIFIC_SCOPE.md). Group inference requires aligned analysis centers and windows; individual source meshes cannot simply be stacked across subjects.

The package does not perform acquisition preprocessing, artifact removal, forward/inverse modeling, automatic anatomical registration, or raw FIF reading inside MATLAB. Automatic best-lag selection, historical HTK training/export pipelines and paper-specific optimizations remain separate migration targets. No historical CBU email/cluster machinery is included.

## Provenance and citation

This branch consolidates the *analysis logic* of LiSuLab/RSA-MEG, the `feature-meg-base` fork and later mapping applications in a new `+rsa_meeg` API. It does not copy all historical files into the path. Each old file is assigned a migration disposition in [provenance/migration.tsv](provenance/migration.tsv); pinned sources are in [source-lock.json](provenance/source-lock.json).

Cite the software version/commit used, **Nili et al. (2014)** for the RSA toolbox foundations, **Su et al. (2014)** and the prior work it cites for source-space ssRSA, and the specific inference/distance methods used. See [CITATION.cff](CITATION.cff) and [references](docs/REFERENCES.md). A DOI/tagged stable release and final author attribution are still to be established by the maintainers.

LGPL-3.0-only for this release; original notices are retained in [LICENSES](LICENSES). See [COPYING.LESSER](COPYING.LESSER), [COPYING](COPYING) and [NOTICE](NOTICE). Historical branches remain available for provenance. Please report issues with the version, minimal input, parameters, expected result and actual result; do not post private subject data.
