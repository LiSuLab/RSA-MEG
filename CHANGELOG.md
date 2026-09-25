# Changelog

## 0.1.0-dev — 2026-09-25

- Standalone `+rsa_meeg` package with no private datasets, CBU infrastructure or mandatory Statistics Toolbox.
- General sensor/source metadata and neighborhoods, explicit inclusive window times and nonconsecutive identifiers.
- ROI/searchlight correlation, normalized squared Euclidean and independent-fold crossnobis; separate EEG/mag/grad units.
- Condition-aligned static/dynamic models, explicit lag seconds, descriptive static GLM.
- Joint-family condition/subject permutation, two-sided max-t/cluster mass/TFCE and separate BH FDR.
- In-memory EEGLAB/FieldTrip and native STC/FreeSurfer adapters; synthetic and explicitly retrieved real data examples.
- Numerical, independent cross-review, IO and end-to-end tests; predeclared Gaussian null calibration.

This release intentionally changes some estimators and invalid legacy behaviors. It does not promise numerical identity with all historical scripts. Migration entries distinguish replacements from deferred functions. Inherited RSA/ssRSA methods are not claimed as new methods.
