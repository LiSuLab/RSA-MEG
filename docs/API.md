# API and data contract

Run `help rsa_meeg.dataset` and the corresponding function help for parameter details. Option names use MATLAB name/value syntax. Times are always seconds; neighborhood radii and canonical positions are millimeters. Input amplitudes keep their supplied units.

## Dataset

`ds = rsa_meeg.dataset(data, sample_times_s, condition_ids, ...)`

`data` is finite, real observation × feature × time for **one subject**. `condition_ids` labels observations; stable unique order becomes `condition_order`. Optional `FoldIds` label independent repetitions/runs, not arbitrary splits of dependent time samples. Every condition must occur in every fold for crossnobis. `FeatureIds` are unique numeric/text identifiers; neighborhood indexing is always local MATLAB indexing. Specify `FeatureTypes` (eeg, mag, grad, source), `FeatureUnits`, `Positions`, `PositionUnit`, `Hemisphere`, `BadFeatures`, `Reference`, and `SubjectId` as appropriate. `Reference` records preprocessing; it does not apply rereferencing. NaNs must be handled before construction; no silent zero filling. Bad features are excluded at analysis time.

Do not concatenate unregistered subjects along features. Compute individual RDMs with consistent condition definitions; aligning/grouping maps also requires a common spatial domain. Mixing EEG/MEG units is rejected within a neighborhood. Unknown default metadata permits array calculations, but does not establish device compatibility or justify physiological interpretation.

## Input adapters

- `io.from_eeglab(EEG, condition_ids, ...)`: loaded, epoched EEGLAB struct, channels × times × epochs. Times converted ms→s. No `.fdt` reading, preprocessing or coordinate-unit assumptions. A single epoch requires `EEG.epoch` metadata; continuous arrays are rejected.
- `io.from_fieldtrip(ft, condition_ids, ...)`: trial/time cells with identical grids, or numeric `rpt_chan_time`, or `chan_time` average. Supply types/units and positions explicitly. No implicit resampling.
- `io.from_stc(files, condition_ids, 'Hemisphere', {'lh','rh'}, 'Surfaces', surface_files, ...)`: files are observation × hemisphere. Each hemisphere retains its own vertex set and geometry. Hemisphere is explicit because STC contains no hemisphere field. `Surfaces` is optional, but geometry is needed for coordinate-radius searchlights. Feature IDs become `lh:0`, etc.
- `io.read_stc` / `io.write_stc`: file IDs stay **zero-based**, local MATLAB indices remain one-based. Data is vertex × time; STC has float32 precision and uniform times. Existing files require `Overwrite=true`; single-sample output requires `TimeStep`.
- `io.read_surface`: FreeSurfer binary triangular format; returned faces are one-based. `io.read_label`: ASCII labels retain zero-based vertex IDs. Map IDs with `ismember`, never use vertex numbers directly as local array subscripts.

## Geometry and RDMs

`geometry.neighborhoods(positions_mm,radius_mm,'Hemisphere',hemi,'Centers',local_indices)` returns cells of local indices. Radius neighborhoods are Euclidean, not cortical geodesic neighborhoods. `geometry.mesh_adjacency(triangles,n_vertices)` accepts one-based local faces. `geometry.spacetime_adjacency(A,n_time)` connects the same spatial vertex across adjacent windows and uses A within each window. A must already encode hemisphere isolation; if windows have gaps, define the desired temporal graph explicitly.

`rdm.searchlight(ds,neighbors,...)` options:

| Option | Meaning |
|---|---|
| WindowSamples / StepSamples | Positive integer sample counts; only full windows, inclusive end; width equal to epoch length yields one window |
| Mode | `spatiotemporal`: concatenate features × samples; `spatial`: average samples; `temporal`: average features |
| Metric | `correlation`, `sqeuclidean`, `crossnobis` |
| FeatureTypes | Select one compatible type; no automatic cross-type normalization |
| CenterIds | External labels corresponding to supplied neighborhoods |
| Precision | Required for crossnobis; matrix or one matrix per center; dimensionality is AFTER mode transformation |

Output `R.values` is pair × center × window. `R.pairs` follows `(1,2),(1,3),…,(K-1,K)`, not arbitrary flattening of a square matrix. `R.condition_ids`, `center_ids`, `windows.start_s/center_s/end_s/indices` preserve semantics. MATLAB drops trailing singleton dimensions; query `size(R.values,3)`. `pair_counts` is pair × 2 observation counts, **not** independent statistical degrees of freedom. `diagnostic` records undefined correlation patterns as NaN.

The RDM also records resolved `feature_indices`, `pattern_dimensions`, selected types/units, normalization, fold weighting, reference, sample times and subject ID. Save this metadata with results. `rdm.average({R_subject1,R_subject2,...},'AlignedCenters',true)` aligns condition labels and calculates an equal-subject finite-entry mean with `valid_counts`. All-missing entries remain NaN. Center IDs and explicit windows must match; incompatible metrics, units and analysis metadata fail. Multiple centers require the explicit alignment acknowledgement; the function does not register anatomy. Different normalized pattern dimensions are retained per subject. This is a descriptive FFX RDM mean, not subject-population inference and not an average of p-values.

## Models

`mapping.compare(R,model,'Correlation',...)` returns center × window × model. A labeled struct has `condition_ids` and `values` (pair × model); labels are reordered explicitly. A plain matrix assumes R's exact condition/pair order. Supported similarity: pearson, spearman (average ties), kendall-a (ties contribute zero).

`mapping.dynamic(R,model,model_times_s,lags_s,...)` requires pair × model-time × model. Convention: **neural window center = model time + lag**. Only time matches within `Tolerance` are used; unmatched samples stay NaN, with no extrapolation or lag-index substitution. Choose tolerance from known sampling precision, not to search for better fits. Output is center × neural-window × model × lag.

`mapping.glm(R,models,'Standardize',false)` fits an intercept and static model columns. Optional standardization affects predictors in each fit, not the response. Returns beta/intercept, rank, condition number, residual sum of squares and finite pair count. Rank-deficient designs fail. These are descriptive coefficients, **not independent-edge OLS p-values**.

`mapping.dynamic_glm(R,models,model_times_s,lags_s,...)` applies those same fits separately at each matched model-time/lag, retaining an explicit center × window × model × lag beta array. It does not select a best lag, create crossvalidated performance estimates or perform post-selection inference.

`mapping.searchlight(ds,neighbors,static_model,'BatchSize',1,...)` computes local RDMs in center batches and retains only the compact model maps. It accepts the searchlight RDM options above plus `Correlation`. Per-center precision cells and CenterIds are sliced consistently. Batch size changes memory use, not the estimator; the input Dataset is still in memory, and no disk cache is implicitly created.

## Inference

`inference.group(values,...)`: finite subjects × tests; tests can be the flattened whole space × time × hemisphere × model × lag family. Subject maps/contrasts must be aligned and have a defensible zero null. Options Method (`max-t`, `cluster-mass`, `tfce`), Tail (`two-sided`, `positive`, `negative`), Adjacency, ClusterThreshold (positive t units), NumPermutations, Seed, Exact, Alpha. Exact enumeration is limited to 16 subjects. Output includes signed t, enhanced magnitude, p_fwer, significance, null maxima, cluster labels and assumptions. Construct block-diagonal adjacency for separate hemispheres/models while placing all blocks in one call for a joint family. Do not infer exact onset/location from cluster significance.

`inference.condition_permutation(R,static_model,...)`: condition-label exchangeability FFX, one common relabeling across all centers/windows/models. Exact enumeration up to 7 conditions. It is not a population-of-subjects test. Undefined observed tests are excluded and the actual family size is recorded; a null relabeling that makes a defined test undefined is rejected.

`out = inference.bh(p,alpha)` returns `out.reject`, `out.p_adjusted`, `out.threshold` and `out.alpha`. It is BH FDR, distinct from the max-statistic FWER methods. The empty rejection set rejects none. BH assumptions (independence or suitable positive dependence) must be justified for the supplied p-values.
