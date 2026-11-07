# Scientific estimands and inference

This document specifies calculations implemented by this release. It is not a completed Methods section for a study; experiment-specific preprocessing, participants, forward/inverse models, reference, covariance estimation and validation design still need to be reported. Code organization, file naming and test infrastructure belong in software documentation, not as substitutes for scientific methods.

## Hierarchical pattern and multi-scale Perlin noise simulation

`sim.generate_beta_patterns` constructs a hierarchical condition covariance matrix $\Sigma_{\text{true}} \in \mathbb{R}^{K \times K}$ recursively from a nested cell tree (such as `{0.6, {0.3, 5}, {0.3, 5}, {0.3, 5}, {0.3, 5}}`). In variance-component (`spread`) parameterization, each internal branch with parameter $v_u > 0$ contributes shared cluster-centroid variance $v_u \mathbf{1}_{S_u}\mathbf{1}_{S_u}^\top$ to each of its child blocks $S_u$, and each leaf $\{v_\ell, n_\ell\}$ contributes exemplar-specific variance $v_\ell \mathbf{I}_{n_\ell}$. Because $\Sigma_{\text{true}}$ is a positive combination of block-ones matrices plus a strictly positive diagonal, it is guaranteed to be symmetric positive-definite for any tree depth. Multivariate condition patterns are drawn as $\mathbf{B} = \Sigma_{\text{true}}^{1/2} \mathbf{Z}$, yielding $\mathbb{E}[\frac{1}{P}\mathbf{B}\mathbf{B}^\top] = \Sigma_{\text{true}}$ across $P$ features (or exact sample covariance when `ExactCovariance=true` and $P \ge K$).

`sim.cortical_perlin_noise` synthesizes multi-scale spatially correlated background activity across $S$ spatial scales $r_1, \dots, r_S$ (default 30 scales from 2 mm to 60 mm). At each scale $r_k$, a Gaussian smoothing kernel $K_{k,uv} \propto \exp(-d(u,v)^2 / (2 r_k^2))$ (using Euclidean or triangle-mesh geodesic distances $d(u,v)$, isolated by hemisphere) is normalized to unit row $\ell_2$ norm and weighted by $w_k \propto r_k^\alpha$ ($\sum_k w_k^2 = 1$). Summing $\sum_{k=1}^S w_k K_k \mathbf{Z}_k$ produces a continuous $1/f^\alpha$ spatial texture field, which can be projected through a forward/inverse resolution operator $\mathbf{R}_m = \mathbf{K}_m \mathbf{G}_m$ and mixed with correlated prestimulus sensor noise $\mathbf{L}_m \mathbf{z}$ ($\mathbf{L}_m \mathbf{L}_m^\top = \mathbf{C}_m$).

## Local representations and dissimilarity

For each participant and searchlight/window, the pattern of condition i is the mean across its observations. A spatiotemporal pattern concatenates the selected features and samples; a spatial pattern first averages samples; a temporal pattern first averages features. Thus these modes test different representational hypotheses. Neighborhood radius is Euclidean in the supplied coordinates; source-space interpretation depends on the reconstruction and on leakage between locations. Overlapping windows/neighborhoods are dependent.

For D pattern features, squared Euclidean distance is `sum((mu_i-mu_j).^2)/D`. Correlation distance is one minus Pearson correlation between the two condition patterns after subtracting each pattern's mean across features. Constant patterns give undefined (NaN) distances. Correlation distance removes feature-wise overall offset/scale in each pattern and may be inappropriate when such amplitude differences are the target effect.

Crossnobis uses K independent folds, differences delta(i,j,k), and an independently estimated positive-definite precision P in the same pattern space:

`d(i,j) = sum_{k != l} delta(i,j,k) * P * delta(i,j,l)' / [K*(K-1)*D]`.

Every condition must occur in every fold. Equal weighting is over ordered fold pairs, not over all trial pairs. Negative distances are valid; truncating them biases the estimator. Identity P gives crossvalidated squared Euclidean distance, not empirical noise whitening. If time samples are concatenated, the precision must cover that entire feature × sample pattern. More features/neighbors or different units change the estimand unless normalization/whitening is chosen appropriately.

`rdm.shrinkage_precision` estimates $\mathbf{P} = \hat{\Sigma}^{-1}$ from within-condition residual patterns (or prestimulus baseline intervals) using analytical Ledoit–Wolf shrinkage, Oracle Approximating Shrinkage (OAS), or fixed shrinkage $\lambda \in [0, 1]$ toward the scaled identity target $\mu \mathbf{I}_D$ ($\mu = \text{tr}(\mathbf{S})/D$). In EEG and source-space searchlights where volume conduction induces strong positive off-diagonal covariance among neighboring features, the off-diagonal negative weights of $\mathbf{P} = \hat{\Sigma}^{-1}$ project out shared common-mode crosstalk and sharpen local spatial specificity (`demo_eeg_meg_spatial_rescue`).

This estimator differs intentionally from the legacy workflow's average of within-session distances and from distances computed after averaging all sessions. Use the documented estimator; do not present corrected/new estimates as bitwise reproduction of old results.

## Model correspondence and conditional fits

Model and neural RDMs are aligned by condition labels. Pearson measures linear correspondence, Spearman compares average-tie ranks, and Kendall tau-a counts concordance minus discordance divided by all pairs of RDM entries; ties contribute zero. RDM entries share conditions and are dependent, so the implementation does not report the usual independent-observation correlation p-value.

`mapping.partial_compare` computes partial Pearson or partial Spearman correspondence between each neural RDM vector $\mathbf{y}$ and target model $\mathbf{x}_j$ controlling for one or more covariate RDM columns $\mathbf{Z}_{\text{ctrl}}$. In Spearman mode, $\mathbf{y}$, $\mathbf{x}_j$, and each column of $\mathbf{Z}_{\text{ctrl}}$ are first converted to average-tie mid-ranks over their common finite pairs. Both $\mathbf{y}$ and $\mathbf{x}_j$ are orthogonalized against the nuisance design $[\mathbf{1}, \mathbf{Z}_{\text{ctrl}}]$ via QR projection before computing the Pearson correlation of their residuals.

Dynamic comparison treats a positive lag as a neural response after the corresponding model time. Comparisons only use matching time coordinates. Searching over lags, models, ROIs or windows creates a selection problem: either hold out independent evaluation data, or include the complete searched family in an appropriate test. OLS with an intercept describes unique linear association conditional on the supplied models; correlated models need rank/conditioning diagnostics and cautious interpretation. It does not establish causality or a unique cognitive mechanism.

## Fixed and random effects

For a descriptive FFX summary, subject RDMs can be averaged with equal subject weight for each finite entry after condition alignment. The output records the number of contributing subjects per entry. Missingness can therefore change the contributing sample across entries and must be examined. Identical center labels do not establish anatomical correspondence; registration or ROI correspondence must precede this operation.

Fixed-effects condition permutation relabels conditions, preserving the RDM structure. The same permutation acts on every tested center/window/model. Validity requires the relevant condition-label exchangeability under the null, and the resulting inference remains conditional on the observed participant/data.

Random-effects inference uses independent participants' aligned maps or within-participant contrasts. Each randomization assigns one sign to each participant and applies it to that participant's entire map. This preserves within-map dependence and requires an appropriate sign-symmetric null; an arbitrary positively biased distance or selected maximum is not automatically centered at zero. `group` does not automatically apply a Fisher transformation; the caller must specify and justify the participant-level estimand. Model correspondence contrasts, raw correlations and Fisher-transformed Pearson correlations are different inputs.

The one-sample statistic is mean divided by its standard error. Numerical column scaling prevents overflow/underflow without changing t. Degenerate zero-variance columns use a relative machine-precision denominator guard; all-zero columns yield zero. These boundary behaviors are tested and should not replace adequate between-subject variability.

For max-t, the maximum statistic spans the entire supplied family. Cluster mass is the sum of **threshold excess**, `sum(|t|-u)`, within connected suprathreshold same-sign components, with threshold u specified in t units. Positive and negative clusters are treated separately; the null maximum includes both when two-sided. This threshold-excess definition is explicit and need not match historical scripts that sum raw t values.

TFCE uses extent exponent E=.5 and height exponent H=2, separately for each sign. For a finite graph, cluster membership changes only at observed statistic heights. The implementation integrates h^2 exactly between these levels, multiplying by cluster extent^(.5); it uses the maximum enhanced magnitude for two-sided correction. Extent counts graph nodes; changing temporal/spatial sampling changes extent and must be considered in cross-resolution comparisons. No claim of resolution invariance is made.

For B Monte Carlo samples, p is `(1 + count(null_max >= observed))/(B+1)`; exact enumeration includes the observed labeling and uses `count/B`. Ties count in the tail. The reported p-values never have spurious zero Monte Carlo probability. Cluster-corrected significance is evidence about the cluster, not every contained vertex or exact onset/offset. BH FDR is a separate procedure, not a synonym for these maximum-statistic tests.

## Evidence and appropriate scientific conclusions

Unit and integration tests verify known numerical answers, label/time geometry, exchangeability operations and software execution. A fixed Gaussian-null calibration evaluates empirical FWER under its stated design. A short real M/EEG excerpt verifies that actual modality-specific arrays can pass through the workflow. None alone establishes superior localization, source-versus-sensor accuracy, resistance to inverse-model leakage, or improved biological model identification. Such claims require fair matched simulations and independent empirical validation as specified in the study protocol.
