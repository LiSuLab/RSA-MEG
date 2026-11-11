# Scientific Scope and Practical Operating Guardrails (`v0.2.0-v2`)

This document summarizes the biophysical scope, methodological estimators, and practical operating guardrails established in the V2 manuscript (*Representational similarity analysis of MEG and EEG data*).

## 1. Multivariate Information Spread Functions (ISF) vs. Univariate Activation Spread Functions (ASF)
- Linear minimum-norm inverse operators ($\text{MNE}$, $\text{dSPM}$, $\text{sLORETA}$) disperse a focal cortical dipole across neighboring gyri and sulci via the univariate Activation Spread Function (ASF). Furthermore, when experimental conditions share equal mean regional activation energy and differ only in fine-grained multi-vertex population geometry, vertex-wise univariate contrasts (e.g., ANOVA or channel-wise ERPs) are mathematically blind (`F = 1.00`, `r = 0.000`), and testing all channel $\times$ condition-pair combinations suffers catastrophic Bonferroni survival collapse (`0.00%` survival across `129,024` parallel ERP $t$-tests in 64-word EEG).
- By combining local searchlight neighborhoods with analytical Ledoit-Wolf shrinkage precision whitening (`rdm.shrinkage_precision`) and cross-validated Mahalanobis (`crossnobis`) distances (`rdm.searchlight`), the multivariate Information Spread Function (ISF) projects out zero-lag volume-conducted crosstalk, recovers iso-energetic population codes, and compresses MEG peak localization error by 55% (`41.4 mm` to `18.5 mm`).

## 2. Scalp EEG Spatial Super-Resolution Rescue & Montage Density
- Three-layer skull volume conduction acts as a spatial low-pass filter on single-latency scalp EEG patterns (`Mode = 'spatial'`, median cortical localization error `36.1 mm`).
- Concatenating short micro-temporal trajectories (`Mode = 'spatiotemporal'`) with Ledoit-Wolf shrinkage precision (`rdm.shrinkage_precision`) exploits millisecond phase-lagged temporal vectors across vertices within a searchlight, rescuing median EEG cortical localization error down to `8.6 mm` (`6.96 mm` mean across well-conditioned head models) and retaining `84.2%` (`32-channel`) and `95.1%` (`19-channel`) of 64-channel natural speech decoding fidelity.

## 3. Biophysical Sulcal-Gyral Double Dissociation & Second-Order Convergence
- Quasi-static Maxwell equations dictate that spherical/BEM MEG sensors are selectively sensitive to **tangential sulcal-wall currents** (such as primary auditory cortex in Heschl's gyrus encoding 24-band cochlear tonotopy), whereas scalp EEG electrodes are sensitive to **radial gyral-crown currents** (such as lateral superior temporal gyrus encoding articulatory manner).
- Despite orthogonal first-order sensor sensitivities (`r = 0.017` waveform correlation), MEG and EEG converge onto a shared second-order 2,016-word-pair representational geometry (`r = 0.148`, `p = 0.0012`) in associative auditory cortex (BanksSTS).

## 4. Recommended Operating Guardrails
Based on systematic sensitivity sweeps across `N = 11` MEG and `N = 8` EEG datasets (`reproducibility/scripts/v2_06_group_inverse_and_preprocessing_benchmark.py`):
1. **Inverse Regularization**: Use standard $\lambda^2 = 1/9$ ($\text{SNR} = 3$) with depth weighting $p = 0.8$.
2. **Temporal Low-Pass Filtering**: Preserve at least **40 Hz** bandwidth (`>= 40 Hz` cutoff); aggressive low-pass filtering at `<= 20 Hz` erases consonantal burst transients and collapses Model 3 articulatory-manner decoding by 58%.
3. **Cortical Spatial Smoothing**: Apply **0 mm FWHM** post-reconstruction cortical smoothing before searchlight RDM calculation; spatial smoothing blurs fine-grained dipole orientation gradients and degrades multivariate ISF specificity.
4. **Searchlight Radius**: Use a **15–20 mm** cortical neighborhood radius (or **25–35 mm** scalp sensor neighborhood radius) paired with `Mode = 'spatiotemporal'` and `Method = 'ledoit-wolf'` precision whitening.
