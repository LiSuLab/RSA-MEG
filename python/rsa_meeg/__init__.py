"""
RSA-MEEG Python Toolbox (version 0.2.0-v2)
==========================================
Open-source Python counterpart to the MATLAB +rsa / +rsa_meeg package for
millisecond sensor- and source-space M/EEG Representational Similarity Analysis:

- `rsa_meeg.sim`: 30-scale cortical geodesic Perlin spatial texture generator,
  64-condition (4x16) and recursive hierarchical pattern simulator, and hybrid
  prestimulus sensor-covariance noise synthesis.
- `rsa_meeg.rdm`: Analytical Ledoit-Wolf / OAS shrinkage precision estimation,
  spatial / temporal / spatiotemporal searchlight crossnobis and correlation RDMs,
  univariate Activation Spread Function (ASF) RDMs, and three-tier zero-tie
  continuous speech model RDM parameterization.
- `rsa_meeg.mapping`: Condition-aligned Spearman / Pearson / Kendall tau-a RDM
  comparison, analytical QR rank-residualized partial RDM regression, and
  cross-modal MEG-EEG second-order convergence.
- `rsa_meeg.stats`: Subject-level sign-flip max-t, cluster-mass, and TFCE
  permutation inference, condition-label permutation, and BH-FDR control.
"""

__version__ = "0.2.0-v2"

import numpy as _np
_np.seterr(divide="ignore", over="ignore", invalid="ignore")

from . import sim, rdm, mapping, stats

__all__ = ["__version__", "sim", "rdm", "mapping", "stats"]
