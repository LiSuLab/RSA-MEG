"""
Simulation module (`rsa_meeg.sim`) for V2 cortical geodesic Perlin spatial frequency
transfer and hierarchical multivariate pattern generation.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def _sqeuclidean_pairwise(X: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distance matrix (K x K) in pure NumPy."""
    X = np.asarray(X, dtype=float)
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
    d2 = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, 0.0)
    return np.maximum(d2, 0.0)


def _euclidean_pairwise(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix (K x K) in pure NumPy."""
    return np.sqrt(_sqeuclidean_pairwise(X))


def _build_hierarchical_covariance(spec: Any, mode: str = "spread") -> np.ndarray:
    """Recursively construct a hierarchical block covariance matrix from a nested specification."""
    if isinstance(spec, np.ndarray):
        cov = np.asarray(spec, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Matrix covariance specification must be square.")
        return cov
    if not isinstance(spec, (list, tuple)) or len(spec) < 2:
        raise ValueError("Hierarchical cluster_spec must be a sequence [param, child1, ...].")

    param = float(spec[0])
    if len(spec) == 2 and isinstance(spec[1], int):
        n_leaves = int(spec[1])
        if n_leaves < 1 or param <= 0:
            raise ValueError("Leaf specification requires positive variance and count >= 1.")
        return param * np.eye(n_leaves, dtype=float)

    blocks = []
    for child in spec[1:]:
        child_cov = _build_hierarchical_covariance(child, mode=mode)
        if mode == "spread":
            child_cov = child_cov + param * np.ones_like(child_cov)
        blocks.append(child_cov)

    total = sum(b.shape[0] for b in blocks)
    cov = np.zeros((total, total), dtype=float)
    offset = 0
    for b in blocks:
        k = b.shape[0]
        cov[offset : offset + k, offset : offset + k] = b
        offset += k
    return cov


def generate_beta_patterns(
    cluster_spec: Any = (0.6, (0.3, 16), (0.3, 16), (0.3, 16), (0.3, 16)),
    n_features: int = 64,
    *,
    normalize_diagonal: bool = True,
    exact_covariance: bool = False,
    zero_mean_across_features: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate multivariate condition patterns B (K x P), true condition covariance
    matrix cov_true (K x K), and ground-truth RDM dictionary rdm_true.

    Default `cluster_spec` generates the canonical V2 64-condition (4 clusters x 16
    exemplars, 2,016 condition pairs) hierarchy.
    """
    rng = np.random.default_rng(seed)
    cov_true = _build_hierarchical_covariance(cluster_spec, mode="spread")
    k_cond = cov_true.shape[0]

    if normalize_diagonal:
        d = np.sqrt(np.diag(cov_true))
        cov_true = cov_true / np.outer(d, d)

    eigvals, eigvecs = np.linalg.eigh(cov_true)
    eigvals = np.clip(eigvals, 1e-12, None)
    sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    z = rng.standard_normal((k_cond, n_features))
    if exact_covariance and n_features >= k_cond:
        z = z - z.mean(axis=1, keepdims=True)
        u, _, vt = np.linalg.svd(z, full_matrices=False)
        z = np.sqrt(n_features) * (u @ vt)

    B = sqrt_cov @ z
    if zero_mean_across_features:
        B = B - B.mean(axis=1, keepdims=True)

    pairs = np.array([(i, j) for i in range(k_cond) for j in range(i + 1, k_cond)], dtype=int)
    sq_mat = _sqeuclidean_pairwise(B) / float(n_features)
    rdm_vec = sq_mat[pairs[:, 0], pairs[:, 1]]

    rdm_true = {
        "condition_ids": np.arange(1, k_cond + 1),
        "pairs": pairs,
        "values": rdm_vec,
        "square": sq_mat,
    }
    return B, cov_true, rdm_true


def cortical_perlin_noise(
    positions_mm: np.ndarray,
    n_observations: int,
    n_times: int = 1,
    *,
    scales_mm: Optional[Sequence[float]] = None,
    spectral_slope: float = 1.0,
    hemisphere: Optional[Sequence[str]] = None,
    distance_matrix: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Synthesize 30-scale (2-60 mm) continuous cortical Perlin background texture
    with 1/f^alpha spatial spectral weighting.
    """
    rng = np.random.default_rng(seed)
    pos = np.asarray(positions_mm, dtype=float)
    n_vertices = pos.shape[0]
    if scales_mm is None:
        scales = np.linspace(2.0, 60.0, 30)
    else:
        scales = np.asarray(scales_mm, dtype=float)

    if distance_matrix is not None:
        dist_mat = np.asarray(distance_matrix, dtype=float)
    else:
        dist_mat = _euclidean_pairwise(pos)

    if hemisphere is not None:
        hemi = np.asarray(hemisphere)
        cross_hemi = hemi[:, None] != hemi[None, :]
        dist_mat = np.where(cross_hemi, np.inf, dist_mat)

    weights = scales ** float(spectral_slope)
    weights = weights / np.linalg.norm(weights)

    n_draws = int(n_observations) * int(n_times)
    field = np.zeros((n_vertices, n_draws), dtype=float)

    for w, r_mm in zip(weights, scales):
        K = np.exp(-0.5 * (dist_mat / float(r_mm)) ** 2)
        row_norms = np.linalg.norm(K, axis=1, keepdims=True)
        K = K / np.maximum(row_norms, 1e-12)
        z = rng.standard_normal((n_vertices, n_draws))
        field += w * (K @ z)

    field = field - field.mean(axis=1, keepdims=True)
    std = field.std(axis=1, keepdims=True)
    field = field / np.maximum(std, 1e-12)

    noise_data = field.T.reshape(int(n_observations), int(n_times), n_vertices).transpose(0, 2, 1)
    info = {
        "scales_mm": scales,
        "weights": weights,
        "spectral_slope": float(spectral_slope),
        "n_vertices": n_vertices,
    }
    return noise_data, info
