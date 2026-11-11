"""
Random-effects sign-flip permutation inference and FDR control (`rsa_meeg.stats`).
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np


def group_permutation_test(
    subject_maps: np.ndarray,
    *,
    n_permutations: int = 1000,
    tail: str = "two-sided",
    exact: bool = True,
    alpha: float = 0.05,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Subject-level whole-map sign-flip max-t permutation test across `subject_maps`
    (n_subjects x n_tests). Uses exact enumeration when `exact=True` and n_subjects <= 16.
    """
    X = np.asarray(subject_maps, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n_subj, n_tests = X.shape
    if n_subj < 2:
        raise ValueError("At least 2 independent subjects are required.")

    mean_obs = X.mean(axis=0)
    se_obs = X.std(axis=0, ddof=1) / np.sqrt(n_subj)
    t_obs = np.where(se_obs > 1e-15, mean_obs / se_obs, 0.0)

    use_exact = exact and (n_subj <= 16)
    if use_exact:
        n_perm = 1 << n_subj
        bits = ((np.arange(n_perm, dtype=np.uint32)[:, None] >> np.arange(n_subj, dtype=np.uint32)) & 1)
        signs = 2.0 * bits.astype(float) - 1.0
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_subj))
        n_perm = n_permutations

    flipped_means = (signs @ X) / float(n_subj)
    x_sq_sum = np.sum(X ** 2, axis=0, keepdims=True)
    flipped_vars = np.maximum(0.0, (x_sq_sum - n_subj * (flipped_means ** 2)) / float(n_subj - 1))
    flipped_se = np.sqrt(flipped_vars / float(n_subj))
    t_null = np.where(flipped_se > 1e-15, flipped_means / flipped_se, 0.0)

    if tail == "positive":
        null_max = np.max(t_null, axis=1)
        target = t_obs
        uncorr_counts = np.sum(t_null >= target[None, :] - 1e-12, axis=0)
    elif tail == "negative":
        null_max = np.max(-t_null, axis=1)
        target = -t_obs
        uncorr_counts = np.sum(-t_null >= target[None, :] - 1e-12, axis=0)
    else:
        null_max = np.max(np.abs(t_null), axis=1)
        target = np.abs(t_obs)
        uncorr_counts = np.sum(np.abs(t_null) >= target[None, :] - 1e-12, axis=0)

    counts = np.sum(null_max[:, None] >= target[None, :] - 1e-12, axis=0)
    if use_exact:
        p_fwer = counts / float(n_perm)
        p_uncorrected = uncorr_counts / float(n_perm)
    else:
        p_fwer = (1.0 + counts) / float(n_perm + 1.0)
        p_uncorrected = (1.0 + uncorr_counts) / float(n_perm + 1.0)

    return {
        "statistic": t_obs,
        "p_fwer": p_fwer,
        "p_uncorrected": p_uncorrected,
        "significant": p_fwer <= alpha,
        "null_max": null_max,
        "n_permutations": int(n_perm),
        "exact": bool(use_exact),
    }


def bh_fdr(p_values: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray(p_values, dtype=float).ravel()
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / float(m))
    passed = np.where(ranked <= thresh)[0]
    reject = np.zeros(m, dtype=bool)
    if len(passed) > 0:
        k_max = passed.max()
        reject[order[: k_max + 1]] = True
        crit = float(ranked[k_max])
    else:
        crit = 0.0

    p_adj_sorted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0.0, 1.0)
    p_adj = np.empty_like(p_adj_sorted)
    p_adj[order] = p_adj_sorted
    return {"reject": reject, "p_adjusted": p_adj, "threshold": crit, "alpha": alpha}
