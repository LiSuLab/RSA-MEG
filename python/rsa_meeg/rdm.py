"""
RDM construction and Ledoit-Wolf shrinkage precision estimation (`rsa_meeg.rdm`).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def _vec_to_square(values: np.ndarray, k_cond: int, pairs: np.ndarray) -> np.ndarray:
    sq = np.zeros((k_cond, k_cond), dtype=float)
    sq[pairs[:, 0], pairs[:, 1]] = values
    sq[pairs[:, 1], pairs[:, 0]] = values
    return sq


def shrinkage_precision(
    residuals: np.ndarray,
    *,
    method: str = "ledoit-wolf",
    shrinkage: Optional[float] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Estimate symmetric positive-definite noise precision matrix P = Sigma_hat^{-1}
    from residual patterns (n_samples x n_features) using analytical Ledoit-Wolf,
    OAS, fixed shrinkage, or diagonal variance standardization.
    """
    X = np.asarray(residuals, dtype=float)
    if X.ndim != 2:
        raise ValueError("residuals must be a 2D array (n_samples x n_features).")
    n, p = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    S = (X.T @ X) / max(n, 1)
    mu = float(np.trace(S)) / max(p, 1)
    if mu <= 0:
        mu = 1.0

    m = method.lower()
    if m == "diagonal":
        diag_var = np.maximum(np.diag(S), eps)
        P = np.diag(1.0 / diag_var)
        lam = 1.0
    else:
        if m == "fixed":
            if shrinkage is None:
                raise ValueError("shrinkage in [0, 1] required when method='fixed'.")
            lam = float(np.clip(shrinkage, 0.0, 1.0))
        elif m in ("ledoit-wolf", "lw"):
            delta = np.sum((S - mu * np.eye(p)) ** 2) / p
            if delta <= 1e-15 or n <= 1:
                lam = 1.0
            else:
                x2 = X ** 2
                beta_bar = (np.sum(x2.T @ x2) / n - np.sum(S ** 2)) / (p * n)
                beta = min(delta, max(0.0, beta_bar))
                lam = float(np.clip(beta / delta, 0.0, 1.0))
        elif m == "oas":
            tr_s2 = float(np.sum(S ** 2))
            tr2_s = float(np.trace(S) ** 2)
            denom = (n + 1.0 - 2.0 / p) * (tr_s2 - tr2_s / p)
            if abs(denom) <= 1e-15:
                lam = 1.0
            else:
                num = (1.0 - 2.0 / p) * tr_s2 + tr2_s
                lam = float(np.clip(num / denom, 0.0, 1.0))
        else:
            raise ValueError(f"Unsupported shrinkage method: {method}")

        sigma_shrunk = (1.0 - lam) * S + lam * mu * np.eye(p)
        eigvals, eigvecs = np.linalg.eigh(sigma_shrunk)
        eigvals = np.maximum(eigvals, eps * max(mu, 1e-6))
        P = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        P = 0.5 * (P + P.T)

    return P, {"shrinkage": lam, "mu": mu, "n_samples": n, "n_features": p, "method": m}


def compute_rdm(
    patterns: np.ndarray,
    *,
    metric: str = "crossnobis",
    precision: Optional[np.ndarray] = None,
    condition_ids: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Compute a Representational Dissimilarity Matrix (RDM) from condition patterns.

    Parameters
    ----------
    patterns : ndarray
        Either (K, D) condition patterns (for 'sqeuclidean' or 'correlation')
        or (n_folds, K, D) fold-wise condition patterns (for 'crossnobis').
    """
    arr = np.asarray(patterns, dtype=float)
    m = metric.lower()

    if m == "crossnobis":
        if arr.ndim != 3:
            raise ValueError("crossnobis requires 3D patterns array (n_folds x n_conditions x n_features).")
        n_folds, k_cond, d_feat = arr.shape
        if n_folds < 2:
            raise ValueError("crossnobis requires at least 2 independent folds.")
        P = np.eye(d_feat) if precision is None else np.asarray(precision, dtype=float)
        pairs = np.array([(i, j) for i in range(k_cond) for j in range(i + 1, k_cond)], dtype=int)
        diffs = arr[:, pairs[:, 0], :] - arr[:, pairs[:, 1], :]  # (F, n_pairs, D)
        sum_diffs = diffs.sum(axis=0)  # (n_pairs, D)
        total_quad = np.einsum("pd,de,pe->p", sum_diffs, P, sum_diffs)
        diag_quad = np.einsum("fpd,de,fpe->p", diffs, P, diffs)
        values = (total_quad - diag_quad) / float(n_folds * (n_folds - 1) * d_feat)
        sq = _vec_to_square(values, k_cond, pairs)
    else:
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        k_cond, d_feat = arr.shape
        pairs = np.array([(i, j) for i in range(k_cond) for j in range(i + 1, k_cond)], dtype=int)
        if m == "sqeuclidean":
            diffs = arr[pairs[:, 0], :] - arr[pairs[:, 1], :]
            values = np.sum(diffs ** 2, axis=1) / float(d_feat)
        elif m == "correlation":
            centered = arr - arr.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            unit = centered / np.maximum(norms, 1e-12)
            corr_mat = unit @ unit.T
            values = 1.0 - corr_mat[pairs[:, 0], pairs[:, 1]]
        elif m == "asf":
            mean_act = arr.mean(axis=1)
            values = (mean_act[pairs[:, 0]] - mean_act[pairs[:, 1]]) ** 2
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        sq = _vec_to_square(values, k_cond, pairs)

    c_ids = np.arange(1, k_cond + 1) if condition_ids is None else np.asarray(condition_ids)
    return {"condition_ids": c_ids, "pairs": pairs, "values": values, "square": sq, "metric": m}


def build_three_tier_speech_rdms(
    loudness_envelopes: np.ndarray,
    cochlear_24band_specs: np.ndarray,
    articulatory_6manner_envelopes: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Construct the V2 three-tier zero-tie continuous speech model RDMs across K spoken words:
      - Model 1 (Acoustic Loudness Envelope): Continuous Hilbert broadband envelope L2 distance
      - Model 2 (24-Band Gammatone Cochlear Tonotopy): Continuous cochlear filterbank distance
      - Model 3 (Articulatory-Manner Envelope Trajectory): 6-channel manner-modulated trajectory distance
    """
    m1 = compute_rdm(np.asarray(loudness_envelopes, dtype=float), metric="sqeuclidean")
    m2 = compute_rdm(np.asarray(cochlear_24band_specs, dtype=float), metric="sqeuclidean")
    m3 = compute_rdm(np.asarray(articulatory_6manner_envelopes, dtype=float), metric="sqeuclidean")

    for name, r in [("model1_loudness", m1), ("model2_tonotopy", m2), ("model3_articulatory", m3)]:
        vals = np.round(r["values"], 12)
        _, counts = np.unique(vals, return_counts=True)
        r["max_tie_fraction"] = float((counts.max() - 1) / max(len(vals), 1)) if len(vals) else 0.0
        r["name"] = name

    return {"model1_loudness": m1, "model2_tonotopy": m2, "model3_articulatory": m3}
