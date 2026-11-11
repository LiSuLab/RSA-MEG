"""
Model correspondence and analytical partial rank regression (`rsa_meeg.mapping`).
"""

from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Compute 1-based mid-ranks with average ties in pure NumPy."""
    arr = np.asarray(x, dtype=float).ravel()
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _kendall_tau_a(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Nili et al. (2014) Kendall tau-a rank correlation in pure NumPy."""
    n = len(x)
    if n < 2:
        return 0.0
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(n, k=1)
    return float(np.sum(dx[iu] * dy[iu]) / float(n * (n - 1) // 2))


def _extract_vector(rdm_or_vec: Union[np.ndarray, dict]) -> np.ndarray:
    if isinstance(rdm_or_vec, dict):
        return np.asarray(rdm_or_vec["values"], dtype=float).ravel()
    return np.asarray(rdm_or_vec, dtype=float).ravel()


def compare_rdm(
    neural_rdm: Union[np.ndarray, dict],
    model_rdm: Union[np.ndarray, dict],
    *,
    correlation: str = "spearman",
) -> float:
    """Compare a neural RDM vector against a candidate model RDM vector."""
    y = _extract_vector(neural_rdm)
    x = _extract_vector(model_rdm)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    if len(y) < 3 or np.std(y) <= 1e-15 or np.std(x) <= 1e-15:
        return 0.0

    c = correlation.lower()
    if c == "spearman":
        ry = _average_ranks(y)
        rx = _average_ranks(x)
        return float(np.corrcoef(ry, rx)[0, 1])
    if c == "pearson":
        return float(np.corrcoef(y, x)[0, 1])
    if c in ("kendall", "kendall-a", "tau-a"):
        return _kendall_tau_a(y, x)
    raise ValueError(f"Unsupported correlation type: {correlation}")


def partial_compare_rdm(
    neural_rdm: Union[np.ndarray, dict],
    target_model: Union[np.ndarray, dict],
    control_models: Union[np.ndarray, dict, Sequence[Union[np.ndarray, dict]]],
    *,
    correlation: str = "spearman",
) -> float:
    """
    Compute partial Spearman (rank-residualized QR projection) or partial Pearson
    correlation between `neural_rdm` and `target_model` controlling for one or more
    covariate RDMs in `control_models`.
    """
    y = _extract_vector(neural_rdm)
    x = _extract_vector(target_model)

    if isinstance(control_models, (dict, np.ndarray)):
        ctrl_arr = _extract_vector(control_models)[:, None] if isinstance(control_models, dict) else np.asarray(control_models, dtype=float)
        if ctrl_arr.ndim == 1:
            ctrl_arr = ctrl_arr[:, None]
    else:
        ctrl_arr = np.column_stack([_extract_vector(cm) for cm in control_models])

    mask = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(ctrl_arr), axis=1)
    y, x, ctrl_arr = y[mask], x[mask], ctrl_arr[mask]
    n = len(y)
    if n <= ctrl_arr.shape[1] + 2:
        raise ValueError("Insufficient finite RDM pairs for partial correlation.")

    c = correlation.lower()
    if c == "spearman":
        y = _average_ranks(y)
        x = _average_ranks(x)
        ctrl_arr = np.column_stack([_average_ranks(ctrl_arr[:, j]) for j in range(ctrl_arr.shape[1])])
    elif c != "pearson":
        raise ValueError(f"Unsupported partial correlation type: {correlation}")

    Z = np.column_stack([np.ones(n, dtype=float), ctrl_arr])
    Q, _ = np.linalg.qr(Z, mode="reduced")
    res_y = y - Q @ (Q.T @ y)
    res_x = x - Q @ (Q.T @ x)

    ny = np.linalg.norm(res_y)
    nx = np.linalg.norm(res_x)
    if ny <= 1e-12 or nx <= 1e-12:
        return 0.0
    return float(np.dot(res_y, res_x) / (ny * nx))
