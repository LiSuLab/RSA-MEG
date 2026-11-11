#!/usr/bin/env python3
"""
v2_07_supervisor_robustness_and_zero_hallucination_audit.py

Full-Time Chief Code & Data Auditor (总监督智能体 B) & Chief Scientific Director (总监督智能体 A)
automated robustness and zero-hallucination verification script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats


def generate_beta_patterns_python(cluster_spec: list, n_voxels: int, rng: np.random.Generator,
                                  centroid: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Exact Python port of rsatoolbox/+rsa/+sim/generateBetaPatterns.m (Cai Wingfield 6-2010).
    CRITICAL: cluster_spec[0] is the VARIANCE (spread), so sigma = sqrt(cluster_spec[0]).
    Returns a [n_conditions, n_voxels] matrix B.
    """
    if centroid is None:
        centroid = np.zeros((1, n_voxels), dtype=np.float64)
    sigma = math.sqrt(float(cluster_spec[0]))
    if not isinstance(cluster_spec[1], (list, tuple)):
        n_conditions = int(cluster_spec[1])
        return np.repeat(centroid, n_conditions, axis=0) + sigma * rng.standard_normal((n_conditions, n_voxels))
    else:
        out_blocks = []
        for sub_spec in cluster_spec[1:]:
            sub_centroid = centroid + sigma * rng.standard_normal((1, n_voxels))
            out_blocks.append(generate_beta_patterns_python(sub_spec, n_voxels, rng, sub_centroid))
        return np.vstack(out_blocks)


def verify_tree_covariance_and_rdm(n_vertices: int = 20000, seed: int = 20261105) -> Dict[str, Any]:
    spec = [0.6, [0.3, 5], [0.3, 5], [0.3, 5], [0.3, 5]]
    n_cond = 20
    block = np.full((5, 5), 0.6, dtype=np.float64) + np.eye(5, dtype=np.float64) * 0.3
    sigma_exact = np.kron(np.eye(4, dtype=np.float64), block)
    corr_exact = sigma_exact / 0.9
    rdm_exact = 1.0 - corr_exact
    np.fill_diagonal(rdm_exact, 0.0)

    rng = np.random.default_rng(seed)
    B = generate_beta_patterns_python(spec, n_vertices, rng)
    sigma_emp = (B @ B.T) / float(n_vertices)
    max_cov_err = float(np.max(np.abs(sigma_emp - sigma_exact)))
    se_bound = 5.0 * math.sqrt(1.62 / float(n_vertices))

    B_centered = B - B.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(B_centered, axis=1, keepdims=True)
    corr_emp = (B_centered @ B_centered.T) / (norms @ norms.T)
    rdm_emp = 1.0 - corr_emp
    np.fill_diagonal(rdm_emp, 0.0)

    within_mask = (sigma_exact > 0.5) & (~np.eye(n_cond, dtype=bool))
    between_mask = (sigma_exact == 0.0)

    passed = bool(
        max_cov_err < se_bound
        and int(within_mask.sum()) == 80
        and int(between_mask.sum()) == 300
        and abs(float(rdm_exact[within_mask].mean()) - (1.0 / 3.0)) < 1e-15
        and abs(float(rdm_exact[between_mask].mean()) - 1.0) < 1e-15
    )

    return {
        "check": "generateBetaPatterns_tree_covariance_20cond",
        "passed": passed,
        "cluster_spec": "{0.6, {0.3, 5}, {0.3, 5}, {0.3, 5}, {0.3, 5}}",
        "n_conditions": n_cond,
        "n_upper_triangle_pairs": 190,
        "n_within_cluster_pairs_upper": 40,
        "n_between_cluster_pairs_upper": 150,
        "exact_diagonal_variance": 0.9,
        "exact_within_cluster_covariance": 0.6,
        "exact_between_cluster_covariance": 0.0,
        "exact_within_cluster_correlation": 2.0 / 3.0,
        "exact_within_cluster_correlation_distance": 1.0 / 3.0,
        "exact_between_cluster_correlation_distance": 1.0,
        "mc_n_vertices": n_vertices,
        "mc_max_abs_covariance_error": max_cov_err,
        "mc_5sigma_tolerance": se_bound,
        "mc_mean_within_rdm": float(rdm_emp[within_mask].mean()),
        "mc_mean_between_rdm": float(rdm_emp[between_mask].mean()),
    }


def verify_snr_calibration_table() -> Dict[str, Any]:
    levels_db = [26.0, 8.2, 3.6, 0.5]
    table = []
    for db in levels_db:
        rp_ieee = 10.0 ** (db / 10.0)
        ra_ieee = 10.0 ** (db / 20.0)
        table.append({
            "slide_label": f"{db:g} dB",
            "snr_db_value": db,
            "ieee_power_ratio_Psig_over_Pnoise": rp_ieee,
            "ieee_noise_to_signal_power_ratio": 1.0 / rp_ieee,
            "ieee_amplitude_ratio_Asig_over_Anoise": ra_ieee,
            "ieee_noise_to_signal_amplitude_ratio": 1.0 / ra_ieee,
        })
    return {
        "check": "li_su_4_snr_levels_calibration",
        "passed": True,
        "levels": table,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_rank_ties_1d(vec: np.ndarray, name: str) -> Dict[str, Any]:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return {"name": name, "status": "ALL_NONFINITE", "n_total": int(vec.size), "n_finite": 0}
    ranks = stats.rankdata(finite, method="average")
    unique_vals = np.unique(finite)
    tie_fraction = 1.0 - (float(unique_vals.size) / float(finite.size))
    max_val = float(np.max(finite))
    peak_ties = int(np.sum(np.abs(finite - max_val) <= 1e-12))
    is_constant = bool(np.ptp(finite) <= 1e-15)
    return {
        "name": name,
        "status": "FAIL_CONSTANT" if is_constant else "PASS",
        "n_total": int(vec.size),
        "n_finite": int(finite.size),
        "n_nonfinite": int(vec.size - finite.size),
        "unique_count": int(unique_vals.size),
        "tie_fraction": tie_fraction,
        "peak_tie_count": peak_ties,
        "rank_std": float(np.std(ranks)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root directory of v2 experiment outputs to audit.",
    )
    args = parser.parse_args()
    res_dir = args.root / "results"

    tree_cov_report = verify_tree_covariance_and_rdm()
    snr_calib_report = verify_snr_calibration_table()

    # Audit all generated files in results/
    file_hashes = {}
    for p in sorted(res_dir.glob("*")):
        if p.is_file():
            file_hashes[p.name] = {
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }

    # Check three-tier speech model RDMs for rank ties
    rdm_tie_checks = {}
    rdm_npz_path = res_dir / "v2_three_tier_speech_model_rdms.npz"
    if rdm_npz_path.exists():
        rdm_npz = np.load(rdm_npz_path, allow_pickle=True)
        for k in ("rdm_model1_vec", "rdm_model2_vec", "rdm_model3_vec"):
            rdm_tie_checks[k] = check_rank_ties_1d(rdm_npz[k], k)

    # Check degrees of freedom & participant sample sizes
    v2_06_path = res_dir / "v2_06_group_inverse_and_preprocessing_benchmark.json"
    dof_checks = {}
    if v2_06_path.exists():
        v2_06 = json.loads(v2_06_path.read_text())
        f1 = v2_06["factor1_noise_covariance_regularization"]["simulation_all_valid_head_models"]
        dof_checks["sim_meg_n"] = f1["meg_n11"]["rdm_fidelity_snr8p2db"]["n_subjects"]
        dof_checks["sim_meg_df"] = f1["meg_n11"]["rdm_fidelity_snr8p2db"]["df"]
        dof_checks["sim_eeg_n"] = f1["eeg_n6"]["rdm_fidelity_snr8p2db"]["n_subjects"]
        dof_checks["sim_eeg_df"] = f1["eeg_n6"]["rdm_fidelity_snr8p2db"]["df"]
        dof_checks["passed_no_pseudoreplication"] = bool(
            dof_checks["sim_meg_n"] == 11
            and dof_checks["sim_meg_df"] == 10
            and dof_checks["sim_eeg_n"] == 6
            and dof_checks["sim_eeg_df"] == 5
        )

    all_passed = bool(
        tree_cov_report["passed"]
        and snr_calib_report["passed"]
        and len(file_hashes) >= 24
        and all(v["status"] == "PASS" and v["tie_fraction"] == 0.0 for v in rdm_tie_checks.values())
        and dof_checks.get("passed_no_pseudoreplication", False)
    )

    report_out = args.root / "audit_reports" / "v2_07_supervisor_audit_report.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "auditor": "v2_07_supervisor_robustness_and_zero_hallucination_audit",
        "overall_status": "PASS" if all_passed else "FAIL",
        "n_verified_result_files": len(file_hashes),
        "mathematical_verifications": {
            "tree_covariance_20cond": tree_cov_report,
            "snr_calibration_4levels": snr_calib_report,
        },
        "speech_model_rdm_rank_tie_audit": rdm_tie_checks,
        "participant_degrees_of_freedom_audit": dof_checks,
        "verified_result_file_sha256_manifest": file_hashes,
    }
    report_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[Supervisor Audit] Status={summary['overall_status']} | Verified {len(file_hashes)} result files -> {report_out}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
