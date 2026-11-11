#!/usr/bin/env python3
"""Unit tests for the Python `rsa_meeg` package (v0.2.0-v2)."""

import unittest
import numpy as np
import rsa_meeg


class TestRSAMEEGPython(unittest.TestCase):
    def test_version_and_64condition_hierarchy(self):
        self.assertEqual(rsa_meeg.__version__, "0.2.0-v2")
        B, cov_true, rdm_true = rsa_meeg.sim.generate_beta_patterns(
            (0.6, (0.3, 16), (0.3, 16), (0.3, 16), (0.3, 16)),
            n_features=80,
            exact_covariance=True,
            seed=42,
        )
        self.assertEqual(B.shape, (64, 80))
        self.assertEqual(cov_true.shape, (64, 64))
        self.assertEqual(rdm_true["values"].shape, (2016,))
        cov_emp = (B @ B.T) / 80.0
        self.assertLess(np.linalg.norm(cov_emp - cov_true), 1e-10)
        self.assertLess(rdm_true["square"][0, 1], rdm_true["square"][0, 20])

    def test_30scale_perlin_and_shrinkage_crossnobis(self):
        pos = np.column_stack([np.arange(20) * 4.0, np.zeros((20, 2))])
        noise, info = rsa_meeg.sim.cortical_perlin_noise(pos, n_observations=40, n_times=1, seed=7)
        self.assertEqual(noise.shape, (40, 20, 1))
        self.assertEqual(len(info["scales_mm"]), 30)

        P, p_info = rsa_meeg.rdm.shrinkage_precision(noise[:, :, 0], method="ledoit-wolf")
        self.assertEqual(P.shape, (20, 20))
        self.assertGreaterEqual(p_info["shrinkage"], 0.0)
        self.assertLessEqual(p_info["shrinkage"], 1.0)
        eigvals = np.linalg.eigvalsh(P)
        self.assertGreater(eigvals.min(), 0.0)

        # Unbiased zero-signal crossnobis expectation
        rng = np.random.default_rng(123)
        folds_noise = rng.standard_normal((4, 16, 20))
        rdm_null = rsa_meeg.rdm.compute_rdm(folds_noise, metric="crossnobis", precision=np.eye(20))
        self.assertLess(abs(rdm_null["values"].mean()), 0.15)

    def test_three_tier_zero_tie_and_partial_compare(self):
        rng = np.random.default_rng(99)
        m1_feat = rng.standard_normal((64, 25))
        m2_feat = m1_feat @ rng.standard_normal((25, 24)) + 0.8 * rng.standard_normal((64, 24))
        m3_feat = rng.standard_normal((64, 30))
        models = rsa_meeg.rdm.build_three_tier_speech_rdms(m1_feat, m2_feat, m3_feat)
        for k, m in models.items():
            self.assertEqual(m["values"].shape, (2016,))
            self.assertEqual(m["max_tie_fraction"], 0.0)

        # Neural RDM driven purely by Model 1 + Model 3
        y_sim = 0.7 * models["model1_loudness"]["values"] + 0.6 * models["model3_articulatory"]["values"]
        r_m2_raw = rsa_meeg.mapping.compare_rdm(y_sim, models["model2_tonotopy"])
        r_m2_part = rsa_meeg.mapping.partial_compare_rdm(
            y_sim, models["model2_tonotopy"], models["model1_loudness"]
        )
        r_m3_part = rsa_meeg.mapping.partial_compare_rdm(
            y_sim,
            models["model3_articulatory"],
            [models["model1_loudness"], models["model2_tonotopy"]],
        )
        self.assertLess(abs(r_m2_part), abs(r_m2_raw) + 0.05)
        self.assertGreater(r_m3_part, 0.35)

    def test_group_sign_flip_and_bh(self):
        rng = np.random.default_rng(2026)
        subj_maps_8 = rng.standard_normal((8, 6))
        subj_maps_8[:, 0] += 3.5
        out8 = rsa_meeg.stats.group_permutation_test(subj_maps_8, exact=True)
        self.assertEqual(out8["n_permutations"], 256)
        self.assertTrue(out8["significant"][0])
        fdr8 = rsa_meeg.stats.bh_fdr(out8["p_uncorrected"], alpha=0.05)
        self.assertTrue(fdr8["reject"][0])

        subj_maps_11 = rng.standard_normal((11, 10))
        subj_maps_11[:, 0] += 3.5
        out11 = rsa_meeg.stats.group_permutation_test(subj_maps_11, exact=True)
        self.assertEqual(out11["n_permutations"], 2048)
        self.assertTrue(out11["significant"][0])
        fdr11 = rsa_meeg.stats.bh_fdr(out11["p_uncorrected"], alpha=0.05)
        self.assertTrue(fdr11["reject"][0])


if __name__ == "__main__":
    unittest.main()
