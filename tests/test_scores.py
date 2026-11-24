import unittest

import numpy as np
from scipy.special import softmax

from aic_finegrained.scores import calibrate_scores, fuse_scores, synthetic_comparison
from aic_finegrained.transport import self_test


class ScoreTests(unittest.TestCase):
    def test_solver_against_independent_optimum(self):
        result = self_test()
        self.assertLess(result["independent_primal_SLSQP_max_error"], 2e-6)

    def test_unconstrained_and_balanced_have_distinct_marginals(self):
        scores = np.array([[2.0, 0.1], [2.3, -0.1], [3.3, 1.0]])
        raw, _ = calibrate_scores(scores, "none")
        soft, _ = calibrate_scores(scores, "sot")
        hard, _ = calibrate_scores(scores, "balanced")
        np.testing.assert_allclose(raw, softmax(scores, 1), atol=1e-12)
        np.testing.assert_allclose(hard.sum(0), [1.5, 1.5], atol=2e-9)
        self.assertGreater(soft[:, 0].sum(), 1.5)
        self.assertLess(soft[:, 0].sum(), raw[:, 0].sum())

    def test_fusion_clamps_zero_neighbor_probability(self):
        fused = fuse_scores(np.array([[0.0, 0.0]]), np.array([[0.0, 1.0]]))
        self.assertEqual(fused.dtype, np.float64)
        np.testing.assert_allclose(fused, [[0.2 * np.log(1e-12), 0.0]], atol=1e-6)

    def test_synthetic_report_uses_identical_cohort(self):
        report = synthetic_comparison()
        self.assertEqual(report["synthetic_class_counts"], [120, 90, 60, 30])
        self.assertEqual([row["method"] for row in report["results"]], ["none", "balanced", "sot"])
        self.assertEqual(report["results"][1]["soft_class_mass"], [75.0] * 4)


if __name__ == "__main__":
    unittest.main()
