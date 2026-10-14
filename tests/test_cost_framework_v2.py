import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from benchmarks.acc_cost_fje_v2 import (
    PERFORMANCE_NAMES,
    evaluate_performance_vector,
    sahib_contribution_weights,
    scalar_cost_from_performance,
)
from core.config import load_manifest
from core.scenarios_v2 import load_scenario_suite
from optimizers.mopso import MOEval, run_mopso
from optimizers.pso import run_pso


class V22FJEManifestTests(unittest.TestCase):
    def test_main_contract(self):
        m = load_manifest()
        self.assertEqual(m["benchmark"], "acc_cost_sahib_fje_v2")
        self.assertEqual(m["framework_version"], "2.2-FJE")
        self.assertEqual(m["controller"]["decision_vector"], ["k_ve", "k_de", "k_vrel"])
        self.assertEqual(m["optimization"]["objectives"], ["proposed_cost_fje"])
        self.assertEqual(m["cost_framework"]["performance_vector"], list(PERFORMANCE_NAMES))
        self.assertEqual(len(m["optimization"]["algorithms"]), 6)
        self.assertEqual(m["optimization"]["independent_runs"], 30)
        self.assertEqual(m["optimization"]["max_iterations"], 30)
        self.assertEqual(m["optimization"]["population_size"], 100)
        self.assertEqual(m["optimization"]["total_jobs"], 180)
        self.assertFalse(m["cost_framework"]["energy_definition"]["negative_tractive_power_credit"])
        self.assertFalse(m["cost_framework"]["energy_definition"]["fuel_or_battery_claim"])

    def test_draft_scenarios_cannot_be_research_suite(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "scenario.json"
            path.write_text(json.dumps({
                "status": "draft",
                "sampling_hz": 60,
                "calibration": {"cruise": [], "following": []},
            }), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_scenario_suite(path, development=False, require_frozen=True)


class V22FJEMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = load_scenario_suite(development=True, require_frozen=False)

    def test_development_metrics_are_finite_and_four_dimensional(self):
        perf = evaluate_performance_vector([1.0, 1.0, 1.0], self.suite)
        self.assertTrue(perf.feasible)
        self.assertEqual(perf.as_array().shape, (4,))
        self.assertTrue(np.all(np.isfinite(perf.as_array())))
        self.assertTrue(np.all(perf.as_array() >= 0.0))
        self.assertGreater(perf.positive_tractive_energy_wh_per_km, 0.0)

    def test_sahib_weights_inverse_mean_and_sum_to_one(self):
        front = np.asarray([
            [1.0, 10.0, 100.0, 1000.0],
            [1.0, 10.0, 100.0, 1000.0],
        ])
        w = sahib_contribution_weights(front)
        self.assertEqual(w.shape, (4,))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[2])
        self.assertGreater(w[2], w[3])

    def test_scalar_cost_rejects_non_normalized_or_wrong_dimension_weights(self):
        perf = evaluate_performance_vector([1.0, 1.0, 1.0], self.suite)
        with self.assertRaises(ValueError):
            scalar_cost_from_performance(perf, [1.0, 1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            scalar_cost_from_performance(perf, [1/3, 1/3, 1/3])


class CalibrationOptimizerTests(unittest.TestCase):
    def test_pso_is_seed_deterministic(self):
        f = lambda x: float(np.sum((np.asarray(x) - 1.5) ** 2))
        a = run_pso(f, [-5, -5], [5, 5], population_size=8, max_iter=4, seed=7)
        b = run_pso(f, [-5, -5], [5, 5], population_size=8, max_iter=4, seed=7)
        self.assertTrue(np.allclose(a[0], b[0]))
        self.assertAlmostEqual(a[1], b[1], places=12)

    def test_mopso_supports_four_objective_archive(self):
        def f(x):
            x0 = float(x[0])
            return MOEval(np.asarray([
                x0 * x0,
                (x0 - 2.0) ** 2,
                (x0 + 1.0) ** 2,
                (x0 - 0.5) ** 2,
            ]), True, 0.0)

        front = run_mopso(
            f,
            [-1.0],
            [3.0],
            population_size=10,
            max_iter=4,
            archive_size=20,
            grid_divisions=5,
            seed=11,
        )
        self.assertGreaterEqual(len(front), 2)
        for item in front:
            self.assertTrue(item.evaluation.feasible)
            self.assertEqual(item.evaluation.values.shape, (4,))


if __name__ == "__main__":
    unittest.main()
