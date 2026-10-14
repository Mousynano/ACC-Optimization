import unittest

import numpy as np

from benchmarks.acc_cost_fje_v2 import evaluate_performance_vector as evaluate_fje
from benchmarks.acc_cost_v2 import evaluate_performance_vector as evaluate_fj_legacy
from core.energy import trajectory_traction_energy
from core.scenarios_v2 import load_scenario_suite


class EnergyPhysicsTests(unittest.TestCase):
    def test_positive_energy_never_gets_braking_credit(self):
        e = trajectory_traction_energy([20.0, 20.0], [-3.0, -3.0], 0.1)
        self.assertEqual(e.positive_energy_j, 0.0)
        self.assertGreater(e.braking_energy_j, 0.0)

    def test_constant_speed_requires_positive_roadload_energy(self):
        e = trajectory_traction_energy([20.0] * 20, [0.0] * 20, 0.1)
        self.assertGreater(e.positive_energy_j, 0.0)
        self.assertGreater(e.positive_wh_per_km, 0.0)


class FJEFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = load_scenario_suite(development=True, require_frozen=False)

    def test_fje_preserves_v21_fj_terms(self):
        for gains in ([3.0, 5.0, 0.8], [1.0, 1.0, 1.0], [5.0, 2.0, 1.5]):
            with self.subTest(gains=gains):
                base = evaluate_fj_legacy(gains, self.suite)
                ext = evaluate_fje(gains, self.suite)
                np.testing.assert_allclose(base.as_array(), ext.as_array()[:3], rtol=1e-12, atol=1e-12)
                self.assertGreater(ext.positive_tractive_energy_wh_per_km, 0.0)


if __name__ == "__main__":
    unittest.main()
