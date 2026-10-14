import math
import unittest

import numpy as np

from benchmarks.sisken_prastiyanto import lean_simulate_system
from core.fast_acc import (
    OBJECTIVE_IAE,
    OBJECTIVE_ISE,
    OBJECTIVE_ITAE,
    OBJECTIVE_ITSE,
    fast_acc_objective,
)
from core.utils import iae, ise, itae, itse


class NumbaP0EquivalenceTests(unittest.TestCase):
    """Numerical-equivalence guard between reference Python and P0 JIT backend."""

    OBJECTIVES = [
        (iae, OBJECTIVE_IAE),
        (ise, OBJECTIVE_ISE),
        (itae, OBJECTIVE_ITAE),
        (itse, OBJECTIVE_ITSE),
    ]

    def assert_equivalent(self, params, obj_fun, obj_id):
        reference = lean_simulate_system(params, obj_fun)
        accelerated = fast_acc_objective(params, obj_id)
        self.assertTrue(math.isfinite(reference))
        self.assertTrue(math.isfinite(accelerated))
        self.assertTrue(
            math.isclose(reference, accelerated, rel_tol=1e-12, abs_tol=1e-12),
            msg=(
                f"Reference/JIT mismatch for {params}: "
                f"reference={reference!r}, accelerated={accelerated!r}, "
                f"abs_diff={abs(reference-accelerated)!r}"
            ),
        )

    def test_boundary_vectors_all_objectives(self):
        vectors = [
            [-100.0, -100.0, -100.0],
            [100.0, 100.0, 100.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
            [0.0, 0.0, 0.0],
        ]
        for params in vectors:
            for obj_fun, obj_id in self.OBJECTIVES:
                with self.subTest(params=params, objective=obj_fun.__name__):
                    self.assert_equivalent(params, obj_fun, obj_id)

    def test_seeded_random_vectors_all_objectives(self):
        rng = np.random.default_rng(20260920)
        for params in rng.uniform(-100.0, 100.0, size=(32, 3)):
            params = params.tolist()
            for obj_fun, obj_id in self.OBJECTIVES:
                with self.subTest(params=params, objective=obj_fun.__name__):
                    self.assert_equivalent(params, obj_fun, obj_id)

    def test_seeded_sensor_noise_matches_reference(self):
        cases = [
            ([1.0, 0.5, 1.0], 0.25, 123),
            ([4.0, 2.0, 0.75], 0.50, 20260925),
            ([10.0, 1.0, 3.0], 1.00, 7),
        ]
        for params, noise_std, seed in cases:
            for obj_fun, obj_id in self.OBJECTIVES:
                with self.subTest(
                    params=params, objective=obj_fun.__name__, seed=seed
                ):
                    reference = lean_simulate_system(
                        params,
                        obj_fun,
                        sim_time=2.0,
                        sensor_noise_std_m=noise_std,
                        sensor_seed=seed,
                    )
                    accelerated = fast_acc_objective(
                        params,
                        obj_id,
                        sim_time=2.0,
                        sensor_noise_std_m=noise_std,
                        sensor_seed=seed,
                    )
                    self.assertTrue(
                        math.isclose(
                            reference, accelerated, rel_tol=1e-12, abs_tol=1e-12
                        ),
                        msg=(
                            f"Noisy reference/JIT mismatch for {params}, "
                            f"std={noise_std}, seed={seed}: "
                            f"reference={reference!r}, accelerated={accelerated!r}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
