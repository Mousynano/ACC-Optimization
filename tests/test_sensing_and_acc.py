import inspect
import math
import unittest

from benchmarks.sisken_prastiyanto import (
    BENCHMARKS_ACC,
    SPACING_ERROR_SCALE,
    SPEED_ERROR_SCALE,
    lean_simulate_system,
    normalized_error_cost,
    pd_acc_control,
)
from core.config import load_json_relative
from core.sensing import RangeOnlySensor, RangeSensorConfig
from core.utils import iae, ise


class RangeOnlySensorTests(unittest.TestCase):
    def test_first_sample_has_no_range_rate(self):
        sensor = RangeOnlySensor(dt=0.1)
        reading = sensor.sense(50.0)
        self.assertFalse(reading.rate_valid)
        self.assertEqual(reading.relative_velocity, 0.0)

    def test_range_rate_comes_from_sensed_distance_difference(self):
        sensor = RangeOnlySensor(dt=0.1)
        sensor.sense(50.0)
        reading = sensor.sense(49.7)
        self.assertTrue(reading.rate_valid)
        self.assertAlmostEqual(reading.relative_velocity, -3.0, places=10)

    def test_nominal_sensor_is_unbiased_above_clamp(self):
        sensor = RangeOnlySensor(dt=0.1)
        reading = sensor.sense(50.0)
        self.assertEqual(reading.distance, 50.0)
        self.assertEqual(reading.injected_noise, 0.0)
        self.assertEqual(reading.measurement_error, 0.0)

    def test_sensor_clamps_negative_measurement(self):
        sensor = RangeOnlySensor(dt=0.1)
        reading = sensor.sense(-2.0)
        self.assertEqual(reading.distance, 0.0)
        self.assertEqual(reading.measurement_error, 2.0)

    def test_seeded_gaussian_noise_is_reproducible(self):
        cfg = RangeSensorConfig(noise_std_m=0.5, seed=42)
        sensor_a = RangeOnlySensor(dt=0.1, config=cfg)
        sensor_b = RangeOnlySensor(dt=0.1, config=cfg)
        seq_a = [sensor_a.sense(50.0).distance for _ in range(5)]
        seq_b = [sensor_b.sense(50.0).distance for _ in range(5)]
        self.assertEqual(seq_a, seq_b)
        self.assertTrue(any(abs(x - 50.0) > 0.0 for x in seq_a))


class ControllerBoundaryTests(unittest.TestCase):
    def test_controller_signature_has_no_privileged_lead_state(self):
        names = set(inspect.signature(pd_acc_control).parameters)
        forbidden = {"x_lead", "v_lead", "a_lead", "xlead", "vlead", "alead"}
        self.assertTrue(names.isdisjoint(forbidden))

    def test_gap_sign_has_expected_effect(self):
        a_cmd, _d_safe, _speed_error, gap_error = pd_acc_control(
            k_v=1.0,
            k_p=0.5,
            k_d=1.0,
            v_ref=25.0,
            v_ego=22.0,
            d_sensed=30.0,
            v_rel_est=-2.0,
        )
        self.assertLess(gap_error, 0.0)
        self.assertLess(a_cmd, 0.0)


class NormalizedObjectiveTests(unittest.TestCase):
    def test_reference_scale_derivation(self):
        self.assertAlmostEqual(SPACING_ERROR_SCALE, 50.0)
        self.assertAlmostEqual(SPEED_ERROR_SCALE, 25.0)

    def test_equal_normalized_iae_terms(self):
        self.assertAlmostEqual(normalized_error_cost(50.0, 25.0, 0.0, iae), 2.0)

    def test_equal_normalized_ise_terms(self):
        self.assertAlmostEqual(normalized_error_cost(50.0, 25.0, 0.0, ise), 2.0)

    def test_manifest_benchmark_namespace(self):
        manifest = load_json_relative("experiment_manifest_acc_range_pd_norm_v3.json")
        self.assertEqual(set(BENCHMARKS_ACC), {manifest["benchmark"]})


class SimulationSmokeTests(unittest.TestCase):
    def test_main_simulation_exposes_sensor_diagnostics(self):
        objective, history = lean_simulate_system(
            [1.0, 0.5, 1.0], iae, return_history=True, sim_time=1.0
        )
        self.assertTrue(math.isfinite(objective))
        self.assertEqual(len(history["time"]), 60)
        self.assertFalse(history["range_rate_valid"][0])
        self.assertTrue(all(history["range_rate_valid"][1:]))
        self.assertEqual(history["dist"], history["dist_sensed"])

    def test_noisy_simulation_is_seeded(self):
        kwargs = dict(
            params=[1.0, 0.5, 1.0],
            obj_fun=iae,
            return_history=True,
            sim_time=1.0,
            sensor_noise_std_m=0.25,
            sensor_seed=123,
        )
        obj_a, history_a = lean_simulate_system(**kwargs)
        obj_b, history_b = lean_simulate_system(**kwargs)
        self.assertEqual(obj_a, obj_b)
        self.assertEqual(history_a["dist_sensed"], history_b["dist_sensed"])
        self.assertTrue(any(abs(e) > 0.0 for e in history_a["sensor_error"]))


if __name__ == "__main__":
    unittest.main()
