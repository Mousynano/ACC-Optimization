import unittest

from core.campaign_eta import (
    RuntimeModel,
    blend_runtime,
    format_duration,
    schedule_makespan_seconds,
)

HOUR = 3600.0


class CampaignEtaTests(unittest.TestCase):
    def test_history_live_blend(self):
        self.assertAlmostEqual(
            blend_runtime(6 * HOUR, 5 * HOUR, 0.70) / HOUR,
            5.7,
            places=8,
        )

    def test_exact_algorithm_objective_history_is_used(self):
        model = RuntimeModel()
        key = ("acc", "denps", "itse")
        model.extend_completed(key, [5.5 * HOUR, 6.5 * HOUR])
        live = {key: [5.0 * HOUR]}
        self.assertAlmostEqual(model.predict_total_seconds(key, live) / HOUR, 5.7, places=8)

    def test_serial_five_job_example(self):
        model = RuntimeModel()
        key = ("acc", "denps", "itse")
        model.extend_completed(key, [5.5 * HOUR, 6.5 * HOUR])
        predicted = model.predict_total_seconds(key, {key: [5.0 * HOUR]})
        eta = schedule_makespan_seconds([], [predicted] * 5, concurrency=1)
        self.assertAlmostEqual(eta / HOUR, 28.5, places=8)

    def test_parallel_makespan_not_serial_sum(self):
        eta = schedule_makespan_seconds([], [5, 6.5, 5, 6.5, 5], concurrency=2)
        self.assertAlmostEqual(eta, 15.0, places=8)

    def test_active_job_remaining_is_included(self):
        eta = schedule_makespan_seconds([2 * HOUR, 3 * HOUR], [5 * HOUR], concurrency=2)
        self.assertAlmostEqual(eta / HOUR, 7.0, places=8)

    def test_duration_format_is_single_value(self):
        self.assertEqual(format_duration(29.3 * HOUR), "1d 05h 18m")
        self.assertEqual(format_duration(42.0), "42s")
        self.assertEqual(format_duration(754.0), "12m 34s")


if __name__ == "__main__":
    unittest.main()
