import tempfile
from pathlib import Path
import unittest

import numpy as np

from core.data_sources_v2 import (
    detect_openacc_leader_speed_column,
    read_openacc_csv,
    select_excitation_window,
    write_processed_window,
)


class OpenACCPreprocessingTests(unittest.TestCase):
    def _write_mock(self, path: Path):
        lines = [
            "Date,04,07,2019\n",
            "Vehicle_order,leader,follower\n",
            "Number_of_vehicles,2\n",
            "ACC,1\n",
            "Distance_setting,min\n",
            "Time,Speed1,Speed2,IVS1\n",
        ]
        # 10 Hz, 120 s; the last half is deliberately more excited.
        for i in range(1201):
            t = i / 10.0
            amp = 0.2 if t < 60 else 3.0
            v1 = 20.0 + amp * np.sin(0.25 * t)
            v2 = 19.5
            lines.append(f"{t:.1f},{v1:.8f},{v2:.8f},40.0\n")
        path.write_text("".join(lines), encoding="utf-8")

    def test_openacc_metadata_header_and_speed_detection(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mock.csv"
            self._write_mock(p)
            df = read_openacc_csv(p)
            self.assertIn("Time", df.columns)
            self.assertEqual(detect_openacc_leader_speed_column(df), "Speed1")

    def test_excitation_rule_is_deterministic(self):
        t = np.arange(0.0, 121.0, 0.1)
        v = 20 + np.where(t < 60, 0.2, 3.0) * np.sin(0.25 * t)
        a = select_excitation_window(t, v, window_s=40.0, stride_s=5.0)
        b = select_excitation_window(t, v, window_s=40.0, stride_s=5.0)
        self.assertEqual(a, b)
        self.assertGreaterEqual(a[0], 55.0)

    def test_processed_window_is_simple_two_column_csv(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "raw.csv"
            out = td / "processed.csv"
            self._write_mock(raw)
            result = write_processed_window(
                source_key="mock",
                raw_path=raw,
                processed_path=out,
                source_url="https://example.invalid/mock.csv",
                source_name="mock",
                selection="max_leader_kinematic_excitation",
                window_s=40.0,
            )
            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(text, "time_s,lead_speed_mps")
            self.assertGreater(result.duration_s, 39.0)
            self.assertEqual(len(result.processed_sha256), 64)

    def test_casale_fixed_window_is_relative_to_part_start(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "part6.csv"
            out = td / "processed.csv"
            lines = [
                "Date,27,10,2020\n",
                "Vehicle_order,leader,follower\n",
                "Number_of_vehicles,2\n",
                "ACC,1\n",
                "Distance_setting,min\n",
                "Time,Speed1,Speed2,IVS1\n",
            ]
            # Casale parts share the experiment time base. Simulate part6 starting
            # at t=1800s rather than resetting to zero.
            for i in range(1401):
                t = 1800.0 + i / 10.0
                lines.append(f"{t:.1f},20.0,19.5,40.0\n")
            raw.write_text("".join(lines), encoding="utf-8")
            result = write_processed_window(
                source_key="casale_validation",
                raw_path=raw,
                processed_path=out,
                source_url="https://example.invalid/part6.csv",
                source_name="mock Casale",
                selection="fixed_offset_50_120_s",
                window_s=70.0,
            )
            self.assertAlmostEqual(result.start_s, 1850.0)
            self.assertAlmostEqual(result.end_s, 1920.0)
            arr = np.genfromtxt(out, delimiter=",", names=True)
            self.assertAlmostEqual(float(arr["time_s"][0]), 0.0)
            self.assertGreater(float(arr["time_s"][-1]), 69.9)


if __name__ == "__main__":
    unittest.main()
