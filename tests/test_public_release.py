import json
from pathlib import Path
import unittest

import numpy as np

from core.config import ROOT_DIR, load_manifest, resolve_project_path
from core.energy import trajectory_traction_energy
from optimizers.de import run_de
from optimizers.de_nps import run_de_nps
from optimizers.fvrade import run_fvrade
from optimizers.ga import run_ga
from optimizers.nshade import run_nshade
from optimizers.rnegde import run_rnegde


class PublicReleaseContractTests(unittest.TestCase):
    def test_frozen_manifest_and_profile_are_consistent(self):
        manifest = load_manifest()
        self.assertEqual(manifest["benchmark"], "acc_cost_sahib_fje_v2")
        self.assertEqual(manifest["framework_version"], "2.2-FJE")
        self.assertEqual(manifest["controller"]["decision_vector"], ["k_ve", "k_de", "k_vrel"])
        self.assertEqual(len(manifest["optimization"]["algorithms"]), 6)
        self.assertEqual(manifest["optimization"]["independent_runs"], 30)

        profile_path = resolve_project_path(manifest["cost_framework"]["cost_profile_path"])
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        weights = np.asarray(list(profile["weights"].values()), dtype=float)
        self.assertEqual(profile["status"], "frozen")
        self.assertEqual(profile["performance_vector"], manifest["cost_framework"]["performance_vector"])
        self.assertTrue(np.all(weights > 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)

    def test_archived_canonical_evidence_has_no_reported_issues(self):
        status_path = ROOT_DIR / "research_outputs" / "canonical" / "evidence_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status.get("issues"), [])
        self.assertIn("complete", status["RQ4"]["status"])


class EnergyDefinitionTests(unittest.TestCase):
    def test_braking_does_not_reduce_positive_tractive_energy(self):
        e = trajectory_traction_energy([20.0, 20.0], [-3.0, -3.0], 0.1)
        self.assertEqual(e.positive_energy_j, 0.0)
        self.assertGreater(e.braking_energy_j, 0.0)

    def test_constant_speed_has_positive_roadload_energy(self):
        e = trajectory_traction_energy([20.0] * 20, [0.0] * 20, 0.1)
        self.assertGreater(e.positive_energy_j, 0.0)
        self.assertGreater(e.positive_wh_per_km, 0.0)


class OptimizerImportTests(unittest.TestCase):
    def test_all_six_public_optimizers_are_importable(self):
        runners = [run_ga, run_de, run_fvrade, run_rnegde, run_nshade, run_de_nps]
        self.assertTrue(all(callable(runner) for runner in runners))


if __name__ == "__main__":
    unittest.main()
