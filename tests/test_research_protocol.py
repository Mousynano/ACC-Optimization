import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import analysis_pipeline as legacy
import benchmarks.acc_cost_fje_v2 as cost
from core.config import ROOT_DIR, load_manifest
from core.energy import applied_force_energy
from core.physics import VehiclePhysics
from core.research_simulation import ActuatorConfig, force_step_jit, force_step_reference, simulate_cruise, simulate_following
from core.research_statistics import distribution_summary, independent_comparisons, spearman_permutation
from core.scenarios_v2 import load_scenario_suite
from optimizers.de import run_de
from research_pipeline import bounded_recovery_scenario, recovery_events, evaluate_batch
from run_objective_ablation import ablation_weights, build_contract, run_block_job, validate_checkpoint


class ResearchSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=load_manifest()
        cls.suite=load_scenario_suite(development=True,require_frozen=False)

    def test_nominal_history_matches_reference_and_compiled_objective(self):
        gains=[.4,80.,.83]
        for scenario,fast,reference in [(self.suite.cruise[0],simulate_cruise,legacy.simulate_cruise_reference),
                                        (self.suite.following[0],simulate_following,legacy.simulate_following_reference)]:
            actual=fast(gains,scenario,self.manifest); expected=reference(gains,scenario,self.manifest)
            self.assertFalse(actual['_status']['failed'])
            for key in expected:
                np.testing.assert_allclose(actual[key],expected[key],rtol=1e-10,atol=1e-10)
        hist=simulate_cruise(gains,self.suite.cruise[0],self.manifest)
        from core.fast_acc_energy import cruise_metrics_energy_kernel_jit
        s=self.suite.cruise[0]; p=self.manifest['physics'];hz=self.manifest['execution_backend']['sampling_hz']
        result=cruise_metrics_energy_kernel_jit(gains[0],s.duration_s,s.initial_ego_speed_mps,
            s.event_times_s,s.set_speeds_mps,hz,p['road_slope_deg'],p['nominal_mass_kg'],p['air_density_kg_m3'],
            p['drag_coefficient'],p['frontal_area_m2'],p['rolling_resistance'],p['gravity_m_per_s2'],p['max_longitudinal_force_n'])
        self.assertAlmostEqual(np.mean(np.square(hist['speed_error'])),result[0],places=10)
        # Production jerk excludes the artificial first zero sample.
        self.assertAlmostEqual(np.mean(np.square(hist['jerk'][1:])),result[1],places=10)
        self.assertAlmostEqual(applied_force_energy(hist['v_ego'],hist['traction_force_n'],1/hz).positive_wh_per_km,result[2],places=9)

    def test_optional_actuator_reference_jit_and_power_bounds(self):
        physics=np.array([1500.,1.225,.36,2.42,.015,9.81,.1])
        actuator=ActuatorConfig(.2,5000.,7000.,100000.).resolved(6000.)
        previous=0.
        for speed,command in [(25.,3.),(35.,6.),(20.,-5.),(0.,1.)]:
            ref=force_step_reference(speed,command,1/60,physics,actuator,previous)
            jit=force_step_jit(speed,command,1/60,physics,actuator,previous)
            np.testing.assert_allclose(jit,ref,rtol=1e-12,atol=1e-12)
            self.assertGreaterEqual(jit[2],-7000.)
            self.assertLessEqual(jit[2],5000.)
            self.assertLessEqual(max(jit[2]*speed,0),100000.+1e-8)
            previous=jit[2]

    def test_nominal_force_step_matches_existing_plant_with_saturation(self):
        for slope in [-10.,0.,10.]:
            physical=VehiclePhysics(theta=slope)
            vector=np.array([1500.,1.225,.36,2.42,.015,9.81,np.deg2rad(slope)])
            for speed,command in [(0.,-5.),(25.,0.),(25.,5.),(25.,-5.)]:
                actual=force_step_jit(speed,command,1/60,vector,ActuatorConfig().resolved(6000),123.)
                expected=physical.step_acceleration_command(speed,command,1/60)
                np.testing.assert_allclose(actual[:3],expected,rtol=1e-12,atol=1e-12)

    def test_actual_force_energy_and_collision_are_not_hidden(self):
        energy=applied_force_energy([20.,20.],[1000.,-1000.],.1)
        self.assertEqual(energy.positive_energy_j,2000.)
        self.assertEqual(energy.braking_energy_j,2000.)
        scenario=replace(self.suite.following[0],initial_gap_m=-.1)
        hist=simulate_following([1.,1.,1.],scenario,self.manifest)
        self.assertTrue(hist['_status']['failed'])
        self.assertEqual(hist['_status']['reason'],'collision')

    def test_recovery_last_admissible_hold_and_unrecovered_window(self):
        protocol=json.loads((ROOT_DIR/'research_protocol.json').read_text())
        dt=.1;t=np.arange(150)*dt
        relative=np.full(150,1.);relative[130:]=0.
        history={'time':t,'v_rel_true':relative,'spacing_error':np.zeros(150),'d_safe':np.full(150,50.),
                 '_status':{'failed':False}}
        table=recovery_events(history,protocol,bounded=True)
        self.assertTrue(table.iloc[0].recovered)
        self.assertAlmostEqual(table.iloc[0].tte_s,0.)
        relative[149]=1.
        table=recovery_events(history,protocol,bounded=True)
        self.assertFalse(table.iloc[0].recovered)
        self.assertTrue(np.isnan(table.iloc[0].tte_s))


class ResearchAblationTests(unittest.TestCase):
    def test_mask_preserves_frozen_ratios_and_infeasibility(self):
        frozen=np.array([.4,.4,.15,.05])
        w=ablation_weights(frozen,'without_jerk')
        self.assertEqual(w[2],0.)
        self.assertAlmostEqual(w[0]/w[3],frozen[0]/frozen[3])
        perf=cost.PerformanceVectorFJE(1.,1.,1.,1.,False,2.)
        self.assertGreater(cost.scalar_cost_from_performance(perf,w),1e12)
        for profile in ['full','without_jerk','without_energy','tracking_only']:
            self.assertAlmostEqual(ablation_weights(frozen,profile).sum(),1.)

    def test_production_still_rejects_zero_and_nonfinite_weights(self):
        payload=json.loads(cost.load_cost_profile().path.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'profile.json'
            for value in [0.,float('nan')]:
                modified=copy.deepcopy(payload);modified['weights'][cost.PERFORMANCE_NAMES[0]]=value
                path.write_text(json.dumps(modified))
                with patch.object(cost,'resolve_project_path',return_value=path):
                    with self.assertRaises(RuntimeError): cost.load_cost_profile()

    def test_explicit_population_is_reproducible_and_validated(self):
        population=np.random.default_rng(5).uniform(-1,1,(8,3))
        kwargs=dict(func=lambda x:float(np.dot(x,x)),min_params=[-1]*3,max_params=[1]*3,
                    population_size=8,max_iter=2,seed=8,initial_population=population)
        first=run_de(**kwargs);second=run_de(**kwargs)
        np.testing.assert_array_equal(first[0],second[0]);self.assertEqual(first[1:],second[1:])
        np.testing.assert_array_equal(population,np.random.default_rng(5).uniform(-1,1,(8,3)))
        with self.assertRaises(ValueError): run_de(**{**kwargs,'initial_population':np.ones((7,3))})

    def test_campaign_population_pairing_and_tampered_resume(self):
        contract=build_contract(smoke=True)
        full=run_block_job(contract,'full',0);alternate=run_block_job(contract,'without_jerk',0)
        self.assertEqual(full['initial_population_sha256'],alternate['initial_population_sha256'])
        self.assertEqual(full['optimizer_seed'],alternate['optimizer_seed'])
        self.assertEqual(full['function_evaluations'],24)
        self.assertFalse(full['research'])
        validate_checkpoint(full,contract,'full',0)
        changed={**full,'initial_population_sha256':'wrong'}
        with self.assertRaises(RuntimeError): validate_checkpoint(changed,contract,'full',0)


class ResearchAnalysisTests(unittest.TestCase):
    def test_counts_include_missing_infinite_and_failed_controllers(self):
        frame=pd.DataFrame({'algorithm':['a']*4,'value':[1.,2.,np.nan,np.inf],'failed':[False,False,True,False]})
        row=distribution_summary(frame,['algorithm'],['value'],100).iloc[0]
        self.assertEqual(row.n_total,4);self.assertEqual(row.n_finite,2)
        self.assertEqual(row.n_positive_infinite,1);self.assertEqual(row.failed_controllers,1)

    def test_independent_tests_do_not_invent_run_pairing(self):
        frame=pd.DataFrame({'algorithm':['a']*4+['b']*3,'metric':[1.,2.,3.,4.,2.,3.,5.]})
        omni,pairs=independent_comparisons(frame,'algorithm',['metric'],100)
        self.assertEqual(omni.iloc[0].pairing,'none')
        self.assertEqual(pairs.iloc[0].n_a,4);self.assertEqual(pairs.iloc[0].n_b,3)

    def test_small_sample_correlation_and_constant_input(self):
        n,rho,p=spearman_permutation(np.arange(8),np.arange(8),5,199)
        self.assertEqual(n,8);self.assertAlmostEqual(rho,1.);self.assertLess(p,.05)
        self.assertTrue(np.isnan(spearman_permutation([1]*8,np.arange(8))[1]))

    def test_batch_resume_preserves_controller_identity(self):
        protocol=json.loads((ROOT_DIR/'research_protocol.json').read_text())
        manifest=load_manifest();suite=load_scenario_suite(development=True,require_frozen=False)
        scenario=replace(suite.cruise[0],duration_s=1.)
        controllers=pd.DataFrame([{'controller_id':'test:de:0','algorithm':'de','run_id':0,'seed':5,
                                  'k_ve':.4,'k_de':80.,'k_vrel':.83}])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            first,_,_=evaluate_batch(controllers,[scenario],manifest,protocol,out,'test-contract')
            with patch('research_pipeline.evaluate_record',side_effect=AssertionError('cached record must be reused')):
                second,_,_=evaluate_batch(controllers,[scenario],manifest,protocol,out,'test-contract')
            pd.testing.assert_frame_equal(first,second)

    def test_equal_trajectory_bytes_do_not_merge_named_profiles(self):
        protocol=json.loads((ROOT_DIR/'research_protocol.json').read_text())
        manifest=load_manifest();suite=load_scenario_suite(development=True,require_frozen=False)
        scenario=replace(suite.cruise[0],duration_s=1.,scenario_id='first')
        alternate=replace(scenario,scenario_id='second')
        controllers=pd.DataFrame([{'controller_id':'test:de:0','algorithm':'de','run_id':0,'seed':5,
                                  'k_ve':.4,'k_de':80.,'k_vrel':.83}])
        with tempfile.TemporaryDirectory() as tmp:
            frame,_,_=evaluate_batch(controllers,[scenario,alternate],manifest,protocol,Path(tmp),'test-contract')
            self.assertEqual(frame.scenario_id.tolist(),['first','second'])
            self.assertFalse(frame.duplicated(['controller_id','scenario_id','condition_id']).any())


if __name__=='__main__':
    unittest.main()
