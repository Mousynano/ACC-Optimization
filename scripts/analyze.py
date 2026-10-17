"""RQ1--RQ4 evidence pipeline over the existing frozen FJE campaign.

Default: all 180 controllers, nominal RQ2 profiles and the 12 physical RQ3
conditions. --all additionally runs/resumes the isolated 120-job RQ4 campaign.
This entry point never downloads data, recalibrates weights, or retunes RQ1.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from core import analysis
from benchmarks.acc_cost_fje_v2 import PERFORMANCE_NAMES, load_cost_profile
from core.config import ROOT_DIR, load_manifest
from core.scenarios_v2 import FollowingScenario, load_scenario_suite, load_validation_scenarios, sha256_file
from core.research_simulation import ActuatorConfig, simulate_cruise, simulate_following
from core.research_statistics import distribution_summary, holm_adjust, median_ci, spearman_permutation, stable_seed
from core.utils import compute_ttc_series, step_response_metrics
from scripts.ablation import atomic_json, build_contract, validate_checkpoint

METRICS = ["cruise_speed_rmse_mps","spacing_rmse_m","speed_rmse_mps","jerk_rms_mps3",
    "positive_tractive_energy_wh_per_km","ttc_min_s","ttc_below_threshold_fraction",
    "min_spacing_m","recovery_fraction","tte_recovered_median_s","recovery_window_s"]


def _safe_json(value):
    if isinstance(value,dict): return {str(k):_safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_safe_json(v) for v in value]
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.integer,)): return int(value)
    return value


def bounded_recovery_scenario(manifest, protocol):
    r = protocol["recovery"]; hz = manifest["execution_backend"]["sampling_hz"]
    t = np.arange(int(round(r["total_duration_s"]*hz)))/hz
    speed = r["initial_equilibrium_speed_mps"]
    lead = speed+r["lead_acceleration_mps2"]*np.clip(t+1/hz-r["disturbance_start_s"],0,r["disturbance_duration_s"])
    gap = manifest["controller"]["standstill_distance_m"]+manifest["controller"]["time_gap_s"]*speed
    return FollowingScenario("SYN_BOUNDED_RECOVERY",r["total_duration_s"],speed,speed,gap,lead,
        provenance="Study-defined finite braking disturbance; initial CTG equilibrium, followed by constant leader speed")


def recovery_events(history, protocol, bounded=False):
    time = np.asarray(history["time"],float)
    if len(time)<3: return pd.DataFrame()
    dt = float(np.median(np.diff(time))); r = protocol["recovery"]
    planned_end=r["disturbance_start_s"]+r["disturbance_duration_s"]
    if bounded and time[-1]+dt <= planned_end:
        return pd.DataFrame([{"event_id":0,"disturbance_start_s":r["disturbance_start_s"],
            "disturbance_end_s":planned_end,"observation_window_s":0.,"hold_window_s":r["hold_window_s"],
            "recovered":False,"tte_s":np.nan,"hold_completion_delay_s":np.nan,
            "nonrecovery_reason":"simulation_failed_before_disturbance_end" if history.get("_status",{}).get("failed")
                else "observation_ends_before_disturbance_end"}])
    hold = max(1,int(round(r["hold_window_s"]/dt)))
    if bounded:
        start = int(round(r["disturbance_start_s"]/dt))
        end = int(round((r["disturbance_start_s"]+r["disturbance_duration_s"])/dt))
        events = [(min(start,len(time)-1),min(end,len(time)-1))]
    else:
        events = analysis.detect_disturbances(time,history["v_lead"])
    relative = np.abs(np.asarray(history["v_rel_true"]))
    spacing = np.abs(np.asarray(history["spacing_error"]))
    band = np.maximum(r["spacing_absolute_tolerance_m"],r["spacing_relative_tolerance_fraction"]*np.abs(history["d_safe"]))
    within = np.isfinite(relative)&np.isfinite(spacing)&(relative<=r["relative_speed_tolerance_mps"])&(spacing<=band)
    rows=[]
    for index,(start,end) in enumerate(events):
        next_start = events[index+1][0] if index+1 < len(events) else len(time)
        first = end if bounded else end+1
        last = next_start-hold
        found = None
        if first <= last:
            # Rolling hold test, including the last admissible start sample.
            sums = np.convolve(within[first:next_start].astype(int),np.ones(hold,dtype=int),mode="valid")
            candidates = np.flatnonzero(sums==hold)
            if len(candidates): found = first+int(candidates[0])
        end_time = float(time[end]); horizon = float(time[next_start-1]+dt-end_time)
        failed = history.get("_status",{}).get("failed",False)
        rows.append({"event_id":index,"disturbance_start_s":float(time[start]),"disturbance_end_s":end_time,
            "observation_window_s":horizon,"hold_window_s":r["hold_window_s"],
            "recovered":found is not None and not failed,
            "tte_s":float(time[found]-end_time) if found is not None and not failed else np.nan,
            "hold_completion_delay_s":float(time[found]-end_time+r["hold_window_s"]) if found is not None and not failed else np.nan,
            "nonrecovery_reason":"" if found is not None and not failed else
                ("simulation_failed" if failed else "window_shorter_than_hold" if first>last else "no_qualifying_hold_within_window")})
    return pd.DataFrame(rows)


def transition_metrics(history, scenario):
    time=np.asarray(history["time"]); speed=np.asarray(history["v_ego"]); rows=[]
    for index in range(1,len(scenario.event_times_s)):
        start=scenario.event_times_s[index]
        end=scenario.event_times_s[index+1] if index+1<len(scenario.event_times_s) else scenario.duration_s
        mask=(time>=start)&(time<end)
        if mask.sum()<3: continue
        result=step_response_metrics(time[mask]-start,speed[mask],scenario.set_speeds_mps[index],y0=speed[mask][0])
        rows.append({"transition_index":index,"direction":"up" if scenario.set_speeds_mps[index]>scenario.set_speeds_mps[index-1] else "down",
            "start_s":float(start),"end_s":float(end),**result})
    return rows


def evaluate_record(row, scenario, manifest, protocol, condition_id="S0", override=None, actuator=None,
                    sensor_override=None, sensor_processing=None, compensation=None):
    gains=np.asarray([row[n] for n in analysis.GAIN_NAMES],float)
    following=isinstance(scenario,FollowingScenario)
    history=(simulate_following(gains,scenario,manifest,override,sensor_override,actuator,
                               sensor_processing,compensation) if following
             else simulate_cruise(gains,scenario,manifest,override,actuator=actuator,compensation=compensation))
    status=history["_status"]; failed=status["failed"]
    record={"controller_id":row["controller_id"],"algorithm":row["algorithm"],"run_id":int(row["run_id"]),
        "optimizer_seed":_safe_json(row.get("seed")),"sensor_seed":sensor_override.seed if sensor_override else manifest["sensing"]["nominal_noise_seed"],
        "scenario_id":scenario.scenario_id,"condition_id":condition_id,"failed":bool(failed),"failure_reason":status["reason"],
        "steps_completed":status["steps_completed"],"expected_steps":status["expected_steps"],"model":status["model"],
        "role":"held_out" if scenario.scenario_id in ("V1_CASALE","V2_WLTC") else "regression" if scenario.scenario_id.startswith("V") else "synthetic_diagnostic",
        "no_closing_samples":False}
    if not failed:
        record.update(analysis.following_metrics(history,manifest,override) if following else analysis.cruise_metrics(history,manifest,override))
        if following:
            ttc=compute_ttc_series(history["dist"],history["v_ego"],history["v_lead"])
            record["no_closing_samples"]=not np.isfinite(ttc).any()
            threshold=protocol["evaluation"]["ttc_reporting_threshold_s"]
            record["ttc_below_threshold_fraction"]=float(np.mean(ttc<threshold))
            finite=ttc[np.isfinite(ttc)]
            record["ttc_closing_below_threshold_fraction"]=float(np.mean(finite<threshold)) if len(finite) else 0.0
            record["ttc_threshold_crossed"]=bool(np.any(ttc<protocol["evaluation"]["ttc_reporting_threshold_s"]))
            record["ttc_below_threshold_duration_s"]=float(np.sum(ttc<protocol["evaluation"]["ttc_reporting_threshold_s"])/manifest["execution_backend"]["sampling_hz"])
    else:
        record.update({name:np.nan for name in METRICS})
        record["ttc_threshold_crossed"]=bool(status["reason"]=="collision")
        if following and len(history["dist"]): record["min_spacing_m"]=float(np.nanmin(history["dist"]))
    if "estimated_road_load_force_n" in history:
        diagnostics={"force_saturation_fraction":float(np.mean(np.abs(history["requested_force_n"]-history["target_force_n"])>1e-8)),
            "actuator_tracking_rmse_n":float(np.sqrt(np.mean((history["traction_force_n"]-history["target_force_n"])**2))),
            "compensation_road_load_rmse_n":float(np.sqrt(np.mean((history["estimated_road_load_force_n"]-history["road_load_force_n"])**2)))}
        if following:
            diagnostics.update(raw_range_error_rmse_m=float(np.sqrt(np.mean((history["dist_raw"]-history["dist"])**2))),
                processed_range_error_rmse_m=float(np.sqrt(np.mean((history["dist_sensed"]-history["dist"])**2))),
                range_rate_error_rmse_mps=float(np.sqrt(np.mean((history["v_rel_est"]-history["v_rel_true"])**2))),
                sensor_update_fraction=float(np.mean(history["sensor_updated"])),
                measurement_age_max_s=float(np.max(history["measurement_age_s"])))
        record.update({k:np.nan if failed else v for k,v in diagnostics.items()})
    events=[]
    if following and scenario.scenario_id in ("V1_CASALE","SYN_BOUNDED_RECOVERY"):
        table=recovery_events(history,protocol,bounded=scenario.scenario_id=="SYN_BOUNDED_RECOVERY")
        if not table.empty:
            events=table.to_dict("records")
            record["disturbances"]=len(table); record["recovered_events"]=int(table.recovered.sum())
            record["nonrecovered_events"]=int((~table.recovered).sum())
            record["recovery_fraction"]=float(table.recovered.mean())
            record["tte_recovered_median_s"]=float(table.tte_s.median()) if table.tte_s.notna().any() else np.nan
            record["recovery_window_s"]=float(table.observation_window_s.min())
    return record,events,transition_metrics(history,scenario) if not following and not failed else [],history


def _scenario_hash(scenario):
    if isinstance(scenario,FollowingScenario):
        vector=np.asarray(scenario.lead_speed_mps,dtype="<f8").tobytes()
        initial=[scenario.initial_ego_speed_mps,scenario.initial_lead_speed_mps,scenario.initial_gap_m]
    else:
        vector=np.asarray([scenario.event_times_s,scenario.set_speeds_mps],dtype="<f8").tobytes()
        initial=[scenario.initial_ego_speed_mps]
    return hashlib.sha256(vector+json.dumps(initial).encode()).hexdigest()


def evaluate_batch(controllers, scenarios, manifest, protocol, out, contract_hash, conditions=None, telemetry_ids=()):
    cache=out/"cache"; cache.mkdir(parents=True,exist_ok=True)
    conditions=conditions or [("S0",None,None)]
    records,events,transitions=[],[],[]
    total=len(controllers)*len(scenarios)*len(conditions); completed=0
    for condition_id,override,actuator in conditions:
        for scenario in scenarios:
            scenario_hash=_scenario_hash(scenario)
            for _,row in controllers.iterrows():
                key={"contract":contract_hash,"controller":row.controller_id,"gains":[row[n] for n in analysis.GAIN_NAMES],
                    "scenario_id":scenario.scenario_id,"scenario":scenario_hash,"condition":condition_id,"physics":asdict(override) if override else None,
                    "actuator":asdict(actuator) if actuator else None}
                digest=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest(); path=cache/(digest+".json")
                save_trace=row.controller_id in telemetry_ids and condition_id=="S0"
                trace=out/"telemetry"/(digest+".npz")
                if path.exists() and (not save_trace or trace.exists()):
                    payload=json.loads(path.read_text())
                    if payload["key"] != key: raise RuntimeError("Evaluation cache identity mismatch")
                    rec,ev,tr=payload["record"],payload["events"],payload["transitions"]
                else:
                    rec,ev,tr,history=evaluate_record(row,scenario,manifest,protocol,condition_id,override,actuator)
                    atomic_json(path,_safe_json({"key":key,"record":rec,"events":ev,"transitions":tr}))
                    if save_trace:
                        trace.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(trace,**{k:v for k,v in history.items() if not k.startswith("_")})
                if rec.get("no_closing_samples") and not rec["failed"]: rec["ttc_min_s"]=np.inf
                records.append(rec)
                identity={k:rec[k] for k in ("controller_id","algorithm","run_id","scenario_id","condition_id")}
                events.extend({**identity,**r} for r in ev); transitions.extend({**identity,**r} for r in tr)
                completed+=1
            print(f"Evaluated {completed}/{total}: {condition_id} / {scenario.scenario_id}",flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(out/"controller_metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"recovery_events.csv",index=False)
    pd.DataFrame(transitions).to_csv(out/"cruise_transitions.csv",index=False)
    return frame,pd.DataFrame(events),pd.DataFrame(transitions)


def metric_plots(frame, group_column, out, prefix):
    for scenario,group in frame.groupby("scenario_id",sort=False):
        for metric in ["cruise_speed_rmse_mps","spacing_rmse_m","jerk_rms_mps3","positive_tractive_energy_wh_per_km"]:
            if metric not in group: continue
            arrays=[];labels=[]
            for label,g in group.groupby(group_column,sort=False):
                vals=pd.to_numeric(g[metric],errors="coerce").to_numpy(float); vals=vals[np.isfinite(vals)]
                if len(vals): arrays.append(vals); labels.append(label)
            if not arrays: continue
            fig,ax=plt.subplots(figsize=(10,5))
            ax.boxplot(arrays,tick_labels=labels,showfliers=True)
            ax.set_title(f"{scenario}: distribution across controllers"); ax.set_ylabel(metric)
            ax.tick_params(axis="x",labelrotation=15); ax.grid(axis="y",alpha=.2)
            fig.tight_layout(); fig.savefig(out/f"{prefix}_{scenario}_{metric}.png",dpi=220); plt.close(fig)


def correlation_outputs(components, nominal, protocol, out):
    training=components[["algorithm","run_id","fitness",*PERFORMANCE_NAMES]]
    held=nominal[nominal.role=="held_out"]
    joined=held.merge(training,on=["algorithm","run_id"],validate="many_to_one")
    rows=[]
    for (algorithm,scenario),group in joined.groupby(["algorithm","scenario_id"],sort=False):
        current=[]
        for source in ["fitness",*PERFORMANCE_NAMES]:
            for outcome in ["spacing_rmse_m","jerk_rms_mps3","positive_tractive_energy_wh_per_km","ttc_min_s"]:
                # Training and held-out energy share a name; merge adds suffixes.
                src=source+"_y" if source=="positive_tractive_energy_wh_per_km" else source
                dst=outcome+"_x" if outcome=="positive_tractive_energy_wh_per_km" else outcome
                n,rho,p=spearman_permutation(group[src],group[dst],stable_seed(f"{algorithm}:{scenario}:{source}:{outcome}"),
                    protocol["evaluation"]["correlation_permutations"])
                current.append({"algorithm":algorithm,"scenario_id":scenario,"training_metric":source,
                    "heldout_metric":outcome,"n_controllers":n,"spearman_rho":rho,"p_permutation":p,
                    "interpretation":"exploratory association; not causal attribution"})
        adjusted=holm_adjust([r["p_permutation"] for r in current])
        rows.extend({**r,"p_holm":p} for r,p in zip(current,adjusted))
    pd.DataFrame(rows).to_csv(out/"training_heldout_spearman.csv",index=False)


def robustness_relative(frame):
    metrics=[m for m in METRICS if m in frame]
    nominal=frame[frame.condition_id=="S0"][["controller_id","scenario_id",*metrics]].rename(columns={m:m+"_nominal" for m in metrics})
    merged=frame.merge(nominal,on=["controller_id","scenario_id"],validate="many_to_one")
    rows=[]
    for _,row in merged.iterrows():
        for metric in metrics:
            value,baseline=row[metric],row[metric+"_nominal"]
            finite=pd.notna(value) and pd.notna(baseline) and np.isfinite(value) and np.isfinite(baseline)
            ratio=value/baseline if finite and abs(baseline)>1e-12 else np.nan
            rows.append({"controller_id":row.controller_id,"algorithm":row.algorithm,"run_id":row.run_id,
                "scenario_id":row.scenario_id,"condition_id":row.condition_id,"metric":metric,"value":value,"nominal":baseline,
                "delta":value-baseline if finite else np.nan,"ratio":ratio,"degradation_pct":100*(ratio-1) if np.isfinite(ratio) else np.nan,
                "failed":row.failed})
    return pd.DataFrame(rows)


def _ablation_contract_compatible(existing, current):
    """Accept completed RQ4 campaigns whose scientific contract is unchanged.

    Scope-only metadata (notably research_protocol.json hash) may differ after the
    2026-10-09 omission cleanup. Critical optimizer/simulator sources and every
    numerical design field must still match exactly.
    """
    keys = [
        "research", "optimizer", "runs", "population_size", "iterations",
        "base_seed", "mutation_factor", "crossover_rate", "profiles",
        "min_bounds", "max_bounds", "frozen_weights", "weights",
        "scenario_manifest_sha256", "cost_profile_sha256", "physics",
        "controller", "sensing", "sampling_hz", "pairing", "weight_policy",
        "model",
    ]
    if any(existing.get(k) != current.get(k) for k in keys):
        return False
    critical_sources = set(current.get("source_sha256", {})) - {"research_protocol.json"}
    return all(existing.get("source_sha256", {}).get(name) == current["source_sha256"].get(name)
               for name in critical_sources)


def find_compatible_ablation_campaign():
    current = build_contract(False)
    root = ROOT_DIR / "research_ablation"
    candidates = sorted(root.glob("research_*"), key=lambda x: x.stat().st_mtime, reverse=True) if root.exists() else []
    for campaign in candidates:
        completion = campaign / "completion.json"
        manifest = campaign / "campaign.json"
        if not completion.exists() or not manifest.exists():
            continue
        try:
            done = json.loads(completion.read_text())
            contract = json.loads(manifest.read_text())
        except Exception:
            continue
        if done.get("complete") and done.get("jobs") == contract.get("runs", 0) * len(contract.get("profiles", [])) \
                and _ablation_contract_compatible(contract, current):
            return campaign
    return None


def load_ablation(campaign):
    contract=json.loads((campaign/"campaign.json").read_text())
    if not contract.get("research"): raise RuntimeError("Smoke ablation cannot enter research evidence")
    current=build_contract(False)
    if not _ablation_contract_compatible(contract,current):
        raise RuntimeError("Ablation scientific contract does not match current frozen inputs/code")
    rows=[]
    for profile in contract["profiles"]:
        for run in range(contract["runs"]):
            path=campaign/f"{profile}_run_{run:02d}.json"
            if not path.exists(): raise RuntimeError(f"Incomplete RQ4 campaign: {path.name}")
            result=json.loads(path.read_text()); validate_checkpoint(result,contract,profile,run)
            rows.append({"controller_id":result["controller_id"],"algorithm":"de","run_id":run,
                "objective_profile":profile,"seed":result["optimizer_seed"],"initial_population_sha256":result["initial_population_sha256"],
                "fitness_within_profile":result["best_fitness"],"full_objective_score":result["full_objective_score"],
        "runtime_s":result["duration"],"function_evaluations":result["function_evaluations"],
                "failed":not result["feasible"],**result["performance"],
                **dict(zip(analysis.GAIN_NAMES,result["best_solution"]))})
    frame=pd.DataFrame(rows)
    frame.loc[frame.failed, list(PERFORMANCE_NAMES)] = np.nan
    if (frame.groupby("run_id").initial_population_sha256.nunique()!=1).any():
        raise RuntimeError("RQ4 pairing failed: initial populations differ within a run block")
    return contract,frame


def ablation_comparisons(frame, metrics, out):
    rows=[]; interactions=[]
    scopes=[("calibration",frame)] if "scenario_id" not in frame else list(frame.groupby("scenario_id",sort=False))
    for scenario,group in scopes:
        for metric in metrics:
            if metric not in group: continue
            pivot=group.pivot(index="run_id",columns="objective_profile",values=metric)
            for alternate in ["without_jerk","without_energy","tracking_only"]:
                pair=pivot[["full",alternate]].apply(pd.to_numeric,errors="coerce")
                pair=pair[np.isfinite(pair).all(axis=1)]
                diff=(pair[alternate]-pair["full"]).to_numpy(float)
                if not len(diff): continue
                statistic,p=stats.wilcoxon(diff,zero_method="wilcox") if np.any(diff!=0) else (0.,1.)
                lo,hi=median_ci(diff,stable_seed(f"{scenario}:{metric}:{alternate}"))
                rows.append({"scenario_id":scenario,"metric":metric,"comparison":alternate+" minus full",
                    "n_total_blocks":len(pivot),"n_finite_pairs":len(diff),"paired_median_difference":np.median(diff),
                    "difference_ci_low":lo,"difference_ci_high":hi,"W":statistic,"p_raw":p,
                    "rank_biserial_alternate_minus_full":analysis._rank_biserial_paired(pair[alternate],pair["full"]),
                    "pairing":"verified identical initial population and optimizer seed"})
            complete=pivot.apply(pd.to_numeric,errors="coerce")
            complete=complete[np.isfinite(complete).all(axis=1)]
            if len(complete):
                interaction=(complete["full"]-complete["without_jerk"]-complete["without_energy"]+complete["tracking_only"]).to_numpy(float)
                lo,hi=median_ci(interaction,stable_seed(f"interaction:{scenario}:{metric}"))
                interactions.append({"scenario_id":scenario,"metric":metric,"n_complete_blocks":len(complete),
                    "interaction_median":np.median(interaction),"ci_low":lo,"ci_high":hi,"interpretation":"exploratory 2x2 objective interaction"})
    result=pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"]=result.groupby(["scenario_id","metric"],sort=False).p_raw.transform(lambda p:holm_adjust(p.to_numpy()))
    result.to_csv(out/"paired_ablation_comparisons.csv",index=False)
    pd.DataFrame(interactions).to_csv(out/"exploratory_objective_interactions.csv",index=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT_DIR/"research_outputs")
    parser.add_argument("--all",action="store_true",help="run/resume RQ4 and generate all four RQ evidence sets")
    parser.add_argument("--ablation-dir",type=Path,help="specific completed research ablation campaign")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--allow-incomplete",action="store_true",help="label partial checkpoint coverage; not a completed study")
    parser.add_argument("--actuator-config",type=Path,help="optional separate reporting scenarios; never changes tuning")
    args=parser.parse_args(argv)
    manifest=load_manifest(); protocol=json.loads((ROOT_DIR/"research_protocol.json").read_text())
    if manifest["benchmark"]!="acc_cost_sahib_fje_v2": raise RuntimeError("RQ pipeline requires the frozen FJE benchmark")
    if manifest["sensing"]["nominal_noise_std_m"]!=0 or manifest["sensing"]["nominal_bias_m"]!=0:
        raise RuntimeError("Frozen nominal training manifest must stay noise/bias free; sensor uncertainty is outside the canonical scope")
    suite=load_scenario_suite(manifest["cost_framework"]["scenario_manifest_path"],require_frozen=True)
    validations=load_validation_scenarios(manifest["cost_framework"]["scenario_manifest_path"],require_frozen=True)
    profile=load_cost_profile(require_frozen=True)
    if profile.scenario_manifest_sha256!=suite.manifest_sha256: raise RuntimeError("Frozen profile/scenario mismatch")
    runs=analysis.load_run_frame(manifest)
    expected=manifest["optimization"]["independent_runs"]
    if runs.empty or runs.duplicated(["algorithm","run_id"]).any(): raise RuntimeError("Missing or duplicate controller identity")
    coverage={a:len(runs[runs.algorithm==a]) for a in manifest["optimization"]["algorithms"]}
    complete=all(n==expected for n in coverage.values())
    if not complete and not args.allow_incomplete: raise RuntimeError(f"Incomplete campaign: {coverage}")
    if any(set(runs.loc[runs.algorithm==a,"run_id"])!=set(range(expected)) for a in coverage) and not args.allow_incomplete:
        raise RuntimeError("Run IDs do not match the expected campaign")
    for _,row in runs.iterrows():
        if len(row.solution)!=3 or not np.isfinite(row.solution).all() or not np.isfinite(row.fitness):
            raise RuntimeError(f"Invalid training checkpoint: {row.controller_id}")
    checkpoint_root=Path(analysis.load_all_job_checkpoints.__globals__["CHECKPOINT_DIR"])
    source_files=["scripts/analyze.py","core/analysis.py","core/research_simulation.py","core/research_statistics.py",
        "core/energy.py","core/physics.py","core/fast_acc.py","core/fast_acc_energy.py","benchmarks/acc_cost_fje_v2.py"]
    contract={"protocol":protocol,"training_manifest":manifest,"scenario_sha256":suite.manifest_sha256,
        "cost_profile_sha256":sha256_file(profile.path),"coverage":coverage,"complete_training_coverage":complete,
        "source_sha256":{name:sha256_file(ROOT_DIR/name) for name in source_files},
        "checkpoint_sha256":{p.name:sha256_file(p) for p in sorted(checkpoint_root.glob(manifest["benchmark"]+"_run_*.pkl"))},
        "actuator_config_sha256":sha256_file(args.actuator_config) if args.actuator_config else None}
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    out=args.out_dir.resolve()/digest[:12];out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"evaluation_contract.json",contract)
    dirs={name:out/name for name in ["RQ1_optimizer","RQ2_behavior","RQ3_physical","RQ4_objective"]}
    for directory in dirs.values(): directory.mkdir(exist_ok=True)
    reps=analysis.select_observed_median_representatives(runs)
    telemetry_ids={f"{manifest['benchmark']}:{r.algorithm}:{int(r.run_id)}" for _,r in reps.iterrows()}
    print("RQ1: all-run optimization evidence",flush=True)
    components=analysis.evaluate_all_components(runs,suite,profile)
    if components.cost_reconstruction_abs_error.max()>1e-8:
        raise RuntimeError("Training checkpoint scores do not match frozen inputs; refusing mixed-model evaluation")
    analysis.objective_outputs(runs,components,reps,manifest,profile,dirs["RQ1_optimizer"])
    analysis.convergence_outputs(runs,manifest,dirs["RQ1_optimizer"])
    (dirs["RQ1_optimizer"]/"qq").mkdir(exist_ok=True)
    analysis.statistical_outputs(runs,components,manifest,dirs["RQ1_optimizer"])
    distribution_summary(runs,["algorithm"],["fitness","runtime_s","function_evaluations"]).to_csv(dirs["RQ1_optimizer"]/"quality_cost_confidence.csv",index=False)
    periodic=analysis.build_periodic_cruise()
    scenarios=[periodic,analysis.build_sinusoidal_following(manifest["execution_backend"]["sampling_hz"]),
        bounded_recovery_scenario(manifest,protocol),*validations]
    print("RQ2: all-controller nominal evaluation",flush=True)
    nominal,events,transitions=evaluate_batch(runs,scenarios,manifest,protocol,dirs["RQ2_behavior"],digest,telemetry_ids=telemetry_ids)
    distribution_summary(nominal,["algorithm","scenario_id"],METRICS).to_csv(dirs["RQ2_behavior"]/"controller_distributions.csv",index=False)
    if not transitions.empty:
        # First aggregate repeated transitions within controller/direction, then bootstrap controllers.
        transition_names=["rise_time","settling_time","overshoot_pct","steady_state_error","peak_time"]
        per_controller=transitions.groupby(["algorithm","controller_id","direction"],as_index=False)[transition_names].median()
        distribution_summary(per_controller,["algorithm","direction"],transition_names).to_csv(dirs["RQ2_behavior"]/"directional_transient_distributions.csv",index=False)
    rep_motion=dirs["RQ2_behavior"]/"representative_motion"
    rep_safety=dirs["RQ2_behavior"]/"representative_safety"
    rep_motion.mkdir(exist_ok=True);rep_safety.mkdir(exist_ok=True)
    analysis.motion_safety_outputs(reps,manifest,suite,rep_motion,rep_safety)
    correlation_outputs(components,nominal,protocol,dirs["RQ2_behavior"])
    metric_plots(nominal,"algorithm",dirs["RQ2_behavior"],"nominal")
    casale=next((s for s in validations if s.scenario_id=="V1_CASALE"),None)
    if casale is None: raise RuntimeError("Casale validation required for physical evaluation")
    physical=[(sid,analysis.PhysicsOverride(manifest["physics"]["nominal_mass_kg"]*mass,
        manifest["physics"]["rolling_resistance"]*crr,slope),None) for sid,_desc,slope,mass,crr in analysis.ROBUSTNESS_SCENARIOS]
    print("RQ3: all-controller physical sensitivity; ideal compensation remains explicit",flush=True)
    robustness,_,_=evaluate_batch(runs,[periodic,casale],manifest,protocol,dirs["RQ3_physical"],digest,physical)
    distribution_summary(robustness,["algorithm","scenario_id","condition_id"],METRICS).to_csv(dirs["RQ3_physical"]/"physical_distributions.csv",index=False)
    relative=robustness_relative(robustness)
    relative.to_csv(dirs["RQ3_physical"]/"relative_to_each_controller_nominal.csv",index=False)
    summary=relative.groupby(["algorithm","scenario_id","condition_id","metric"],as_index=False)[["delta","ratio","degradation_pct"]].median()
    summary.to_csv(dirs["RQ3_physical"]/"median_degradation.csv",index=False)
    for metric in ["cruise_speed_rmse_mps","spacing_rmse_m","positive_tractive_energy_wh_per_km"]:
        for scenario,g in summary[summary.metric==metric].groupby("scenario_id"):
            grid=g.pivot(index="algorithm",columns="condition_id",values="degradation_pct")
            grid=grid.reindex(index=manifest["optimization"]["algorithms"],
                              columns=[s[0] for s in analysis.ROBUSTNESS_SCENARIOS])
            if not np.isfinite(grid.to_numpy(float)).any(): continue
            fig,ax=plt.subplots(figsize=(12,4.5))
            img=ax.imshow(grid.to_numpy(float),aspect="auto",cmap="coolwarm")
            ax.set_xticks(range(len(grid.columns)),grid.columns);ax.set_yticks(range(len(grid.index)),grid.index)
            ax.set_title(f"{scenario}: median change from each controller's nominal {metric}")
            fig.colorbar(img,ax=ax,label="Change (%)");fig.tight_layout()
            fig.savefig(dirs["RQ3_physical"]/f"degradation_{scenario}_{metric}.png",dpi=220);plt.close(fig)
    if args.actuator_config:
        entries=json.loads(args.actuator_config.read_text())
        conditions=[]
        for entry in entries:
            if set(entry)-{"id","lag_s","traction_limit_n","braking_limit_n","max_power_w","initial_force_n"}:
                raise ValueError("Only explicit actuator parameters permitted; no sensor or mismatch overrides")
            identifier=entry["id"]
            if not identifier or identifier=="S0" or identifier in [c[0] for c in conditions]: raise ValueError("Unique non-nominal actuator IDs required")
            actuator=ActuatorConfig(**{k:v for k,v in entry.items() if k!="id"});actuator.resolved(manifest["physics"]["max_longitudinal_force_n"])
            conditions.append((identifier,None,actuator))
        extension=dirs["RQ3_physical"]/"optional_actuator";extension.mkdir(exist_ok=True)
        evaluate_batch(runs,[periodic,casale],manifest,protocol,extension,digest,[("S0",None,None),*conditions])
    campaign=args.ablation_dir
    if args.all:
        campaign=find_compatible_ablation_campaign()
        if campaign is None:
            subprocess.run([sys.executable,"-m","scripts.ablation","--workers",str(args.workers)],cwd=ROOT_DIR,check=True)
            campaign=ROOT_DIR/"research_ablation"/("research_"+build_contract(False)["contract_sha256"][:12])
        else:
            print(f"RQ4: reusing compatible completed campaign {campaign.name}",flush=True)
    ablation_status="not executed"
    if campaign:
        ab_contract,controllers=load_ablation(campaign.resolve())
        controllers.to_csv(dirs["RQ4_objective"]/"training_components_and_gains.csv",index=False)
        ablation_comparisons(controllers,[*PERFORMANCE_NAMES,*analysis.GAIN_NAMES,"full_objective_score"],dirs["RQ4_objective"])
        calibration_comparisons=dirs["RQ4_objective"]/"paired_ablation_comparisons.csv"
        calibration_comparisons.replace(dirs["RQ4_objective"]/"paired_calibration_comparisons.csv")
        (dirs["RQ4_objective"]/"exploratory_objective_interactions.csv").replace(dirs["RQ4_objective"]/"calibration_objective_interactions.csv")
        behavior,_,_=evaluate_batch(controllers,scenarios,manifest,protocol,dirs["RQ4_objective"],digest+ab_contract["contract_sha256"])
        behavior=behavior.merge(controllers[["controller_id","objective_profile"]],on="controller_id",validate="many_to_one")
        behavior.to_csv(dirs["RQ4_objective"]/"controller_metrics.csv",index=False)
        distribution_summary(behavior,["objective_profile","scenario_id"],METRICS).to_csv(dirs["RQ4_objective"]/"ablation_distributions.csv",index=False)
        ablation_comparisons(behavior,METRICS,dirs["RQ4_objective"])
        metric_plots(behavior,"objective_profile",dirs["RQ4_objective"],"ablation")
        atomic_json(dirs["RQ4_objective"]/"campaign_contract.json",ab_contract)
        ablation_status="complete: 4 objective profiles x 30 paired DE blocks"
    write_evidence_report(out,protocol,runs,components,nominal,robustness,ablation_status)
    print(f"Research evidence: {out}",flush=True)
    return 0


def write_evidence_report(out,protocol,runs,components,nominal,robustness,ablation_status):
    max_error=float(components.cost_reconstruction_abs_error.max())
    issues=[]
    if max_error>1e-8: issues.append("Frozen score reconstruction mismatch")
    for label,frame in [("RQ2",nominal),("RQ3",robustness)]:
        if frame.duplicated(["controller_id","scenario_id","condition_id"]).any(): issues.append(label+" duplicate evaluation identity")
    status={"RQ1":{"controllers":len(runs),"score_reconstruction_max_abs_error":max_error},
        "RQ2":{"evaluations":len(nominal),"heldout_evaluations":int((nominal.role=="held_out").sum()),
            "failed_evaluations":int(nominal.failed.sum()),"unit":"controller; profiles/transitions are repeated measurements"},
        "RQ3":{"evaluations":len(robustness),"conditions":int(robustness.condition_id.nunique()),
            "failed_evaluations":int(robustness.failed.sum()),"compensation":"ideal matched road-load compensation; no mismatch claims"},
        "RQ4":{"status":ablation_status,"scope":"canonical DE only"},"issues":issues}
    atomic_json(out/"evidence_status.json",status)
    rq1=runs.groupby("algorithm",sort=False).agg(score=("fitness","median"),runtime=("runtime_s","mean"),FE=("function_evaluations","mean"))
    lines=["# Research-question evidence\n", "Generated from the declared frozen simulation protocol; no manuscript files were edited.\n",
        "## RQ1: optimizer quality and cost\n","| Optimizer | Median score | Mean runtime (s) | Mean FE |\n|---|---:|---:|---:|\n"+"\n".join(f"| {a} | {r.score:.9g} | {r.runtime:.5g} | {r.FE:.5g} |" for a,r in rq1.iterrows()),
        "\nIndependent-run inference is in RQ1_optimizer/. Runtime/FE are recorded effort, not an equal-FE comparison.\n",
        "## RQ2: closed-loop behavior\n",f"Evaluated {len(nominal)} controller-profile combinations, including {(nominal.role=='held_out').sum()} held-out evaluations. Failed evaluations: {nominal.failed.sum()}.",
        "\ncontroller_distributions.csv reports finite counts, tails, and controller-bootstrap intervals. Recovery fractions include nonrecovered events; recovered-only TTE is conditional. Correlations are exploratory.\n",
        "## RQ3: physical sensitivity\n",f"Evaluated {len(robustness)} controller-profile-condition combinations across {robustness.condition_id.nunique()} configurations. Failed evaluations: {robustness.failed.sum()}.",
        "\nRelative changes compare each fixed-gain controller with its own nominal result. Compensation uses the same physical parameters as the plant, so these are matched-model sensitivities.\n",
        "## RQ4: objective contribution\n",ablation_status,
        "\nUse common physical metrics and paired differences, not comparisons of unlike ablated scalar fitness values. Results are conditional on canonical DE.\n",
        "## Intro--Methods--Results--Conclusion traceability\n",
        "| RQ | Intro problem | Methods | Results evidence | Conclusion boundary |",
        "|---|---|---|---|---|",
        "| RQ1 | Stochastic quality-cost trade-off | 30 runs/optimizer, independent inference | RQ1_optimizer | Tested optimizers and declared budget |",
        "| RQ2 | Scores may not translate to motion | Frozen gains, held-out profiles, diagnostics | RQ2_behavior | Tested controllers and profiles |",
        "| RQ3 | Nominal behavior does not establish condition sensitivity | Fixed gains, 12 matched-model configurations | RQ3_physical | Ideal compensation and tested grid |",
        "| RQ4 | Full-objective components do not isolate objective contributions | Paired DE retuning, four objective profiles | RQ4_objective | Conditional on DE and declared model |"]
    (out/"RQ_evidence.md").write_text("\n".join(lines),encoding="utf-8")
    if issues: raise RuntimeError("Research evidence audit failed: "+"; ".join(issues))


if __name__=="__main__":
    raise SystemExit(main())
