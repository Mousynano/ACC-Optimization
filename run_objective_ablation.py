"""RQ4: isolated, resumable canonical-DE objective ablation.

Never writes production weights/checkpoints. Full objective is rerun in the
same paired design. Smoke jobs have a separate namespace and research=false.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")

import numpy as np
from benchmarks.acc_cost_fje_v2 import PERFORMANCE_NAMES, evaluate_performance_vector, load_cost_profile, scalar_cost_from_performance
from core.config import ROOT_DIR, load_manifest
from core.scenarios_v2 import load_scenario_suite, sha256_file
from core.research_statistics import stable_seed
from optimizers.de import run_de

PROFILE_MASKS = {
    "full":(1,1,1,1), "without_jerk":(1,1,0,1),
    "without_energy":(1,1,1,0), "tracking_only":(1,1,0,0),
}


def ablation_weights(frozen_weights, profile):
    if profile not in PROFILE_MASKS: raise ValueError("Unknown ablation profile")
    w = np.asarray(frozen_weights,float)
    if w.shape != (4,) or not np.isfinite(w).all() or np.any(w <= 0) or not np.isclose(w.sum(),1):
        raise ValueError("Ablation must start from four positive frozen weights")
    masked = w*np.asarray(PROFILE_MASKS[profile])
    return masked/masked.sum()


def initial_population(seed, population, bounds_low, bounds_high):
    return np.random.default_rng(seed).uniform(bounds_low,bounds_high,size=(population,3))


def population_hash(population):
    return hashlib.sha256(np.asarray(population,dtype="<f8").tobytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def build_contract(smoke=False):
    protocol = json.loads((ROOT_DIR/"research_protocol.json").read_text())
    manifest = load_manifest(); ab = protocol["ablation"]
    suite = load_scenario_suite(manifest["cost_framework"]["scenario_manifest_path"],development=smoke,require_frozen=not smoke)
    profile = load_cost_profile(require_frozen=True)
    if not smoke and suite.manifest_sha256 != profile.scenario_manifest_sha256:
        raise RuntimeError("Frozen objective and scenario hashes disagree")
    if manifest["sensing"]["nominal_noise_std_m"] != 0 or manifest["sensing"]["nominal_bias_m"] != 0:
        raise RuntimeError("Sensor uncertainty is outside the approved protocol")
    paths = ["benchmarks/acc_cost_fje_v2.py","core/fast_acc_energy.py","core/fast_acc.py",
             "optimizers/de.py","run_objective_ablation.py","research_protocol.json"]
    contract = {"protocol_version":protocol["protocol_version"],"research":not smoke,
        "optimizer":"de","runs":2 if smoke else ab["runs"],
        "population_size":8 if smoke else ab["population_size"],"iterations":2 if smoke else ab["iterations"],
        "base_seed":ab["base_seed"],"mutation_factor":ab["mutation_factor"],"crossover_rate":ab["crossover_rate"],
        "profiles":list(PROFILE_MASKS),"min_bounds":manifest["controller"]["min_bounds"],
        "max_bounds":manifest["controller"]["max_bounds"],"frozen_weights":profile.weights.tolist(),
        "weights":{name:ablation_weights(profile.weights,name).tolist() for name in PROFILE_MASKS},
        "scenario_manifest_sha256":suite.manifest_sha256,"cost_profile_sha256":sha256_file(profile.path),
        "physics":manifest["physics"],"controller":manifest["controller"],"sensing":manifest["sensing"],
        "sampling_hz":manifest["execution_backend"]["sampling_hz"],
        "source_sha256":{name:sha256_file(ROOT_DIR/name) for name in paths},
        "pairing":"explicit shared initial_population + local DE seed per block",
        "weight_policy":ab["weight_rule"],"model":"ideal_compensation_with_force_limits_v1"}
    if not smoke:
        for key,manifest_key in [("runs","independent_runs"),("population_size","population_size"),("iterations","max_iterations")]:
            if contract[key] != manifest["optimization"][manifest_key]:
                raise RuntimeError(f"Ablation {key} differs from approved main protocol")
    contract["contract_sha256"] = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    return contract


def run_block_job(contract, profile_name, run_id):
    manifest = load_manifest()
    suite = load_scenario_suite(manifest["cost_framework"]["scenario_manifest_path"],
        development=not contract["research"],require_frozen=contract["research"])
    population_seed = stable_seed(f"ablation-population:{run_id}",contract["base_seed"])
    optimizer_seed = stable_seed(f"ablation-DE:{run_id}",contract["base_seed"])
    population = initial_population(population_seed,contract["population_size"],contract["min_bounds"],contract["max_bounds"])
    weights = np.array(contract["weights"][profile_name],float)
    calls = 0
    evaluate_performance_vector(population[0],suite)  # exclude compilation from optimizer timing
    def objective(params):
        nonlocal calls
        calls += 1
        return scalar_cost_from_performance(evaluate_performance_vector(params,suite),weights)
    started = time.perf_counter()
    solution,score,curve = run_de(objective,contract["min_bounds"],contract["max_bounds"],
        population_size=contract["population_size"],max_iter=contract["iterations"],
        seed=optimizer_seed,initial_population=population,mutation_factor=contract["mutation_factor"],
        crossover_rate=contract["crossover_rate"],verbose=False)
    runtime = time.perf_counter()-started
    perf = evaluate_performance_vector(solution,suite)
    return {"contract_sha256":contract["contract_sha256"],"research":contract["research"],
        "profile":profile_name,"algorithm":"de","run_id":run_id,
        "controller_id":f"ablation:{contract['contract_sha256'][:12]}:{profile_name}:{run_id}",
        "population_seed":population_seed,"optimizer_seed":optimizer_seed,
        "initial_population_sha256":population_hash(population),"best_solution":np.asarray(solution).tolist(),
        "best_fitness":float(score),"curve":list(map(float,curve)),"duration":runtime,"function_evaluations":calls,
        "performance":perf.as_dict(),"feasible":bool(perf.feasible),"constraint_violation":float(perf.constraint_violation),
        "full_objective_score":scalar_cost_from_performance(perf,contract["frozen_weights"]),
        "weights":dict(zip(PERFORMANCE_NAMES,weights.tolist()))}


def validate_checkpoint(payload, contract, profile, run_id):
    if payload.get("contract_sha256") != contract["contract_sha256"] or payload.get("profile") != profile or payload.get("run_id") != run_id:
        raise RuntimeError("Ablation checkpoint contract/identity mismatch")
    seed = stable_seed(f"ablation-population:{run_id}",contract["base_seed"])
    expected = population_hash(initial_population(seed,contract["population_size"],contract["min_bounds"],contract["max_bounds"]))
    if payload.get("initial_population_sha256") != expected:
        raise RuntimeError("Ablation checkpoint initial-population mismatch")
    expected_optimizer = stable_seed(f"ablation-DE:{run_id}",contract["base_seed"])
    if payload.get("optimizer_seed") != expected_optimizer or len(payload.get("curve",[])) != contract["iterations"]:
        raise RuntimeError("Ablation checkpoint seed/iteration mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT_DIR/"research_ablation")
    parser.add_argument("--workers",type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument("--smoke-test",action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1: parser.error("workers must be positive")
    contract = build_contract(args.smoke_test)
    root = args.out_dir.resolve()/(("smoke_" if args.smoke_test else "research_")+contract["contract_sha256"][:12])
    root.mkdir(parents=True,exist_ok=True)
    manifest_path = root/"campaign.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != contract:
        raise RuntimeError("Campaign contract differs; use a different output directory")
    atomic_json(manifest_path,contract)
    jobs = []
    for profile in contract["profiles"]:
        for run_id in range(contract["runs"]):
            path = root/f"{profile}_run_{run_id:02d}.json"
            if path.exists(): validate_checkpoint(json.loads(path.read_text()),contract,profile,run_id)
            else: jobs.append((profile,run_id,path))
    print(f"RQ4 campaign: {len(jobs)} pending / {len(contract['profiles'])*contract['runs']} jobs; {root}",flush=True)
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_block_job,contract,p,r):(p,r,path) for p,r,path in jobs}
            for index,future in enumerate(as_completed(futures),1):
                p,r,path = futures[future]; result = future.result()
                validate_checkpoint(result,contract,p,r); atomic_json(path,result)
                print(f"[{index}/{len(jobs)}] {p} run={r} score={result['best_fitness']:.7g}",flush=True)
    atomic_json(root/"completion.json",{"complete":True,"research":contract["research"],
        "jobs":len(contract["profiles"])*contract["runs"],"contract_sha256":contract["contract_sha256"]})
    print(root,flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
