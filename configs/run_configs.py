from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    n_runs: int = 1
    max_iter: int = 50
    population_size: int = 30
    max_concurrent_jobs: int = 20