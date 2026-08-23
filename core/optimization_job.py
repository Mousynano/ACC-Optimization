from dataclasses import dataclass
@dataclass
class OptimizationJob:
    fun_name: str
    fitness_fn: callable
    run_id: int
    algo_name: str
    obj_name: str
    obj_fn: callable