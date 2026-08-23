from tqdm import tqdm
import time

def display_progress(algorithms, objectives, progress, sub_progress):
    # Build main tqdm bars
    bars_alg = {}
    bars_obj = {}

    pos = 0
    
    # top-level (algorithms)
    for algo in algorithms:
        bars_alg[algo] = tqdm(
            total=4,
            desc=f"[ALGO] {algo.upper()}",
            position=pos,
            ncols=100
        )
        pos += 1
        
        # sub-objectives
        bars_obj[algo] = {}
        for obj in objectives:
            bars_obj[algo][obj] = tqdm(
                total=300,  # your max_iter
                desc=f"  > {obj.upper()}",
                position=pos,
                ncols=100
            )
            pos += 1

    # Live update loop
    running = True
    while running:
        running = any(v < 4 for v in progress.values())

        # update algorithm bars
        for algo in algorithms:
            bars_alg[algo].n = progress[algo]
            bars_alg[algo].refresh()

            # update objective bars
            for obj in objectives:
                bars_obj[algo][obj].n = sub_progress[algo][obj]
                bars_obj[algo][obj].refresh()

        time.sleep(0.05)

def job_progress_fraction(progress_dict, fun_name: str, run_id: int, algo_name: str, objectives: Dict[str, Any], max_iter: int) -> float:
    """
    Return 0.0..1.0
    Progress job dihitung sebagai mean progress dari semua objective.
    """
    if not objectives:
        return 0.0

    total = 0.0
    for obj_name in objectives.keys():
        key = (fun_name, run_id, algo_name, obj_name)
        step = progress_dict.get(key, 0)
        # clamp
        if step < 0:
            step = 0
        if step > max_iter:
            step = max_iter
        total += step / float(max_iter)

    return total / float(len(objectives))

def slot_progress_fraction(progress_dict, fun_name, run_id, algo_name, obj_name, max_iter):
    step = progress_dict.get((fun_name, run_id, algo_name, obj_name), 0)
    if step < 0: step = 0
    if step > max_iter: step = max_iter
    return step / float(max_iter)