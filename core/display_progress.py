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
