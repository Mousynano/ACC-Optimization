import os
import random
import hashlib
import numpy as np

def seed_everything(seed: int, xp=None):
    """
    Global seeding for:
    - python random
    - numpy
    - xp backend (numpy / cupy)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if xp is not None and hasattr(xp, "random"):
        xp.random.seed(seed)

def seed_for_run(fun_name: str, run_id: int, base_seed: int = 20260111) -> int:
    s = f"{base_seed}:{fun_name}:{run_id}".encode("utf-8")
    digest = hashlib.md5(s).hexdigest()  # stabil lintas run/machine
    return int(digest[:8], 16)  # 32-bit