import os
import random
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
