"""Deterministic process-local seeding for optimization jobs."""

import os
import random
import numpy as np


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
