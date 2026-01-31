# core/xp.py
# Backend selector: NumPy or CuPy
import os

USE_GPU = os.getenv("USE_GPU", "0") == "1"

if USE_GPU:
    try:
        import cupy as xp
        import cupy as cp
        GPU_AVAILABLE = True
    except Exception:
        import numpy as xp
        import numpy as cp
        GPU_AVAILABLE = False
else:
    import numpy as xp
    import numpy as cp
    GPU_AVAILABLE = False

def to_cpu(arr):
    """Convert xp array to NumPy for fitness or multiprocessing."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return arr
