from typing import Callable, Dict
from core.utils import ise

import numpy as np
import math

# ============================================================
# 1. Benchmark Callable[[float], float]s (3-DIM)
# ============================================================

def rastrigin(x: np.ndarray, obj_fun=ise) -> float:
    """3D Rastrigin (global minimum = 0 at x = [0,0,0])"""
    x = np.asarray(x)
    A = 10
    res = 3 * A + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return obj_fun(res)


def sphere(x: np.ndarray, obj_fun: Callable[[float], float]) -> float:
    """3D Sphere (convex, unimodal)"""
    x = np.asarray(x)
    res = np.sum(x**2)
    return obj_fun(res)


def ackley(x: np.ndarray, obj_fun: Callable[[float], float]) -> float:
    """3D Ackley (multimodal)"""
    x = np.asarray(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    res = term1 + term2 + a + math.e
    return obj_fun(res)


def griewank(x: np.ndarray, obj_fun: Callable[[float], float]) -> float:
    """3D Griewank (many local minima, mild)"""
    x = np.asarray(x)
    i = np.arange(1, len(x) + 1)
    res = np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0
    return obj_fun(res)


def rosenbrock(x: np.ndarray, obj_fun: Callable[[float], float]) -> float:
    """3D Rosenbrock (narrow valley, non-convex)"""
    x = np.asarray(x)
    res = np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)
    return obj_fun(res)


def zakharov(x: np.ndarray, obj_fun: Callable[[float], float]) -> float:
    """3D Zakharov"""
    x = np.asarray(x)
    i = np.arange(1, len(x) + 1)
    term1 = np.sum(x**2)
    term2 = np.sum(0.5 * i * x)
    res = term1 + term2**2 + term2**4
    return obj_fun(res)


def schwefel(x: np.ndarray, obj_fun: Callable[[float], float]) -> float:
    """3D Schwefel"""
    x = np.asarray(x)
    res = 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return obj_fun(res)


def salomon(x: np.ndarray, obj_fun: Callable[[float], float]) -> float:
    """3D Salomon"""
    x = np.asarray(x)
    r = np.sqrt(np.sum(x**2))
    res = 1 - np.cos(2 * np.pi * r) + 0.1 * r
    return obj_fun(res)


def cosine_mixture(x: np.ndarray, obj_fun: Callable[[float], float]) -> float:
    """3D Cosine mixture"""
    x = np.asarray(x)
    res = -0.1 * np.sum(np.cos(5 * np.pi * x)) + np.sum(x**2)
    return obj_fun(res)

# Map nama → fungsi, biar gampang dipilih di pipeline
BENCHMARKS: Dict[str, Callable[[np.ndarray, Callable[[float], float]], float]] = {
    "rastrigin": rastrigin,
    # "sphere": sphere,
    # "ackley": ackley,
    # "griewank": griewank,
    # "rosenbrock": rosenbrock,
    # "zakharov": zakharov,
    # "schwefel": schwefel,
    # "salomon": salomon,
    # "cosine_mixture": cosine_mixture,
}