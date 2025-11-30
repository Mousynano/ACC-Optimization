import numpy as np

def exp_reward(x):
    return np.exp(-np.sum(x ** 2))

def cosine_reward(x):
    return np.prod(np.cos(x))

def gaussian_peak(x, target=1.0):
    return np.exp(-np.sum((x - target) ** 2))

def rastrigin(x):
    x = np.array(x)
    A = 10
    return -(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def rastrigin(x):
    x = to_numpy(x)
    return np.sum(x**2 - 10 * np.cos(2*np.pi*x) + 10)

# ============================================
# Basic helper for converting list → ndarray
# ============================================
def to_numpy(x):
    return np.array(x, dtype=float)


# ============================================
# 1. Sphere
# f1(x) = Σ xi^2
# ============================================
def sphere(x):
    x = to_numpy(x)
    return np.sum(x**2)


# ============================================
# 2. Schwefel 1.2
# f2(x) = Σ ( Σ_{j=1}^i x_j )^2
# ============================================
def schwefel_12(x):
    x = to_numpy(x)
    cumulative_sum = np.cumsum(x)
    return np.sum(cumulative_sum**2)


# ============================================
# 3. Rosenbrock
# f3(x) = Σ [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
# ============================================
def rosenbrock(x):
    x = to_numpy(x)
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)


# ============================================
# 4. Step Function
# f4(x) = Σ (xi + 0.5)^2
# ============================================
def step(x):
    x = to_numpy(x)
    return np.sum((x + 0.5)**2)


# ============================================
# 5. Schwefel
# f5(x) = - Σ xi * sin( sqrt(|xi|) )
# ============================================
def schwefel(x):
    x = to_numpy(x)
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))))


# ============================================
# 6. Rastrigin
# f6(x) = Σ [xi^2 - 10cos(2πxi) + 10]
# ============================================
def rastrigin(x):
    x = to_numpy(x)
    return np.sum(x**2 - 10 * np.cos(2*np.pi*x) + 10)


# ============================================
# 7. Griewank
# f7(x) = (1/4000) Σ xi^2 - Π cos(xi / sqrt(i)) + 1
# ============================================
def griewank(x):
    x = to_numpy(x)
    i = np.arange(1, len(x)+1)
    return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(i))) + 1


# ============================================
# 8. Penalized Function
# f8(x) = π/d { 10 sin^2(πy1) + Σ (yi−1)^2 [1 + 10 sin^2(πyi+1)] }
#         + (yd−1)^2 + Σ u(xi, a, k, m)
# where yi = 1 + (xi+1)/4
#
# u(x) =  k(x−a)^m   if x > a
#       = k(−x−a)^m  if x < −a
#       = 0          otherwise
# ============================================

def u_penalty(x, a=10, k=100, m=4):
    if x > a:
        return k * (x - a)**m
    elif x < -a:
        return k * (-x - a)**m
    return 0

def penalized(x):
    x = to_numpy(x)
    d = len(x)
    yi = 1 + (x + 1) / 4

    # main summation part
    term1 = 10 * np.sin(np.pi * yi[0])**2
    term2 = np.sum((yi[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * yi[1:])**2))
    term3 = (yi[-1] - 1)**2

    # penalty
    penalty = np.sum([u_penalty(xi) for xi in x])

    return (np.pi / d) * (term1 + term2 + term3) + penalty


BENCHMARKS = {
    "exp": exp_reward,
    "cosine": cosine_reward,
    "gaussian": gaussian_peak,
    "rastrigin": rastrigin,
}
