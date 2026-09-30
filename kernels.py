from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def sinc_pi(x: float) -> float:
    if abs(x) < 1e-8:
        return 1.0

    x_pi = math.pi * x
    return math.sin(x_pi) / x_pi


@njit(cache=True, fastmath=True)
def lanczos_kernel(x: float, a: int = 6) -> float:
    ax = abs(x)
    if ax < 1e-8:
        return 1.0
    if ax >= a:
        return 0.0

    return sinc_pi(x) * sinc_pi(x / a)


@njit(cache=True, fastmath=True)
def cubic_kernel(x: float, a: float = -0.5) -> float:
    ax = abs(x)

    if ax < 1.0:
        return (a + 2.0) * ax * ax * ax - (a + 3.0) * ax * ax + 1.0

    if ax < 2.0:
        return a * ax * ax * ax - 5.0 * a * ax * ax + 8.0 * a * ax - 4.0 * a

    return 0.0


@njit(cache=True, fastmath=True)
def cubic_weights(frac: float) -> np.ndarray:
    weights = np.empty(4, dtype=np.float32)
    weights[0] = cubic_kernel(1.0 + frac)
    weights[1] = cubic_kernel(frac)
    weights[2] = cubic_kernel(1.0 - frac)
    weights[3] = cubic_kernel(2.0 - frac)

    weight_sum = weights[0] + weights[1] + weights[2] + weights[3]
    if abs(weight_sum) > 1e-8:
        weights /= weight_sum
    return weights