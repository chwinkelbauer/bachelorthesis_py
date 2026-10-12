"""
Robust aggregation across seeds (Agarwal et al. 2021, "Deep RL at the edge of the statistical precipice").
"""

from typing import Callable, Tuple

import numpy as np


def iqm(values) -> float:
    """Interquartile mean: mean of the middle 50% of values."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(v[lo:hi].mean()) if hi > lo else float(v.mean())


def bootstrap_ci(values, statistic: Callable = iqm, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval of a statistic."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = [statistic(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))
