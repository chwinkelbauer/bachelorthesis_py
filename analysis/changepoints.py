"""
Change-point detection on training curves (e.g. no-use share over iterations) to locate the step at which an
economy switches stage. Binary segmentation for shifts in the mean with a BIC-type penalty
2 * sigma^2 * ln(n), where sigma is estimated robustly from first differences (MAD).
"""

from typing import List

import numpy as np


def _cost(cs: np.ndarray, cs2: np.ndarray, i: int, j: int) -> float:
    n = j - i
    s = cs[j] - cs[i]
    return (cs2[j] - cs2[i]) - s * s / n


def binary_segmentation(series, min_size: int = 5, max_changes: int = 5) -> List[int]:
    """Returns sorted indices t where a new segment starts (series[t] is the first value after the change)."""
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n < 2 * min_size:
        return []
    diffs = np.diff(y)
    sigma = 1.4826 * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2) if n > 2 else 0.0
    penalty = 2.0 * max(sigma, 1e-12) ** 2 * np.log(n)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    cs2 = np.concatenate([[0.0], np.cumsum(y * y)])

    segments, changes = [(0, n)], []
    while len(changes) < max_changes:
        best = None
        for (i, j) in segments:
            if j - i < 2 * min_size:
                continue
            base = _cost(cs, cs2, i, j)
            for t in range(i + min_size, j - min_size + 1):
                gain = base - _cost(cs, cs2, i, t) - _cost(cs, cs2, t, j)
                if best is None or gain > best[0]:
                    best = (gain, t, (i, j))
        if best is None or best[0] <= penalty:
            break
        _, t, seg = best
        segments.remove(seg)
        segments += [(seg[0], t), (t, seg[1])]
        changes.append(t)
    return sorted(changes)
