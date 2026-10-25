"""The seam check the user confirmed, ported without changing what it decides.

The tile is laid 2x2 in memory. A join's ratio is the colour jump across it over
the typical neighbour jump elsewhere, with that baseline floored at 1.0. A join
passes only when the overall ratio is within the threshold and the worst of 20
sections is within twice it.

Names follow the borders, which is what a repair touches:
  lr  the vertical line in the 2x2: the left edge against the right edge
  tb  the horizontal line: the top edge against the bottom edge
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from seamfix import config


@dataclass(frozen=True)
class Join:
    overall: float
    worst: float
    sections: list[tuple[int, int, float]]  # (start, end, ratio) along the 2x2 line
    ok: bool


@dataclass(frozen=True)
class SeamCheck:
    lr: Join
    tb: Join
    threshold: float

    @property
    def ok(self) -> bool:
        return self.lr.ok and self.tb.ok

    @property
    def failing(self) -> tuple[str, ...]:
        return tuple(axis for axis in ("lr", "tb") if not getattr(self, axis).ok)

    def summary(self) -> dict:
        return {
            axis: {
                "overall": round(j.overall, 3),
                "worst": round(j.worst, 3),
                "ok": j.ok,
                "sections": [round(r, 2) for _, _, r in j.sections],
            }
            for axis, j in (("lr", self.lr), ("tb", self.tb))
        } | {"ok": self.ok, "threshold": self.threshold}


def measure_seam(arr: np.ndarray, segments: int, floor: float = config.FLOOR):
    """arr is (L, N, 3) with the seam between N//2-1 and N//2 along axis 1.
    Returns (overall_ratio, [(start, end, ratio), ...]). Verbatim from v1."""
    length, n, _ = arr.shape
    c = n // 2
    seam = np.abs(arr[:, c - 1].astype(np.int16) - arr[:, c].astype(np.int16)).mean(axis=1)
    base_sum = np.zeros(length, dtype=np.float64)
    count = 0
    for j in range(n - 1):
        if j == c - 1:
            continue
        base_sum += np.abs(arr[:, j + 1].astype(np.int16) - arr[:, j].astype(np.int16)).mean(axis=1)
        count += 1
    baseline = base_sum / max(count, 1)
    overall = seam.mean() / max(baseline.mean(), floor)
    results = []
    for idx in np.array_split(np.arange(length), segments):
        ratio = seam[idx].mean() / max(baseline[idx].mean(), floor)
        results.append((int(idx[0]), int(idx[-1]), float(ratio)))
    return float(overall), results


def check(
    rgb: np.ndarray,
    threshold: float = config.THRESHOLD,
    floor: float = config.FLOOR,
    segments: int = config.SEGMENTS,
) -> SeamCheck:
    grid = np.tile(rgb, (2, 2, 1))
    joins = []
    for arr in (grid, grid.transpose(1, 0, 2)):
        overall, sections = measure_seam(arr, segments, floor)
        worst = max(r for _, _, r in sections)
        joins.append(Join(overall, worst, sections, overall <= threshold and worst <= 2 * threshold))
    return SeamCheck(joins[0], joins[1], threshold)


def profile(rgb: np.ndarray, axis: str, floor: float = config.FLOOR) -> np.ndarray:
    """Ratio along one join at every position of the tile, smoothed over a tenth
    of its length with wrap-around, so a bad stretch across a corner is one
    stretch. Locates what to repaint; it decides nothing."""
    arr = rgb if axis == "lr" else rgb.transpose(1, 0, 2)
    a = arr.astype(np.int16)
    seam = np.abs(a[:, -1] - a[:, 0]).mean(axis=1)
    baseline = np.abs(np.diff(a, axis=1)).mean(axis=(1, 2))
    size = max(3, arr.shape[0] // 10)
    seam = uniform_filter1d(seam.astype(np.float64), size, mode="wrap")
    baseline = uniform_filter1d(baseline.astype(np.float64), size, mode="wrap")
    return seam / np.maximum(baseline, floor)
