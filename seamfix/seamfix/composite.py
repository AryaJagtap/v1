"""Putting a painter's output back: align it, match its colour, blend it in only
inside the band.

v1 feathered with a gaussian that spread past the band, so pixels outside it
moved. Here the ramp runs inward from the band edge, and outside the band the
original pixel is kept bit for bit. A model that redraws the whole crop also
shifts and tints it a little; both are measured on the untouched surroundings
and undone before anything is kept.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, shift

from seamfix import config

REACH = 6  # px searched for a shift. a model that moved the picture further redrew it


@dataclass(frozen=True)
class Fit:
    dy: float
    dx: float
    gain: tuple[float, ...]
    bias: tuple[float, ...]
    drift: float  # mean jump just outside the band after the fit, over the neighbour jump

    def as_dict(self) -> dict:
        return {"dy": self.dy, "dx": self.dx, "gain": self.gain, "bias": self.bias, "drift": self.drift}


def alpha(mask: np.ndarray, feather: int = config.FEATHER) -> np.ndarray:
    """1 deep inside the band, ramping to 0 at its edges, 0 everywhere outside.
    Where the band runs off the crop border it carries on in the next window,
    and the distance transform, blind past the border, leaves it unramped."""
    if feather <= 0:
        return mask.astype(np.float32)
    return np.clip(distance_transform_edt(mask) / feather, 0, 1).astype(np.float32)


def ring(mask: np.ndarray, inner: int = 4, outer: int = 24) -> np.ndarray:
    """A strip of untouched pixels just outside the band."""
    return binary_dilation(mask, iterations=outer) & ~binary_dilation(mask, iterations=inner)


def _offset(before: np.ndarray, painted: np.ndarray, keep: np.ndarray) -> tuple[float, float]:
    """The (dy, dx) that lays `painted` back onto `before`, by trying every shift
    within REACH on the kept pixels, then a parabola through the best."""
    a = before.astype(np.float32).mean(axis=2)
    b = painted.astype(np.float32).mean(axis=2)
    r = REACH
    core = np.zeros_like(keep)
    core[r:-r, r:-r] = keep[r:-r, r:-r]
    ys, xs = np.nonzero(core[::2, ::2])
    ys, xs = ys * 2, xs * 2
    cost = np.empty((2 * r + 1, 2 * r + 1))
    for i, dy in enumerate(range(-r, r + 1)):
        for j, dx in enumerate(range(-r, r + 1)):
            cost[i, j] = np.abs(a[ys, xs] - b[ys - dy, xs - dx]).mean()
    i, j = np.unravel_index(np.argmin(cost), cost.shape)

    def vertex(m, c, p):
        d = m - 2 * c + p
        return 0.0 if d <= 1e-9 else float(np.clip(0.5 * (m - p) / d, -0.5, 0.5))

    dy = i - r + (vertex(cost[i - 1, j], cost[i, j], cost[i + 1, j]) if 0 < i < 2 * r else 0)
    dx = j - r + (vertex(cost[i, j - 1], cost[i, j], cost[i, j + 1]) if 0 < j < 2 * r else 0)
    return float(dy), float(dx)


def fit(painted: np.ndarray, before: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, Fit]:
    """Align and colour-match a painter's crop to the original on the pixels it
    was told to keep. Returns the corrected crop and what was corrected."""
    keep = ~binary_dilation(mask, iterations=2)
    dy, dx = _offset(before, painted, keep)
    out = painted.astype(np.float32)
    if abs(dy) > 0.2 or abs(dx) > 0.2:
        out = shift(out, (dy, dx, 0), order=3, mode="nearest")
    near = ring(mask)
    gain, bias = [], []
    for c in range(3):
        x, y = out[..., c][near], before[..., c][near].astype(np.float32)
        a = float(np.polyfit(x, y, 1)[0]) if x.std() > 2 else 1.0
        a = float(np.clip(a, 0.85, 1.18))
        b = float(y.mean() - a * x.mean())
        out[..., c] = out[..., c] * a + b
        gain.append(round(a, 3))
        bias.append(round(b, 2))
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    neighbour = max(float(np.abs(np.diff(before.astype(np.int16), axis=1)).mean()), config.FLOOR)
    drift = float(np.abs(out[near].astype(np.int16) - before[near].astype(np.int16)).mean())
    return out, Fit(round(dy, 2), round(dx, 2), tuple(gain), tuple(bias), round(drift / neighbour, 3))


def blend(before: np.ndarray, painted: np.ndarray, mask: np.ndarray, feather: int = config.FEATHER):
    a = alpha(mask, feather)[..., None]
    out = before.astype(np.float32) * (1 - a) + painted.astype(np.float32) * a
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    out[~mask] = before[~mask]  # exact, not merely close
    return out
