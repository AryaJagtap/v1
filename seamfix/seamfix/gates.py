"""Checks a model cannot talk its way past, run on every candidate before any
model is asked to judge it.

The seam ratio alone passed v1's blurred paisley at 1.01, because a blur hides a
join perfectly. So a candidate must also keep the detail of its neighbours, add
no new seam where the band meets the original, and bring no colour the design
does not already hold.
"""

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace

from seamfix import checker, config

# band detail over neighbour detail, measured after a 1px blur so a JPEG's
# compression noise does not count as detail. Calibrated on pattern.jpg: the
# untouched join 1.00, LaMa's smear 0.38, Nano Banana's clean repaint 0.88
DETAIL = (0.75, 1.5)
DETAIL_BLOCK = 0.5  # worst 64-row stretch: catches texture lost in one place
BLOCK = 64
PALETTE_COLOURS = 48
FAR = 28.0  # RGB distance from every design colour that counts as a new colour
STRIP = 16  # px each side of a band edge when measuring a new seam there


@dataclass(frozen=True)
class Gate:
    name: str
    value: float
    limit: str
    ok: bool


@dataclass(frozen=True)
class Gates:
    items: tuple[Gate, ...]

    @property
    def ok(self) -> bool:
        return all(g.ok for g in self.items)

    @property
    def failed(self) -> list[str]:
        return [g.name for g in self.items if not g.ok]

    def value(self, name: str) -> float:
        return next(g.value for g in self.items if g.name == name)

    @property
    def score(self) -> float:
        """Lower is better. Ranks candidates; passing decides nothing on its own."""
        v = {g.name: g.value for g in self.items}
        score = sum(v.get(f"join_{a}", 0) for a in ("lr", "tb")) / config.THRESHOLD
        score += 0.5 * sum(v.get(f"edges_{a}", 0) for a in ("lr", "tb")) / config.THRESHOLD
        score += 2 * sum(abs(np.log(max(v.get(f"detail_{a}", 1), 1e-3))) for a in ("lr", "tb"))
        score += 20 * sum(v.get(f"new_colour_{a}", 0) for a in ("lr", "tb"))
        return round(float(score), 4)

    def as_list(self) -> list[dict]:
        return [asdict(g) for g in self.items]


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def _ratio(arr: np.ndarray, section: int) -> tuple[float, float]:
    segments = max(1, arr.shape[0] // max(section, 1))
    overall, parts = checker.measure_seam(arr, segments)
    return overall, max(r for _, _, r in parts)


def _palette(rgb: np.ndarray) -> np.ndarray:
    small = Image.fromarray(rgb).quantize(PALETTE_COLOURS, method=Image.Quantize.MEDIANCUT)
    return np.asarray(small.getpalette()[: PALETTE_COLOURS * 3], np.float32).reshape(-1, 3)


def _flat(rgb: np.ndarray) -> np.ndarray:
    """Pixels away from any edge. An outline's anti-aliased blend is not a new
    colour; a patch of a colour the design lacks is, and it is flat."""
    g = gaussian_filter(_gray(rgb), 1.0)
    grad = np.hypot(np.diff(g, axis=0, append=g[-1:]), np.diff(g, axis=1, append=g[:, -1:]))
    return grad < 6.0


def _far_share(pixels: np.ndarray, palette: np.ndarray) -> float:
    if len(pixels) == 0:
        return 0.0
    sample = pixels[:: max(1, len(pixels) // 20000)].astype(np.float32)
    d = np.sqrt(((sample[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return float((d > FAR).mean())


def _band(m: np.ndarray, width: int) -> tuple[np.ndarray, int, int]:
    """Rows the band runs along, and its columns, for a join through the centre."""
    c, half = m.shape[1] // 2, width // 2
    lo, hi = max(c - half, STRIP), min(c + half, m.shape[1] - STRIP)
    rows = np.nonzero(m[:, lo:hi].any(axis=1) & ~m.all(axis=1))[0]
    return rows, lo, hi


def _detail(a: np.ndarray, rows: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """Band detail over flank detail, overall and in the worst 64-row stretch."""
    half = (hi - lo) // 2
    lap = np.abs(laplace(gaussian_filter(_gray(a[rows]), 1.0)))
    f = config.FEATHER
    inner = lap[:, lo + f : hi - f] if hi - lo > 2 * f + 2 else lap[:, lo:hi]
    flank = np.concatenate([lap[:, max(0, lo - 2 * half) : lo], lap[:, hi : hi + 2 * half]], axis=1)
    overall = float(inner.mean() / max(flank.mean(), 1e-3))
    blocks = [
        inner[i : i + BLOCK].mean() / max(flank[i : i + BLOCK].mean(), 1.0)
        for i in range(0, max(1, len(rows) - BLOCK + 1), BLOCK)
    ]
    return overall, float(min(blocks))


def detail(after: np.ndarray, mask: np.ndarray, axes: tuple[str, ...], band: dict[str, int]) -> float:
    """The lowest band detail over its flanks across the window's joins."""
    values = []
    for axis in axes:
        a, m = (after, mask) if axis == "lr" else (after.transpose(1, 0, 2), mask.T)
        rows, lo, hi = _band(m, band[axis])
        if len(rows) >= 8:
            values.append(_detail(a, rows, lo, hi)[0])
    return min(values) if values else 1.0


def window(
    before: np.ndarray, after: np.ndarray, mask: np.ndarray, axes: tuple[str, ...],
    band: dict[str, int], section: int,
) -> Gates:
    """Gates for one window. The join of each axis runs through the crop's centre."""
    items: list[Gate] = []
    for axis in axes:
        b, a, m = before, after, mask
        if axis == "tb":
            b, a, m = b.transpose(1, 0, 2), a.transpose(1, 0, 2), m.T
        rows, lo, hi = _band(m, band[axis])
        half = (hi - lo) // 2
        if len(rows) < 8:
            continue
        overall, worst = _ratio(a[rows], section)
        items.append(Gate(
            f"join_{axis}", round(overall, 3), f"<= {config.THRESHOLD}, worst <= {2 * config.THRESHOLD}",
            overall <= config.THRESHOLD and worst <= 2 * config.THRESHOLD,
        ))
        edge = 0.0
        edge_worst = 0.0
        for e in (lo, hi):
            o, w = _ratio(a[rows, e - STRIP : e + STRIP], section)
            edge, edge_worst = max(edge, o), max(edge_worst, w)
        items.append(Gate(
            f"edges_{axis}", round(edge, 3), f"<= {1.25 * config.THRESHOLD:.2f}",
            edge <= 1.25 * config.THRESHOLD and edge_worst <= 2.5 * config.THRESHOLD,
        ))
        detail, worst = _detail(a, rows, lo, hi)
        items.append(Gate(
            f"detail_{axis}", round(detail, 3), f"{DETAIL[0]}..{DETAIL[1]}",
            DETAIL[0] <= detail <= DETAIL[1],
        ))
        items.append(Gate(f"detail_local_{axis}", round(worst, 3), f">= {DETAIL_BLOCK}", worst >= DETAIL_BLOCK))
        palette = _palette(b)
        fb, fa = _flat(b[rows]), _flat(a[rows])
        sides = np.r_[max(0, lo - 2 * half) : lo, hi : min(b.shape[1], hi + 2 * half)]
        known = _far_share(b[rows][:, sides][fb[:, sides]], palette)
        new = _far_share(a[rows, lo:hi][fa[:, lo:hi]], palette)
        allowed = max(2 * known, 0.02)
        items.append(Gate(f"new_colour_{axis}", round(new, 4), f"<= {allowed:.3f}", new <= allowed))
    return Gates(tuple(items))


def tile(
    original: np.ndarray, repaired: np.ndarray, claimed: np.ndarray, before: checker.SeamCheck,
) -> tuple[Gates, checker.SeamCheck]:
    """Gates for the assembled tile: nothing outside the bands moved, the joins
    pass the confirmed check, and no join got worse."""
    moved = int((original != repaired).any(axis=2)[~claimed].sum())
    after = checker.check(repaired)
    items = [Gate("untouched", moved, "== 0 px changed outside the bands", moved == 0)]
    for axis in ("lr", "tb"):
        j0, j1 = getattr(before, axis), getattr(after, axis)
        items.append(Gate(
            f"seam_{axis}", round(j1.overall, 3),
            f"<= {config.THRESHOLD}, worst {j1.worst:.2f} <= {2 * config.THRESHOLD}", j1.ok,
        ))
        if j0.ok:
            items.append(Gate(f"held_{axis}", round(j1.overall - j0.overall, 3), "<= 0.05", j1.overall - j0.overall <= 0.05))
    return Gates(tuple(items)), after
