"""One attempt at one window: cut the crop, have a painter propose, align and
blend inside the band, gate. Shared by the graph, the CLI and the tests."""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from seamfix import composite, gates, imaging
from seamfix.painters import Painter, Request
from seamfix.plan import Plan, Window


@dataclass
class Candidate:
    window: int
    attempt: int
    painter: str
    after: np.ndarray  # the blended crop
    raw: np.ndarray  # what the painter returned, before alignment
    gates: gates.Gates
    fit: composite.Fit | None
    seconds: float
    verdict: dict = field(default_factory=dict)
    sharpen: float = 0.0  # unsharp amount that matched the band's detail to its neighbours

    @property
    def accepted(self) -> bool:
        return self.gates.ok and self.verdict.get("accept", False)


DETAIL_TARGET = 0.95  # band detail over its neighbours, after matching
MOST_SHARPEN = 1.0  # unsharp cap: firms a soft repaint, cannot rescue a smear


def match_detail(before: np.ndarray, painted: np.ndarray, win: Window, band: dict) -> tuple[np.ndarray, float]:
    """Sharpen a soft repaint until its band carries the detail of the artwork
    beside it, by the same measure the gate uses. An image model draws clean
    anti-aliased edges against a JPEG's crisp ones; this closes that gap and no
    more. Returns the crop and the unsharp amount used."""
    f = painted.astype(np.float32)
    edges = f - gaussian_filter(f, (1.0, 1.0, 0))

    def level(amount: float) -> tuple[np.ndarray, float]:
        out = np.clip(f + amount * edges + 0.5, 0, 255).astype(np.uint8)
        return out, gates.detail(composite.blend(before, out, win.mask), win.mask, win.axes, band)

    if level(0.0)[1] >= DETAIL_TARGET:
        return painted, 0.0
    if level(MOST_SHARPEN)[1] < DETAIL_TARGET:
        return level(MOST_SHARPEN)[0], MOST_SHARPEN
    lo, hi = 0.0, MOST_SHARPEN
    for _ in range(6):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if level(mid)[1] < DETAIL_TARGET else (lo, mid)
    return level(hi)[0], round(hi, 3)


def crop(rgb: np.ndarray, win: Window) -> np.ndarray:
    return imaging.take(rgb, win.y0, win.x0, win.size, win.size)


def attempt(
    rgb: np.ndarray, plan: Plan, win: Window, painter: Painter, number: int, request: Request,
) -> Candidate:
    before = request.crop
    started = time.time()
    raw = painter.paint(request)
    if raw.shape != before.shape:
        raise ValueError(f"{painter.name} returned {raw.shape}, expected {before.shape}")
    fitted, fit = (raw, None) if not painter.paid else composite.fit(raw, before, win.mask)
    sharpen = 0.0
    if painter.paid:
        fitted, sharpen = match_detail(before, fitted, win, plan.band)
    after = composite.blend(before, fitted, win.mask)
    section = max(16, max(rgb.shape[:2]) // 20)
    checks = gates.window(before, after, win.mask, win.axes, plan.band, section)
    if fit is not None and fit.drift > 1.0:
        checks = gates.Gates(checks.items + (gates.Gate("drift", fit.drift, "<= 1.0", False),))
    return Candidate(win.index, number, painter.name, after, raw, checks, fit,
                     round(time.time() - started, 1), sharpen=sharpen)


def apply(rgb: np.ndarray, win: Window, after: np.ndarray) -> None:
    imaging.put(rgb, win.y0, win.x0, after, win.mask)
