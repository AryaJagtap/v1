"""What to repaint: the fewest stretches of a join that let it pass, cut into
square windows at native resolution.

Coordinates are the tile's own. The lr join is the line between x = W-1 and
x = 0, so a window centred on it starts at x = -S/2 and wraps; the tb join is
the same between y = H-1 and y = 0. A window over the corner where both joins
cross gets a cross-shaped mask and is painted first, so the corner is painted
once, and every later window sees it as finished context.
"""

from dataclasses import dataclass, field

import numpy as np

from seamfix import config

BINS = 20  # stretches per join when choosing what to repaint


@dataclass
class Window:
    index: int
    y0: int  # top-left in tile coordinates, may be negative: crops wrap
    x0: int
    size: int
    mask: np.ndarray  # bool size x size, True where it may be repainted
    axes: tuple[str, ...]  # the joins that cross this window
    band: int  # band width across the join, px

    @property
    def box(self) -> dict:
        return {"y0": self.y0, "x0": self.x0, "size": self.size, "axes": list(self.axes)}


@dataclass
class Plan:
    windows: list[Window]
    spans: dict[str, list[tuple[int, int]]]  # per axis, (start, length) along the join
    band: dict[str, int]
    claimed: np.ndarray = field(repr=False)  # every pixel any window may repaint


def _bins(rgb: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per stretch along a join: the jump across it, the typical neighbour jump,
    and where the stretch starts. Same quantities as the checker, on the tile."""
    arr = (rgb if axis == "lr" else rgb.transpose(1, 0, 2)).astype(np.int16)
    seam = np.abs(arr[:, -1] - arr[:, 0]).mean(axis=1)
    base = np.abs(np.diff(arr, axis=1)).mean(axis=(1, 2))
    edges = np.linspace(0, arr.shape[0], BINS + 1).astype(int)
    s = np.array([seam[a:b].sum() for a, b in zip(edges[:-1], edges[1:], strict=True)])
    b = np.array([base[a:b].sum() for a, b in zip(edges[:-1], edges[1:], strict=True)])
    return s, b, edges


def spans(rgb: np.ndarray, axis: str, whole: bool, threshold: float = config.THRESHOLD):
    """The stretches to repaint, worst first, until the join would pass with room:
    overall under 0.9 of the threshold, every section under 0.8 of its limit."""
    length = rgb.shape[0] if axis == "lr" else rgb.shape[1]
    if whole:
        return [(0, length)]
    s, b, edges = _bins(rgb, axis)
    floor = config.FLOOR * np.diff(edges)
    b = np.maximum(b, floor)
    chosen = np.zeros(BINS, bool)

    def passes():
        fixed = np.where(chosen, b * 0.95, s)
        overall = fixed.sum() / b.sum()
        pairs = (fixed[0::2] + fixed[1::2]) / (b[0::2] + b[1::2])  # the checker's sections
        return overall <= 0.9 * threshold and pairs.max() <= 0.8 * 2 * threshold

    while not passes() and not chosen.all():
        excess = np.where(chosen, -np.inf, s - b)
        chosen[int(np.argmax(excess))] = True
    if chosen.sum() > 0.6 * BINS:
        return [(0, length)]
    # pad each chosen stretch by half a bin and merge, around the corner too
    marked = np.zeros(length, bool)
    pad = length // (2 * BINS)
    for i in np.nonzero(chosen)[0]:
        idx = np.arange(edges[i] - pad, edges[i + 1] + pad) % length
        marked[idx] = True
    if marked.all():
        return [(0, length)]
    start = int(np.argmin(marked))  # begin on an unmarked position so runs do not wrap
    rolled = np.roll(marked, -start)
    out, run = [], None
    for i, m in enumerate(rolled):
        if m and run is None:
            run = i
        elif not m and run is not None:
            out.append(((run + start) % length, i - run))
            run = None
    if run is not None:
        out.append(((run + start) % length, length - run))
    return out


def reach(rgb: np.ndarray, axis: str, spans_: list[tuple[int, int]], most: int) -> int:
    """How far from the join a defect runs, in px. A strip pasted next to the
    edge leaves a straight seam parallel to the join: a jump at the same column
    across most broken rows, which drawn artwork almost never has. The band must
    reach past it, or its inner edge stays behind as a new seam."""
    arr = rgb if axis == "lr" else rgb.transpose(1, 0, 2)
    length, side = arr.shape[:2]
    rows = np.nonzero(_in_spans(np.arange(length), spans_, length))[0]
    if len(rows) < 8:
        return 0
    cols = np.arange(-most, most) % side  # the join between -1 and 0
    a = arr[np.ix_(rows, cols)].astype(np.int16)
    jump = np.abs(np.diff(a, axis=1)).mean(axis=2)  # rows x (2*most - 1)
    typical = np.median(jump, axis=1, keepdims=True) + 1.0
    straight = (jump > 4 * typical).mean(axis=0)  # share of rows jumping at each column
    found = 0
    for i, share in enumerate(straight):
        d = i + 1 - most  # the boundary between offsets i-most and i+1-most
        if share >= 0.6 and d != 0:
            found = max(found, abs(d) + (1 if d < 0 else 0))
    return found


def band_width(side: int, difficulty: str) -> int:
    width = round(config.BAND_SHARE[difficulty] * side)
    width = max(config.BAND_MIN, min(config.BAND_MAX, width, side // 4))
    return width - width % 2


def _in_spans(positions: np.ndarray, spans_: list[tuple[int, int]], length: int) -> np.ndarray:
    hit = np.zeros(positions.shape, bool)
    for start, size in spans_:
        hit |= ((positions - start) % length) < size
    return hit


def make(
    rgb: np.ndarray, failing: tuple[str, ...], difficulty: str, whole: bool = False,
    widen: float = 1.0,
) -> Plan:
    h, w = rgb.shape[:2]
    size = min(config.WINDOW, h, w)
    half = size // 2
    # a window never paints to its own border along the join: the last eighth at
    # each end is context. so the band meets the next window's work with both
    # sides in view, including where a whole-length band closes on itself
    margin = size // 8
    inside = np.zeros(size, bool)
    inside[margin : size - margin] = True
    found = {axis: spans(rgb, axis, whole) for axis in failing}
    band = {}
    for axis, side in (("lr", w), ("tb", h)):
        width = band_width(side, difficulty) * widen
        if axis in found:  # wide enough to swallow a seam pasted beside the join
            width = max(width, 2 * (reach(rgb, axis, found[axis], side // 8) + 6))
        band[axis] = min(width, side // 4)
    band = {k: int(v) - int(v) % 2 for k, v in band.items()}
    claimed = np.zeros((h, w), bool)
    windows: list[Window] = []

    def add(y0: int, x0: int, axes: tuple[str, ...]) -> None:
        rows = np.arange(y0, y0 + size) % h
        cols = np.arange(x0, x0 + size) % w
        mask = np.zeros((size, size), bool)
        if "lr" in axes:  # band of columns around the lr join, in its stretches
            near = np.abs(((cols + w // 2) % w) - w // 2 + 0.5) <= band["lr"] / 2
            mask |= (_in_spans(rows, found["lr"], h) & inside)[:, None] & near[None, :]
        if "tb" in axes:
            near = np.abs(((rows + h // 2) % h) - h // 2 + 0.5) <= band["tb"] / 2
            mask |= near[:, None] & (_in_spans(cols, found["tb"], w) & inside)[None, :]
        mask &= ~claimed[np.ix_(rows, cols)]
        if mask.sum() < 16:
            return
        view = claimed[np.ix_(rows, cols)] | mask
        claimed[np.ix_(rows, cols)] = view
        windows.append(Window(len(windows), y0, x0, size, mask, axes, max(band.values())))

    if len(failing) == 2:
        add(-half, -half, ("lr", "tb"))
    step = size - 2 * margin  # painted parts of neighbouring windows abut
    for axis in failing:
        for start, span in found[axis]:
            if span <= step:  # fits one window, with context both ends
                positions = [start + span // 2 - half]
            else:
                positions = [start - margin + k * step for k in range(int(np.ceil(span / step)))]
            for along in positions:
                if axis == "lr":
                    add(along, -half, ("lr",))
                else:
                    add(-half, along, ("tb",))
    return Plan(windows, found, band, claimed)
