"""A repeat smaller than the image: the exact fix for a tile cropped too big.

`peacock.jpg` holds 1.6 repeats across and 1.4 down. Repainting its borders
would invent pixels for a tile that was never broken; cutting out the repeat it
already contains is lossless. Textile repeats are often half-drop, so the
vertical repeat may come with a sideways shift, and the straight tile is then
two or more of them tall.

Similar motifs also match at half a repeat, so a match score alone picks wrong.
Each candidate is cut out and the confirmed seam check decides. Any window the
size of the repeat is the same repeat, so the cut may start where its joins run
through quiet ground, as a designer would place it.
"""

from dataclasses import dataclass
from itertools import product
from math import ceil

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter

from seamfix import checker, config

SMALLEST = 0.15  # a repeat shorter than this share of a side is texture, not a tile
EXACT = 0.02  # a repeat this close continues pixel for pixel: seamless by construction
MOST_ROWS = 4  # half-drop, third-drop, quarter-drop
OFFSETS = 4  # cut positions tried per axis


@dataclass(frozen=True)
class Repeat:
    width: int  # the repeat across
    height: int  # the repeat down
    drop: int  # sideways shift of the next row of repeats, 0 for a block
    rows: int  # repeats stacked to make a straight tile, 2 for half-drop
    match: float  # worst mean jump against the shifted self, over the neighbour jump
    across: bool  # found a repeat across, so the cut may move sideways
    down: bool

    @property
    def block(self) -> tuple[int, int]:
        return self.width, self.height * self.rows

    def describe(self) -> str:
        kind = "block" if self.drop == 0 else f"drop {self.drop}px per row"
        w, h = self.block
        return f"repeat {self.width}x{self.height} ({kind}) -> straight tile {w}x{h}"


@dataclass(frozen=True)
class Cut:
    repeat: Repeat
    offset: tuple[int, int]  # (y, x) where the straight tile starts in the image
    image: Image.Image
    check: checker.SeamCheck
    wrap: dict  # per axis: source pixels just past the cut against its own first ones, or None

    @property
    def score(self) -> float:
        return max(self.check.lr.overall, self.check.tb.overall)

    @property
    def proven(self) -> tuple[str, ...]:
        """Joins that continue pixel for pixel, whatever the seam check reads: the
        image just past the cut repeats the cut's own first column (lr) or row
        (tb). The check compares a join with the average neighbour jump, so an
        outline lying along a perfect join reads as a seam. A cut that spans the
        whole image on an axis proves nothing on it."""
        if self.repeat.match > EXACT:
            return ()
        return tuple(a for a in ("lr", "tb") if self.wrap[a] is not None and self.wrap[a] <= EXACT)

    @property
    def ok(self) -> bool:
        """Every join either proven or passing the seam check."""
        return all(a in self.proven or getattr(self.check, a).ok for a in ("lr", "tb"))


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def _neighbour(g: np.ndarray) -> float:
    jump = (np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2
    return max(float(jump), config.FLOOR)


def _shift_scores(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean squared difference of g against itself at every shift, over the
    overlap only, by FFT. Index [dy % 2h, dx % 2w]."""
    h, w = g.shape
    shape = (2 * h, 2 * w)
    fa, fa2, f1 = (np.fft.rfft2(x, s=shape) for x in (g, g * g, np.ones_like(g)))

    def corr(x, y):
        return np.fft.irfft2(np.conj(x) * y, s=shape)

    n = np.rint(corr(f1, f1))
    msd = (corr(fa2, f1) + corr(f1, fa2) - 2 * corr(fa, fa)) / np.maximum(n, 1)
    return np.maximum(msd, 0), n


def _jump(g: np.ndarray, dy: int, dx: int) -> float:
    """Mean absolute difference between g and g shifted by (dy, dx), overlap only."""
    h, w = g.shape
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    return float(np.abs(g[y0:y1, x0:x1] - g[y0 + dy : y1 + dy, x0 + dx : x1 + dx]).mean())


def _refine(g, dy, dx, reach, base, move_dy):
    tried = [
        (_jump(g, y, x) / base, y, x)
        for y in (range(dy - reach, dy + reach + 1) if move_dy else [dy])
        for x in range(dx - reach, dx + reach + 1)
        if 0 <= y < g.shape[0] and abs(x) < g.shape[1]
    ]
    return min(tried) if tried else None


def _minima(score: np.ndarray, allowed: np.ndarray, keep: int) -> list[tuple[int, int]]:
    """Positions of the `keep` deepest local minima of score where allowed."""
    masked = np.where(allowed, score, np.inf)
    local = (masked == minimum_filter(masked, size=5, mode="wrap")) & np.isfinite(masked)
    ys, xs = np.nonzero(local)
    order = np.argsort(masked[ys, xs])[:keep]
    return [(int(ys[i]), int(xs[i])) for i in order]


def candidates(rgb: np.ndarray, keep: int = 3) -> list[Repeat]:
    g = _gray(rgb)
    h, w = g.shape
    base = _neighbour(g)
    factor = max(1, ceil(max(h, w) / 320))
    small = np.asarray(
        Image.fromarray(g).resize((max(8, w // factor), max(8, h // factor)), Image.BOX), np.float32
    )
    sh, sw = small.shape
    msd, n = _shift_scores(small)
    sbase = max(
        float(np.square(np.diff(small, axis=0)).mean() + np.square(np.diff(small, axis=1)).mean())
        / 2,
        1.0,
    )
    ratio = msd / sbase
    dys = np.arange(2 * sh)[:, None]
    dxs = np.arange(2 * sw)[None, :]
    dxs = np.where(dxs >= sw, dxs - 2 * sw, dxs)  # signed
    enough = (n >= 0.12 * sh * sw) & (ratio < 0.8)
    across_ok = enough & (dys == 0) & (dxs >= ceil(SMALLEST * sw))
    down_ok = enough & (dys >= ceil(SMALLEST * sh)) & (dys < sh) & (np.abs(dxs) <= sw // 2)

    across = []
    for _, x in _minima(ratio, across_ok, keep):
        r = _refine(g, 0, int(dxs[0, x]) * factor, factor + 1, base, move_dy=False)
        if r and r[0] < config.REPEAT_MATCH:
            across.append(r)
    down = []
    for y, x in _minima(ratio, down_ok, keep):
        r = _refine(g, y * factor, int(dxs[0, x]) * factor, factor + 1, base, move_dy=True)
        if r and r[0] < config.REPEAT_MATCH:
            down.append(r)

    found = []
    for a, d in product(across or [None], down or [None]):
        if a is None and d is None:
            continue
        width = a[2] if a else w
        height, drop = (d[1], d[2]) if d else (h, 0)
        if a:
            drop = ((drop + width // 2) % width) - width // 2
        elif abs(drop) > 1:
            continue  # a drop with no repeat across: no straight tile is exact
        rows = next(
            (k for k in range(1, MOST_ROWS + 1) if min((k * drop) % width, -(k * drop) % width) <= 1),
            None,
        )
        if rows is None:
            continue
        if abs(drop) <= 1:
            drop, rows = 0, 1
        match = max(a[0] if a else 0.0, d[0] if d else 0.0)
        found.append(Repeat(width, height, drop, rows, round(match, 3), bool(a), bool(d)))
    unique = {(r.width, r.height, r.drop): r for r in found}
    return sorted(unique.values(), key=lambda r: (r.block[0] * r.block[1], r.match))


def extract(image: Image.Image, rep: Repeat, offset: tuple[int, int] = (0, 0)) -> Image.Image:
    """The straight tile, pixel for pixel from the image, in the image's own mode.
    Built through the repeat, so it may start anywhere and be taller than the image."""
    bw, bh = rep.block
    oy, ox = offset
    if oy + bh <= image.height and ox + bw <= image.width:
        return image.crop((ox, oy, ox + bw, oy + bh))
    src = np.asarray(image)
    ys, xs = np.mgrid[oy : oy + bh, ox : ox + bw]
    k = ys // rep.height
    out = src[ys - k * rep.height, (xs - k * rep.drop) % rep.width]
    tile = Image.frombytes(image.mode, (bw, bh), np.ascontiguousarray(out).tobytes())
    if image.mode == "P":
        tile.putpalette(image.getpalette())
    tile.info = dict(image.info)
    return tile


def _wrap(src: np.ndarray, tile: np.ndarray, oy: int, ox: int) -> dict:
    """Per axis, how far the image just past the cut differs from the cut's own
    first column (lr) or row (tb), over the neighbour jump. None where the cut
    reaches the image's edge and there is nothing past it to compare."""
    bh, bw = tile.shape[:2]
    base = _neighbour(_gray(src))
    out = {"lr": None, "tb": None}
    if ox + bw < src.shape[1]:
        out["lr"] = float(np.abs(src[oy : oy + bh, ox + bw].astype(np.int16) - tile[:, 0]).mean() / base)
    if oy + bh < src.shape[0]:
        out["tb"] = float(np.abs(src[oy + bh, ox : ox + bw].astype(np.int16) - tile[0]).mean() / base)
    return out


def best_cut(image: Image.Image, rgb: np.ndarray) -> Cut | None:
    """The cut that passes the seam check, smallest first, at the quietest start.
    Failing that, the best cut if it is far better than the image as it is, so a
    narrow repair can finish it. None when the image holds no repeat."""
    before = checker.check(rgb)
    original = max(before.lr.overall, before.tb.overall)
    cuts = []
    for rep in candidates(rgb):
        bw, bh = rep.block
        # a plain crop never stitches the image to itself. stitching is exact only
        # for an exact repeat, and a resized design repeats at a fraction of a
        # pixel, so the cut moves only within the image
        room_y, room_x = image.height - bh, image.width - bw
        ys = sorted({round(i * room_y / (OFFSETS - 1)) for i in range(OFFSETS)}) if rep.down and room_y > 0 else [0]
        xs = sorted({round(i * room_x / (OFFSETS - 1)) for i in range(OFFSETS)}) if rep.across and room_x > 0 else [0]
        for oy, ox in product(ys, xs):
            tile = extract(image, rep, (oy, ox))
            cut_rgb = np.asarray(tile.convert("RGB"))
            cuts.append(Cut(rep, (oy, ox), tile, checker.check(cut_rgb), _wrap(rgb, cut_rgb, oy, ox)))
    if not cuts:
        return None
    passing = [c for c in cuts if c.ok]
    if passing:
        smallest = min(c.repeat.block[0] * c.repeat.block[1] for c in passing)
        same = [c for c in passing if c.repeat.block[0] * c.repeat.block[1] == smallest]
        # the start the designer chose, when it passes, otherwise the quietest
        return next((c for c in same if c.offset == (0, 0)), min(same, key=lambda c: c.score))
    best = min(cuts, key=lambda c: c.score)
    return best if best.score < min(original, config.UNRELATED_OVERALL) * 0.75 else None
