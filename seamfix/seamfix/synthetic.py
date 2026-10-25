"""Tiles with a known answer, for the tests and the visual cases.

`seamless` draws a textile-like repeat in which every shape is drawn at its
wrapped positions too, so it tiles exactly. The defects then break one join, a
corner, or a stretch across the wrap, by pasting in artwork from elsewhere in
the tile, the way a careless crop or edit breaks a real one.
"""

import numpy as np
from PIL import Image, ImageDraw

GROUND = (214, 200, 176)
INKS = [(122, 52, 50), (92, 64, 34), (250, 248, 240), (60, 48, 36), (176, 150, 112)]


def seamless(w: int = 512, h: int = 512, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), GROUND)
    draw = ImageDraw.Draw(img)

    def wrapped(shape, box, **kw):
        # integers first: PIL truncates a negative float the other way, which
        # put the wrapped copy of a shape a pixel off its original
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        for dx in (-w, 0, w):
            for dy in (-h, 0, h):
                getattr(draw, shape)((x0 + dx, y0 + dy, x1 + dx, y1 + dy), **kw)

    # fine ground texture: an offset dot lattice that divides the tile exactly.
    # an even number of rows, or the offset rows meet each other across the wrap
    cols, rows = max(1, round(w / 12)), max(2, 2 * round(h / 24))
    step_x, step_y = w / cols, h / rows
    for i in range(cols):
        for j in range(rows):
            x, y = (i + 0.5 * (j % 2)) * step_x, j * step_y
            wrapped("ellipse", (x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(198, 184, 158))
    side = min(w, h)
    for _ in range(14):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rx, ry = rng.uniform(0.06, 0.16) * side, rng.uniform(0.05, 0.12) * side
        ink = INKS[int(rng.integers(0, len(INKS)))]
        wrapped("ellipse", (cx - rx, cy - ry, cx + rx, cy + ry), fill=ink, outline=(52, 40, 30), width=max(2, side // 160))
        for k in range(8):  # a ring of dots around each motif
            a = 2 * np.pi * k / 8
            px, py = cx + 1.3 * rx * np.cos(a), cy + 1.3 * ry * np.sin(a)
            r = max(2, side // 110)
            wrapped("ellipse", (px - r, py - r, px + r, py + r), fill=INKS[0])
    return img


def break_lr(img: Image.Image, start: float = 0.3, length: float = 0.3, width: int = 0) -> Image.Image:
    """The right edge, over a stretch, replaced by artwork from the middle."""
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    width = width or max(6, w // 20)
    rows = (np.arange(int(start * h), int((start + length) * h))) % h
    src = (rows + h // 3) % h
    a[rows, w - width :] = a[src, w // 2 - width // 2 : w // 2 - width // 2 + width]
    return Image.fromarray(a)


def break_tb(img: Image.Image, start: float = 0.4, length: float = 0.35) -> Image.Image:
    return Image.fromarray(np.asarray(break_lr(img.transpose(Image.Transpose.TRANSPOSE), start, length))).transpose(
        Image.Transpose.TRANSPOSE
    )


def break_corner(img: Image.Image, size: float = 0.08) -> Image.Image:
    """The top-left corner block replaced: both joins break, only near the corner."""
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    s = max(8, int(size * min(w, h)))
    a[:s, :s] = a[h // 2 : h // 2 + s, w // 3 : w // 3 + s]
    return Image.fromarray(a)


def over_cropped(tile: Image.Image, across: float = 1.45, down: float = 1.3) -> Image.Image:
    w, h = tile.size
    big = Image.new("RGB", (3 * w, 3 * h))
    for i in range(3):
        for j in range(3):
            big.paste(tile, (i * w, j * h))
    return big.crop((w // 5, h // 7, w // 5 + int(across * w), h // 7 + int(down * h)))


def half_drop(tile: Image.Image, across: float = 2.6, down: float = 1.5) -> Image.Image:
    """Columns of the tile, each dropped by half a repeat: the straight tile is
    two repeats wide."""
    w, h = tile.size
    cols, rows = int(np.ceil(across)) + 1, int(np.ceil(down)) + 2
    big = Image.new("RGB", (cols * w, rows * h))
    for i in range(cols):
        for j in range(-1, rows):
            big.paste(tile, (i * w, j * h + (h // 2 if i % 2 else 0)))
    return big.crop((0, 0, int(across * w), int(down * h)))


def unrelated(w: int = 480, h: int = 560, seed: int = 3) -> Image.Image:
    """Artwork whose opposite edges were never made to meet."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (232, 226, 214))
    draw = ImageDraw.Draw(img)
    for _ in range(26):
        x, y = rng.uniform(-40, w), rng.uniform(-40, h)
        r = rng.uniform(20, 90)
        draw.ellipse((x, y, x + r * 1.6, y + r), fill=tuple(int(v) for v in rng.integers(40, 230, 3)))
    return img
