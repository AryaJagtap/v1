"""Reading and writing tiles, the 2x2 preview, the diamond, and crops that wrap.

The pipeline works on an RGB copy. What it writes back goes into the image in
its own mode, and only where the repair touched, so a palette file stays on its
palette and a CMYK file stays CMYK everywhere the band did not reach.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# modes written back natively. anything else is converted to RGB, and says so
NATIVE = {"RGB", "RGBA", "L", "LA", "P", "CMYK"}


@dataclass
class Tile:
    path: Path
    image: Image.Image  # as stored, upright, in its own mode
    rgb: np.ndarray  # H x W x 3 uint8 working copy
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def opaque(self) -> bool:
        if "A" in self.image.mode:
            return int(np.asarray(self.image.getchannel("A")).min()) == 255
        return "transparency" not in self.image.info


def load(path: Path) -> Tile:
    path = Path(path)
    with Image.open(path) as raw:
        raw.load()
        info = dict(raw.info)
        upright = ImageOps.exif_transpose(raw)
    notes = []
    if upright.size != raw.size or upright.tobytes() != raw.tobytes():
        notes.append("rotated upright from its EXIF orientation")
    if upright.mode not in NATIVE:
        notes.append(f"{upright.mode} converted to RGB")
        upright = upright.convert("RGB")
    upright.info = {k: info[k] for k in ("dpi", "icc_profile", "transparency") if k in info}
    return Tile(path, upright, np.asarray(upright.convert("RGB")).copy(), notes)


def write_back(tile: Tile, rgb: np.ndarray, changed: np.ndarray) -> Image.Image:
    """The tile with `rgb` written in only where `changed`, in the tile's own mode."""
    source = tile.image
    mode = source.mode
    if not changed.any():
        return source.copy()
    if mode == "P":
        painted = Image.fromarray(rgb).quantize(palette=source, dither=Image.Dither.NONE)
    elif mode in ("RGBA", "LA"):
        painted = Image.fromarray(rgb).convert(mode[:-1])
    else:
        painted = Image.fromarray(rgb).convert(mode)
    out = np.asarray(source).copy()
    new = np.asarray(painted)
    if mode in ("RGBA", "LA"):
        out[..., :-1][changed] = new.reshape(*changed.shape, -1)[changed]
    else:
        out[changed] = new[changed]
    image = Image.frombytes(mode, source.size, np.ascontiguousarray(out).tobytes())
    if mode == "P":
        image.putpalette(source.getpalette())
    image.info = dict(source.info)
    return image


def save(image: Image.Image, path: Path) -> Path:
    """PNG for everything PNG can hold, TIFF for CMYK. Keeps dpi and ICC."""
    path = Path(path).with_suffix(".tif" if image.mode == "CMYK" else ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {k: image.info[k] for k in ("dpi", "icc_profile") if k in image.info}
    if image.mode == "CMYK":
        kwargs["compression"] = "tiff_lzw"
    image.save(path, **kwargs)
    return path


def tile_2x2(image: Image.Image) -> Image.Image:
    w, h = image.size
    grid = Image.new(image.mode, (w * 2, h * 2))
    if image.mode == "P":
        grid.putpalette(image.getpalette())
    for x, y in ((0, 0), (w, 0), (0, h), (w, h)):
        grid.paste(image, (x, y))
    return grid


def diamond(preview: Image.Image) -> Image.Image:
    """The preview with the diamond and its dotted diagonals. Never on a design."""
    scale = 4
    base = preview.convert("RGBA")
    w, h = base.size
    line = max(2, w // 400) * scale
    layer = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    big_w, big_h = w * scale, h * scale
    top, right = (big_w // 2, 0), (big_w - 1, big_h // 2)
    bottom, left = (big_w // 2, big_h - 1), (0, big_h // 2)
    draw.line([top, right, bottom, left, top], fill="black", width=line, joint="curve")
    radius, spacing = line / 2, line * 4
    for p0, p1 in ((top, bottom), (left, right)):
        steps = max(1, int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / spacing))
        for i in range(steps + 1):
            x = p0[0] + (p1[0] - p0[0]) * i / steps
            y = p0[1] + (p1[1] - p0[1]) * i / steps
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="black")
    return Image.alpha_composite(base, layer.resize((w, h), Image.LANCZOS)).convert("RGB")


def mark_joins(preview: Image.Image, check) -> Image.Image:
    """Red over the failing stretches of each join, on the clean 2x2, never the diamond."""
    out = preview.convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    width = max(3, w // 300)
    limit = 2 * check.threshold
    for s, e, r in check.lr.sections:
        if r > limit:
            draw.line([(w // 2, s), (w // 2, e)], fill="red", width=width)
    for s, e, r in check.tb.sections:
        if r > limit:
            draw.line([(s, h // 2), (e, h // 2)], fill="red", width=width)
    return out


def take(arr: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """An h x w crop at (y0, x0) that wraps around the tile edges, as a copy."""
    rows = np.arange(y0, y0 + h) % arr.shape[0]
    cols = np.arange(x0, x0 + w) % arr.shape[1]
    return arr[np.ix_(rows, cols)].copy()


def put(arr: np.ndarray, y0: int, x0: int, patch: np.ndarray, where: np.ndarray) -> None:
    """Write `patch` back over a wrapped crop, only where `where` is set."""
    rows = np.arange(y0, y0 + patch.shape[0]) % arr.shape[0]
    cols = np.arange(x0, x0 + patch.shape[1]) % arr.shape[1]
    view = arr[np.ix_(rows, cols)]
    view[where] = patch[where]
    arr[np.ix_(rows, cols)] = view


def thumbnail(image: Image.Image | np.ndarray, side: int = 1024) -> Image.Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    scale = side / max(image.size)
    if scale >= 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
