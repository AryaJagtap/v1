"""The judge sees before and after side by side in one image, zoomed on the
strip, with the strip's edges marked in the margin and never on the artwork,
and scores what a printer would reject. Its notes feed the next attempt;
whether to accept is decided in `Verdict.accept`, in code.

Three separate 768px images at the default media resolution hid a tone step of
a few pixels: the judge passed a join the gates measured at 2.0. Side by side,
zoomed, at high resolution, it sees what a person comparing them sees.
"""

import numpy as np
from google.genai import types
from PIL import Image, ImageDraw

from seamfix import config
from seamfix.agents import llm, prompts
from seamfix.agents.schemas import Verdict

LONGEST = 1536  # px, the panel's longest side
MARGIN = 24  # white margin carrying the strip marks
GAP = 16


def join_words(axes: tuple[str, ...]) -> str:
    words = {"lr": "vertical line", "tb": "horizontal line"}
    return " and the ".join(words[a] for a in axes)


def design_text(brief: dict) -> str:
    if not brief:
        return "no description available"
    keys = ("style", "motifs", "linework", "palette", "ground")
    return "\n".join(
        f"{k}: {', '.join(brief[k]) if isinstance(brief[k], list) else brief[k]}" for k in keys if k in brief
    )


def focus(mask: np.ndarray, margin: int | None = None) -> tuple[slice, slice]:
    """The painted pixels and as much again on each side across the strip."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    across = min(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    margin = margin if margin is not None else max(48, across // 2)
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + 1 + margin)
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + 1 + margin)
    return slice(y0, y1), slice(x0, x1)


def panel(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> Image.Image:
    """BEFORE | AFTER on white, zoomed to LONGEST, strip edges ticked in the margins."""
    ys, xs = focus(mask)
    b, a, m = before[ys, xs], after[ys, xs], mask[ys, xs]
    h, w = m.shape
    scale = min(3.0, LONGEST / max(2 * w + GAP, h))  # smooth zoom: nearest made JPEG look broken
    zw, zh = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (2 * zw + GAP + 2 * MARGIN, zh + 2 * MARGIN), "white")
    draw = ImageDraw.Draw(canvas)
    cols = np.nonzero(m.any(axis=0) & ~m.all(axis=0))[0]
    rows = np.nonzero(m.any(axis=1) & ~m.all(axis=1))[0]
    for i, crop in enumerate((b, a)):
        left = MARGIN + i * (zw + GAP)
        canvas.paste(Image.fromarray(crop).resize((zw, zh), Image.LANCZOS), (left, MARGIN))
        draw.text((left, 4), ("BEFORE", "AFTER")[i], fill="black")
        for edge in (cols.min(), cols.max() + 1) if len(cols) and len(cols) < w else ():
            x = left + round(edge * scale)
            draw.line([(x, MARGIN - 10), (x, MARGIN - 2)], fill=(0, 160, 0), width=3)
            draw.line([(x, MARGIN + zh + 2), (x, MARGIN + zh + 10)], fill=(0, 160, 0), width=3)
        for edge in (rows.min(), rows.max() + 1) if len(rows) and len(rows) < h else ():
            y = MARGIN + round(edge * scale)
            draw.line([(left - 10, y), (left - 2, y)], fill=(0, 160, 0), width=3)
    return canvas


def inspect(
    before: np.ndarray, after: np.ndarray, mask: np.ndarray, axes, brief: dict,
    model: str = config.JUDGE_MODEL, thinking: str = "high",
) -> Verdict:
    return llm.ask(
        model,
        prompts.JUDGE_ROLE,
        prompts.JUDGE_TASK.format(join=join_words(axes), design=design_text(brief)),
        [panel(before, after, mask)],
        Verdict,
        what="judge",
        thinking=thinking,
        media=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    )
