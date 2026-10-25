"""Nano Banana (Gemini image models) as a painter.

It redraws the whole crop it is given, so what it returns is aligned,
colour-matched and blended back inside the strip only; everything else in the
crop is discarded. How Nano Banana 2.1 reads a mask is not documented, so the
strip is shown one of two ways and the probes decide:

  mask    the crop untouched, plus a white-on-black image of the strip
  erased  the crop with the strip's core painted flat magenta, nothing else

Only the core, the centre half where the two edges clash, is erased. A motif
the join cuts mostly sits within the strip, and erasing all of it made the
model invent a new one; with the rest visible it continues the one that is
there. The whole strip is still taken from its output.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from seamfix import config, imaging
from seamfix.agents import llm, prompts
from seamfix.agents.judge import join_words
from seamfix.painters import Request

MAGENTA = np.array([255, 0, 255], np.uint8)
CORE = 0.5  # share of the band width erased, around the join


def _listed(value) -> str:
    return ", ".join(value) if isinstance(value, list) else str(value or "as seen in image 1")


def core(mask: np.ndarray, share: float = CORE) -> np.ndarray:
    """The centre `share` of the band: pixels deeper inside it than the rest."""
    depth = distance_transform_edt(mask)
    half = np.percentile(depth[mask], 99) if mask.any() else 0
    return mask & (depth >= max(1.0, half * (1 - share)))


class NanoBanana:
    paid = True

    def __init__(
        self, model: str = config.PAINT_MODEL, show: str = "erased", thinking: str | None = None,
        resolution: str | None = None,
    ):
        """resolution "2K" on a 1024 crop draws at twice the size; scaling it down
        keeps edges and fine ground texture crisp."""
        self.model, self.show, self.thinking, self.resolution = model, show, thinking, resolution
        self.name = f"{model}:{show}" + (f":{resolution}" if resolution else "")

    def task(self, request: Request) -> tuple[str, list]:
        join = join_words(request.axes)
        if self.show == "erased":
            crop = request.crop.copy()
            crop[core(request.mask)] = MAGENTA
            images = [crop]
            lines = prompts.PAINT_IMAGES_ERASED.format(join=join)
        else:
            strip = (request.mask * 255).astype(np.uint8)
            images = [request.crop, np.stack([strip] * 3, axis=2)]
            lines = prompts.PAINT_IMAGES_MASK.format(join=join)
        if request.reference is not None:
            images.append(np.asarray(imaging.thumbnail(request.reference, 768)))
            lines += "\n" + prompts.PAINT_REFERENCE
        b = request.brief
        text = prompts.PAINT_TASK.format(
            images=lines,
            style=_listed(b.get("style")),
            motifs=_listed(b.get("motifs")),
            linework=_listed(b.get("linework")),
            palette=_listed(b.get("palette")),
            ground=_listed(b.get("ground")),
            across=b.get("across_join") or "continue whatever reaches the strip from both sides",
            feedback=request.feedback or prompts.FIRST_ATTEMPT,
        )
        return text, images

    def paint(self, request: Request) -> np.ndarray:
        text, images = self.task(request)
        h, w = request.mask.shape
        return llm.draw(
            self.model, prompts.PAINT_ROLE, text, images, (w, h), f"paint {self.name}",
            self.thinking, self.resolution,
        )
