"""Painters propose pixels for a band. They decide nothing: every proposal is
aligned, blended inside the band only, gated, and judged by others.

Each returns a crop the same size as the one it was given.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Request:
    crop: np.ndarray  # H x W x 3 uint8, the join through its centre
    mask: np.ndarray  # bool H x W, True where it may paint
    axes: tuple[str, ...]  # the joins crossing the crop: lr vertical, tb horizontal
    brief: dict = field(default_factory=dict)  # the analyst's description of the design
    feedback: str = ""  # what the judge or the gates said about the last attempt
    reference: np.ndarray | None = None  # the whole tile, small, for its motifs and palette


class Painter(Protocol):
    name: str
    paid: bool

    def paint(self, request: Request) -> np.ndarray: ...
