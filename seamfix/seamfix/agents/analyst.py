"""The analyst looks once: what kind of image this is, how hard its join is,
and how it is drawn, so the painter's prompt is written from this design and
not from a sentence that fits every design badly."""

from PIL import Image

from seamfix import config, imaging
from seamfix.agents import llm, prompts
from seamfix.agents.schemas import Diagnosis
from seamfix.checker import SeamCheck


def measured(check: SeamCheck, notes: list[str]) -> str:
    lines = []
    for axis, label in (("lr", "left edge against right edge (vertical line)"),
                        ("tb", "top edge against bottom edge (horizontal line)")):
        j = getattr(check, axis)
        lines.append(f"{label}: overall {j.overall:.2f}, worst stretch {j.worst:.2f}, "
                     f"{'passes' if j.ok else 'fails'} (limit {check.threshold}, worst {2 * check.threshold})")
    return "\n".join(lines + notes)


def diagnose(tile: Image.Image, marked_2x2: Image.Image, check: SeamCheck, notes: list[str]) -> Diagnosis:
    return llm.ask(
        config.ANALYST_MODEL,
        prompts.ANALYST_ROLE,
        prompts.ANALYST_TASK.format(measured=measured(check, notes)),
        [imaging.thumbnail(tile, 1024), imaging.thumbnail(marked_2x2, 1024)],
        Diagnosis,
        what="analyst",
        thinking="low",
    )
