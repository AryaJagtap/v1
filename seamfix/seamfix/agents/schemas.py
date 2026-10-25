"""The shapes bound to each model call. Field descriptions document them; the
prompts never repeat them."""

from typing import Literal

from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    kind: Literal["repeat_tile", "artwork_not_repeat", "photo_or_mockup"]
    difficulty: Literal["easy", "medium", "hard"]
    reason: str = Field(description="one sentence: why this kind and difficulty")
    style: str = Field(description="technique and finish in printer's words")
    motifs: list[str] = Field(description="recurring elements with their colours, at most 8")
    linework: str = Field(description="outline colour and weight, how dots and fills are drawn")
    palette: list[str] = Field(description="distinct colours as plain names, at most 10")
    ground: str = Field(description="background colour and texture")
    across_join: str = Field(description="what reaches the failing join from each side and how it continues")


DefectKind = Literal[
    "visible_join", "broken_line", "ghosting", "invented_motif", "smear", "blur",
    "colour_shift", "texture_mismatch", "text_or_mark", "distortion",
]


class Defect(BaseModel):
    kind: DefectKind
    where: Literal["top", "middle", "bottom", "left", "right", "whole"]
    severity: Literal["minor", "major"]
    note: str = Field(description="what exactly is wrong, in a few words")


class Verdict(BaseModel):
    # described before scored, in this order, so the scores rest on a comparison
    finish_outside: str = Field(description="edges and texture of the untouched artwork beside the strip")
    finish_inside: str = Field(description="edges and texture inside the strip, in the same terms")
    join_visible: bool = Field(description="a break, step or tone change still runs along the join")
    continuity: int = Field(ge=1, le=5, description="lines and motifs meet across the strip")
    fidelity: int = Field(ge=1, le=5, description="nothing in the strip is new")
    finish: int = Field(ge=1, le=5, description="the strip matches the finish around it")
    defects: list[Defect]
    fix: str = Field(description="at most two sentences for the painter's next attempt, or empty")

    @property
    def accept(self) -> bool:
        """Decided here, not by the model. Calibrated on untouched artwork: Gemini
        3.1 Pro scores a clean strip's finish 3 with only minor notes, so finish
        needs 3; continuity and fidelity need 4; no major defect; join gone."""
        scores = self.continuity >= 4 and self.fidelity >= 4 and self.finish >= 3
        majors = any(d.severity == "major" for d in self.defects)
        return scores and not majors and not self.join_visible
