"""The only module that talks to Gemini. Every call passes through the ledger.

`ask` returns a pydantic shape the provider enforces; `draw` returns an image.
Neither prompt carries a JSON skeleton: the schema is bound to the call.
"""

import io
import time
from functools import cache

import numpy as np
from google import genai
from google.genai import errors, types
from PIL import Image

from seamfix import config
from seamfix.budget import LEDGER, cost

RETRIES = 2
TIMEOUT_MS = 180_000


@cache
def client() -> genai.Client:
    return genai.Client(api_key=config.gemini_key(), http_options=types.HttpOptions(timeout=TIMEOUT_MS))


def part(image: np.ndarray | Image.Image) -> types.Part:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def _call(model: str, contents: list, cfg: types.GenerateContentConfig, what: str, estimate: float):
    held = LEDGER.reserve(estimate, what)
    usd, entry = 0.0, {"model": model, "what": what}
    try:
        for attempt in range(RETRIES + 1):
            try:
                response = client().models.generate_content(model=model, contents=contents, config=cfg)
                break
            except errors.APIError as e:
                transient = getattr(e, "code", 0) in (429, 500, 502, 503, 504)
                if not transient or attempt == RETRIES:
                    raise
                time.sleep(4 * (attempt + 1))
        if response.usage_metadata:
            usd, entry = cost(model, response.usage_metadata)
            entry["what"] = what
        return response
    finally:
        LEDGER.settle(held, usd, entry)


def ask(
    model: str, role: str, task: str, images: list, schema, what: str, thinking: str = "low",
    media: types.MediaResolution | None = None,
):
    cfg = types.GenerateContentConfig(
        system_instruction=role,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=types.ThinkingConfig(thinking_level=thinking),
        media_resolution=media,
    )
    contents = [*(part(i) for i in images), task]
    estimate = config.ESTIMATE_USD["judge" if "judge" in what else "analyst"]
    response = _call(model, contents, cfg, what, estimate)
    if response.parsed is None:
        raise RuntimeError(f"{what}: no structured answer ({response.text[:200] if response.text else 'empty'})")
    return response.parsed


def draw(
    model: str, role: str, task: str, images: list, size: tuple[int, int], what: str,
    thinking: str | None = None, resolution: str | None = None,
) -> np.ndarray:
    """One image back, resized to `size` (w, h) if the model returned another size.
    Interim thought images are skipped."""
    w, h = size
    cfg = types.GenerateContentConfig(
        system_instruction=role,
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=_aspect(w, h),
            image_size=resolution or ("1K" if max(w, h) <= 1024 else "2K"),
        ),
        thinking_config=types.ThinkingConfig(thinking_level=thinking) if thinking else None,
    )
    contents = [*(part(i) for i in images), task]
    response = _call(model, contents, cfg, what, config.ESTIMATE_USD["paint"])
    found = None
    for cand in response.candidates or []:
        for p in (cand.content.parts if cand.content else None) or []:
            if p.inline_data is not None and p.inline_data.data and not p.thought:
                found = p.inline_data.data
    if found is None:
        reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
        raise RuntimeError(f"{what}: no image returned ({reason})")
    image = Image.open(io.BytesIO(found)).convert("RGB")
    if image.size != (w, h):
        image = image.resize((w, h), Image.LANCZOS)
    return np.asarray(image)


ASPECTS = {"1:1": 1, "4:5": 0.8, "5:4": 1.25, "3:4": 0.75, "4:3": 4 / 3, "2:3": 2 / 3, "3:2": 1.5,
           "9:16": 9 / 16, "16:9": 16 / 9}


def _aspect(w: int, h: int) -> str:
    return min(ASPECTS, key=lambda k: abs(np.log(ASPECTS[k] / (w / h))))
