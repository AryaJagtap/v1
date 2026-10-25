"""Every knob in one place. Keys come from the environment or the repo's .env,
never from code, and are never printed."""

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent  # seamfix/
REPO = ROOT.parent  # v1/, where the sample designs live
RUNS = Path(os.environ.get("SEAMFIX_RUNS", ROOT / "runs"))
CACHE = Path(os.environ.get("SEAMFIX_CACHE", Path.home() / ".cache" / "seamfix"))


def load_env(path: Path = REPO / ".env") -> None:
    """Fill os.environ from .env without overriding what the shell already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env()

# the seam check the user confirmed: ratio of the jump across a join to the
# typical neighbour jump, floored so flat areas do not explode
THRESHOLD = 1.1
FLOOR = 1.0
SEGMENTS = 20

# a join this far gone along at least half its length has edges that were
# never related: class C, a wide band along the whole join
UNRELATED_OVERALL = 2.0

# a repeat found inside the image must match its shifted self this closely,
# as mean jump over the typical neighbour jump. noise from JPEG sits near 0.3
REPEAT_MATCH = 0.6

# repaint band across the join, as a share of the side it cuts into
BAND_SHARE = {"easy": 0.04, "medium": 0.055, "hard": 0.18}
BAND_MIN, BAND_MAX = 24, 400
# a window is cut at native resolution, never scaled down, so nothing blurs
WINDOW = 1024
FEATHER = 4  # px ramp inside the band edge; outside the band nothing changes

# attempts per window: free LaMa first, then Nano Banana with the judge's notes
ATTEMPTS = int(os.environ.get("SEAMFIX_ATTEMPTS", "3"))
ROUNDS = 2  # whole-tile re-checks, step 8 of the workflow

PAINT_MODEL = os.environ.get("SEAMFIX_PAINT_MODEL", "gemini-nano-banana-2.1")
ANALYST_MODEL = os.environ.get("SEAMFIX_ANALYST_MODEL", "gemini-3.8-flash")
JUDGE_MODEL = os.environ.get("SEAMFIX_JUDGE_MODEL", "gemini-3.1-pro-preview")

# a hard cap across every run, kept in runs/ledger.json. no call starts that
# could cross it
BUDGET_USD = float(os.environ.get("SEAMFIX_BUDGET_USD", "1.50"))

# USD per million tokens: (input, text output, image output). Published rates
# where known, otherwise the dearest sibling, so the ledger over-counts
PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-nano-banana-2.1": (0.50, 3.00, 60.00),
    "gemini-3.1-flash-image": (0.50, 3.00, 60.00),
    "gemini-3-pro-image": (2.00, 12.00, 120.00),
    "gemini-3.8-flash": (0.50, 3.00, 0.0),
    "gemini-3.5-flash": (0.50, 3.00, 0.0),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.0),
}
# what one call is assumed to cost before it runs, for the reservation
ESTIMATE_USD = {"paint": 0.08, "judge": 0.04, "analyst": 0.01}

LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_PATH = Path(os.environ.get("LAMA_MODEL", CACHE / "big-lama.pt"))


def gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return key
