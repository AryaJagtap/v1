"""Paid probe, kept small: one window of one sample, each painter variant once,
gated and judged. The analyst's brief is cached so it is paid for once.

    python experiments/nb_probe.py pattern.jpg 1 mask erased
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from seamfix import checker, imaging, plan, repair
from seamfix.agents import analyst, judge
from seamfix.budget import LEDGER
from seamfix.painters import Request
from seamfix.painters.nano_banana import NanoBanana

name, index, variants = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
stem = Path(name).stem
out = Path("experiments/out") / f"nb_{stem}"
out.mkdir(parents=True, exist_ok=True)
tile = imaging.load(Path("..") / name)
check = checker.check(tile.rgb)

cached = out / "brief.json"
if cached.exists():
    brief = json.loads(cached.read_text())
else:
    preview = imaging.mark_joins(imaging.tile_2x2(tile.image), check)
    brief = analyst.diagnose(tile.image, preview, check, tile.notes).model_dump()
    cached.write_text(json.dumps(brief, indent=1))
print("brief:", {k: brief[k] for k in ("kind", "difficulty", "style")})

difficulty = "hard" if brief["difficulty"] == "hard" else brief["difficulty"]
p = plan.make(tile.rgb, check.failing, difficulty)
win = p.windows[index]
before = repair.crop(tile.rgb, win)
print(f"window {win.box} band {p.band}")
for variant in variants:
    # variant: [model@]show[+resolution], e.g. erased+2K or gemini-3-pro-image@erased
    model, _, rest = variant.rpartition("@") if "@" in variant else ("", "", variant)
    show, _, resolution = rest.partition("+")
    painter = NanoBanana(model=model or NanoBanana().model, show=show, resolution=resolution or None)
    request = Request(before, win.mask, win.axes, brief, reference=tile.rgb)
    raw_path = out / f"raw_{painter.name.replace(':', '_')}_w{index}.png"
    if raw_path.exists():  # paid once already: replay it
        painter.paint = lambda _r, p=raw_path: np.asarray(Image.open(p).convert("RGB"))
    else:
        paint = painter.paint
        def painter_paint(r, paint=paint, p=raw_path):
            arr = paint(r)
            Image.fromarray(arr).save(p)
            return arr
        painter.paint = painter_paint
    cand = repair.attempt(tile.rgb, p, win, painter, 1, request)
    judge.panel(before, cand.after, win.mask).save(out / f"panel_{painter.name.replace(':', '_')}_w{index}.png")
    verdict = judge.inspect(before, cand.after, win.mask, win.axes, brief)
    tag = f"{painter.name.replace(':', '_')}"
    ys, xs = judge.focus(win.mask)
    Image.fromarray(np.concatenate([before[ys, xs], cand.after[ys, xs], cand.raw[ys, xs]], axis=1)).save(out / f"{tag}_w{index}.png")
    print(f"{painter.name}: gates ok={cand.gates.ok} "
          + " ".join(f"{g.name}={g.value}{'' if g.ok else '!'}" for g in cand.gates.items)
          + f" fit={cand.fit.as_dict() if cand.fit else None}")
    print(f"  judge accept={verdict.accept} join_visible={verdict.join_visible} c/f/f={verdict.continuity}/{verdict.fidelity}/{verdict.finish} "
          f"defects={[(d.kind, d.where, d.severity) for d in verdict.defects]} fix={verdict.fix!r}")
print(f"spent ${LEDGER.spent:.3f} of ${LEDGER.cap:.2f}")
