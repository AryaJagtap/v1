"""Before and every candidate of one window at the same pixels, zoomed, from the
raw painter outputs, so a comparison never lines up different regions.

    python experiments/compare.py pattern.jpg 1 380:680 out.png raw_a.png raw_b.png
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from seamfix import checker, composite, imaging, plan, repair

name, index, rows, dest, *raws = sys.argv[1:]
r0, r1 = map(int, rows.split(":"))
tile = imaging.load(Path("..") / name)
check = checker.check(tile.rgb)
brief = json.loads((Path("experiments/out") / f"nb_{Path(name).stem}" / "brief.json").read_text())
p = plan.make(tile.rgb, check.failing, brief["difficulty"])
win = p.windows[int(index)]
before = repair.crop(tile.rgb, win)
cols = np.nonzero(win.mask.any(axis=0))[0]
c0, c1 = max(0, cols.min() - 80), min(win.size, cols.max() + 81)
panels = [("before", before)]
for raw in raws:
    painted = np.asarray(Image.open(raw).convert("RGB"))
    fitted, _ = composite.fit(painted, before, win.mask)
    panels.append((Path(raw).stem.removeprefix("raw_")[-28:], composite.blend(before, fitted, win.mask)))
zoomed = []
for label, arr in panels:
    sub = Image.fromarray(arr[r0:r1, c0:c1])
    z = sub.resize((sub.width * 2, sub.height * 2), Image.LANCZOS)
    d = ImageDraw.Draw(z)
    for x in (cols.min() - c0, cols.max() + 1 - c0):
        d.line([(2 * x, 0), (2 * x, 10)], fill=(0, 200, 0), width=3)
    d.text((6, 6), label, fill="black")
    zoomed.append(z)
sheet = Image.new("RGB", (sum(z.width + 8 for z in zoomed), zoomed[0].height), "white")
x = 0
for z in zoomed:
    sheet.paste(z, (x, 0))
    x += z.width + 8
sheet.save(dest)
print(sheet.size)
