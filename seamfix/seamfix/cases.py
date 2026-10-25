"""The edge cases, run through the whole workflow and laid out to be looked at.

    python -m seamfix.cases            # synthetic cases, free (LaMa, no paid call)
    python -m seamfix.cases --samples  # also the repo's sample designs, free

Writes test_cases/<case>/ with the input, every step's images and report.json,
and test_cases/index.html: for each case what it tests, what should happen,
what did, and the 2x2 before and after. A case passes when the outcome is one
it allows and its checks hold.
"""

import argparse
import filecmp
import html
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from seamfix import config, imaging, pipeline, synthetic

OUT = config.ROOT / "test_cases"
FREE = {"painters": ["lama"], "judge": False, "attempts": 1, "analyst": False}


@dataclass
class Case:
    name: str
    tests: str
    expect: tuple[str, ...]  # allowed outcomes
    make: object  # () -> (Image, filename)
    checks: list[str] = field(default_factory=list)


def _lattice(size: int = 48, cell: int = 12) -> Image.Image:
    a = np.full((size, size, 3), (214, 200, 176), np.uint8)
    for y in range(cell // 2, size, cell):
        for x in range(cell // 2, size, cell):
            a[y - 2 : y + 2, x - 2 : x + 2] = (122, 52, 50)
    return Image.fromarray(a)


def _tiny_break() -> Image.Image:
    """The lattice started so its dots straddle the left-right join, then the
    dots' right halves moved off their rows."""
    a = np.roll(np.asarray(_lattice()), 6, axis=1).copy()
    a[:, -3:] = np.roll(a[:, -3:], 5, axis=0)
    return Image.fromarray(a)


def _odd() -> Image.Image:
    # the same seamless tile, started where its joins run through quiet ground
    return Image.fromarray(np.roll(np.asarray(synthetic.seamless(333, 517)), (101, 77), axis=(0, 1)))


def _transparent() -> Image.Image:
    a = np.asarray(synthetic.break_lr(synthetic.seamless(256, 256)).convert("RGBA")).copy()
    a[:20, :20, 3] = 0
    return Image.fromarray(a)


def _exif(path: Path) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6
    synthetic.seamless(512, 512).save(path, exif=exif.tobytes(), quality=95)


base = synthetic.seamless
CASES = [
    Case("seamless", "a tile whose joins already pass", ("returned",),
         lambda: (base(), "seamless.png"), ["byte_identical"]),
    Case("left_right_break", "left edge against right edge broken over a stretch", ("fixed", "review"),
         lambda: (synthetic.break_lr(base()), "lr.png"), ["middle_untouched"]),
    Case("top_bottom_break", "top edge against bottom edge broken over a stretch", ("fixed", "review"),
         lambda: (synthetic.break_tb(base()), "tb.png"), ["middle_untouched"]),
    Case("corner_break", "both joins broken only where they cross: the corner painted once", ("fixed", "review"),
         lambda: (synthetic.break_corner(base()), "corner.png"), ["middle_untouched"]),
    Case("break_across_wrap", "a broken stretch that runs off the bottom and on at the top", ("fixed", "review"),
         lambda: (synthetic.break_lr(base(), start=0.9, length=0.2), "wrap.png"), ["middle_untouched"]),
    Case("over_cropped", "1.45 x 1.3 repeats in one image: cut to the repeat, no AI", ("cut",),
         lambda: (synthetic.over_cropped(base()), "over.png"), ["size_is_repeat"]),
    Case("half_drop", "a half-drop repeat: straight tile two repeats wide, built pixel for pixel", ("cut",),
         lambda: (synthetic.half_drop(base(300, 260)), "halfdrop.png"), []),
    Case("odd_size", "333 x 517, odd on both sides, left-right broken", ("fixed", "review"),
         lambda: (synthetic.break_lr(_odd()), "odd.png"), ["middle_untouched", "same_size"]),
    Case("tiny", "48 x 48 dot lattice, left-right broken. It also repeats every 12 rows: a cut may only shorten it, never hide the broken join",
         ("fixed", "review"),
         lambda: (_tiny_break(), "tiny.png"), ["same_size"]),
    Case("flat_colour", "one flat colour: the floor keeps the ratio sane", ("returned",),
         lambda: (Image.new("RGB", (200, 160), (190, 40, 60)), "flat.png"), ["byte_identical"]),
    Case("unrelated_edges", "artwork never made to repeat: wide band, whole joins", ("fixed", "review"),
         lambda: (synthetic.unrelated(), "unrelated.png"), ["middle_untouched"]),
    Case("palette_mode", "an indexed-colour tile stays indexed, on its own palette", ("fixed", "review"),
         lambda: (synthetic.break_lr(base(256, 256)).quantize(48), "palette.png"), ["mode_kept"]),
    Case("grayscale", "a grayscale tile stays grayscale", ("fixed", "review"),
         lambda: (synthetic.break_lr(base(256, 256)).convert("L"), "gray.png"), ["mode_kept"]),
    Case("cmyk", "a CMYK tile comes back CMYK, as TIFF", ("fixed", "review"),
         lambda: (synthetic.break_lr(base(256, 256)).convert("CMYK"), "cmyk.tif"), ["mode_kept"]),
    Case("rgba_opaque", "RGBA with every pixel opaque is repaired like RGB", ("fixed", "review"),
         lambda: (synthetic.break_lr(base(256, 256)).convert("RGBA"), "rgba.png"), ["mode_kept"]),
    Case("transparent", "transparent pixels: refused, nothing is painted", ("refused",),
         lambda: (_transparent(), "transparent.png"), []),
    Case("exif_rotated", "a JPEG shown rotated by its EXIF tag: checked upright, returned as is", ("returned",),
         None, ["byte_identical"]),
]


def run_case(case: Case, folder: Path) -> dict:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    if case.make is None:
        src = folder / "exif.jpg"
        _exif(src)
    else:
        image, name = case.make()
        src = folder / name
        image.save(src)
    result = pipeline.run(src, FREE, run_id="run", root=folder)
    run = folder / "run"
    checks = {}
    files = result.get("files", {})
    design = run / files["design"] if files.get("design") else None
    tile_in = imaging.load(src)
    if "byte_identical" in case.checks:
        checks["byte_identical"] = bool(design and filecmp.cmp(src, design, shallow=False))
    if design and result["status"] == "fixed":
        tile_out = imaging.load(design)
        if "mode_kept" in case.checks:
            checks["mode_kept"] = tile_out.image.mode == tile_in.image.mode
        if "same_size" in case.checks or "middle_untouched" in case.checks:
            checks["same_size"] = tile_out.size == tile_in.size
        if "middle_untouched" in case.checks:
            h, w = tile_in.rgb.shape[:2]
            m = max(config.BAND_MAX, w // 4, h // 4) // 2
            checks["middle_untouched"] = bool(np.array_equal(tile_in.rgb[m:-m, m:-m], tile_out.rgb[m:-m, m:-m])) if h > 2 * m and w > 2 * m else True
    if "size_is_repeat" in case.checks and design:
        checks["size_is_repeat"] = imaging.load(design).size == (512, 512)
    ok = result["status"] in case.expect and all(checks.values())
    return {"name": case.name, "tests": case.tests, "expect": case.expect, "status": result["status"],
            "reason": result.get("reason", ""), "checks": checks, "ok": ok,
            "input": src.name, "files": files, "windows": result.get("windows", []),
            "size_in": list(tile_in.size), "size_out": result.get("output_size")}


def run_sample(name: str) -> dict:
    folder = OUT / f"sample_{Path(name).stem}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    src = folder / name
    shutil.copyfile(config.REPO / name, src)
    result = pipeline.run(src, FREE, run_id="run", root=folder)
    return {"name": f"sample: {name}", "tests": "repo sample, free mode (LaMa only, no analyst)",
            "expect": ("any",), "status": result["status"], "reason": result.get("reason", ""),
            "checks": {}, "ok": True, "input": name, "files": result.get("files", {}),
            "windows": result.get("windows", []), "size_in": list(imaging.load(src).size),
            "size_out": result.get("output_size")}


def page(rows: list[dict]) -> str:
    def img(path: Path, alt: str) -> str:
        return f'<img src="{html.escape(str(path))}" alt="{html.escape(alt)}" loading="lazy">' if path.exists() else '<span class="none">—</span>'

    body = []
    for r in rows:
        key = r["name"].removeprefix("sample: ")
        folder = Path(f"sample_{Path(key).stem}") if r["name"].startswith("sample") else Path(key)
        run = folder / "run"
        first = "cut" if (OUT / run / "cut_2x2.png").exists() and r["status"] == "cut" else "input"
        after = run / ("fixed_2x2.png" if r["status"] == "fixed" else f"{'cut' if r['status'] == 'cut' else 'input'}_2x2.png")
        checks = " ".join(f'<span class="chip {"ok" if v else "bad"}">{html.escape(k)}</span>' for k, v in r["checks"].items())
        wins = sum(1 for w in r["windows"] if w.get("accepted"))
        body.append(f"""
<article class="{'pass' if r['ok'] else 'fail'}">
  <header><h2>{html.escape(r['name'])}</h2><span class="verdict">{'PASS' if r['ok'] else 'FAIL'}</span></header>
  <p class="tests">{html.escape(r['tests'])}</p>
  <dl><dt>expected</dt><dd>{html.escape(' or '.join(r['expect']))}</dd>
      <dt>outcome</dt><dd><b>{html.escape(r['status'])}</b> {html.escape(r['reason'])}</dd>
      <dt>size</dt><dd>{r['size_in'][0]}x{r['size_in'][1]} → {r['size_out'][0] if r['size_out'] else '?'}x{r['size_out'][1] if r['size_out'] else '?'}</dd>
      <dt>windows</dt><dd>{wins} accepted of {len(r['windows'])}</dd>
      <dt>checks</dt><dd>{checks or '—'}</dd></dl>
  <div class="pair"><figure>{img(OUT / run / 'input_2x2.png', 'input 2x2')}<figcaption>input, 2x2</figcaption></figure>
  <figure>{img(OUT / run / 'input_marked.png', 'failing joins')}<figcaption>failing stretches in red</figcaption></figure>
  <figure>{img(OUT / after, 'result 2x2')}<figcaption>result, 2x2 ({html.escape(first if r['status'] == 'cut' else r['status'])})</figcaption></figure></div>
  <p class="files"><a href="{html.escape(str(run))}/report.json">report.json</a> · <a href="{html.escape(str(run))}/events.jsonl">events.jsonl</a></p>
</article>""")
    passed = sum(r["ok"] for r in rows)
    return f"""<!doctype html><meta charset="utf-8"><title>seamfix test cases</title>
<style>
:root{{--ink:#1d1d1f;--mute:#6e6e73;--line:#e5e5ea;--ok:#1f8a4c;--bad:#c62828}}
body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,Inter,system-ui,sans-serif;color:var(--ink);background:#fafafa;margin:0;padding:32px}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--mute);margin:0 0 24px}}
article{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px;margin:0 0 16px}}
article.fail{{border-color:#f3c0c0}} header{{display:flex;justify-content:space-between;align-items:center}}
h2{{font-size:16px;margin:0}} .verdict{{font-weight:600;font-size:12px;letter-spacing:.04em}}
.pass .verdict{{color:var(--ok)}} .fail .verdict{{color:var(--bad)}} .tests{{color:var(--mute);margin:4px 0 12px}}
dl{{display:grid;grid-template-columns:90px 1fr;gap:2px 12px;margin:0 0 12px}} dt{{color:var(--mute)}} dd{{margin:0}}
.chip{{display:inline-block;border-radius:99px;padding:0 8px;font-size:12px;margin-right:4px}}
.chip.ok{{background:#e6f4ea;color:var(--ok)}} .chip.bad{{background:#fdecea;color:var(--bad)}}
.pair{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} figure{{margin:0}}
img{{width:100%;border-radius:8px;border:1px solid var(--line);image-rendering:auto}} figcaption{{color:var(--mute);font-size:12px;margin-top:4px}}
.files a{{color:var(--mute)}} .none{{color:var(--mute)}}
</style>
<h1>seamfix test cases</h1><p class="sub">{passed} of {len(rows)} pass · free mode: LaMa only, no paid call · regenerate with <code>python -m seamfix.cases</code></p>
{''.join(body)}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    for case in CASES:
        if args.only and case.name not in args.only:
            continue
        r = run_case(case, OUT / case.name)
        print(f"{'PASS' if r['ok'] else 'FAIL'} {case.name}: {r['status']} {r['checks']} {r['reason'][:80]}", flush=True)
        rows.append(r)
    if args.samples:
        for name in ["flower.jpeg", "abcd.jpg", "peacock.jpg", "sample.jpg", "test.jpg", "design.jpeg"]:
            r = run_sample(name)
            print(f"     sample {name}: {r['status']} {r['reason'][:80]}", flush=True)
            rows.append(r)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    (OUT / "index.html").write_text(page(rows))
    print(f"{sum(r['ok'] for r in rows)}/{len(rows)} pass -> {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
