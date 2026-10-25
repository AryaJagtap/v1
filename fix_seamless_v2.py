"""Check a repeat tile and, if it is not seamless, repair ONLY the mismatched spots with gpt-image-2.5-flare.

v2 changes (to remove ghosting / overlapping motifs):
  * Only the sections of the join that actually fail the check are repainted
    (not the whole cross), so correct artwork is never regenerated.
  * Hard-edged paste-back (tiny feather) - no cross-fading of two slightly
    misaligned drawings, which is what creates doubled outlines.
  * Colour of the generated patch is matched to the artwork around it.
  * Native resolution by default (no down/up-scaling blur).
  * Several candidates per attempt; the one with the best seam score is kept.
  * Short, concrete prompt.

Setup:
    pip install openai numpy pillow
    Put your key in API_KEY below.

Usage:
    python fix_seamless.py flower.jpeg
    python fix_seamless.py flower.jpeg --candidates 3 --strip 0.10 -o fixed.jpg
"""
import argparse
import base64
import io
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

API_KEY = "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"
BASE_URL = None  # only change if your key belongs to a gateway/provider other than OpenAI
MODEL = "gpt-image-2.5-flare"
FINE_SEGMENTS = 40  # how finely the join is scanned to find the bad spots

PROMPT = (

    "This image is a repeat tile for a print design (textile or wallpaper style). "
    "The transparent cross-shaped area is where the tile edges were joined and the pattern "
    "does not line up. Repaint ONLY that transparent area so the design flows continuously "
    "across it in every direction: complete motifs that are cut off, align mismatched ones, "
    "and fill gaps by reusing the motifs, shapes, line work, textures, colours, contrast and "
    "scale already present in this image. Do not add new kinds of objects, new colours, text, "
    "borders, frames, shadows or watermarks, and leave no visible line, blur or colour shift "
    "along the join. Do not change anything outside the transparent area."
)


# ---------- seam check (same logic as check_seamless.py) ----------
def check_seam(arr, segments):
    length, n, _ = arr.shape
    c = n // 2
    seam = np.abs(arr[:, c - 1].astype(np.int16) - arr[:, c].astype(np.int16)).mean(axis=1)

    base_sum = np.zeros(length, dtype=np.float64)
    count = 0
    for j in range(n - 1):
        if j == c - 1:
            continue
        base_sum += np.abs(arr[:, j + 1].astype(np.int16) - arr[:, j].astype(np.int16)).mean(axis=1)
        count += 1
    baseline = base_sum / max(count, 1)

    floor = 1.0
    overall = seam.mean() / max(baseline.mean(), floor)
    results = []
    for idx in np.array_split(np.arange(length), segments):
        results.append((int(idx[0]), int(idx[-1]),
                        float(seam[idx].mean() / max(baseline[idx].mean(), floor))))
    return overall, results


def seam_report(img, threshold, segments):
    a = np.tile(np.array(img.convert("RGB")), (2, 2, 1))
    v_overall, v_segs = check_seam(a, segments)
    h_overall, h_segs = check_seam(a.transpose(1, 0, 2), segments)
    v_worst = max(s[2] for s in v_segs)
    h_worst = max(s[2] for s in h_segs)
    limit = threshold * 2
    ok = v_overall <= threshold and h_overall <= threshold and v_worst <= limit and h_worst <= limit
    score = max(v_overall, h_overall, v_worst / 2, h_worst / 2)  # lower is better
    text = (f"up-down {v_overall:.2f} (worst {v_worst:.2f}), "
            f"left-right {h_overall:.2f} (worst {h_worst:.2f})")
    return ok, score, text


# ---------- repair ----------
def png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def model_size(cw, ch, max_side):
    """WxH for the API: multiples of 16, within assumed model limits
    (min ~655k pixels, max 3840 long edge / ~8.3M pixels). Adjust if the API rejects it."""
    scale = min(1.0, max_side / max(cw, ch))
    if cw * ch * scale * scale < 655_360:
        scale = (655_360 / (cw * ch)) ** 0.5
    if cw * ch * scale * scale > 8_294_400:
        scale = (8_294_400 / (cw * ch)) ** 0.5
    mw = max(16, int(round(cw * scale / 16)) * 16)
    mh = max(16, int(round(ch * scale / 16)) * 16)
    return mw, mh


def flagged_ranges(R, threshold):
    """Find the stretches of the join that fail. R is the tile rolled so the join is at its centre."""
    _, v_segs = check_seam(R, FINE_SEGMENTS)
    _, h_segs = check_seam(R.transpose(1, 0, 2), FINE_SEGMENTS)
    v_bad = [(s, e) for s, e, r in v_segs if r > threshold]
    h_bad = [(s, e) for s, e, r in h_segs if r > threshold]
    if not v_bad and not h_bad:  # fall back to the 3 worst stretches overall
        worst = sorted([("v", s, e, r) for s, e, r in v_segs] +
                       [("h", s, e, r) for s, e, r in h_segs], key=lambda t: -t[3])[:3]
        v_bad = [(s, e) for d, s, e, r in worst if d == "v"]
        h_bad = [(s, e) for d, s, e, r in worst if d == "h"]
    return v_bad, h_bad


def repair(img, client, strip, args):
    """Returns (score, tile, ok, text) for the best of args.candidates results."""
    a = np.array(img.convert("RGB"))
    h, w, _ = a.shape

    # Wrapped canvas centred on the tile join, with extra context on every side.
    big = np.tile(a, (2, 2, 1))
    mx, my = w // 4, h // 4
    wx0, wy0 = w - w // 2, h - h // 2          # puts the join exactly at the window centre
    x0, y0 = wx0 - mx, wy0 - my
    cw, ch = w + 2 * mx, h + 2 * my
    canvas = big[y0:y0 + ch, x0:x0 + cw]
    sx, sy = w - x0, h - y0                    # join position inside the canvas
    R = canvas[my:my + h, mx:mx + w]

    # Mask only the stretches of the join that actually fail.
    v_bad, h_bad = flagged_ranges(R, args.threshold)
    print(f"  Repainting {len(v_bad)} up-down and {len(h_bad)} left-right section(s) of the join")
    sw, sh = max(4, int(w * strip / 2)), max(4, int(h * strip / 2))
    pad = max(24, min(w, h) // 40)
    m = np.zeros((ch, cw), np.uint8)
    for s, e in v_bad:
        m[max(0, s + my - pad):e + my + pad + 1, sx - sw:sx + sw] = 255
    for s, e in h_bad:
        m[sy - sh:sy + sh, max(0, s + mx - pad):e + mx + pad + 1] = 255

    # Send to the model (transparent = area to repaint).
    mw, mh = model_size(cw, ch, args.max_side)
    small = Image.fromarray(canvas).resize((mw, mh), Image.LANCZOS)
    alpha = ImageOps.invert(Image.fromarray(m).resize((mw, mh), Image.NEAREST))
    mask_rgba = Image.new("RGBA", (mw, mh), (0, 0, 0, 255))
    mask_rgba.putalpha(alpha)

    resp = client.images.edit(
        model=MODEL,
        image=("tile.png", png_bytes(small), "image/png"),
        mask=("mask.png", png_bytes(mask_rgba), "image/png"),
        prompt=PROMPT,
        size=f"{mw}x{mh}",
        quality=args.quality,
        n=args.candidates,
    )

    # Hard-edged paste of the patch only (tiny feather), plus colour match from the ring around it.
    mask_img = Image.fromarray(m)
    soft = np.array(mask_img.filter(ImageFilter.GaussianBlur(args.feather)),
                    dtype=np.float32)[..., None] / 255.0
    ring = (np.array(mask_img.filter(ImageFilter.MaxFilter(31))) > 0) & (m == 0)

    best = None
    for item in resp.data:
        gen = Image.open(io.BytesIO(base64.b64decode(item.b64_json))).convert("RGB")
        g = np.array(gen.resize((cw, ch), Image.LANCZOS)).astype(np.float32)
        if ring.sum() > 100:
            g = g + (canvas[ring].astype(np.float32).mean(0) - g[ring].mean(0))
        out = canvas.astype(np.float32) * (1 - soft) + g * soft
        tile = Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)[my:my + h, mx:mx + w])
        ok, score, text = seam_report(tile, args.threshold, args.segments)
        if best is None or score < best[0]:
            best = (score, tile, ok, text)
    return best


def save(im, path):
    if path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(path, quality=100, subsampling=0)
    else:
        im.save(path)


def main():
    p = argparse.ArgumentParser(description="Check a tile and repair it if it is not seamless.")
    p.add_argument("input", help="Path to the single tile image")
    p.add_argument("-o", "--output", help="Output path (default: <input>_fixed.<ext>, or <input>_seamless.<ext> if already seamless)")
    p.add_argument("--threshold", type=float, default=1.1, help="Seam check threshold (default 1.1)")
    p.add_argument("--segments", type=int, default=20, help="Sections per seam for the pass/fail check (default 20)")
    p.add_argument("--strip", type=float, default=0.12, help="Width of each repainted spot as a fraction of the tile (default 0.12)")
    p.add_argument("--retries", type=int, default=1, help="Extra attempts with a wider strip if still not seamless (default 1)")
    p.add_argument("--candidates", type=int, default=2, help="Images requested per attempt; the best is kept (default 2)")
    p.add_argument("--feather", type=float, default=2.0, help="Edge softness of the paste-back in px (default 2; keep small)")
    p.add_argument("--quality", default="high", choices=["auto", "low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-side", type=int, default=3840, help="Longest side sent to the model, max 3840 (default 3840)")
    args = p.parse_args()
    args.max_side = min(args.max_side, 3840)

    src = Path(args.input)
    img = Image.open(src).convert("RGB")
    w, h = img.size

    ok, _, text = seam_report(img, args.threshold, args.segments)
    print(f"Check: {text}")
    if ok:
        out_path = Path(args.output) if args.output else src.with_name(f"{src.stem}_seamless{src.suffix}")
        shutil.copyfile(src, out_path)
        print(f"YES - the design pattern is seamless. Original returned: {out_path}")
        return 0

    print("NO - the design pattern is not seamless. Repairing with " + MODEL + " ...")
    if API_KEY == "input ur api key":
        sys.exit("Put your API key in API_KEY at the top of the script first.")

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    best = None  # (score, image, ok)
    for attempt in range(args.retries + 1):
        strip = min(0.4, args.strip + 0.08 * attempt)
        print(f"Attempt {attempt + 1}: strip {strip:.0%}, {args.candidates} candidate(s) ...")
        try:
            score, fixed, ok, text = repair(img, client, strip, args)
        except Exception as e:
            print(f"  API call failed: {e}")
            continue
        print(f"  Best candidate: {text} -> {'seamless' if ok else 'still a mismatch'}")
        if best is None or score < best[0]:
            best = (score, fixed, ok)
        if ok:
            break

    if best is None:
        sys.exit("No image was generated.")

    score, fixed, ok = best
    assert fixed.size == (w, h)
    out_path = Path(args.output) if args.output else src.with_name(f"{src.stem}_fixed{src.suffix}")
    save(fixed, out_path)
    print(f"Saved {w}x{h} image to {out_path}")
    print("YES - the repaired design is seamless." if ok else
          "NO - the repaired design still does not pass the check (best attempt saved).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
