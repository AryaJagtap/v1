"""Check a repeat tile and, if it is not seamless, repair ONLY the worst join spots with
GPT (gpt-image-2.5-flare) or Gemini (gemini-nano-banana-2.1).

Why this version exists
  * Retrying with a WIDER strip (old script: 15% -> 25% -> 35% -> 40%) replaces more and more
    of the original artwork, and each new strip is a new chance to misalign. More attempts made
    the result worse. Here the repaint band stays narrow (96px). A retry re-rolls the same
    narrow spot instead of widening it.
  * Only genuinely bad spots are repainted (--spot-threshold), worst first. Artwork that is
    already seamless is never touched.
  * Each spot is repaired in its own 1:1 crop (1024px), not the whole canvas.
  * Gemini has no mask input, so the band is painted solid magenta INSIDE the image, which
    Gemini follows far better than a separate mask picture. GPT gets a real mask.
  * Every generated patch is checked before it is used: the area just outside the band must
    still match the original (catches Gemini shifting or re-rendering the picture), no magenta
    may be left, and the seam must improve. Otherwise the original pixels are kept and the spot
    is retried (--tries).
  * --dry-run previews the spots for free; --debug-dir saves every crop/mask/model output.

Setup:
    pip install openai google-genai numpy pillow scipy
    Keys: set environment variables OPENAI_API_KEY / GEMINI_API_KEY (recommended), e.g. in
    PowerShell:   $env:OPENAI_API_KEY="..."   $env:GEMINI_API_KEY="..."
    (or paste them into API_KEY / GEMINI_API_KEY below - but never share the file)

Usage:
    python seamless_gemini.py pattern.jpg --dry-run
    python seamless_gemini.py pattern.jpg --provider gemini --debug-dir debug
    python seamless_gemini.py pattern.jpg --provider gpt --floor 3
"""
import argparse
import base64
import io
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter

API_KEY = os.environ.get("OPENAI_API_KEY", "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")
BASE_URL = None  # only change if your key belongs to a gateway/provider other than OpenAI
MODEL = "gpt-image-2.5-flare"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "Q.Ab8RN6IgVotnM2C1GLeSlpXPyYi4LX69XRCYsbdFSnrhGpYBcw")
GEMINI_MODEL = "gemini-nano-banana-2.1"

FINE = 64      # how finely the join is scanned to find the bad spots
FLOOR = 1.0    # minimum "normal pixel change" used by the seam test (--floor)
MARKER = (255, 0, 255)  # colour painted over the repaint band for Gemini

GPT_PROMPT = (
    "Repeat tile of a print design. Fill the transparent band so the artwork continues "
    "straight across it: extend the outlines, shapes, fills and textures that touch the band's "
    "edges until they meet their counterparts on the other side, in exactly the same drawing "
    "style, line weight, colours and texture as the neighbouring artwork. Add no new motifs. "
    "Change nothing outside the band."
)

GEMINI_PROMPT = (
    "This is a repeat tile of a print design. The area filled with solid magenta (#FF00FF) is a "
    "band where two tile edges were joined and the artwork does not line up. Replace the magenta "
    "completely: extend the outlines, shapes, fills and textures that touch the band until they "
    "meet their counterparts on the other side, in exactly the same drawing style, line weight, "
    "colours and texture as the artwork around it. Add no new motifs. Do not change anything "
    "outside the magenta area; keep the exact same framing, size and position of everything. "
    "No magenta may remain. Return only the image."
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

    overall = seam.mean() / max(baseline.mean(), FLOOR)
    results = []
    for idx in np.array_split(np.arange(length), segments):
        results.append((int(idx[0]), int(idx[-1]),
                        float(seam[idx].mean() / max(baseline[idx].mean(), FLOOR))))
    return overall, results


def seam_report(img, threshold, segments):
    a = np.tile(np.array(img.convert("RGB")), (2, 2, 1))
    v_overall, v_segs = check_seam(a, segments)
    h_overall, h_segs = check_seam(a.transpose(1, 0, 2), segments)
    v_worst = max(s[2] for s in v_segs)
    h_worst = max(s[2] for s in h_segs)
    limit = threshold * 2
    ok = v_overall <= threshold and h_overall <= threshold and v_worst <= limit and h_worst <= limit
    text = (f"up-down {v_overall:.2f} (worst {v_worst:.2f}), "
            f"left-right {h_overall:.2f} (worst {h_worst:.2f})")
    return ok, text


# ---------- spot finding ----------
def find_spots(Rt, spot_thr, max_spots, crop):
    """Rt is the tile rolled by half, so the original join sits in the centre cross.
    Returns [(axis, start, end, ratio)], axis 'v' = up-down join, 'h' = left-right join."""
    _, v = check_seam(Rt, FINE)
    _, hz = check_seam(Rt.transpose(1, 0, 2), FINE)

    spots = []
    for axis, segs in (("v", v), ("h", hz)):
        run = None
        for s, e, r in segs:
            if r > spot_thr:
                if run:
                    run[1], run[2] = e, max(run[2], r)
                else:
                    run = [s, e, r]
            elif run:
                spots.append((axis, *run))
                run = None
        if run:
            spots.append((axis, *run))

    if not spots:  # overall check fails but nothing is extreme: take the 3 worst stretches
        allsegs = [("v", s, e, r) for s, e, r in v] + [("h", s, e, r) for s, e, r in hz]
        spots = sorted(allsegs, key=lambda t: -t[3])[:3]

    out = []
    maxlen = max(64, int(crop * 0.6))
    for axis, s, e, r in spots:  # split very long spots so each fits in one crop
        n = max(1, -(-(e - s + 1) // maxlen))
        edges = np.linspace(s, e + 1, n + 1).astype(int)
        for i in range(n):
            out.append((axis, int(edges[i]), int(edges[i + 1] - 1), r))
    out.sort(key=lambda t: -t[3])
    return out[:max_spots]


# ---------- providers ----------
def png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def gpt_generate(client, crop, mask, args):
    """crop: (S,S,3) uint8, mask: (S,S) bool (True = repaint). Returns a PIL RGB image."""
    S = crop.shape[0]
    api = 1024 if S < 1024 else S
    send = Image.fromarray(crop)
    alpha = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8))  # transparent = repaint
    if api != S:
        send, alpha = send.resize((api, api), Image.LANCZOS), alpha.resize((api, api), Image.BILINEAR)
    mask_rgba = Image.new("RGBA", (api, api), (0, 0, 0, 255))
    mask_rgba.putalpha(alpha)
    resp = client.images.edit(
        model=MODEL,
        image=("tile.png", png_bytes(send), "image/png"),
        mask=("mask.png", png_bytes(mask_rgba), "image/png"),
        prompt=GPT_PROMPT,
        size=f"{api}x{api}",
        quality=args.quality,
    )
    return Image.open(io.BytesIO(base64.b64decode(resp.data[0].b64_json))).convert("RGB")


def gemini_generate(client, crop, mask, args):
    from google.genai import types
    S = crop.shape[0]
    marked = crop.copy()
    marked[mask] = MARKER  # Gemini has no mask input, so mark the band inside the image
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[GEMINI_PROMPT, Image.fromarray(marked)],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="1:1",
                image_size="1K" if S <= 1024 else "2K" if S <= 2048 else "4K"),
        ),
    )
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else None) or []:
            if part.inline_data is not None and part.inline_data.data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    raise RuntimeError("Gemini returned no image.")


# ---------- repair ----------
def estimate_shift(orig, g, valid, max_shift=24):
    """Find the whole-pixel (dy, dx) so that np.roll(g, (dy, dx)) best matches orig over `valid`.
    Image models often return the picture shifted by a few pixels; undoing that is what makes
    the patch line up with the surrounding artwork. Returns (0, 0) if no clear shift exists."""
    S = orig.shape[0]
    go = gaussian_filter(orig.mean(-1), 1.5)
    gg = gaussian_filter(g.mean(-1), 1.5)
    ys, xs = np.nonzero(valid)
    if len(ys) > 120_000:
        pick = np.random.default_rng(0).choice(len(ys), 120_000, replace=False)
        ys, xs = ys[pick], xs[pick]
    ref = go[ys, xs]

    def err(dy, dx):  # np.roll(gg, (dy, dx))[y, x] == gg[y - dy, x - dx]
        return float(np.abs(ref - gg[(ys - dy) % S, (xs - dx) % S]).mean())

    zero = err(0, 0)
    best = min(((err(dy, dx), dy, dx)
                for dy in range(-max_shift, max_shift + 1, 2)
                for dx in range(-max_shift, max_shift + 1, 2)))
    _, by, bx = best
    best = min(((err(by + dy, bx + dx), by + dy, bx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)))
    e, dy, dx = best
    return (dy, dx) if (dy, dx) != (0, 0) and e < zero * 0.9 else (0, 0)


def crop_geometry(shape, spot, args):
    h, w = shape[:2]
    axis, s, e, _ = spot
    S = max(64, (min(args.crop, h, w) // 32) * 32)
    cy, cx = ((s + e) // 2, w // 2) if axis == "v" else (h // 2, (s + e) // 2)
    rows = (np.arange(S) + cy - S // 2) % h   # modulo = wraps around the tile edge
    cols = (np.arange(S) + cx - S // 2) % w
    half = args.band // 2
    L0 = max(0, S // 2 - (e - s) // 2 - args.pad)
    L1 = min(S, S // 2 + (e - s) // 2 + 1 + args.pad)
    m = Image.new("L", (S, S), 0)
    box = (S // 2 - half, L0, S // 2 + half, L1) if axis == "v" else (L0, S // 2 - half, L1, S // 2 + half)
    ImageDraw.Draw(m).rounded_rectangle(box, radius=half, fill=255)
    return S, rows, cols, np.array(m) > 127, (L0, L1)


def local_ratio(crop, axis, span):
    oc = crop if axis == "v" else crop.transpose(1, 0, 2)
    return check_seam(oc[span[0]:span[1]], 1)[0]


def repair_spot(Rt, spot, generate, args, idx):
    axis, s, e, ratio = spot
    S, rows, cols, mask, span = crop_geometry(Rt.shape, spot, args)
    ix = np.ix_(rows, cols)
    crop = Rt[ix].copy()
    orig = crop.astype(np.float32)
    base = local_ratio(crop, axis, span)
    name = f"{'up-down' if axis == 'v' else 'left-right'} {s}-{e}"

    dbg = Path(args.debug_dir) if args.debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(dbg / f"spot{idx:03d}_in.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(dbg / f"spot{idx:03d}_mask.png")

    dist = distance_transform_edt(~mask)
    ring = (dist > 0) & (dist <= 24)
    ringf = ring.astype(np.float32)
    soft = np.clip(gaussian_filter(mask.astype(np.float32), args.feather), 0, 1)[..., None]
    sigma = max(8.0, args.band / 3)
    den = gaussian_filter(ringf, sigma)[..., None]
    marker = np.array(MARKER, np.float32)

    for attempt in range(1, args.tries + 1):
        try:
            gen = generate(crop, mask, args)
        except Exception as ex:
            print(f"  spot {idx} ({name}) try {attempt}: API call failed: {ex}")
            continue
        if gen.size != (S, S):
            gen = gen.resize((S, S), Image.LANCZOS)
        g = np.array(gen).astype(np.float32)
        if dbg:
            gen.save(dbg / f"spot{idx:03d}_try{attempt}_raw.png")

        # undo any whole-pixel shift the model introduced, so the patch lines up with its surroundings
        dy, dx = estimate_shift(orig, g, dist > 20)
        if (dy, dx) != (0, 0):
            g = np.roll(g, (dy, dx), (0, 1))
            print(f"  spot {idx} ({name}) try {attempt}: model output was shifted, realigned by dy={dy}, dx={dx}")

        # reject if marker colour is left in the band
        left = float((np.linalg.norm(g - marker, axis=-1)[mask] < 70).mean())

        # low-frequency colour correction so the patch matches the artwork around it
        diff = orig - g
        num = np.stack([gaussian_filter(diff[..., c] * ringf, sigma) for c in range(3)], -1)
        g = g + np.where(den > 0.02, num / np.maximum(den, 1e-3), 0)

        # how well does the area just outside the band still match the original?
        # (high = the model shifted / re-rendered the picture, so the patch would not line up)
        changed = float(np.abs(orig - g)[ring].mean() / 255)

        comp = np.clip(orig * (1 - soft) + g * soft + 0.5, 0, 255).astype(np.uint8)
        new = local_ratio(comp, axis, span)
        if dbg:
            Image.fromarray(comp).save(dbg / f"spot{idx:03d}_try{attempt}_result.png")

        problems = []
        if left > 0.01:
            problems.append(f"{left:.0%} magenta left")
        if changed > args.max_change:
            problems.append(f"surroundings changed {changed:.1%} > {args.max_change:.0%} (misaligned)")
        if new >= base * 0.95:
            problems.append("seam did not improve")
        status = "accepted" if not problems else "rejected: " + "; ".join(problems)
        print(f"  spot {idx} ({name}) try {attempt}: local seam {base:.2f} -> {new:.2f}, "
              f"surroundings changed {changed:.1%} -> {status}")
        if not problems:
            Rt[ix] = comp
            return True
    print(f"  spot {idx} ({name}): kept original pixels")
    return False


def spots_overlay(Rt, spots, args, path):
    im = Image.fromarray(Rt).convert("RGB")
    d = ImageDraw.Draw(im)
    h, w = Rt.shape[:2]
    half, t = args.band // 2, max(3, w // 400)
    for axis, s, e, r in spots:
        if axis == "v":
            d.rounded_rectangle((w // 2 - half, s - args.pad, w // 2 + half, e + args.pad), half, outline="red", width=t)
        else:
            d.rounded_rectangle((s - args.pad, h // 2 - half, e + args.pad, h // 2 + half), half, outline="red", width=t)
    im.save(path)


def save(im, path):
    if path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(path, quality=100, subsampling=0)
    else:
        im.save(path)


def run(img, generate, args):
    """Returns (fixed_tile_image, ok, text)."""
    a = np.array(img.convert("RGB"))
    h, w, _ = a.shape
    Rt = np.roll(a, (h // 2, w // 2), (0, 1))  # original joins now sit in the centre cross
    fixed, ok, text = img, False, ""
    for rnd in range(args.rounds):
        spots = find_spots(Rt, args.spot_threshold, args.max_spots, args.crop)
        print(f"Round {rnd + 1}: {len(spots)} spot(s) to repair")
        improved = False
        for i, sp in enumerate(spots, 1):
            improved |= repair_spot(Rt, sp, generate, args, i + rnd * 100)
        fixed = Image.fromarray(np.roll(Rt, (-(h // 2), -(w // 2)), (0, 1)))
        ok, text = seam_report(fixed, args.threshold, args.segments)
        print(f"  Re-check: {text} -> {'seamless' if ok else 'still a mismatch'}")
        if ok or not improved:
            break
    return fixed, ok, text


def ask_provider():
    while True:
        choice = input(f"Which model to use?\n  1) GPT    ({MODEL})\n  2) Gemini ({GEMINI_MODEL})\n"
                       "Enter 1/2 or gpt/gemini: ").strip().lower()
        if choice in ("1", "gpt"):
            return "gpt"
        if choice in ("2", "gemini"):
            return "gemini"
        print("Please enter 1, 2, gpt or gemini.")


def main():
    global FLOOR
    p = argparse.ArgumentParser(description="Check a tile and repair its worst join spots if it is not seamless.")
    p.add_argument("input", help="Path to the single tile image")
    p.add_argument("-o", "--output", help="Output path (default: <gpt|gemini>_<fixed|unfix>_<input>.png, or <input>_seamless.<ext> if already seamless)")
    p.add_argument("--provider", choices=["gpt", "gemini"], help="Skip the question and use this model")
    p.add_argument("--threshold", type=float, default=1.1, help="Seam check threshold (default 1.1)")
    p.add_argument("--floor", type=float, default=1.0,
                   help="Smallest 'normal pixel change' the test divides by (default 1.0). Raise to 2-3 for soft/painterly "
                        "designs, where the default is stricter than the eye can see")
    p.add_argument("--segments", type=int, default=20, help="Sections per seam for the pass/fail check (default 20)")
    p.add_argument("--spot-threshold", type=float, default=1.8, help="Only join stretches above this ratio are repaired (default 1.8)")
    p.add_argument("--max-spots", type=int, default=10, help="Repair at most this many spots per round, worst first (default 10)")
    p.add_argument("--band", type=int, default=96, help="Width in px of the repainted band (default 96)")
    p.add_argument("--pad", type=int, default=32, help="Extra length in px added to each end of a spot (default 32)")
    p.add_argument("--crop", type=int, default=1024, help="Crop size in px sent to the model for each spot (default 1024)")
    p.add_argument("--tries", type=int, default=2, help="Attempts per spot, only used if an attempt is rejected (default 2)")
    p.add_argument("--rounds", type=int, default=2, help="Repair rounds, re-scanning after each (default 2)")
    p.add_argument("--max-change", type=float, default=0.08,
                   help="Reject a result if the area just outside the band changed by more than this fraction (default 0.08)")
    p.add_argument("--feather", type=float, default=3.0, help="Edge softness of the paste-back in px (default 3)")
    p.add_argument("--quality", default="high", choices=["auto", "low", "medium", "high", "xhigh", "max"], help="GPT only")
    p.add_argument("--dry-run", action="store_true", help="Show the spots that would be repaired; no API call")
    p.add_argument("--debug-dir", help="Save every crop, mask, model output and result here")
    args = p.parse_args()
    FLOOR = args.floor

    src = Path(args.input)
    img = Image.open(src).convert("RGB")
    w, h = img.size

    ok, text = seam_report(img, args.threshold, args.segments)
    print(f"Check: {text}")
    if ok:
        out_path = Path(args.output) if args.output else src.with_name(f"{src.stem}_seamless{src.suffix}")
        shutil.copyfile(src, out_path)
        print(f"YES - the design pattern is seamless. Original returned: {out_path}")
        return 0
    print("NO - the design pattern is not seamless.")

    if args.dry_run:
        a = np.array(img)
        Rt = np.roll(a, (h // 2, w // 2), (0, 1))
        spots = find_spots(Rt, args.spot_threshold, args.max_spots, args.crop)
        for axis, s, e, r in spots:
            print(f"  {'up-down' if axis == 'v' else 'left-right'} join {s}-{e}: ratio {r:.2f}")
        path = src.with_name(f"{src.stem}_spots.png")
        spots_overlay(Rt, spots, args, path)
        print(f"{len(spots)} spot(s) would be repaired ({len(spots)} image(s) requested per round, more only on rejects).")
        print(f"Preview saved to {path} (tile shifted by half so the joins are in the centre; red = repaint areas)")
        return 0

    provider = args.provider or ask_provider()
    if provider == "gemini":
        if GEMINI_API_KEY == "input ur gemini api key":
            sys.exit("Set GEMINI_API_KEY (environment variable or at the top of the script) first.")
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        generate = lambda crop, mask, a: gemini_generate(client, crop, mask, a)
        label = GEMINI_MODEL
    else:
        if API_KEY == "input ur api key":
            sys.exit("Set OPENAI_API_KEY (environment variable or at the top of the script) first.")
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        generate = lambda crop, mask, a: gpt_generate(client, crop, mask, a)
        label = MODEL

    print(f"Repairing with {label} ...")
    fixed, ok, text = run(img, generate, args)
    assert fixed.size == (w, h)
    status = "fixed" if ok else "unfix"
    out_path = Path(args.output) if args.output else src.with_name(f"{provider}_{status}_{src.stem}.png")
    save(fixed, out_path)
    print(f"Saved {w}x{h} image to {out_path}")
    print("YES - the repaired design is seamless." if ok else
          "NO - the repaired design still does not pass the check (result saved anyway).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
