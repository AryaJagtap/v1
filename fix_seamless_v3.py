"""Check a repeat tile and, if it is not seamless, repair ONLY the worst join spots with gpt-image-2.5-flare.

v3 - what changed and why
  * v2 compared every small section against the *overall* threshold (1.1) instead of the
    per-section limit, so it flagged more than half of each join and repainted huge
    rectangles. v3 flags only genuinely bad spots (--spot-threshold, default 1.8), worst first.
  * Each spot is repaired in its own 1:1 crop (default 1024px) instead of sending the whole
    3000px canvas to the model, so fine line art is not downscaled and re-invented.
  * The repaint area is a narrow rounded band along the join (default 96px), not a wide box,
    so there are no rectangular patch edges and far less original artwork is replaced.
  * The patch is colour-corrected (low frequency) to the artwork around it before it is pasted.
  * Candidates are ranked on seam match AND on how little the model changed the area just
    outside the band. A spot is only accepted if it actually improves the seam; otherwise the
    original pixels are kept.
  * --dry-run shows exactly which spots would be repaired (no API call, no cost).
  * --debug-dir saves every crop, mask, model output and result so you can inspect them.

Setup:
    pip install openai numpy pillow scipy
    Put your key in API_KEY below.

Usage:
    python fix_seamless.py pattern.jpg --dry-run            # free preview of the spots
    python fix_seamless.py pattern.jpg -o fixed.png --debug-dir debug
"""
import argparse
import base64
import io
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter

API_KEY = "input ur api key"
BASE_URL = None  # only change if your key belongs to a gateway/provider other than OpenAI
MODEL = "gpt-image-2.5-flare"
FINE = 64  # how finely the join is scanned to find the bad spots

PROMPT = (
    "Repeat tile of a paisley textile print. Fill the transparent band so the artwork continues "
    "straight across it: extend the outlines, fills and dot rows that touch the band's edges "
    "until they meet their counterparts on the other side, with the same line weight, colours "
    "and flat crisp print style as the neighbouring artwork, and the same fine background "
    "texture. Add no new motifs. Change nothing outside the band."
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


# ---------- repair ----------
def png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


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


def repair_spot(Rt, spot, client, args, idx):
    axis, s, e, ratio = spot
    S, rows, cols, mask, span = crop_geometry(Rt.shape, spot, args)
    ix = np.ix_(rows, cols)
    crop = Rt[ix].copy()
    orig = crop.astype(np.float32)
    base = local_ratio(crop, axis, span)

    api = 1024 if S < 1024 else S
    send = Image.fromarray(crop)
    alpha = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8))
    if api != S:
        send, alpha = send.resize((api, api), Image.LANCZOS), alpha.resize((api, api), Image.BILINEAR)
    mask_rgba = Image.new("RGBA", (api, api), (0, 0, 0, 255))
    mask_rgba.putalpha(alpha)

    dbg = Path(args.debug_dir) if args.debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(dbg / f"spot{idx:02d}_in.png")
        mask_rgba.save(dbg / f"spot{idx:02d}_mask.png")

    resp = client.images.edit(
        model=MODEL,
        image=("tile.png", png_bytes(send), "image/png"),
        mask=("mask.png", png_bytes(mask_rgba), "image/png"),
        prompt=PROMPT,
        size=f"{api}x{api}",
        quality=args.quality,
        n=args.candidates,
    )

    dist = distance_transform_edt(~mask)
    ring = (dist > 0) & (dist <= 24)
    ringf = ring.astype(np.float32)
    soft = np.clip(gaussian_filter(mask.astype(np.float32), args.feather), 0, 1)[..., None]
    sigma = max(8.0, args.band / 3)
    den = gaussian_filter(ringf, sigma)[..., None]

    best = None
    for k, item in enumerate(resp.data):
        gen = Image.open(io.BytesIO(base64.b64decode(item.b64_json))).convert("RGB")
        if gen.size != (S, S):
            gen = gen.resize((S, S), Image.LANCZOS)
        g = np.array(gen).astype(np.float32)
        changed = float(np.abs(orig - g)[ring].mean() / 255)  # how much the model altered the surroundings

        # low-frequency colour correction so the patch matches the artwork around it
        diff = orig - g
        num = np.stack([gaussian_filter(diff[..., c] * ringf, sigma) for c in range(3)], -1)
        g = g + np.where(den > 0.02, num / np.maximum(den, 1e-3), 0)

        comp = np.clip(orig * (1 - soft) + g * soft + 0.5, 0, 255).astype(np.uint8)
        new = local_ratio(comp, axis, span)
        score = new + 4 * changed
        if dbg:
            Image.fromarray(comp).save(dbg / f"spot{idx:02d}_cand{k}.png")
        if best is None or score < best[0]:
            best = (score, new, changed, comp)

    _, new, changed, comp = best
    accepted = new < base * 0.95
    if accepted:
        Rt[ix] = comp
    print(f"  spot {idx} ({'up-down' if axis == 'v' else 'left-right'} {s}-{e}): "
          f"local seam {base:.2f} -> {new:.2f}, surroundings changed {changed:.1%} -> "
          f"{'accepted' if accepted else 'kept original (no improvement)'}")
    return accepted


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


def run(img, client, args):
    """Returns (fixed_tile_image, ok, text)."""
    a = np.array(img.convert("RGB"))
    h, w, _ = a.shape
    Rt = np.roll(a, (h // 2, w // 2), (0, 1))  # original joins now sit in the centre cross
    for rnd in range(args.rounds):
        spots = find_spots(Rt, args.spot_threshold, args.max_spots, args.crop)
        print(f"Round {rnd + 1}: {len(spots)} spot(s) to repair")
        improved = False
        for i, sp in enumerate(spots, 1):
            improved |= repair_spot(Rt, sp, client, args, i + rnd * 100)
        fixed = Image.fromarray(np.roll(Rt, (-(h // 2), -(w // 2)), (0, 1)))
        ok, text = seam_report(fixed, args.threshold, args.segments)
        print(f"  Re-check: {text} -> {'seamless' if ok else 'still a mismatch'}")
        if ok or not improved:
            break
    return fixed, ok, text


def main():
    p = argparse.ArgumentParser(description="Check a tile and repair its worst join spots if it is not seamless.")
    p.add_argument("input", help="Path to the single tile image")
    p.add_argument("-o", "--output", help="Output path (default: <input>_fixed.<ext>, or <input>_seamless.<ext> if already seamless)")
    p.add_argument("--threshold", type=float, default=1.1, help="Seam check threshold (default 1.1)")
    p.add_argument("--segments", type=int, default=20, help="Sections per seam for the pass/fail check (default 20)")
    p.add_argument("--spot-threshold", type=float, default=1.8, help="Only join stretches above this ratio are repaired (default 1.8)")
    p.add_argument("--max-spots", type=int, default=10, help="Repair at most this many spots per round, worst first (default 10)")
    p.add_argument("--band", type=int, default=96, help="Width in px of the repainted band (default 96)")
    p.add_argument("--pad", type=int, default=32, help="Extra length in px added to each end of a spot (default 32)")
    p.add_argument("--crop", type=int, default=1024, help="Crop size in px sent to the model for each spot (default 1024)")
    p.add_argument("--candidates", type=int, default=2, help="Images requested per spot; the best is kept (default 2)")
    p.add_argument("--rounds", type=int, default=2, help="Repair rounds, re-scanning after each (default 2)")
    p.add_argument("--feather", type=float, default=3.0, help="Edge softness of the paste-back in px (default 3)")
    p.add_argument("--quality", default="high", choices=["auto", "low", "medium", "high", "xhigh", "max"])
    p.add_argument("--dry-run", action="store_true", help="Show the spots that would be repaired; no API call")
    p.add_argument("--debug-dir", help="Save every crop, mask, model output and result here")
    args = p.parse_args()

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
        print(f"{len(spots)} spot(s) would be repaired ({len(spots) * args.candidates} images requested per round).")
        print(f"Preview saved to {path} (tile shifted by half so the joins are in the centre; red = repaint areas)")
        return 0

    if API_KEY == "input ur api key":
        sys.exit("Put your API key in API_KEY at the top of the script first.")
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"Repairing with {MODEL} ...")
    fixed, ok, text = run(img, client, args)
    assert fixed.size == (w, h)
    out_path = Path(args.output) if args.output else src.with_name(f"{src.stem}_fixed{src.suffix}")
    save(fixed, out_path)
    print(f"Saved {w}x{h} image to {out_path}")
    print("YES - the repaired design is seamless." if ok else
          "NO - the repaired design still does not pass the check (result saved anyway).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())