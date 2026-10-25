"""Check a repeat tile and, if it is not seamless, repair its borders with
gpt-image-2.5-flare or gemini-nano-banana-2.1 (asked at start-up).

Flow:
  1. Check the tile (same test as check_seamless.py).
  2. Seamless  -> save a copy of the original and stop.
  3. Not seamless -> build a wrapped canvas where the tile's edges meet in the
     centre, ask the model to repaint ONLY that centre cross (mask) using the
     design already in the image, paste the result back (everything outside the
     strip stays pixel-identical to the original), cut the tile back out at
     the original size, and re-check. Retries with a wider strip if still not seamless.

Setup:
    pip install openai google-genai numpy pillow
    Put your keys in API_KEY (GPT) and GEMINI_API_KEY (Gemini) below.

Usage:
    python seamless_gemini.py test.jpg
    python seamless_gemini.py test.jpg --threshold 1.1 --quality high -o fixed.jpg
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

GEMINI_API_KEY = "AQ.Ab8RN6IgVotnM2C1GLeSlpXPyYi4LX69XRCYsbdFSnrhGpYBcw"
GEMINI_MODEL = "gemini-nano-banana-2.1"

# Gemini has no mask parameter, so the mask is sent as a second image with this note.
GEMINI_MASK_NOTE = (
    "Two images are attached. The first is the pattern canvas to repair. The second is a "
    "black-and-white mask of the same size: WHITE marks the only area you may repaint, "
    "BLACK must stay exactly as in the first image. Return one image with the same framing "
    "and aspect ratio as the first image.\n"
)

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
    """Pick a WxH for the API: multiples of 16, within assumed model limits
    (min ~655k pixels, max 3840 long edge / ~8.3M pixels). Adjust if the API rejects it."""
    scale = min(1.0, max_side / max(cw, ch))
    if cw * ch * scale * scale < 655_360:
        scale = (655_360 / (cw * ch)) ** 0.5
    if cw * ch * scale * scale > 8_294_400:
        scale = (8_294_400 / (cw * ch)) ** 0.5
    mw = max(16, int(round(cw * scale / 16)) * 16)
    mh = max(16, int(round(ch * scale / 16)) * 16)
    return mw, mh


def gemini_edit(client, image, mask, max_side):
    """Send canvas + mask to Gemini and return the generated image (PIL, RGB)."""
    from google.genai import types
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[GEMINI_MASK_NOTE + PROMPT, image, mask],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                image_size="1K" if max_side <= 1024 else "2K" if max_side <= 2048 else "4K"),
        ),
    )
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else None) or []:
            if part.inline_data is not None and part.inline_data.data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    raise RuntimeError("Gemini returned no image.")


def repair(img, client, strip, quality, max_side, provider="gpt"):
    a = np.array(img.convert("RGB"))
    h, w, _ = a.shape

    # Wrapped canvas centred on the tile join, with extra context on every side.
    big = np.tile(a, (2, 2, 1))
    mx, my = w // 4, h // 4
    x0, y0 = w // 2 - mx, h // 2 - my
    cw, ch = w + 2 * mx, h + 2 * my
    canvas = big[y0:y0 + ch, x0:x0 + cw]
    sx, sy = w - x0, h - y0  # join position inside the canvas

    # Cross-shaped mask over the join.
    sw, sh = max(4, int(w * strip / 2)), max(4, int(h * strip / 2))
    m = np.zeros((ch, cw), np.uint8)
    m[:, sx - sw:sx + sw] = 255
    m[sy - sh:sy + sh, :] = 255

    # Send to the model (transparent = area to repaint).
    mw, mh = model_size(cw, ch, max_side)
    small = Image.fromarray(canvas).resize((mw, mh), Image.LANCZOS)
    alpha = ImageOps.invert(Image.fromarray(m).resize((mw, mh), Image.BILINEAR))
    mask_rgba = Image.new("RGBA", (mw, mh), (0, 0, 0, 255))
    mask_rgba.putalpha(alpha)

    if provider == "gemini":
        gen = gemini_edit(client, small, Image.fromarray(m).resize((mw, mh), Image.BILINEAR), max_side)
    else:
        resp = client.images.edit(
            model=MODEL,
            image=("tile.png", png_bytes(small), "image/png"),
            mask=("mask.png", png_bytes(mask_rgba), "image/png"),
            prompt=PROMPT,
            size=f"{mw}x{mh}",
            quality=quality,
        )
        gen = Image.open(io.BytesIO(base64.b64decode(resp.data[0].b64_json))).convert("RGB")
    gen = np.array(gen.resize((cw, ch), Image.LANCZOS)).astype(np.float32)

    # Paste only the strip (feathered) onto the untouched full-resolution canvas.
    soft = np.array(Image.fromarray(m).filter(ImageFilter.GaussianBlur(max(2, min(sw, sh) // 5))),
                    dtype=np.float32) / 255.0
    out = canvas.astype(np.float32) * (1 - soft[..., None]) + gen * soft[..., None]
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    # Cut the tile back out at the original size.
    return Image.fromarray(out[my:my + h, mx:mx + w])


def save(im, path):
    if path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(path, quality=100, subsampling=0)
    else:
        im.save(path)


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
    p = argparse.ArgumentParser(description="Check a tile and repair it if it is not seamless.")
    p.add_argument("input", help="Path to the single tile image")
    p.add_argument("-o", "--output", help="Output path (default: <gpt|gemini>_<fixed|unfix>_<input>.png, or <input>_seamless.<ext> if already seamless)")
    p.add_argument("--threshold", type=float, default=1.1, help="Seam check threshold (default 1.1)")
    p.add_argument("--segments", type=int, default=20, help="Sections per seam for local checks (default 20)")
    p.add_argument("--strip", type=float, default=0.15, help="Width of the repaired strip as a fraction of the tile (default 0.15)")
    p.add_argument("--retries", type=int, default=4, help="Extra attempts with a wider strip if still not seamless (default 4)")
    p.add_argument("--quality", default="high", choices=["auto", "low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-side", type=int, default=2048, help="Longest side sent to the model, max 3840 (default 2048)")
    args = p.parse_args()

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

    provider = ask_provider()
    print("NO - the design pattern is not seamless. Repairing with "
          + (GEMINI_MODEL if provider == "gemini" else MODEL) + " ...")
    if provider == "gemini":
        if GEMINI_API_KEY == "input ur gemini api key":
            sys.exit("Put your Gemini API key in GEMINI_API_KEY at the top of the script first.")
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        if API_KEY == "input ur api key":
            sys.exit("Put your API key in API_KEY at the top of the script first.")
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    best = None  # (score, image, ok)
    for attempt in range(args.retries + 1):
        strip = min(0.4, args.strip + 0.1 * attempt)
        print(f"Attempt {attempt + 1}: repainting a {strip:.0%} strip along the joins ...")
        try:
            fixed = repair(img, client, strip, args.quality, min(args.max_side, 3840), provider)
        except Exception as e:
            print(f"  API call failed: {e}")
            continue
        ok, score, text = seam_report(fixed, args.threshold, args.segments)
        print(f"  Re-check: {text} -> {'seamless' if ok else 'still a mismatch'}")
        if best is None or score < best[0]:
            best = (score, fixed, ok)
        if ok:
            break

    if best is None:
        sys.exit("No image was generated.")

    score, fixed, ok = best
    assert fixed.size == (w, h)
    status = "fixed" if ok else "unfix"
    out_path = Path(args.output) if args.output else src.with_name(f"{provider}_{status}_{src.stem}.png")
    save(fixed, out_path)
    print(f"Saved {w}x{h} image to {out_path}")
    print("YES - the repaired design is seamless." if ok else
          "NO - the repaired design still does not pass the check (best attempt saved).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
