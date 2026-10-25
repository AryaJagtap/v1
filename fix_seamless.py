"""Check a repeat tile and, if it is not seamless, repair its borders with gpt-image-2.5-flare.

Flow:
  1. Check the tile (same test as check_seamless.py).
  2. Seamless  -> save a copy of the original and stop.
  3. Not seamless -> build a wrapped canvas where the tile's edges meet in the
     centre, ask the model to repaint ONLY that centre cross (mask) using the
     design already in the image, paste the result back (everything outside the
     strip stays pixel-identical to the original), cut the tile back out at
     the original size, and re-check. Retries with a wider strip if still not seamless.

Setup:
    pip install openai numpy pillow
    Put your key in API_KEY below.

Usage:
    python fix_seamless.py test.jpg
    python fix_seamless.py test.jpg --threshold 1.1 --quality high -o fixed.jpg
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

PROMPT = (
    """
TASK: PRECISION RESTORATION OF A SEAMLESS TEXTILE PATTERN

You are a precision ornamental-art restoration system, not a creative
image generator. Repair the supplied pattern tile to produce a clean,
professional, genuinely seamless textile or wallpaper repeat.

The original image is the only authoritative reference. Preserve its
existing artwork, motif geometry, colors, linework, layering, spacing,
and decorative details. The objective is to repair the damaged joining
area, NOT to redesign, embellish, or regenerate the pattern.

STRICTLY follow the instructions below.

==================================================
1. PRESERVE THE ORIGINAL DESIGN
==================================================

Modify only the designated transparent or damaged repair region.

Treat all correct, existing artwork as immutable. Do not redraw, replace,
move, resize, rotate, duplicate, simplify, or modify any intact motif.

Preserve the original composition, motif count, relative positions,
scale, proportions, background, and decorative density.

Do not regenerate the complete image. Do not change the original canvas
dimensions, aspect ratio, or crop.

Every reconstructed detail must be consistent with evidence visible in
the original reference. Never substitute artistic imagination for
reference-based reconstruction.

==================================================
2. CRITICAL: ELIMINATE HALF-PRINTED AND OVERLAPPING MOTIFS
==================================================

This is the highest-priority repair requirement.

The original artwork contains intricate, closely interwoven ornamental
shapes. Reconstruct their existing geometry accurately instead of
drawing additional decorative shapes over them.

NEVER place a new flower, petal, leaf, paisley, curl, flourish, scallop,
or ornamental fragment over an existing incomplete motif to hide it.

NEVER conceal a broken contour by adding another decorative element.

NEVER merge two neighboring motifs into one malformed shape.

NEVER create a second interpretation of an existing motif on top of
the original motif.

NEVER allow the outline of one motif to continue accidentally through
the fill, outline, or interior details of another motif.

NEVER create duplicate outlines, ghost shapes, doubled edges, floating
fragments, unfinished petals, disconnected leaves, or unexplained
overlapping decorative patches.

Each individual motif must have one coherent, traceable geometry.
Its outer contour, internal linework, fill regions, and decorative
details must belong to the same recognizable shape.

Where a motif is partially missing, reconstruct only its missing
portion. Continue the existing contour from its actual endpoint and
follow its original direction, curvature, thickness, and proportions.

Do not draw a complete new motif when only a small section of an
existing motif requires repair.

Where two motifs meet, preserve their original boundary. Keep their
contours distinct and follow the original spacing and layering.

If the reference does not establish how an ambiguous motif continues,
do not invent a complicated completion. Preserve the existing
structure and use a conservative continuation of the surrounding
background or ornamental linework.

A clean, simple, reference-consistent repair is preferable to an
elaborate but incorrect reconstruction.

==================================================
3. PRESERVE ORIGINAL LAYERING AND NEGATIVE SPACE
==================================================

Determine which shapes are in the foreground and which belong to the
background by examining the surrounding undamaged artwork.

Preserve the original foreground-background relationship exactly.

If a foreground petal, paisley, leaf, or decorative border covers part
of another shape, maintain that same relationship. Do not reverse
their layering, expose hidden portions incorrectly, or draw background
lines across foreground shapes.

Continue background contours only through regions where they should
actually be visible.

Preserve the original gaps, narrow channels, enclosed spaces, and
negative space between neighboring ornamental shapes.

Do not fill every empty space with additional decoration. Do not
connect separate motifs merely because they are close together.

Do not create artificial intersections where existing contours pass
near one another.

Every intersection, overlap, and concealed boundary must look
intentional and consistent with the original illustration.

==================================================
4. REPAIR CONTOURS BEFORE DETAILS
==================================================

Reconstruct the damaged area in this logical order:

1. Establish the boundaries of existing motifs.
2. Continue interrupted outer contours.
3. Restore the original separation and layering of neighboring shapes.
4. Reconstruct the correct interior fills.
5. Continue internal decorative lines, dots, veins, scallops, and
   fine ornamental details.
6. Restore the background texture and local color transitions.
7. Verify continuity at the tile boundaries.

Do not add internal details until their parent shape has been
reconstructed correctly.

Every line must connect to the correct existing line or end naturally
where the reference indicates that it should end.

Maintain consistent outline thickness, curvature, edge smoothness,
and color. Do not create a second contour alongside an existing one.

Preserve the original distinction between filled shapes, outlined
shapes, background decorations, and overlapping ornamental layers.

Avoid abrupt line endings, malformed junctions, disconnected strokes,
crossing outlines, doubled borders, and irregular contour thickness.

==================================================
5. NO INVENTED OR UNNECESSARY MOTIFS
==================================================

Use only the flowers, leaves, paisleys, petals, curls, ornamental
ribbons, scallops, dots, flourishes, and background marks already
present in the supplied reference.

Do not invent new flowers, additional petals, new leaf arrangements,
extra decorative clusters, new medallions, or replacement ornaments.

Do not duplicate nearby motifs to disguise a damaged area.

Do not increase the number of motifs or make the composition denser.

A repeated motif may be reconstructed only when its presence and
appearance are supported by the original pattern.

Use other occurrences of the same motif as references for missing
details, but do not copy an entire motif into a location where it
does not belong.

Preserve original asymmetry, irregularities, spacing, and stylistic
differences. Do not force artificial symmetry.

==================================================
6. EXACT ORIGINAL COLOR MATCHING
==================================================

The original image is the exclusive color reference.

Sample or match colors from the nearest corresponding undamaged
regions wherever possible.

Preserve the exact visual relationships between the existing ivory,
cream, beige, taupe, brown, olive, muted red, maroon, white, and
other colors actually present in the reference.

Do not introduce new hues, saturation changes, color grading,
unrequested gradients, artificial shadows, or brighter highlights.

Match each fill and outline independently. Do not paint an entire
reconstructed motif with a single approximate color.

Continue the original background color and fine texture without
halos, color bleeding, patches, or abrupt tonal transitions.

Do not globally enhance, recolor, sharpen, blur, or otherwise
process the undamaged artwork.

==================================================
7. PRESERVE BACKGROUND TEXTURE
==================================================

Continue the existing background texture with matching detail size,
density, spacing, direction, contrast, and color.

Preserve fine floral marks, stippling, tiny dots, and subtle decorative
textures where they are visible in the surrounding reference.

Do not replace detailed areas with flat fills, blurred patches,
generic noise, newly invented patterns, or oversized decorative marks.

Prevent background texture from bleeding across foreground contours.

Maintain clean boundaries between the background and every
ornamental shape.

==================================================
8. REPAIR MASK AND EDGE CONTROL
==================================================

The transparent or designated repair region is the only editable area.

Use the surrounding original artwork as context, with sufficient
nearby detail to reconstruct interrupted contours accurately.

Do not unnecessarily expand the repair region into correct artwork.

Where the mask touches an existing shape, continue its actual
boundary instead of creating a replacement shape.

Where the mask intersects a line, preserve the line's original
trajectory and reconnect it smoothly.

Where the mask intersects several neighboring motifs, reconstruct
each motif independently while respecting their original layering.

Do not interpret the mask boundary as an artistic outline, border,
or new motif boundary.

The finished artwork must contain no unintended transparent holes,
unfinished fills, visible mask edges, or repair patches.

==================================================
9. TRUE SEAMLESS REPEAT
==================================================

Ensure that the finished tile repeats continuously in both horizontal
and vertical directions, including diagonal repetition.

Match the left edge to the right edge and the top edge to the bottom
edge. Check all four corners and their transitions.

Preserve the position, shape, scale, color, thickness, and direction
of every feature crossing a tile boundary.

Do not create new motifs merely to make opposite edges appear similar.

Maintain the original composition while correcting boundary
discontinuities.

The result must not show visible seams, broken contours, duplicated
edge fragments, mismatched textures, sudden color changes, or
obvious repeating artifacts.

==================================================
10. PRINT-READY OUTPUT
==================================================

Produce clean, detailed, high-fidelity flat artwork suitable for
professional textile and wallpaper printing.

Preserve the source resolution, canvas dimensions, original palette,
fine linework, and intricate ornamental details.

Avoid blur, pixelation, jagged contours, compression artifacts,
smudges, sharpening halos, and loss of detail.

Do not invent additional texture or detail to simulate higher quality.

Do not add text, borders, frames, watermarks, mockups, or backgrounds
that are not present in the original artwork.

Return only the repaired pattern tile.

==================================================
11. FINAL QUALITY-CONTROL RULE
==================================================

Before finalizing, inspect every reconstructed motif against the
original image.

Reject any reconstruction that introduces:
- A new or duplicated decorative motif.
- A half-printed shape hidden beneath another shape.
- Two accidentally fused or intersecting motifs.
- A doubled, broken, or misaligned outline.
- An incorrect foreground-background relationship.
- An inconsistent color, fill, texture, or contour.
- A visible patch or discontinuity at a tile boundary.

If any proposed reconstruction conflicts with the existing artwork,
preserve the original artwork and choose the least invasive repair.

PRIORITY:
Reference fidelity > correct motif geometry > correct layering >
original colors and texture > seamless continuity > added detail.

FINAL COMMAND:
Restore the existing artwork; do not create new artwork.
Repair the missing geometry of existing motifs instead of covering
errors with additional designs. Every reconstructed element must
belong to the original composition, with clean contours, correct
layering, no accidental overlaps, and no invented decorative motifs.
"""
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


def repair(img, client, strip, quality, max_side):
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


def main():
    p = argparse.ArgumentParser(description="Check a tile and repair it if it is not seamless.")
    p.add_argument("input", help="Path to the single tile image")
    p.add_argument("-o", "--output", help="Output path (default: <input>_fixed.<ext>, or <input>_seamless.<ext> if already seamless)")
    p.add_argument("--threshold", type=float, default=1.1, help="Seam check threshold (default 1.1)")
    p.add_argument("--segments", type=int, default=20, help="Sections per seam for local checks (default 20)")
    p.add_argument("--strip", type=float, default=0.15, help="Width of the repaired strip as a fraction of the tile (default 0.15)")
    p.add_argument("--retries", type=int, default=2, help="Extra attempts with a wider strip if still not seamless (default 2)")
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

    print("NO - the design pattern is not seamless. Repairing with " + MODEL + " ...")
    if API_KEY == "input ur api key":
        sys.exit("Put your API key in API_KEY at the top of the script first.")

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    best = None  # (score, image, ok)
    for attempt in range(args.retries + 1):
        strip = min(0.4, args.strip + 0.1 * attempt)
        print(f"Attempt {attempt + 1}: repainting a {strip:.0%} strip along the joins ...")
        try:
            fixed = repair(img, client, strip, args.quality, min(args.max_side, 3840))
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
    out_path = Path(args.output) if args.output else src.with_name(f"{src.stem}_fixed{src.suffix}")
    save(fixed, out_path)
    print(f"Saved {w}x{h} image to {out_path}")
    print("YES - the repaired design is seamless." if ok else
          "NO - the repaired design still does not pass the check (best attempt saved).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
