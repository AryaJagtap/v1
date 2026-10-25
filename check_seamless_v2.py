"""Check whether a tiled design is seamless along its centre join lines.

The input should be the clean 2x2 result (no diamond or dotted lines drawn on it).
The script compares pixels on either side of the vertical (up-down) and
horizontal (left-right) centre lines. If the jump across a join is about the
same as the normal pixel-to-pixel change elsewhere in the image, the join is
invisible, i.e. the design is seamless.

Usage:
    python check_seamless.py pattern_2x2.png
    python check_seamless.py pattern.png --single-tile     # input is one tile, not a 2x2
    python check_seamless.py pattern_2x2.png --threshold 1.3 --debug
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def check_seam(arr, segments):
    """arr shape: (L, N, 3). The seam lies between index N//2-1 and N//2 along axis 1.

    Returns (overall_ratio, [(start, end, ratio), ...]).
    ratio = jump across the seam / typical jump between neighbouring pixels elsewhere.
    ~1.0 means the seam is invisible; much larger means a visible mismatch.
    """
    length, n, _ = arr.shape
    c = n // 2

    seam = np.abs(arr[:, c - 1].astype(np.int16) - arr[:, c].astype(np.int16)).mean(axis=1)

    # Baseline: average neighbour difference for every other column pair (row by row).
    base_sum = np.zeros(length, dtype=np.float64)
    count = 0
    for j in range(n - 1):
        if j == c - 1:
            continue
        base_sum += np.abs(arr[:, j + 1].astype(np.int16) - arr[:, j].astype(np.int16)).mean(axis=1)
        count += 1
    baseline = base_sum / max(count, 1)

    floor = 1.0  # avoids huge ratios in perfectly flat areas
    overall = seam.mean() / max(baseline.mean(), floor)

    results = []
    for idx in np.array_split(np.arange(length), segments):
        ratio = seam[idx].mean() / max(baseline[idx].mean(), floor)
        results.append((int(idx[0]), int(idx[-1]), float(ratio)))
    return overall, results


def main():
    p = argparse.ArgumentParser(description="Check if a tiled design is seamless.")
    p.add_argument("input", help="Path to the clean 2x2 result image")
    p.add_argument("--single-tile", action="store_true",
                   help="Input is a single tile; tile it 2x2 internally before checking")
    p.add_argument("--threshold", type=float, default=1.1,
                   help="Max allowed seam/baseline ratio (default 1.5; lower = stricter)")
    p.add_argument("--segments", type=int, default=20,
                   help="Number of pieces each seam is split into for local checks (default 20)")
    p.add_argument("--debug", action="store_true",
                   help="Save an image with failing seam sections marked in red")
    p.add_argument("--original",
                   help="Original image to return if the design is seamless "
                        "(default: the input image itself)")
    p.add_argument("-o", "--output",
                   help="Where to save the returned original (default: <original>_seamless.<ext>)")
    args = p.parse_args()

    src = Path(args.input)
    img = Image.open(src).convert("RGB")
    a = np.array(img)
    if args.single_tile:
        a = np.tile(a, (2, 2, 1))
        img = Image.fromarray(a)

    h, w, _ = a.shape
    if w % 2 or h % 2:
        print("Warning: image width/height is odd, so the centre line is not an exact tile join.")

    local_limit = args.threshold * 2  # single sections are noisier, so allow more slack
    v_overall, v_segs = check_seam(a, args.segments)                       # up-down seam
    h_overall, h_segs = check_seam(a.transpose(1, 0, 2), args.segments)    # left-right seam

    def report(name, overall, segs):
        worst = max(segs, key=lambda s: s[2])
        ok = overall <= args.threshold and worst[2] <= local_limit
        print(f"{name}: overall ratio {overall:.2f}, worst section {worst[2]:.2f} "
              f"(positions {worst[0]}-{worst[1]}) -> {'OK' if ok else 'MISMATCH'}")
        return ok

    v_ok = report("Up-down seam   ", v_overall, v_segs)
    h_ok = report("Left-right seam", h_overall, h_segs)

    if args.debug:
        dbg = img.copy()
        d = ImageDraw.Draw(dbg)
        t = max(3, w // 300)
        for s, e, r in v_segs:
            if r > local_limit:
                d.line([(w // 2, s), (w // 2, e)], fill="red", width=t)
        for s, e, r in h_segs:
            if r > local_limit:
                d.line([(s, h // 2), (e, h // 2)], fill="red", width=t)
        dbg_path = src.with_name(f"{src.stem}_seamcheck{src.suffix}")
        dbg.save(dbg_path)
        print(f"Debug image saved to {dbg_path}")

    print()
    if v_ok and h_ok:
        print("YES - the design pattern is seamless.")
        original = Path(args.original) if args.original else src
        out_path = Path(args.output) if args.output else original.with_name(
            f"{original.stem}_seamless{original.suffix}")
        shutil.copyfile(original, out_path)  # byte-for-byte copy, no re-encoding
        print(f"Original image returned: {out_path}")
        sys.exit(0)
    print("NO - the design pattern is not seamless.")
    sys.exit(1)


if __name__ == "__main__":
    main()
