"""Draw a diamond and its dotted diagonals on an image.

The diamond joins the midpoints of the four edges. The dotted lines run
top-to-bottom and left-to-right through the centre (the diamond's diagonals).

Usage:
    python draw_diamond.py pattern_2x2.png
    python draw_diamond.py pattern_2x2.png -o out.png -w 6 -c black
    python draw_diamond.py pattern_2x2.png --no-dotted
"""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

SS = 4  # supersampling factor for smooth lines


def dotted_line(d, p0, p1, radius, spacing, color):
    """Draw round dots along the segment p0 -> p1."""
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n = max(1, int(length / spacing))
    for i in range(n + 1):
        t = i / n
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_diamond(img, width, color, dotted=True, dot_gap=3.0):
    base = img.convert("RGBA")
    w, h = base.size

    layer = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    W, H = w * SS, h * SS
    top, right, bottom, left = (W // 2, 0), (W - 1, H // 2), (W // 2, H - 1), (0, H // 2)

    # Solid diamond outline
    d.line([top, right, bottom, left, top], fill=color, width=width * SS, joint="curve")

    # Dotted diagonals (vertical + horizontal through the centre)
    if dotted:
        radius = width * SS / 2
        spacing = width * SS * (1 + dot_gap)
        dotted_line(d, top, bottom, radius, spacing, color)
        dotted_line(d, left, right, radius, spacing, color)

    layer = layer.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(base, layer)


def main():
    p = argparse.ArgumentParser(description="Draw a diamond with dotted diagonals on an image.")
    p.add_argument("input", help="Path to the input image (e.g. the 2x2 result)")
    p.add_argument("-o", "--output", help="Output path (default: <input>_diamond.<ext>)")
    p.add_argument("-w", "--width", type=int, help="Line width in px (default: scales with image size)")
    p.add_argument("-c", "--color", default="black", help="Line color (default: black)")
    p.add_argument("--dot-gap", type=float, default=3.0, help="Gap between dots, in multiples of dot size (default: 3)")
    p.add_argument("--no-dotted", action="store_true", help="Skip the dotted diagonals")
    args = p.parse_args()

    src = Path(args.input)
    img = Image.open(src)
    width = args.width or max(2, img.width // 400)

    out = draw_diamond(img, width, args.color, dotted=not args.no_dotted, dot_gap=args.dot_gap)

    out_path = Path(args.output) if args.output else src.with_name(f"{src.stem}_diamond{src.suffix}")
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        out.convert("RGB").save(out_path, quality=100)
    else:
        out.save(out_path)
    print(f"Saved {out.size[0]}x{out.size[1]} image to {out_path}")


if __name__ == "__main__":
    main()
