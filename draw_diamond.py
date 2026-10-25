"""Draw a diamond (rhombus) on an image, joining the midpoints of its four edges.

Usage:
    python draw_diamond.py pattern_2x2.png
    python draw_diamond.py pattern_2x2.png -o out.png -w 6 -c black
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

SS = 4  # supersampling factor for smooth diagonal lines


def draw_diamond(img: Image.Image, width: int, color: str) -> Image.Image:
    base = img.convert("RGBA")
    w, h = base.size

    # Draw on a larger transparent layer, then shrink it for anti-aliasing.
    layer = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    W, H = w * SS, h * SS
    top, right, bottom, left = (W // 2, 0), (W - 1, H // 2), (W // 2, H - 1), (0, H // 2)
    pts = [top, right, bottom, left, top]
    d.line(pts, fill=color, width=width * SS, joint="curve")

    layer = layer.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(base, layer)


def main():
    p = argparse.ArgumentParser(description="Draw a diamond on an image.")
    p.add_argument("input", help="Path to the input image (e.g. the 2x2 result)")
    p.add_argument("-o", "--output", help="Output path (default: <input>_diamond.<ext>)")
    p.add_argument("-w", "--width", type=int, help="Line width in px (default: scales with image size)")
    p.add_argument("-c", "--color", default="black", help="Line color (default: black)")
    args = p.parse_args()

    src = Path(args.input)
    img = Image.open(src)
    width = args.width or max(2, img.width // 400)

    out = draw_diamond(img, width, args.color)

    out_path = Path(args.output) if args.output else src.with_name(f"{src.stem}_diamond{src.suffix}")
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        out.convert("RGB").save(out_path, quality=100)
    else:
        out.save(out_path)
    print(f"Saved {out.size[0]}x{out.size[1]} image to {out_path}")


if __name__ == "__main__":
    main()
