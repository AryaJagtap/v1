"""Tile an image into a seamless 2x2 grid (no gaps, no overlap).

Usage:
    python tile_2x2.py input.jpg
    python tile_2x2.py input.jpg -o output.png
"""
import argparse
from pathlib import Path

from PIL import Image


def tile_2x2(img: Image.Image) -> Image.Image:
    w, h = img.size
    # Canvas is exactly 2x the tile size, so tiles butt up against each other.
    canvas = Image.new(img.mode, (w * 2, h * 2))
    for row in range(2):
        for col in range(2):
            canvas.paste(img, (col * w, row * h))
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Create a seamless 2x2 tiled image.")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("-o", "--output", help="Output path (default: <input>_2x2.<ext>)")
    args = parser.parse_args()

    src = Path(args.input)
    img = Image.open(src)
    # Keep transparency if present; otherwise normalise odd modes to RGB.
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")

    out = tile_2x2(img)

    out_path = Path(args.output) if args.output else src.with_name(f"{src.stem}_2x2{src.suffix}")
    if out_path.suffix.lower() in (".jpg", ".jpeg") and out.mode == "RGBA":
        out = out.convert("RGB")
    out.save(out_path, quality=100) if out_path.suffix.lower() in (".jpg", ".jpeg") else out.save(out_path)
    print(f"Saved {out.size[0]}x{out.size[1]} image to {out_path}")


if __name__ == "__main__":
    main()
