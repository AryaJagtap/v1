"""Unified Seamless Pattern Tile Pipeline.

Implements the complete workflow from workflow.png:
  Step 1: Input single tile image.
  Step 2: Generate 2x2 tile preview.
  Step 3: Draw diamond overlay with dotted diagonals.
  Step 4: Objective seam check along up-down and left-right joins.
  Step 5: If seamless -> return pristine original byte-for-byte.
  Step 6: If not seamless -> classify mismatch axis (up-down, left-right, or both).
  Step 7: Repair only the mismatched borders without altering the center:
          - Left-right mismatch: horizontal roll -> vertical band inpaint -> horizontal unroll
          - Up-down mismatch: vertical roll -> horizontal band inpaint -> vertical unroll
          - Both mismatched: sequential 1D inpainting along both axes
          Center pixels are mathematically guaranteed to remain 100% untouched.
  Step 8: Re-feed edited image into Step 2 for verification loop.

Supported AI Models (5 options across 3 providers):
  1. Replicate:
     a) FLUX.1 Fill Pro (black-forest-labs/flux-fill-pro) [Highest fidelity & structural accuracy]
     b) FLUX.1 Fill Dev (black-forest-labs/flux-fill-dev) [Fast open-weights inpainting]
  2. OpenAI GPT:
     a) GPT Image 2.5 Sunburst (gpt-image-2.5-sunburst) [High precision, fine linework]
     b) GPT Image 2.5 Flare (gpt-image-2.5-flare) [Fast generation]
  3. Google Gemini:
     a) Gemini Image 2.1 (gemini-nano-banana-2.1)
     b) Imagen 3 Capability (imagen-3.0-generate-002)

Usage:
  python seamless_generator.py pattern.jpg
  python seamless_generator.py pattern.jpg --preview-only
  python seamless_generator.py pattern.jpg --provider replicate --model flux-pro
  python seamless_generator.py pattern.jpg --provider gpt --model gpt-sunburst
"""

import argparse
import base64
import io
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

# ---------- Load Environment Variables (.env) ----------
def load_dotenv():
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k not in os.environ:
                os.environ[k] = v

load_dotenv()

# ---------- Provider & Model Catalog ----------
MODELS_CATALOG = {
    "replicate": {
        "name": "Replicate",
        "models": {
            "flux-pro": {
                "id": "black-forest-labs/flux-fill-pro",
                "label": "FLUX.1 Fill Pro (black-forest-labs/flux-fill-pro) [Recommended / Highest Fidelity]",
            },
            "flux-dev": {
                "id": "black-forest-labs/flux-fill-dev",
                "label": "FLUX.1 Fill Dev (black-forest-labs/flux-fill-dev) [Fast & Detailed]",
            },
        },
        "default": "flux-pro",
        "key_env": "REPLICATE_API_TOKEN",
    },
    "gpt": {
        "name": "OpenAI GPT",
        "models": {
            "gpt-sunburst": {
                "id": "gpt-image-2.5-sunburst",
                "label": "GPT Image 2.5 Sunburst [High Precision / Print Quality]",
            },
            "gpt-flare": {
                "id": "gpt-image-2.5-flare",
                "label": "GPT Image 2.5 Flare [Fast]",
            },
        },
        "default": "gpt-sunburst",
        "key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "name": "Google Gemini",
        "models": {
            "gemini-2.1": {
                "id": "gemini-nano-banana-2.1",
                "label": "Gemini Image 2.1 (gemini-nano-banana-2.1)",
            },
            "imagen-3": {
                "id": "imagen-3.0-generate-002",
                "label": "Google Imagen 3 (imagen-3.0-generate-002)",
            },
        },
        "default": "gemini-2.1",
        "key_env": "GEMINI_API_KEY",
    },
}


def select_model_interactive(pre_provider=None, pre_model=None):
    """Hierarchical two-step interactive model selector as requested:
    Step 1: Choose provider (1. Replicate, 2. GPT, 3. Gemini)
    Step 2: Choose model within selected provider."""
    # Step 1: Provider
    provider = pre_provider
    if not provider or provider not in MODELS_CATALOG:
        print("\n" + "=" * 62)
        print("                 SELECT AI PROVIDER")
        print("=" * 62)
        print("  1) Replicate       (State-of-the-Art FLUX.1 Fill models)")
        print("  2) OpenAI GPT      (GPT Image 2.5 Sunburst / Flare)")
        print("  3) Google Gemini   (Gemini Image 2.1 / Imagen 3)")
        print("-" * 62)
        while True:
            choice = input("Select provider [1-3] (default 1): ").strip()
            if choice in ("", "1", "replicate"):
                provider = "replicate"
                break
            elif choice in ("2", "gpt", "openai"):
                provider = "gpt"
                break
            elif choice in ("3", "gemini", "google"):
                provider = "gemini"
                break
            print("Invalid selection. Please enter 1, 2, or 3.")

    prov_info = MODELS_CATALOG[provider]
    models = list(prov_info["models"].items())

    # Step 2: Model within provider
    model_key = pre_model
    valid_keys = [k for k, _ in models]
    if not model_key or model_key not in valid_keys:
        print("\n" + "-" * 62)
        print(f"       SELECT MODEL FOR {prov_info['name'].upper()}")
        print("-" * 62)
        for i, (k, info) in enumerate(models, 1):
            print(f"  {i}) {info['label']}")
        print("-" * 62)
        while True:
            choice = input(f"Select model [1-{len(models)}] (default 1): ").strip()
            if choice == "" or choice == "1":
                model_key = models[0][0]
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    model_key = models[idx][0]
                    break
            except ValueError:
                pass
            print(f"Invalid selection. Please enter a number between 1 and {len(models)}.")

    chosen_model = prov_info["models"][model_key]
    print(f"\n[AI Backend Active]: {prov_info['name']} -> {chosen_model['id']}")
    return provider, chosen_model["id"], prov_info["key_env"]


# ---------- Step 2: 2x2 Tile Preview ----------
def create_2x2_tile(img: Image.Image) -> Image.Image:
    """Duplicate input image 4 times in a seamless 2x2 grid."""
    w, h = img.size
    grid = Image.new(img.mode, (w * 2, h * 2))
    grid.paste(img, (0, 0))
    grid.paste(img, (w, 0))
    grid.paste(img, (0, h))
    grid.paste(img, (w, h))
    return grid


# ---------- Step 3: Draw Diamond with Dotted Diagonals ----------
def draw_diamond_overlay(img: Image.Image, width: int = None, color: str = "black", dot_gap: float = 3.0) -> Image.Image:
    """Draw a diamond joining edge midpoints with center dotted diagonals."""
    SS = 4
    base = img.convert("RGBA")
    w, h = base.size
    line_w = width or max(2, w // 400)

    layer = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    W, H = w * SS, h * SS
    top, right, bottom, left = (W // 2, 0), (W - 1, H // 2), (W // 2, H - 1), (0, H // 2)

    # Solid diamond outline
    d.line([top, right, bottom, left, top], fill=color, width=line_w * SS, joint="curve")

    # Dotted diagonals through centre
    radius = (line_w * SS) / 2
    spacing = (line_w * SS) * (1 + dot_gap)

    def dotted_segment(p0, p1):
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        steps = max(1, int(length / spacing))
        for i in range(steps + 1):
            t = i / steps
            x = p0[0] + (p1[0] - p0[0]) * t
            y = p0[1] + (p1[1] - p0[1]) * t
            d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    dotted_segment(top, bottom)
    dotted_segment(left, right)

    layer = layer.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(base, layer)


# ---------- Step 4: Objective Seam Checker ----------
def measure_seam(arr: np.ndarray, segments: int, floor: float = 1.0):
    """arr shape: (L, N, 3). The seam lies between index N//2-1 and N//2 along axis 1.
    Returns (overall_ratio, [(start, end, ratio), ...])."""
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

    overall = seam.mean() / max(baseline.mean(), floor)
    results = []
    for idx in np.array_split(np.arange(length), segments):
        seg_ratio = seam[idx].mean() / max(baseline[idx].mean(), floor)
        results.append((int(idx[0]), int(idx[-1]), float(seg_ratio)))
    return overall, results


def check_seamless_detailed(img: Image.Image, threshold: float = 1.1, segments: int = 20, floor: float = 1.0):
    """Checks up-down and left-right seams of the 2x2 tiled image.
    Returns:
      is_seamless: bool
      v_ok: bool (left-right borders pass)
      h_ok: bool (up-down borders pass)
      v_overall, v_segs, h_overall, h_segs
    """
    tile_arr = np.array(img.convert("RGB"))
    grid_2x2 = np.tile(tile_arr, (2, 2, 1))

    v_overall, v_segs = measure_seam(grid_2x2, segments, floor)                      # up-down seam (tests left-right borders)
    h_overall, h_segs = measure_seam(grid_2x2.transpose(1, 0, 2), segments, floor)   # left-right seam (tests top-bottom borders)

    local_limit = threshold * 2.0
    v_worst = max(v_segs, key=lambda s: s[2])
    h_worst = max(h_segs, key=lambda s: s[2])

    v_ok = (v_overall <= threshold) and (v_worst[2] <= local_limit)
    h_ok = (h_overall <= threshold) and (h_worst[2] <= local_limit)
    is_seamless = v_ok and h_ok

    return is_seamless, v_ok, h_ok, (v_overall, v_worst, v_segs), (h_overall, h_worst, h_segs)


def save_seamcheck_debug(img_2x2: Image.Image, v_segs, h_segs, local_limit: float, out_path: Path):
    """Draw red markers on failing seam segments."""
    dbg = img_2x2.copy()
    d = ImageDraw.Draw(dbg)
    w, h = dbg.size
    t = max(3, w // 300)
    for s, e, r in v_segs:
        if r > local_limit:
            d.line([(w // 2, s), (w // 2, e)], fill="red", width=t)
    for s, e, r in h_segs:
        if r > local_limit:
            d.line([(s, h // 2), (e, h // 2)], fill="red", width=t)
    dbg.save(out_path)


# ---------- Dynamic Vision Prompting Agent ----------
def get_adaptive_prompt(img: Image.Image, custom_prompt: str = None) -> str:
    """Generate a clean, targeted inpainting prompt. If custom_prompt is provided, use it."""
    if custom_prompt:
        return custom_prompt

    # Base prompt engineered for seamless inpainting without concept bleeding
    return (
        "High resolution textile and surface repeat pattern. "
        "Inpaint ONLY the transparent masked band to make the repeat seamless: "
        "smoothly connect and complete the continuing outlines, shapes, motifs, textures, "
        "and background colors across the seam. Exactly match the surrounding drawing style, "
        "line weight, palette, and texture. Do not introduce any new motifs, objects, borders, "
        "or watermarks. Leave untouched pixels completely unaltered."
    )


# ---------- Step 7: Inpainting Backends ----------
def inpaint_replicate(img: Image.Image, mask: Image.Image, prompt: str, model_id: str, api_token: str) -> Image.Image:
    """Inpaint with Replicate (FLUX.1 Fill Pro or Dev).
    Replicate expects mask where white (255) = inpaint, black (0) = preserve."""
    import time
    import replicate
    import httpx

    client = replicate.Client(api_token=api_token)
    orig_w, orig_h = img.size

    # Cap resolution for diffusion inpainting models (optimal: 1024 - 1536px)
    max_dim = 1536
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        mw = max(16, int(round(orig_w * scale / 16)) * 16)
        mh = max(16, int(round(orig_h * scale / 16)) * 16)
        send_img = img.resize((mw, mh), Image.LANCZOS)
        send_mask = mask.resize((mw, mh), Image.NEAREST)
        print(f"  [Replicate Scale]: Optimized resolution {orig_w}x{orig_h} -> {mw}x{mh} for FLUX.1 Fill")
    else:
        mw, mh = orig_w, orig_h
        send_img = img
        send_mask = mask

    # In-memory buffer encoding (instant, no file locking)
    print("  [1/3] Preparing image and mask buffers ...")
    buf_img = io.BytesIO()
    send_img.save(buf_img, format="JPEG", quality=95)
    buf_img.seek(0)

    buf_mask = io.BytesIO()
    send_mask.save(buf_mask, format="PNG")
    buf_mask.seek(0)

    inputs = {
        "image": buf_img,
        "mask": buf_mask,
        "prompt": prompt,
        "guidance": 30.0,
        "output_format": "jpg",
    }
    if "pro" in model_id.lower():
        inputs["steps"] = 35

    print(f"  [2/3] Calling Replicate [{model_id}] (uploading buffers & generating diffusion steps) ...")
    t0 = time.time()
    output = client.run(model_id, input=inputs)
    elapsed = time.time() - t0
    print(f"  [3/3] Generation complete in {elapsed:.1f}s! Downloading result ...")

    # Download / read output
    if hasattr(output, "read"):
        gen_img = Image.open(io.BytesIO(output.read())).convert("RGB")
    elif isinstance(output, list) and len(output) > 0:
        resp = httpx.get(str(output[0]), timeout=90.0)
        resp.raise_for_status()
        gen_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    elif isinstance(output, str):
        resp = httpx.get(output, timeout=90.0)
        resp.raise_for_status()
        gen_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        resp = httpx.get(str(output), timeout=90.0)
        resp.raise_for_status()
        gen_img = Image.open(io.BytesIO(resp.content)).convert("RGB")

    if gen_img.size != (orig_w, orig_h):
        gen_img = gen_img.resize((orig_w, orig_h), Image.LANCZOS)
    return gen_img


def inpaint_openai(img: Image.Image, mask: Image.Image, prompt: str, model_id: str, api_key: str) -> Image.Image:
    """Inpaint with OpenAI images.edit.
    OpenAI expects RGBA mask where alpha=0 is area to inpaint, alpha=255 is area to preserve."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    w, h = img.size

    # Prepare alpha channel (0 = inpaint, 255 = keep)
    mask_np = np.array(mask.convert("L"))
    alpha = np.where(mask_np > 127, 0, 255).astype(np.uint8)
    mask_rgba = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    mask_rgba.putalpha(Image.fromarray(alpha))

    # Resize to acceptable API dimensions if necessary
    scale = min(1.0, 2048 / max(w, h))
    mw = max(16, int(round(w * scale / 16)) * 16)
    mh = max(16, int(round(h * scale / 16)) * 16)

    send_img = img.resize((mw, mh), Image.LANCZOS)
    send_mask = mask_rgba.resize((mw, mh), Image.BILINEAR)

    buf_img = io.BytesIO()
    send_img.save(buf_img, format="PNG")
    buf_mask = io.BytesIO()
    send_mask.save(buf_mask, format="PNG")

    print(f"  Calling OpenAI [{model_id}] (resolution: {mw}x{mh}) ...")
    resp = client.images.edit(
        model=model_id,
        image=("image.png", buf_img.getvalue(), "image/png"),
        mask=("mask.png", buf_mask.getvalue(), "image/png"),
        prompt=prompt,
        size=f"{mw}x{mh}",
        quality="high",
    )

    if resp.data[0].b64_json:
        data = base64.b64decode(resp.data[0].b64_json)
    else:
        import httpx
        data = httpx.get(resp.data[0].url, timeout=90.0).content

    gen_img = Image.open(io.BytesIO(data)).convert("RGB")
    if gen_img.size != (w, h):
        gen_img = gen_img.resize((w, h), Image.LANCZOS)
    return gen_img


def inpaint_gemini(img: Image.Image, mask: Image.Image, prompt: str, model_id: str, api_key: str) -> Image.Image:
    """Inpaint with Google Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    w, h = img.size

    # Mark mask band for Gemini
    gemini_note = (
        "Inpaint the designated repair band in this repeat tile. "
        "Seamlessly continue all motifs, lines, and textures across the seam. "
        "Keep everything else strictly unchanged. "
    )
    full_prompt = gemini_note + prompt

    print(f"  Calling Gemini [{model_id}] ...")
    resp = client.models.generate_content(
        model=model_id,
        contents=[full_prompt, img, mask],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="1:1" if w == h else None,
                image_size="1K" if max(w, h) <= 1024 else "2K"
            )
        )
    )
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else None) or []:
            if part.inline_data is not None and part.inline_data.data:
                gen_img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                if gen_img.size != (w, h):
                    gen_img = gen_img.resize((w, h), Image.LANCZOS)
                return gen_img
    raise RuntimeError("Gemini returned no image.")


# ---------- Inpainting Dispatcher ----------
def call_inpainter(img: Image.Image, mask: Image.Image, prompt: str, provider: str, model_id: str, api_key: str) -> Image.Image:
    if provider == "replicate":
        return inpaint_replicate(img, mask, prompt, model_id, api_key)
    elif provider == "gpt":
        return inpaint_openai(img, mask, prompt, model_id, api_key)
    elif provider == "gemini":
        return inpaint_gemini(img, mask, prompt, model_id, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------- Step 7: Zero-Distortion Boundary Repair Algorithm ----------
def repair_vertical_seam(img: Image.Image, provider: str, model_id: str, api_key: str, prompt: str, band_w: int = 96, feather: float = 3.0) -> Image.Image:
    """Repairs the left-right border mismatch (vertical seam).
    1. Roll horizontally by +W/2 so left and right borders meet at center column W/2.
    2. Inpaint vertical band centered at W/2.
    3. Composite with feathered mask (leaving everything outside the band 100% untouched).
    4. Unroll horizontally by -W/2 back to original position.
    Center is mathematically guaranteed to remain 100% untouched!"""
    orig_np = np.array(img.convert("RGB"))
    h, w, _ = orig_np.shape
    half_w = w // 2

    # 1. Roll horizontally
    rolled_np = np.roll(orig_np, half_w, axis=1)
    rolled_img = Image.fromarray(rolled_np)

    # 2. Mask vertical strip centered at W/2
    bw = min(band_w, w // 4)
    mask_np = np.zeros((h, w), dtype=np.uint8)
    mask_np[:, half_w - bw // 2 : half_w + bw // 2] = 255
    mask_img = Image.fromarray(mask_np)

    # 3. Inpaint
    print(f"  [Repairing Left-Right Borders] Band width: {bw}px ...")
    gen_img = call_inpainter(rolled_img, mask_img, prompt, provider, model_id, api_key)
    gen_np = np.array(gen_img.convert("RGB")).astype(np.float32)

    # 4. Feathered composite
    soft_mask = np.clip(gaussian_filter(mask_np.astype(np.float32) / 255.0, feather), 0, 1)[..., None]
    comp_rolled = np.clip(rolled_np.astype(np.float32) * (1 - soft_mask) + gen_np * soft_mask + 0.5, 0, 255).astype(np.uint8)

    # 5. Unroll horizontally
    repaired_np = np.roll(comp_rolled, -half_w, axis=1)

    # Verification: Verify that the interior center remains 100% identical
    center_slice_orig = orig_np[:, bw : w - bw]
    center_slice_rep = repaired_np[:, bw : w - bw]
    max_drift = np.max(np.abs(center_slice_orig.astype(int) - center_slice_rep.astype(int)))
    assert max_drift == 0, f"Integrity assertion failed: center drifted by {max_drift}"
    print(f"  [Center Protected]: Interior columns [{bw}:{w-bw}] verified bit-identical (drift=0).")

    return Image.fromarray(repaired_np)


def repair_horizontal_seam(img: Image.Image, provider: str, model_id: str, api_key: str, prompt: str, band_h: int = 96, feather: float = 3.0) -> Image.Image:
    """Repairs the up-down border mismatch (horizontal seam).
    1. Roll vertically by +H/2 so top and bottom borders meet at center row H/2.
    2. Inpaint horizontal band centered at H/2.
    3. Composite with feathered mask (leaving everything outside the band 100% untouched).
    4. Unroll vertically by -H/2 back to original position.
    Center is mathematically guaranteed to remain 100% untouched!"""
    orig_np = np.array(img.convert("RGB"))
    h, w, _ = orig_np.shape
    half_h = h // 2

    # 1. Roll vertically
    rolled_np = np.roll(orig_np, half_h, axis=0)
    rolled_img = Image.fromarray(rolled_np)

    # 2. Mask horizontal strip centered at H/2
    bh = min(band_h, h // 4)
    mask_np = np.zeros((h, w), dtype=np.uint8)
    mask_np[half_h - bh // 2 : half_h + bh // 2, :] = 255
    mask_img = Image.fromarray(mask_np)

    # 3. Inpaint
    print(f"  [Repairing Up-Down Borders] Band height: {bh}px ...")
    gen_img = call_inpainter(rolled_img, mask_img, prompt, provider, model_id, api_key)
    gen_np = np.array(gen_img.convert("RGB")).astype(np.float32)

    # 4. Feathered composite
    soft_mask = np.clip(gaussian_filter(mask_np.astype(np.float32) / 255.0, feather), 0, 1)[..., None]
    comp_rolled = np.clip(rolled_np.astype(np.float32) * (1 - soft_mask) + gen_np * soft_mask + 0.5, 0, 255).astype(np.uint8)

    # 5. Unroll vertically
    repaired_np = np.roll(comp_rolled, -half_h, axis=0)

    # Verification: Verify that the interior center remains 100% identical
    center_slice_orig = orig_np[bh : h - bh, :]
    center_slice_rep = repaired_np[bh : h - bh, :]
    max_drift = np.max(np.abs(center_slice_orig.astype(int) - center_slice_rep.astype(int)))
    assert max_drift == 0, f"Integrity assertion failed: center drifted by {max_drift}"
    print(f"  [Center Protected]: Interior rows [{bh}:{h-bh}] verified bit-identical (drift=0).")

    return Image.fromarray(repaired_np)


# ---------- Main Pipeline Controller ----------
def run_pipeline(
    image_path: Path,
    provider: str = None,
    model_name: str = None,
    threshold: float = 1.1,
    floor: float = 1.0,
    band_size: int = 96,
    max_rounds: int = 2,
    preview_only: bool = False,
    custom_prompt: str = None,
    out_dir: Path = None,
):
    src = Path(image_path)
    if not src.exists():
        sys.exit(f"Error: File not found: {src}")

    target_dir = out_dir or src.parent
    stem = src.stem
    ext = src.suffix

    print("\n" + "=" * 62)
    print(f" PROCESSING TILE: {src.name}")
    print("=" * 62)

    img = Image.open(src).convert("RGB")
    w, h = img.size
    print(f"Step 1: Loaded input image ({w}x{h} px)")

    # Step 2: Generate 2x2 tile preview
    preview_2x2 = create_2x2_tile(img)
    p2x2_path = target_dir / f"{stem}_2x2.png"
    preview_2x2.save(p2x2_path)
    print(f"Step 2: Saved 2x2 tile preview -> {p2x2_path.name}")

    # Step 3: Draw Diamond overlay
    diamond = draw_diamond_overlay(preview_2x2)
    dia_path = target_dir / f"{stem}_diamond.png"
    diamond.save(dia_path)
    print(f"Step 3: Saved diamond overlay -> {dia_path.name}")

    # Step 4: Seam Check
    is_seamless, v_ok, h_ok, v_stat, h_stat = check_seamless_detailed(img, threshold=threshold, floor=floor)
    v_overall, v_worst, v_segs = v_stat
    h_overall, h_worst, h_segs = h_stat

    print("\nStep 4: Seam Analysis:")
    print(f"  Up-down seam (Left-Right borders): overall={v_overall:.2f}, worst={v_worst[2]:.2f} (pos {v_worst[0]}-{v_worst[1]}) -> {'OK' if v_ok else 'MISMATCH'}")
    print(f"  Left-right seam (Top-Bottom borders): overall={h_overall:.2f}, worst={h_worst[2]:.2f} (pos {h_worst[0]}-{h_worst[1]}) -> {'OK' if h_ok else 'MISMATCH'}")

    # Save debug seam check
    dbg_path = target_dir / f"{stem}_seamcheck.png"
    save_seamcheck_debug(preview_2x2, v_segs, h_segs, threshold * 2.0, dbg_path)
    print(f"  Seamcheck map saved -> {dbg_path.name}")

    # Step 5: Check if already seamless
    if is_seamless:
        print("\n" + "*" * 62)
        print("  STEP 5: YES - The design pattern is already seamless!")
        print("*" * 62)
        seamless_out = target_dir / f"{stem}_seamless{ext}"
        shutil.copyfile(src, seamless_out)
        print(f"Original image returned byte-for-byte: {seamless_out.name}\n")
        return 0

    if preview_only:
        print("\n[Preview Only Mode]: Seam analysis complete. Exiting without AI repair.")
        return 1

    # Step 6: Classify Mismatch
    print("\nStep 6: Classify Mismatch Axis:")
    if not v_ok and h_ok:
        case = "vertical_only"
        print("  -> Case A: Only Left-Right borders mismatched.")
    elif v_ok and not h_ok:
        case = "horizontal_only"
        print("  -> Case B: Only Top-Bottom borders mismatched.")
    else:
        case = "both"
        print("  -> Case C: Both Left-Right and Top-Bottom borders mismatched.")

    # Select AI Model
    prov, model_id, key_env = select_model_interactive(provider, model_name)
    api_key = os.environ.get(key_env)
    if not api_key:
        api_key = input(f"\nEnter your {key_env}: ").strip()
        if not api_key:
            sys.exit(f"Error: {key_env} is required to use {prov}.")
        os.environ[key_env] = api_key

    prompt = get_adaptive_prompt(img, custom_prompt)
    current_img = img

    # Step 7 & 8: Repair and Verification Loop
    for rnd in range(1, max_rounds + 1):
        print("\n" + "=" * 62)
        print(f" STEP 7: AI REPAIR ROUND {rnd}/{max_rounds}")
        print("=" * 62)

        if case in ("vertical_only", "both"):
            current_img = repair_vertical_seam(current_img, prov, model_id, api_key, prompt, band_w=band_size)

        if case in ("horizontal_only", "both"):
            current_img = repair_horizontal_seam(current_img, prov, model_id, api_key, prompt, band_h=band_size)

        # Step 8: Re-verify repaired image (Back to Step 2 & 4)
        print(f"\nStep 8: Re-evaluating repaired tile (Round {rnd}) ...")
        is_seamless, v_ok, h_ok, v_stat, h_stat = check_seamless_detailed(current_img, threshold=threshold, floor=floor)
        print(f"  Re-check: Left-Right {'OK' if v_ok else 'MISMATCH'} (ratio={v_stat[0]:.2f}), "
              f"Top-Bottom {'OK' if h_ok else 'MISMATCH'} (ratio={h_stat[0]:.2f})")

        if is_seamless:
            print("\n" + "*" * 62)
            print("  SUCCESS: Tile is now 100% genuinely seamless!")
            print("*" * 62)
            break
        elif rnd < max_rounds:
            print("  Tile still has slight boundary discontinuity, running refinement round ...")
            # Update case for round 2
            if not v_ok and h_ok:
                case = "vertical_only"
            elif v_ok and not h_ok:
                case = "horizontal_only"
            else:
                case = "both"

    # Save repaired output
    status = "fixed" if is_seamless else "unfix"
    fixed_path = target_dir / f"{prov}_{status}_{stem}.png"
    if fixed_path.suffix.lower() in (".jpg", ".jpeg"):
        current_img.save(fixed_path, quality=100, subsampling=0)
    else:
        current_img.save(fixed_path)
    print(f"\nSaved repaired image -> {fixed_path.name}")

    # Generate final 2x2 and diamond for user inspection
    final_2x2 = create_2x2_tile(current_img)
    final_2x2_path = target_dir / f"{prov}_{status}_{stem}_2x2.png"
    final_2x2.save(final_2x2_path)
    print(f"Saved final 2x2 preview -> {final_2x2_path.name}")

    final_diamond = draw_diamond_overlay(final_2x2)
    final_dia_path = target_dir / f"{prov}_{status}_{stem}_diamond.png"
    final_diamond.save(final_dia_path)
    print(f"Saved final diamond preview -> {final_dia_path.name}")

    return 0 if is_seamless else 1


def main():
    p = argparse.ArgumentParser(description="Unified Seamless Pattern Tile Pipeline")
    p.add_argument("input", help="Path to single tile image")
    p.add_argument("--provider", choices=["replicate", "gpt", "gemini"], help="AI Provider")
    p.add_argument("--model", help="Model key (e.g. flux-pro, flux-dev, gpt-sunburst, gpt-flare, gemini-2.1, imagen-3)")
    p.add_argument("--threshold", type=float, default=1.1, help="Seam threshold (default 1.1)")
    p.add_argument("--floor", type=float, default=1.0, help="Floor for flat areas (default 1.0)")
    p.add_argument("--band", type=int, default=96, help="Repaint band width/height in px (default 96)")
    p.add_argument("--rounds", type=int, default=3, help="Max repair rounds (default 3)")
    p.add_argument("--prompt", help="Custom prompt override")
    p.add_argument("--preview-only", action="store_true", help="Run Steps 1-4 only (generate 2x2, diamond, seam check, no AI calls)")
    p.add_argument("-o", "--output-dir", help="Directory to save outputs")
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    return run_pipeline(
        image_path=args.input,
        provider=args.provider,
        model_name=args.model,
        threshold=args.threshold,
        floor=args.floor,
        band_size=args.band,
        max_rounds=args.rounds,
        preview_only=args.preview_only,
        custom_prompt=args.prompt,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
