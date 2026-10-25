"""Offline tests: no network, no paid call. The repair path uses LaMa."""

import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from seamfix import checker, composite, config, imaging, period, plan, synthetic

REPO = config.REPO


def v1():
    spec = importlib.util.spec_from_file_location("v1", REPO / "seamless_generator_latest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["flower.jpeg", "pattern.jpg", "sample.jpg", "test.jpg"])
def test_checker_decides_as_v1(name):
    """The port changes nothing the user confirmed."""
    img = Image.open(REPO / name).convert("RGB")
    old = v1().check_seamless_detailed(img)
    new = checker.check(np.asarray(img))
    assert (new.ok, new.lr.ok, new.tb.ok) == old[:3]
    assert new.lr.overall == pytest.approx(old[3][0])
    assert new.tb.overall == pytest.approx(old[4][0])


def test_synthetic_defects_break_only_their_join():
    base = synthetic.seamless()
    assert checker.check(np.asarray(base)).ok
    assert checker.check(np.asarray(synthetic.break_lr(base))).failing == ("lr",)
    assert checker.check(np.asarray(synthetic.break_tb(base))).failing == ("tb",)
    assert checker.check(np.asarray(synthetic.break_corner(base))).failing == ("lr", "tb")


def test_repeat_found_in_over_cropped_tile():
    tile = synthetic.seamless()
    big = synthetic.over_cropped(tile)
    cut = period.best_cut(big, np.asarray(big))
    assert cut is not None and cut.repeat.block == (512, 512) and cut.ok


def test_half_drop_gives_a_straight_tile_twice_as_wide():
    big = synthetic.half_drop(synthetic.seamless(300, 260))
    cut = period.best_cut(big, np.asarray(big))
    assert cut is not None and cut.ok and set(cut.proven) == {"lr", "tb"}  # pixel for pixel
    assert sorted(cut.repeat.block) == [260, 600]
    # the pixels just past the cut in the source are the cut's own first ones
    src, tile = np.asarray(big), np.asarray(cut.image)
    (oy, ox), (bw, bh) = cut.offset, cut.repeat.block
    if ox + bw < src.shape[1]:
        assert np.array_equal(src[oy : oy + bh, ox + bw], tile[:, 0])
    if oy + bh < src.shape[0]:
        assert np.array_equal(src[oy + bh, ox : ox + bw], tile[0])


@pytest.mark.parametrize("name,block", [("abcd.jpg", (355, 327)), ("peacock.jpg", (447, 447))])
def test_real_over_cropped_samples(name, block):
    tile = imaging.load(REPO / name)
    cut = period.best_cut(tile.image, tile.rgb)
    assert cut is not None and cut.repeat.block == block and cut.ok


@pytest.mark.parametrize("name", ["pattern.jpg", "sample.jpg", "design.jpeg"])
def test_no_repeat_where_there_is_none(name):
    tile = imaging.load(REPO / name)
    assert period.best_cut(tile.image, tile.rgb) is None


def test_wrapped_crop_round_trips():
    a = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    crop = imaging.take(a, -3, -4, 5, 6)
    assert crop[3, 4].tolist() == a[0, 0].tolist()
    b = a.copy()
    imaging.put(b, -3, -4, crop + 1, np.ones((5, 6), bool))
    assert (b != a).any(axis=2).sum() == 30
    assert b[0, 0].tolist() == (a[0, 0] + 1).tolist()


def test_blend_changes_nothing_outside_the_band():
    rng = np.random.default_rng(0)
    before = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    painted = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), bool)
    mask[:, 24:40] = True
    out = composite.blend(before, painted, mask)
    assert (out[~mask] == before[~mask]).all()
    assert composite.alpha(mask)[~mask].max() == 0


def test_fit_undoes_a_shift_and_a_tint():
    before = np.asarray(synthetic.seamless(256, 256))
    mask = np.zeros((256, 256), bool)
    mask[:, 112:144] = True
    moved = np.roll(before, (2, -3), axis=(0, 1)).astype(np.float32) * 0.95 + 6
    fixed, fit = composite.fit(np.clip(moved, 0, 255).astype(np.uint8), before, mask)
    assert (fit.dy, fit.dx) == pytest.approx((-2, 3), abs=0.3)
    assert fit.drift < 0.5


def test_plan_paints_the_corner_once_first():
    broken = np.asarray(synthetic.break_corner(synthetic.seamless()))
    p = plan.make(broken, ("lr", "tb"), "easy")
    assert p.windows[0].axes == ("lr", "tb")
    total = sum(int(w.mask.sum()) for w in p.windows)
    assert total == int(p.claimed.sum())  # no pixel claimed twice


def test_plan_stretch_wraps_across_the_corner():
    broken = np.asarray(synthetic.break_lr(synthetic.seamless(), start=0.9, length=0.2))
    spans = plan.spans(broken, "lr", whole=False)
    covered = np.zeros(512, bool)
    for start, size in spans:
        covered[np.arange(start, start + size) % 512] = True
    assert covered[500] and covered[10] and not covered[256]


@pytest.mark.parametrize("mode", ["L", "P", "RGBA", "CMYK", "LA"])
def test_write_back_keeps_mode_and_untouched_pixels(mode, tmp_path):
    src = synthetic.seamless(128, 128)
    native = src.convert(mode) if mode != "P" else src.quantize(32)
    path = tmp_path / f"tile.{'tif' if mode == 'CMYK' else 'png'}"
    native.save(path)
    tile = imaging.load(path)
    rgb = tile.rgb.copy()
    changed = np.zeros(rgb.shape[:2], bool)
    changed[:, :8] = True
    rgb[changed] = 255 - rgb[changed]
    out = imaging.write_back(tile, rgb, changed)
    assert out.mode == tile.image.mode and out.size == tile.size
    a, b = np.asarray(out), np.asarray(tile.image)
    assert (a[~changed] == b[~changed]).all()


def test_exif_rotation_is_applied(tmp_path):
    img = synthetic.seamless(120, 80)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 on display
    path = tmp_path / "rotated.jpg"
    img.save(path, exif=exif.tobytes())
    tile = imaging.load(path)
    assert tile.size == (80, 120) and tile.notes


def test_transparency_is_detected():
    img = synthetic.seamless(64, 64).convert("RGBA")
    a = np.asarray(img).copy()
    a[0, 0, 3] = 0
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    buf.seek(0)
    path = Path("/tmp/seamfix_transparent.png")
    path.write_bytes(buf.getvalue())
    assert not imaging.load(path).opaque
