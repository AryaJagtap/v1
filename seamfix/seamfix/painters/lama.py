"""Big-LaMa, run locally on the CPU. Free, deterministic, and it fills only the
mask: it has no prompt to misread and cannot write text.

The weights are the TorchScript published by simple-lama-inpainting. The pip
package itself pins pillow<10 and numpy<2, which do not install on Python 3.12,
so this does what it does: pad to a multiple of 8, run, crop back.
"""

import os
import threading
import urllib.request

import numpy as np
import torch

from seamfix import config
from seamfix.painters import Request

_model = None
_lock = threading.Lock()


def _load():
    global _model
    with _lock:
        if _model is None:
            path = config.LAMA_PATH
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                part = path.with_suffix(".part")
                urllib.request.urlretrieve(config.LAMA_URL, part)
                part.rename(path)
            torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) - 2)))
            _model = torch.jit.load(str(path), map_location="cpu").eval()
    return _model


class Lama:
    name = "lama"
    paid = False

    def paint(self, request: Request) -> np.ndarray:
        model = _load()
        crop, mask = request.crop, request.mask
        h, w = mask.shape
        ph, pw = (-h) % 8, (-w) % 8
        img = np.pad(crop, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        msk = np.pad(mask, ((0, ph), (0, pw)), mode="symmetric")
        image = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
        holes = torch.from_numpy(msk.astype(np.float32))[None, None]
        with _lock, torch.inference_mode():
            out = model(image, holes)
        out = out[0].permute(1, 2, 0).numpy()[:h, :w]
        return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
