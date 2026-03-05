from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load BMP/PNG as uint8 RGB ndarray (H, W, 3)."""
    path = Path(path)
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image.")
    return arr


def save_rgb_image(path: str | Path, rgb: np.ndarray) -> None:
    """Save uint8 RGB (H, W, 3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    Image.fromarray(rgb, mode="RGB").save(path)


def save_gray_image(path: str | Path, gray: np.ndarray) -> None:
    """Save uint8 grayscale (H, W) as L."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    Image.fromarray(gray, mode="L").save(path)


def make_triptych(before: np.ndarray, middle: np.ndarray, after: np.ndarray) -> Image.Image:
    """
    Create a side-by-side image: before | middle | after
    Input arrays may be (H,W) or (H,W,3).
    """
    def to_pil(a: np.ndarray) -> Image.Image:
        if a.ndim == 2:
            a8 = a if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)
            return Image.fromarray(a8, mode="L").convert("RGB")
        if a.ndim == 3 and a.shape[2] == 3:
            a8 = a if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)
            return Image.fromarray(a8, mode="RGB")
        raise ValueError("Unsupported array shape for montage.")

    im1, im2, im3 = to_pil(before), to_pil(middle), to_pil(after)
    h = max(im1.height, im2.height, im3.height)
    w = im1.width + im2.width + im3.width

    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for im in (im1, im2, im3):
        y = (h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width
    return canvas


def ensure_factors(M: int, N: int) -> Tuple[int, int]:
    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive integers.")
    return int(M), int(N)