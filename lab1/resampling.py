from __future__ import annotations

import numpy as np


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _get_pixel(img: np.ndarray, y: int, x: int) -> np.ndarray:
    h, w = img.shape[:2]
    y = _clamp(y, 0, h - 1)
    x = _clamp(x, 0, w - 1)
    return img[y, x]


def sample_nearest(img: np.ndarray, y: float, x: float) -> np.ndarray:
    return _get_pixel(img, int(round(y)), int(round(x)))


def sample_bilinear(img: np.ndarray, y: float, x: float) -> np.ndarray:
    """
    Bilinear sampling at fractional coords.
    Works for grayscale (H,W) and RGB (H,W,3).
    """
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = y0 + 1
    x1 = x0 + 1

    wy = y - y0
    wx = x - x0

    p00 = _get_pixel(img, y0, x0).astype(np.float32)
    p01 = _get_pixel(img, y0, x1).astype(np.float32)
    p10 = _get_pixel(img, y1, x0).astype(np.float32)
    p11 = _get_pixel(img, y1, x1).astype(np.float32)

    top = p00 * (1.0 - wx) + p01 * wx
    bot = p10 * (1.0 - wx) + p11 * wx
    val = top * (1.0 - wy) + bot * wy
    return val


def upsample(img: np.ndarray, M: int, method: str = "bilinear") -> np.ndarray:
    """
    Stretch image by integer factor M (upsampling).
    method: 'nearest' or 'bilinear'
    """
    if M <= 0:
        raise ValueError("M must be positive integer.")
    h, w = img.shape[:2]
    oh, ow = h * M, w * M

    out = np.zeros((oh, ow) + (() if img.ndim == 2 else (img.shape[2],)), dtype=np.float32)

    sampler = sample_bilinear if method == "bilinear" else sample_nearest

    # Pixel-center mapping
    for oy in range(oh):
        sy = (oy + 0.5) / M - 0.5
        for ox in range(ow):
            sx = (ox + 0.5) / M - 0.5
            out[oy, ox] = sampler(img, sy, sx)

    return np.clip(out, 0, 255).astype(np.uint8)


def decimate(img: np.ndarray, N: int) -> np.ndarray:
    """
    Compress image by integer factor N via subsampling (decimation):
    take every N-th pixel (no prefilter).
    """
    if N <= 0:
        raise ValueError("N must be positive integer.")
    h, w = img.shape[:2]
    oh, ow = max(1, h // N), max(1, w // N)
    if img.ndim == 2:
        return img[0:oh * N:N, 0:ow * N:N].copy()
    return img[0:oh * N:N, 0:ow * N:N, :].copy()


def resample_two_pass(img: np.ndarray, M: int, N: int, up_method: str = "bilinear") -> np.ndarray:
    """
    K = M/N via two passes:
      1) upsample by M
      2) decimate by N
    """
    tmp = upsample(img, M, method=up_method)
    out = decimate(tmp, N)
    return out


def resample_one_pass_rational(img: np.ndarray, M: int, N: int, method: str = "bilinear") -> np.ndarray:
    """
    One-pass resampling by rational factor K = M/N.
    Output size ~ round(H*K), round(W*K).
    Each output pixel is sampled directly from input.
    """
    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive integers.")
    K = M / N

    h, w = img.shape[:2]
    oh = max(1, int(round(h * K)))
    ow = max(1, int(round(w * K)))

    out = np.zeros((oh, ow) + (() if img.ndim == 2 else (img.shape[2],)), dtype=np.float32)
    sampler = sample_bilinear if method == "bilinear" else sample_nearest

    # Pixel-center mapping:
    # sy = (oy+0.5)/K - 0.5, sx = (ox+0.5)/K - 0.5
    for oy in range(oh):
        sy = (oy + 0.5) / K - 0.5
        for ox in range(ow):
            sx = (ox + 0.5) / K - 0.5
            out[oy, ox] = sampler(img, sy, sx)

    return np.clip(out, 0, 255).astype(np.uint8)