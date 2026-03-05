from __future__ import annotations

import numpy as np


_EPS = 1e-8


def split_rgb_gray(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return R,G,B channels as uint8 grayscale (H,W)."""
    r = rgb[:, :, 0].copy()
    g = rgb[:, :, 1].copy()
    b = rgb[:, :, 2].copy()
    return r, g, b


def split_rgb_color(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return R-only, G-only, B-only images as RGB (H,W,3):
      R-only = (R,0,0), etc.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image (H,W,3).")

    r = np.zeros_like(rgb)
    g = np.zeros_like(rgb)
    b = np.zeros_like(rgb)

    r[:, :, 0] = rgb[:, :, 0]
    g[:, :, 1] = rgb[:, :, 1]
    b[:, :, 2] = rgb[:, :, 2]
    return r, g, b


def split_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return R,G,B channels as uint8 grayscale images."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image (H,W,3).")
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return r.copy(), g.copy(), b.copy()


def rgb_to_hsi(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RGB (uint8) -> HSI
      H in [0, 2*pi)
      S in [0, 1]
      I in [0, 1]
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image (H,W,3).")

    rgbf = rgb.astype(np.float32) / 255.0
    R = rgbf[:, :, 0]
    G = rgbf[:, :, 1]
    B = rgbf[:, :, 2]

    I = (R + G + B) / 3.0

    # Saturation in HSI
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1.0 - (3.0 * min_rgb / (R + G + B + _EPS))
    S = np.clip(S, 0.0, 1.0)

    # Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + _EPS
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))  # [0, pi]

    H = np.where(B <= G, theta, 2.0 * np.pi - theta)  # [0, 2pi)
    H = np.mod(H, 2.0 * np.pi)

    return H.astype(np.float32), S.astype(np.float32), I.astype(np.float32)


def hsi_to_rgb(H: np.ndarray, S: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    HSI -> RGB uint8.
    Based on standard piecewise conversion for H in [0, 2pi).
    """
    H = H.astype(np.float32)
    S = np.clip(S.astype(np.float32), 0.0, 1.0)
    I = np.clip(I.astype(np.float32), 0.0, 1.0)

    R = np.zeros_like(I, dtype=np.float32)
    G = np.zeros_like(I, dtype=np.float32)
    B = np.zeros_like(I, dtype=np.float32)

    H2 = np.mod(H, 2.0 * np.pi)

    # Sector 0: 0 <= H < 2pi/3
    s0 = (H2 >= 0.0) & (H2 < 2.0 * np.pi / 3.0)
    h0 = H2[s0]
    B[s0] = I[s0] * (1.0 - S[s0])
    R[s0] = I[s0] * (1.0 + (S[s0] * np.cos(h0) / (np.cos(np.pi / 3.0 - h0) + _EPS)))
    G[s0] = 3.0 * I[s0] - (R[s0] + B[s0])

    # Sector 1: 2pi/3 <= H < 4pi/3
    s1 = (H2 >= 2.0 * np.pi / 3.0) & (H2 < 4.0 * np.pi / 3.0)
    h1 = H2[s1] - 2.0 * np.pi / 3.0
    R[s1] = I[s1] * (1.0 - S[s1])
    G[s1] = I[s1] * (1.0 + (S[s1] * np.cos(h1) / (np.cos(np.pi / 3.0 - h1) + _EPS)))
    B[s1] = 3.0 * I[s1] - (R[s1] + G[s1])

    # Sector 2: 4pi/3 <= H < 2pi
    s2 = (H2 >= 4.0 * np.pi / 3.0) & (H2 < 2.0 * np.pi)
    h2 = H2[s2] - 4.0 * np.pi / 3.0
    G[s2] = I[s2] * (1.0 - S[s2])
    B[s2] = I[s2] * (1.0 + (S[s2] * np.cos(h2) / (np.cos(np.pi / 3.0 - h2) + _EPS)))
    R[s2] = 3.0 * I[s2] - (G[s2] + B[s2])

    rgb = np.stack([R, G, B], axis=2)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def intensity_to_uint8(I: np.ndarray) -> np.ndarray:
    """HSI intensity [0..1] -> uint8 gray."""
    return (np.clip(I, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def invert_intensity_in_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB->HSI, invert I: I' = 1 - I, then HSI->RGB.
    """
    H, S, I = rgb_to_hsi(rgb)
    I2 = 1.0 - I
    return hsi_to_rgb(H, S, I2)