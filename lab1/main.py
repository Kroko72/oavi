from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from utils import load_rgb_image, save_rgb_image, save_gray_image, make_triptych, ensure_factors
from color_models import split_rgb_color, rgb_to_hsi, intensity_to_uint8, invert_intensity_in_rgb
from resampling import upsample, decimate, resample_two_pass, resample_one_pass_rational


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab1: Color models + resampling (no library resizers).")
    p.add_argument("--input", required=True, help="Path to input BMP/PNG (truecolor, 3-channel).")
    p.add_argument("--out", default="out", help="Output directory.")
    p.add_argument("--M", type=int, default=2, help="Integer upsampling factor.")
    p.add_argument("--N", type=int, default=3, help="Integer decimation factor.")
    p.add_argument("--method", choices=["nearest", "bilinear"], default="bilinear",
                   help="Sampling method for interpolation / one-pass rational.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    M, N = ensure_factors(args.M, args.N)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb_image(args.input)
    save_rgb_image(out_dir / "00_original.png", rgb)

    # =========================
    # 1) Цветовые модели
    # =========================
    r, g, b = split_rgb_color(rgb)
    save_rgb_image(out_dir / "01_R.png", r)
    save_rgb_image(out_dir / "02_G.png", g)
    save_rgb_image(out_dir / "03_B.png", b)

    H, S, I = rgb_to_hsi(rgb)
    Iu8 = intensity_to_uint8(I)
    save_gray_image(out_dir / "04_HSI_intensity_I.png", Iu8)

    rgb_invI = invert_intensity_in_rgb(rgb)
    save_rgb_image(out_dir / "05_invert_intensity.png", rgb_invI)

    # For demo "before/after"
    make_triptych(rgb, rgb_invI, rgb).save(out_dir / "05_demo_invertI_before_after.png")

    # =========================
    # 2) Передискретизация
    # =========================
    # 2.1 Stretch by M
    up = upsample(rgb, M, method=args.method)
    save_rgb_image(out_dir / f"10_upsample_M{M}_{args.method}.png", up)
    make_triptych(rgb, up, rgb).save(out_dir / f"10_demo_upsample_M{M}_{args.method}.png")

    # 2.2 Compress by N (decimation)
    down = decimate(rgb, N)
    save_rgb_image(out_dir / f"11_decimate_N{N}.png", down)
    make_triptych(rgb, down, rgb).save(out_dir / f"11_demo_decimate_N{N}.png")

    # 2.3 Two-pass resampling K=M/N
    two = resample_two_pass(rgb, M, N, up_method=args.method)
    save_rgb_image(out_dir / f"12_two_pass_K{M}_over_{N}_{args.method}.png", two)
    make_triptych(rgb, two, rgb).save(out_dir / f"12_demo_two_pass_K{M}_over_{N}_{args.method}.png")

    # 2.4 One-pass resampling K=M/N
    one = resample_one_pass_rational(rgb, M, N, method=args.method)
    save_rgb_image(out_dir / f"13_one_pass_K{M}_over_{N}_{args.method}.png", one)
    make_triptych(rgb, one, rgb).save(out_dir / f"13_demo_one_pass_K{M}_over_{N}_{args.method}.png")

    # A small summary image (optional)
    # (No resizing — just a contact sheet with original + main outputs)
    sheet = Image.new("RGB", (1, 1))
    _ = sheet  # placeholder if you want to extend

    print("Done. Results saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()