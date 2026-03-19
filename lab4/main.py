from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


ALLOWED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}
KAYYALI_GX = np.array(
    [
        [6, 0, -6],
        [0, 0, 0],
        [-6, 0, 6],
    ],
    dtype=np.float32,
)
KAYYALI_GY = np.array(
    [
        [-6, 0, 6],
        [0, 0, 0],
        [6, 0, -6],
    ],
    dtype=np.float32,
)


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    gray = 0.3 * r + 0.59 * g + 0.11 * b
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path)
    array = np.array(image)

    if array.ndim == 2:
        gray = array.astype(np.uint8)
        original_rgb = np.stack([gray, gray, gray], axis=2)
        return original_rgb, gray

    if array.ndim == 3 and array.shape[2] >= 3:
        original_rgb = array[:, :, :3].astype(np.uint8)
        gray = rgb_to_grayscale(original_rgb)
        return original_rgb, gray

    raise ValueError(f"{path.name}: неподдерживаемый формат изображения.")


def convolve_3x3(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padded = np.pad(gray.astype(np.float32), pad_width=1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.sum(windows * kernel[None, None, :, :], axis=(2, 3))


def normalize_signed(values: np.ndarray) -> np.ndarray:
    min_value = float(values.min())
    max_value = float(values.max())

    if max_value == min_value:
        return np.zeros(values.shape, dtype=np.uint8)

    normalized = 255.0 * (values - min_value) / (max_value - min_value)
    return np.clip(np.rint(normalized), 0, 255).astype(np.uint8)


def normalize_magnitude(values: np.ndarray) -> np.ndarray:
    max_value = float(values.max())

    if max_value <= 0:
        return np.zeros(values.shape, dtype=np.uint8)

    normalized = 255.0 * values / max_value
    return np.clip(np.rint(normalized), 0, 255).astype(np.uint8)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=2)


def save_side_by_side(left_rgb: np.ndarray, right_rgb: np.ndarray, out_path: Path) -> None:
    height = max(left_rgb.shape[0], right_rgb.shape[0])
    width = left_rgb.shape[1] + right_rgb.shape[1]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    canvas[: left_rgb.shape[0], : left_rgb.shape[1]] = left_rgb
    canvas[: right_rgb.shape[0], left_rgb.shape[1] :] = right_rgb

    Image.fromarray(canvas, mode="RGB").save(out_path)


def save_grid(images: list[np.ndarray], rows: int, cols: int, out_path: Path) -> None:
    if len(images) != rows * cols:
        raise ValueError("Количество изображений не совпадает с размером сетки.")

    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    canvas = np.full((rows * height, cols * width, 3), 255, dtype=np.uint8)

    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        y0 = row * height
        x0 = col * width
        canvas[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image

    Image.fromarray(canvas, mode="RGB").save(out_path)


def process_image(path: Path, output_dir: Path, threshold: int) -> None:
    original_rgb, gray = load_image(path)

    gx = convolve_3x3(gray, KAYYALI_GX)
    gy = convolve_3x3(gray, KAYYALI_GY)
    g = np.sqrt(gx * gx + gy * gy)

    gx_norm = normalize_signed(gx)
    gy_norm = normalize_signed(gy)
    g_norm = normalize_magnitude(g)
    binary = np.where(g_norm > threshold, 255, 0).astype(np.uint8)

    grayscale_dir = output_dir / "grayscale"
    gx_dir = output_dir / "gx"
    gy_dir = output_dir / "gy"
    g_dir = output_dir / "g"
    binary_dir = output_dir / "binary"
    compare_dir = output_dir / "comparisons"

    for directory in [
        grayscale_dir,
        gx_dir,
        gy_dir,
        g_dir,
        binary_dir,
        compare_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    gray_path = grayscale_dir / f"{path.stem}_gray.bmp"
    gx_path = gx_dir / f"{path.stem}_gx.bmp"
    gy_path = gy_dir / f"{path.stem}_gy.bmp"
    g_path = g_dir / f"{path.stem}_g.bmp"
    binary_path = binary_dir / f"{path.stem}_binary.bmp"

    Image.fromarray(gray, mode="L").save(gray_path, format="BMP")
    Image.fromarray(gx_norm, mode="L").save(gx_path, format="BMP")
    Image.fromarray(gy_norm, mode="L").save(gy_path, format="BMP")
    Image.fromarray(g_norm, mode="L").save(g_path, format="BMP")
    Image.fromarray(binary, mode="L").save(binary_path, format="BMP")

    save_side_by_side(
        original_rgb,
        gray_to_rgb(gray),
        compare_dir / f"{path.stem}_original_to_gray.png",
    )
    save_grid(
        [
            gray_to_rgb(gx_norm),
            gray_to_rgb(gy_norm),
            gray_to_rgb(g_norm),
        ],
        rows=1,
        cols=3,
        out_path=compare_dir / f"{path.stem}_gx_gy_g.png",
    )
    save_side_by_side(
        gray_to_rgb(g_norm),
        gray_to_rgb(binary),
        compare_dir / f"{path.stem}_g_to_binary.png",
    )
    Image.fromarray(gray_to_rgb(binary), mode="RGB").save(
        compare_dir / f"{path.stem}_binary_only.png",
    )
    save_grid(
        [
            original_rgb,
            gray_to_rgb(gray),
            gray_to_rgb(gx_norm),
            gray_to_rgb(gy_norm),
            gray_to_rgb(g_norm),
            gray_to_rgb(binary),
        ],
        rows=2,
        cols=3,
        out_path=compare_dir / f"{path.stem}_summary.png",
    )

    print(f"OK: {path.name}")
    print(f"gray: {gray_path}")
    print(f"gx: {gx_path}")
    print(f"gy: {gy_path}")
    print(f"g: {g_path}")
    print(f"binary: {binary_path}")
    print(f"threshold: {threshold}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ЛР4: выделение контуров оператором Кайяли",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input_images"),
        help="Папка с исходными изображениями",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Папка для результатов",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=64,
        help="Порог бинаризации для нормализованной матрицы G (0..255)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0 <= args.threshold <= 255:
        raise ValueError("Порог должен быть в диапазоне от 0 до 255.")

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {args.input_dir}")

    image_paths = sorted(
        path for path in args.input_dir.iterdir() if path.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not image_paths:
        print(f"В папке {args.input_dir} нет изображений подходящего формата")
        return

    for path in image_paths:
        process_image(path, args.output_dir, args.threshold)

    print("Готово")


if __name__ == "__main__":
    main()
