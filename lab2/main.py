from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


ALLOWED_EXTENSIONS = {".bmp", ".png"}
WINDOW_SIZE = 3


def load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path)
    rgb = np.array(image, dtype=np.uint8)

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"{path.name}: ожидается полноцветное 3-канальное изображение.")

    return rgb[:, :, :3]


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    gray = 0.3 * r + 0.59 * g + 0.11 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def adaptive_threshold_minimax(gray: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    if window_size != 3:
        raise ValueError("Для этого варианта поддерживается окно 3x3.")

    pad = window_size // 2
    padded = np.pad(gray, pad_width=pad, mode="edge")

    neighbors = []
    for dy in range(window_size):
        for dx in range(window_size):
            neighbors.append(padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]])

    local_min = np.minimum.reduce(neighbors).astype(np.float32)
    local_max = np.maximum.reduce(neighbors).astype(np.float32)
    threshold = (local_min + local_max) / 2.0

    return (gray.astype(np.float32) > threshold).astype(np.uint8) * 255


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=2)


def binary_to_rgb(binary: np.ndarray) -> np.ndarray:
    return np.stack([binary, binary, binary], axis=2)


def save_side_by_side(left_rgb: np.ndarray, right_rgb: np.ndarray, out_path: Path) -> None:
    h = max(left_rgb.shape[0], right_rgb.shape[0])
    w = left_rgb.shape[1] + right_rgb.shape[1]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    canvas[: left_rgb.shape[0], : left_rgb.shape[1]] = left_rgb
    canvas[: right_rgb.shape[0], left_rgb.shape[1] :] = right_rgb

    Image.fromarray(canvas, mode="RGB").save(out_path)


def process_image(path: Path, output_dir: Path) -> None:
    rgb = load_rgb_image(path)
    gray = rgb_to_grayscale(rgb)
    binary = adaptive_threshold_minimax(gray, window_size=WINDOW_SIZE)

    grayscale_dir = output_dir / "grayscale"
    binary_dir = output_dir / "binary"
    compare_dir = output_dir / "comparisons"
    grayscale_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    gray_path = grayscale_dir / f"{path.stem}_gray.bmp"
    binary_path = binary_dir / f"{path.stem}_binary.bmp"

    Image.fromarray(gray, mode="L").save(gray_path, format="BMP")
    Image.fromarray(binary.astype(bool)).save(binary_path, format="BMP")

    save_side_by_side(
        rgb,
        gray_to_rgb(gray),
        compare_dir / f"{path.stem}_color_to_gray.png",
    )
    save_side_by_side(
        gray_to_rgb(gray),
        binary_to_rgb(binary),
        compare_dir / f"{path.stem}_gray_to_binary.png",
    )

    print(f"OK: {path.name}")
    print(f"gray: {gray_path}")
    print(f"binary: {binary_path}")
    print(f"compare: {compare_dir / f'{path.stem}_color_to_gray.png'}")
    print(f"compare: {compare_dir / f'{path.stem}_gray_to_binary.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ЛР2: обесцвечивание и бинаризация растровых изображений"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input_images"),
        help="Папка с исходными изображениями (.bmp, .png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Папка для результатов",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {input_dir}")

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not image_paths:
        print(f"В папке {input_dir} нет .bmp или .png изображений")
        return

    for path in image_paths:
        process_image(path, output_dir)

    print("Готово")


if __name__ == "__main__":
    main()
