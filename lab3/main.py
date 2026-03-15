from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


ALLOWED_EXTENSIONS = {".bmp", ".png"}
COMPARISON_SCALE = 4


def load_binary_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    gray = np.array(image, dtype=np.uint8)
    return (gray >= 128).astype(np.uint8)


def build_white_fringe_masks() -> list[np.ndarray]:
    isolated = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    top_edge_variants = [
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.uint8),
    ]

    masks: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    # В лекции заданы изолированный пиксель и 6 апертур для верхнего края.
    # Остальные направления получаются поворотом на 90, 180 и 270 градусов.
    for mask in [isolated, *top_edge_variants]:
        for turns in range(4):
            rotated = np.rot90(mask, turns)
            key = tuple(int(value) for value in rotated.ravel())
            if key not in seen:
                seen.add(key)
                masks.append(rotated)

    return masks


def invert_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    return [1 - mask for mask in masks]


BLACK_FRINGE_MASKS = invert_masks(build_white_fringe_masks())


def erase_black_fringe(binary: np.ndarray) -> np.ndarray:
    padded = np.pad(binary, pad_width=1, mode="constant", constant_values=1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    remove_mask = np.zeros(binary.shape, dtype=bool)

    for mask in BLACK_FRINGE_MASKS:
        matches = np.all(windows == mask, axis=(2, 3))
        remove_mask |= matches

    filtered = binary.copy()
    filtered[remove_mask] = 1

    return filtered


def binary_to_uint8(binary: np.ndarray) -> np.ndarray:
    return (binary * 255).astype(np.uint8)


def binary_to_rgb(binary: np.ndarray) -> np.ndarray:
    gray = binary_to_uint8(binary)
    return np.stack([gray, gray, gray], axis=2)


def upscale_rgb(rgb: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return rgb

    enlarged = np.repeat(rgb, scale, axis=0)
    return np.repeat(enlarged, scale, axis=1)


def save_side_by_side(left: np.ndarray, right: np.ndarray, out_path: Path) -> None:
    left_rgb = upscale_rgb(binary_to_rgb(left), COMPARISON_SCALE)
    right_rgb = upscale_rgb(binary_to_rgb(right), COMPARISON_SCALE)

    height = max(left_rgb.shape[0], right_rgb.shape[0])
    width = left_rgb.shape[1] + right_rgb.shape[1]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    canvas[: left_rgb.shape[0], : left_rgb.shape[1]] = left_rgb
    canvas[: right_rgb.shape[0], left_rgb.shape[1] :] = right_rgb

    Image.fromarray(canvas, mode="RGB").save(out_path)


def process_image(path: Path, output_dir: Path) -> None:
    binary = load_binary_image(path)
    filtered = erase_black_fringe(binary)
    difference = np.bitwise_xor(binary, filtered).astype(np.uint8)

    filtered_dir = output_dir / "filtered"
    difference_dir = output_dir / "difference"
    comparisons_dir = output_dir / "comparisons"

    filtered_dir.mkdir(parents=True, exist_ok=True)
    difference_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = filtered_dir / f"{path.stem}_filtered.bmp"
    difference_path = difference_dir / f"{path.stem}_xor.bmp"

    Image.fromarray(binary_to_uint8(filtered), mode="L").save(filtered_path, format="BMP")
    Image.fromarray(binary_to_uint8(difference), mode="L").save(
        difference_path,
        format="BMP",
    )

    save_side_by_side(
        binary,
        filtered,
        comparisons_dir / f"{path.stem}_original_to_filtered.png",
    )
    save_side_by_side(
        binary,
        difference,
        comparisons_dir / f"{path.stem}_original_to_xor.png",
    )

    changed_pixels = int(np.count_nonzero(difference))
    print(f"OK: {path.name}")
    print(f"filtered: {filtered_path}")
    print(f"xor: {difference_path}")
    print(f"changed_pixels: {changed_pixels}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ЛР3: фильтрация бинарных изображений методом стирания чёрной бахромы",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input_images"),
        help="Папка с бинарными изображениями (.bmp, .png)",
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

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {args.input_dir}")

    image_paths = sorted(
        path for path in args.input_dir.iterdir() if path.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not image_paths:
        print(f"В папке {args.input_dir} нет .bmp или .png изображений")
        return

    for path in image_paths:
        process_image(path, args.output_dir)

    print("Готово")


if __name__ == "__main__":
    main()
