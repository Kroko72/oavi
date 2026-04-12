from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ALLOWED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}
ANGLES = (0, 90, 180, 270)
DEFAULT_DISTANCE = 1
DEFAULT_GAMMA = 0.75
DEFAULT_C = 1.0
DEFAULT_F0 = 0.0
EPS = 1e-12

HISTOGRAM_WIDTH = 768
HISTOGRAM_HEIGHT = 320
PANEL_TILE_WIDTH = 320
PANEL_TILE_HEIGHT = 240
PANEL_MARGIN = 16
LABEL_HEIGHT = 28

CSV_FIELDS = [
    "file",
    "is_color",
    "gamma",
    "distance",
    "angles",
    "pairs_count_before",
    "pairs_count_after",
    "corr_before",
    "corr_after",
    "corr_delta",
    "gray_min_before",
    "gray_max_before",
    "gray_min_after",
    "gray_max_after",
    "gray_mean_before",
    "gray_mean_after",
]


if hasattr(Image, "Resampling"):
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
else:
    RESAMPLE_NEAREST = Image.NEAREST
    RESAMPLE_BICUBIC = Image.BICUBIC


@dataclass
class PreparedImage:
    stem: str
    original_rgb: np.ndarray
    grayscale: np.ndarray
    contrasted_rgb: np.ndarray
    contrasted_grayscale: np.ndarray
    is_color: bool


@dataclass
class TextureResult:
    file: str
    is_color: bool
    gamma: float
    distance: int
    angles: str
    pairs_count_before: int
    pairs_count_after: int
    corr_before: float
    corr_after: float
    corr_delta: float
    gray_min_before: int
    gray_max_before: int
    gray_min_after: int
    gray_max_after: int
    gray_mean_before: float
    gray_mean_after: float


def sanitize_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace(".", "_")


def ensure_directories(base_dir: Path) -> dict[str, Path]:
    directories = {
        "grayscale": base_dir / "grayscale",
        "contrasted_grayscale": base_dir / "contrasted_grayscale",
        "contrasted_color": base_dir / "contrasted_color",
        "histograms": base_dir / "histograms",
        "matrices": base_dir / "matrices",
        "comparisons": base_dir / "comparisons",
        "reports": base_dir / "reports",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def image_to_rgb_array(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image, dtype=np.uint8)


def is_rgb_grayscale(rgb: np.ndarray) -> bool:
    return bool(
        np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
    )


def rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_float = rgb.astype(np.float32) / 255.0
    r = rgb_float[:, :, 0]
    g = rgb_float[:, :, 1]
    b = rgb_float[:, :, 2]

    max_channel = np.max(rgb_float, axis=2)
    min_channel = np.min(rgb_float, axis=2)
    delta = max_channel - min_channel

    lightness = (max_channel + min_channel) / 2.0

    saturation = np.zeros_like(lightness, dtype=np.float32)
    saturation_den = 1.0 - np.abs(2.0 * lightness - 1.0)
    mask = delta > EPS
    saturation[mask] = delta[mask] / np.maximum(saturation_den[mask], EPS)

    hue = np.zeros_like(lightness, dtype=np.float32)
    red_mask = mask & (max_channel == r)
    green_mask = mask & (max_channel == g)
    blue_mask = mask & (max_channel == b)

    hue[red_mask] = np.mod(((g - b)[red_mask] / delta[red_mask]), 6.0)
    hue[green_mask] = ((b - r)[green_mask] / delta[green_mask]) + 2.0
    hue[blue_mask] = ((r - g)[blue_mask] / delta[blue_mask]) + 4.0
    hue = (hue / 6.0) % 1.0

    return hue.astype(np.float32), saturation.astype(np.float32), lightness.astype(np.float32)


def hsl_to_rgb(hue: np.ndarray, saturation: np.ndarray, lightness: np.ndarray) -> np.ndarray:
    hue = np.mod(hue.astype(np.float32), 1.0)
    saturation = np.clip(saturation.astype(np.float32), 0.0, 1.0)
    lightness = np.clip(lightness.astype(np.float32), 0.0, 1.0)

    chroma = (1.0 - np.abs(2.0 * lightness - 1.0)) * saturation
    hue_prime = hue * 6.0
    x = chroma * (1.0 - np.abs(np.mod(hue_prime, 2.0) - 1.0))
    m = lightness - chroma / 2.0

    r1 = np.zeros_like(lightness, dtype=np.float32)
    g1 = np.zeros_like(lightness, dtype=np.float32)
    b1 = np.zeros_like(lightness, dtype=np.float32)

    sector0 = (0.0 <= hue_prime) & (hue_prime < 1.0)
    sector1 = (1.0 <= hue_prime) & (hue_prime < 2.0)
    sector2 = (2.0 <= hue_prime) & (hue_prime < 3.0)
    sector3 = (3.0 <= hue_prime) & (hue_prime < 4.0)
    sector4 = (4.0 <= hue_prime) & (hue_prime < 5.0)
    sector5 = (5.0 <= hue_prime) & (hue_prime <= 6.0)

    r1[sector0], g1[sector0], b1[sector0] = chroma[sector0], x[sector0], 0.0
    r1[sector1], g1[sector1], b1[sector1] = x[sector1], chroma[sector1], 0.0
    r1[sector2], g1[sector2], b1[sector2] = 0.0, chroma[sector2], x[sector2]
    r1[sector3], g1[sector3], b1[sector3] = 0.0, x[sector3], chroma[sector3]
    r1[sector4], g1[sector4], b1[sector4] = x[sector4], 0.0, chroma[sector4]
    r1[sector5], g1[sector5], b1[sector5] = chroma[sector5], 0.0, x[sector5]

    rgb = np.stack([r1 + m, g1 + m, b1 + m], axis=2)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def power_transform(channel: np.ndarray, gamma: float, c: float, f0: float) -> np.ndarray:
    normalized = np.clip(channel.astype(np.float32), 0.0, 1.0)
    transformed = c * np.power(normalized + f0, gamma)
    return np.clip(transformed, 0.0, 1.0).astype(np.float32)


def prepare_image(path: Path, gamma: float, c: float, f0: float) -> PreparedImage:
    rgb = image_to_rgb_array(path)
    color_image = not is_rgb_grayscale(rgb)

    if color_image:
        hue, saturation, lightness = rgb_to_hsl(rgb)
        contrasted_lightness = power_transform(lightness, gamma=gamma, c=c, f0=f0)
        grayscale = np.clip(np.rint(lightness * 255.0), 0, 255).astype(np.uint8)
        contrasted_grayscale = np.clip(np.rint(contrasted_lightness * 255.0), 0, 255).astype(np.uint8)
        contrasted_rgb = hsl_to_rgb(hue, saturation, contrasted_lightness)
    else:
        grayscale = rgb[:, :, 0].copy()
        lightness = grayscale.astype(np.float32) / 255.0
        contrasted_lightness = power_transform(lightness, gamma=gamma, c=c, f0=f0)
        contrasted_grayscale = np.clip(np.rint(contrasted_lightness * 255.0), 0, 255).astype(np.uint8)
        contrasted_rgb = np.repeat(contrasted_grayscale[:, :, None], 3, axis=2)

    return PreparedImage(
        stem=sanitize_stem(path),
        original_rgb=rgb,
        grayscale=grayscale,
        contrasted_rgb=contrasted_rgb,
        contrasted_grayscale=contrasted_grayscale,
        is_color=color_image,
    )


def build_glcm(gray: np.ndarray) -> np.ndarray:
    matrix = np.zeros((256, 256), dtype=np.uint64)

    src0 = gray[:, :-DEFAULT_DISTANCE]
    dst0 = gray[:, DEFAULT_DISTANCE:]
    np.add.at(matrix, (src0.ravel(), dst0.ravel()), 1)

    src90 = gray[DEFAULT_DISTANCE:, :]
    dst90 = gray[:-DEFAULT_DISTANCE, :]
    np.add.at(matrix, (src90.ravel(), dst90.ravel()), 1)

    src180 = gray[:, DEFAULT_DISTANCE:]
    dst180 = gray[:, :-DEFAULT_DISTANCE]
    np.add.at(matrix, (src180.ravel(), dst180.ravel()), 1)

    src270 = gray[:-DEFAULT_DISTANCE, :]
    dst270 = gray[DEFAULT_DISTANCE:, :]
    np.add.at(matrix, (src270.ravel(), dst270.ravel()), 1)

    return matrix


def normalize_glcm(matrix: np.ndarray) -> np.ndarray:
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Невозможно нормализовать пустую матрицу GLCM")
    return matrix.astype(np.float64) / float(total)


def calculate_corr(probability_matrix: np.ndarray) -> float:
    levels = np.arange(probability_matrix.shape[0], dtype=np.float64)
    row_profile = probability_matrix.sum(axis=1)
    column_profile = probability_matrix.sum(axis=0)

    mu_r = float(np.sum(levels * row_profile))
    mu_c = float(np.sum(levels * column_profile))

    sigma_r = float(np.sqrt(np.sum(((levels - mu_r) ** 2) * row_profile)))
    sigma_c = float(np.sqrt(np.sum(((levels - mu_c) ** 2) * column_profile)))

    if sigma_r < EPS or sigma_c < EPS:
        return 0.0

    i_grid, j_grid = np.indices(probability_matrix.shape, dtype=np.float64)
    numerator = np.sum((i_grid - mu_r) * (j_grid - mu_c) * probability_matrix)
    return float(numerator / (sigma_r * sigma_c))


def save_gray_image(array: np.ndarray, out_path: Path) -> None:
    Image.fromarray(array, mode="L").save(out_path)


def save_rgb_image(array: np.ndarray, out_path: Path) -> None:
    Image.fromarray(array, mode="RGB").save(out_path)


def histogram_from_gray(gray: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    return hist.astype(np.int64)


def create_histogram_image(histogram: np.ndarray, title: str) -> Image.Image:
    canvas = Image.new("RGB", (HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    left = 50
    right = HISTOGRAM_WIDTH - 20
    top = 35
    bottom = HISTOGRAM_HEIGHT - 35

    draw.text((left, 8), title, fill="black", font=font)
    draw.line((left, bottom, right, bottom), fill="black", width=1)
    draw.line((left, top, left, bottom), fill="black", width=1)

    max_value = int(histogram.max()) if histogram.size > 0 else 0
    if max_value == 0:
        return canvas

    usable_width = right - left
    usable_height = bottom - top

    for level, value in enumerate(histogram):
        x = left + int(level * usable_width / 255)
        bar_height = int(value * usable_height / max_value)
        draw.line((x, bottom, x, bottom - bar_height), fill="black", width=2)

    for tick in (0, 64, 128, 192, 255):
        x = left + int(tick * usable_width / 255)
        draw.line((x, bottom, x, bottom + 4), fill="black", width=1)
        draw.text((x - 10, bottom + 8), str(tick), fill="black", font=font)

    draw.text((6, top - 4), str(max_value), fill="black", font=font)
    draw.text((8, bottom - 10), "0", fill="black", font=font)
    return canvas


def create_glcm_visualization(matrix: np.ndarray, title: str) -> Image.Image:
    non_zero_count = int(np.count_nonzero(matrix))
    use_log = non_zero_count < int(matrix.size * 0.25)

    values = np.log1p(matrix.astype(np.float64)) if use_log else matrix.astype(np.float64)
    max_value = float(values.max()) if values.size > 0 else 0.0

    if max_value > 0.0:
        normalized = np.clip(np.rint(values / max_value * 255.0), 0, 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(matrix, dtype=np.uint8)

    image = Image.fromarray(normalized, mode="L").resize((512, 512), RESAMPLE_NEAREST).convert("RGB")
    canvas = Image.new("RGB", (560, 560), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    canvas.paste(image, (24, 24))
    draw.rectangle((23, 23, 536, 536), outline="black", width=1)
    draw.text((24, 540), title, fill="black", font=font)
    return canvas


def create_panel(items: list[tuple[str, Image.Image]], out_path: Path) -> None:
    font = ImageFont.load_default()
    columns = 2
    rows = (len(items) + columns - 1) // columns

    width = columns * (PANEL_TILE_WIDTH + PANEL_MARGIN) + PANEL_MARGIN
    height = rows * (PANEL_TILE_HEIGHT + LABEL_HEIGHT + PANEL_MARGIN) + PANEL_MARGIN
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (title, image) in enumerate(items):
        row = index // columns
        column = index % columns
        x0 = PANEL_MARGIN + column * (PANEL_TILE_WIDTH + PANEL_MARGIN)
        y0 = PANEL_MARGIN + row * (PANEL_TILE_HEIGHT + LABEL_HEIGHT + PANEL_MARGIN)

        fitted = ImageOps.contain(image, (PANEL_TILE_WIDTH, PANEL_TILE_HEIGHT), RESAMPLE_BICUBIC)
        px = x0 + (PANEL_TILE_WIDTH - fitted.width) // 2
        py = y0 + (PANEL_TILE_HEIGHT - fitted.height) // 2

        draw.rectangle(
            (x0, y0, x0 + PANEL_TILE_WIDTH, y0 + PANEL_TILE_HEIGHT),
            outline="black",
            width=1,
        )
        canvas.paste(fitted, (px, py))
        draw.text((x0, y0 + PANEL_TILE_HEIGHT + 6), title, fill="black", font=font)

    canvas.save(out_path)


def save_matrix_csv(matrix: np.ndarray, out_path: Path) -> None:
    np.savetxt(out_path, matrix, fmt="%d", delimiter=";")


def describe_delta(delta: float) -> str:
    if delta > 0:
        return "увеличился"
    if delta < 0:
        return "уменьшился"
    return "не изменился"


def process_image(path: Path, output_dirs: dict[str, Path], gamma: float) -> TextureResult:
    prepared = prepare_image(path, gamma=gamma, c=DEFAULT_C, f0=DEFAULT_F0)

    gray_path = output_dirs["grayscale"] / f"{prepared.stem}_gray.png"
    contrasted_gray_path = output_dirs["contrasted_grayscale"] / f"{prepared.stem}_power_gray.png"
    contrasted_color_path = output_dirs["contrasted_color"] / f"{prepared.stem}_power_color.png"

    save_gray_image(prepared.grayscale, gray_path)
    save_gray_image(prepared.contrasted_grayscale, contrasted_gray_path)
    save_rgb_image(prepared.contrasted_rgb, contrasted_color_path)

    hist_before = histogram_from_gray(prepared.grayscale)
    hist_after = histogram_from_gray(prepared.contrasted_grayscale)

    hist_before_image = create_histogram_image(hist_before, "Гистограмма яркости до преобразования")
    hist_after_image = create_histogram_image(hist_after, "Гистограмма яркости после преобразования")

    hist_before_path = output_dirs["histograms"] / f"{prepared.stem}_hist_before.png"
    hist_after_path = output_dirs["histograms"] / f"{prepared.stem}_hist_after.png"
    hist_before_image.save(hist_before_path)
    hist_after_image.save(hist_after_path)

    glcm_before = build_glcm(prepared.grayscale)
    glcm_after = build_glcm(prepared.contrasted_grayscale)
    prob_before = normalize_glcm(glcm_before)
    prob_after = normalize_glcm(glcm_after)

    corr_before = calculate_corr(prob_before)
    corr_after = calculate_corr(prob_after)

    matrix_before_path = output_dirs["matrices"] / f"{prepared.stem}_glcm_before.csv"
    matrix_after_path = output_dirs["matrices"] / f"{prepared.stem}_glcm_after.csv"
    save_matrix_csv(glcm_before, matrix_before_path)
    save_matrix_csv(glcm_after, matrix_after_path)

    glcm_before_image = create_glcm_visualization(glcm_before, "GLCM до преобразования")
    glcm_after_image = create_glcm_visualization(glcm_after, "GLCM после преобразования")

    glcm_before_image_path = output_dirs["matrices"] / f"{prepared.stem}_glcm_before.png"
    glcm_after_image_path = output_dirs["matrices"] / f"{prepared.stem}_glcm_after.png"
    glcm_before_image.save(glcm_before_image_path)
    glcm_after_image.save(glcm_after_image_path)

    original_view = Image.fromarray(prepared.original_rgb, mode="RGB")
    gray_view = Image.fromarray(prepared.grayscale, mode="L").convert("RGB")
    contrasted_gray_view = Image.fromarray(prepared.contrasted_grayscale, mode="L").convert("RGB")
    contrasted_color_view = Image.fromarray(prepared.contrasted_rgb, mode="RGB")

    images_panel_path = output_dirs["comparisons"] / f"{prepared.stem}_images.png"
    analysis_panel_path = output_dirs["comparisons"] / f"{prepared.stem}_analysis.png"

    create_panel(
        [
            ("Исходное изображение", original_view),
            ("Полутоновое изображение", gray_view),
            ("Контрастированное цветное", contrasted_color_view),
            ("Контрастированное полутоновое", contrasted_gray_view),
        ],
        images_panel_path,
    )

    create_panel(
        [
            ("Гистограмма до", hist_before_image),
            ("Гистограмма после", hist_after_image),
            ("GLCM до", glcm_before_image),
            ("GLCM после", glcm_after_image),
        ],
        analysis_panel_path,
    )

    return TextureResult(
        file=path.name,
        is_color=prepared.is_color,
        gamma=gamma,
        distance=DEFAULT_DISTANCE,
        angles="{0,90,180,270}",
        pairs_count_before=int(glcm_before.sum()),
        pairs_count_after=int(glcm_after.sum()),
        corr_before=corr_before,
        corr_after=corr_after,
        corr_delta=corr_after - corr_before,
        gray_min_before=int(prepared.grayscale.min()),
        gray_max_before=int(prepared.grayscale.max()),
        gray_min_after=int(prepared.contrasted_grayscale.min()),
        gray_max_after=int(prepared.contrasted_grayscale.max()),
        gray_mean_before=float(prepared.grayscale.mean()),
        gray_mean_after=float(prepared.contrasted_grayscale.mean()),
    )


def write_results_csv(results: list[TextureResult], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, delimiter=";")
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def write_report(results: list[TextureResult], out_path: Path) -> None:
    gamma_value = results[0].gamma if results else DEFAULT_GAMMA
    lines = [
        "Лабораторная работа №8",
        "Вариант: GLCM, d=1, phi={0,90,180,270}, признак CORR, степенное преобразование",
        f"Параметры степенного преобразования: c={DEFAULT_C}, f0={DEFAULT_F0}, gamma={gamma_value}",
        "",
    ]

    for result in results:
        delta_description = describe_delta(result.corr_delta)
        lines.extend(
            [
                f"Файл: {result.file}",
                f"  Цветное изображение: {'да' if result.is_color else 'нет'}",
                f"  CORR до преобразования: {result.corr_before:.6f}",
                f"  CORR после преобразования: {result.corr_after:.6f}",
                f"  Изменение CORR: {result.corr_delta:+.6f} ({delta_description})",
                f"  Диапазон яркости до: [{result.gray_min_before}, {result.gray_max_before}]",
                f"  Диапазон яркости после: [{result.gray_min_after}, {result.gray_max_after}]",
                f"  Средняя яркость до: {result.gray_mean_before:.2f}",
                f"  Средняя яркость после: {result.gray_mean_after:.2f}",
                "",
            ]
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Лабораторная работа №8: текстурный анализ GLCM и степенное контрастирование",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input_images"),
        help="Папка с входными изображениями",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Папка для результатов",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Показатель степени gamma для преобразования яркости",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if args.gamma <= 0:
        raise ValueError("Параметр gamma должен быть положительным")

    image_paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(f"В папке {input_dir} не найдено поддерживаемых изображений")

    output_dirs = ensure_directories(output_dir)

    results: list[TextureResult] = []
    for image_path in image_paths:
        results.append(process_image(image_path, output_dirs=output_dirs, gamma=args.gamma))

    write_results_csv(results, output_dir / "results.csv")
    write_report(results, output_dirs["reports"] / "report.txt")


if __name__ == "__main__":
    main()
