from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


GREEK_SYMBOLS = [
    ("alpha", "α"),
    ("beta", "β"),
    ("gamma", "γ"),
    ("delta", "δ"),
    ("epsilon", "ε"),
    ("zeta", "ζ"),
    ("eta", "η"),
    ("theta", "θ"),
    ("iota", "ι"),
    ("kappa", "κ"),
    ("lambda", "λ"),
    ("mu", "μ"),
    ("nu", "ν"),
    ("xi", "ξ"),
    ("omicron", "ο"),
    ("pi", "π"),
    ("rho", "ρ"),
    ("sigma", "σ"),
    ("tau", "τ"),
    ("upsilon", "υ"),
    ("phi", "φ"),
    ("chi", "χ"),
    ("psi", "ψ"),
    ("omega", "ω"),
]
DEFAULT_FONT_PATH = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
DEFAULT_FONT_SIZE = 52
BINARY_THRESHOLD = 128
BAR_SIZE = 12
CHART_COLOR = 30
GRID_COLOR = 210
BORDER_COLOR = 0
BACKGROUND_COLOR = 255

CSV_FIELDS = [
    "file",
    "symbol_name",
    "symbol",
    "font_name",
    "font_size",
    "width",
    "height",
    "weight_q1",
    "weight_q2",
    "weight_q3",
    "weight_q4",
    "specific_weight_q1",
    "specific_weight_q2",
    "specific_weight_q3",
    "specific_weight_q4",
    "center_x",
    "center_y",
    "center_x_norm",
    "center_y_norm",
    "inertia_x",
    "inertia_y",
    "inertia_x_norm",
    "inertia_y_norm",
]


def binary_to_uint8(binary: np.ndarray) -> np.ndarray:
    return np.where(binary == 1, 0, 255).astype(np.uint8)


def load_font(font_path: Path, font_size: int) -> ImageFont.FreeTypeFont:
    if not font_path.exists():
        raise FileNotFoundError(f"Файл шрифта не найден: {font_path}")
    return ImageFont.truetype(str(font_path), font_size)


def crop_binary_image(binary: np.ndarray) -> np.ndarray:
    rows, cols = np.where(binary == 1)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("На изображении не найдено черных пикселей")

    top = int(rows.min())
    bottom = int(rows.max()) + 1
    left = int(cols.min())
    right = int(cols.max()) + 1
    return binary[top:bottom, left:right]


def render_symbol(symbol: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    left, top, right, bottom = font.getbbox(symbol)
    margin = max(10, font.size // 2)
    width = right - left + 2 * margin
    height = bottom - top + 2 * margin

    image = Image.new("L", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    draw.text((margin - left, margin - top), symbol, font=font, fill=0)

    gray = np.array(image, dtype=np.uint8)
    binary = (gray < BINARY_THRESHOLD).astype(np.uint8)
    return crop_binary_image(binary)


def save_binary_image(binary: np.ndarray, out_path: Path) -> None:
    Image.fromarray(binary_to_uint8(binary), mode="L").save(out_path)


def split_into_quarters(binary: np.ndarray) -> list[np.ndarray]:
    height, width = binary.shape
    middle_y = height // 2
    middle_x = width // 2

    return [
        binary[:middle_y, :middle_x],
        binary[:middle_y, middle_x:],
        binary[middle_y:, :middle_x],
        binary[middle_y:, middle_x:],
    ]


def calculate_scalar_features(binary: np.ndarray) -> dict[str, int | float]:
    height, width = binary.shape
    weight = float(binary.sum())

    if weight == 0:
        raise ValueError("Невозможно вычислить признаки: вес символа равен нулю")

    quarters = split_into_quarters(binary)
    quarter_weights = [int(part.sum()) for part in quarters]
    quarter_specific_weights = []
    for part, part_weight in zip(quarters, quarter_weights, strict=True):
        area = part.shape[0] * part.shape[1]
        quarter_specific_weights.append(part_weight / area if area else 0.0)

    y_indices, x_indices = np.indices(binary.shape)
    center_x = float((x_indices * binary).sum() / weight)
    center_y = float((y_indices * binary).sum() / weight)

    center_x_norm = center_x / (width - 1) if width > 1 else 0.0
    center_y_norm = center_y / (height - 1) if height > 1 else 0.0

    inertia_x = float((((y_indices - center_y) ** 2) * binary).sum())
    inertia_y = float((((x_indices - center_x) ** 2) * binary).sum())
    normalization = float((width**2) * (height**2))
    inertia_x_norm = inertia_x / normalization if normalization else 0.0
    inertia_y_norm = inertia_y / normalization if normalization else 0.0

    return {
        "width": width,
        "height": height,
        "weight_q1": quarter_weights[0],
        "weight_q2": quarter_weights[1],
        "weight_q3": quarter_weights[2],
        "weight_q4": quarter_weights[3],
        "specific_weight_q1": quarter_specific_weights[0],
        "specific_weight_q2": quarter_specific_weights[1],
        "specific_weight_q3": quarter_specific_weights[2],
        "specific_weight_q4": quarter_specific_weights[3],
        "center_x": center_x,
        "center_y": center_y,
        "center_x_norm": center_x_norm,
        "center_y_norm": center_y_norm,
        "inertia_x": inertia_x,
        "inertia_y": inertia_y,
        "inertia_x_norm": inertia_x_norm,
        "inertia_y_norm": inertia_y_norm,
    }


def calculate_profile_x(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=0).astype(int)


def calculate_profile_y(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=1).astype(int)


def choose_tick_step(max_value: int, target_ticks: int = 5) -> int:
    if max_value <= 0:
        return 1

    raw_step = max_value / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))

    for factor in (1, 2, 5, 10):
        step = int(magnitude * factor)
        if step >= raw_step:
            return max(step, 1)

    return max(int(math.ceil(raw_step)), 1)


def collect_ticks(last_value: int, target_ticks: int = 6) -> list[int]:
    if last_value <= 0:
        return [0]

    step = choose_tick_step(last_value, target_ticks)
    ticks = list(range(0, last_value + 1, step))
    if ticks[-1] != last_value:
        ticks.append(last_value)
    return ticks


def format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def write_features_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def save_profile_x(values: np.ndarray, symbol: str, out_path: Path, font_path: Path) -> None:
    plot_width = max(len(values) * BAR_SIZE, 180)
    plot_height = 260
    left_margin = 68
    right_margin = 30
    top_margin = 52
    bottom_margin = 72
    width = left_margin + plot_width + right_margin
    height = top_margin + plot_height + bottom_margin

    image = Image.new("L", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(font_path), 24)
    label_font = ImageFont.truetype(str(font_path), 16)

    max_value = int(values.max()) if len(values) else 0
    max_value = max(max_value, 1)

    x0 = left_margin
    y0 = height - bottom_margin
    x1 = width - right_margin
    y1 = top_margin

    draw.rectangle((x0, y1, x1, y0), outline=BORDER_COLOR, width=1)

    for tick in collect_ticks(max_value):
        y = y0 - int(round((tick / max_value) * plot_height))
        draw.line((x0, y, x1, y), fill=GRID_COLOR, width=1)
        label = str(tick)
        label_width, label_height = measure_text(draw, label, label_font)
        draw.text((x0 - 10 - label_width, y - label_height // 2), label, fill=0, font=label_font)

    last_index = len(values) - 1
    for tick in collect_ticks(last_index if last_index >= 0 else 0):
        x = x0 + int(round((tick + 0.5) * BAR_SIZE))
        draw.line((x, y0, x, y0 + 5), fill=BORDER_COLOR, width=1)
        label = str(tick)
        label_width, _ = measure_text(draw, label, label_font)
        draw.text((x - label_width // 2, y0 + 10), label, fill=0, font=label_font)

    for index, value in enumerate(values):
        bar_left = x0 + index * BAR_SIZE + 1
        bar_right = x0 + (index + 1) * BAR_SIZE - 1
        bar_height = int(round((int(value) / max_value) * plot_height))
        bar_top = y0 - bar_height
        if value > 0:
            draw.rectangle((bar_left, bar_top, bar_right, y0 - 1), fill=CHART_COLOR, outline=CHART_COLOR)

    title = f"Профиль X: {symbol}"
    title_width, _ = measure_text(draw, title, title_font)
    draw.text(((width - title_width) // 2, 12), title, fill=0, font=title_font)
    draw.text((x0, y0 + 40), "x", fill=0, font=label_font)
    draw.text((x0 - 42, y1 - 24), "Вес", fill=0, font=label_font)

    image.save(out_path)


def save_profile_y(values: np.ndarray, symbol: str, out_path: Path, font_path: Path) -> None:
    plot_width = 320
    plot_height = max(len(values) * BAR_SIZE, 180)
    left_margin = 68
    right_margin = 40
    top_margin = 52
    bottom_margin = 52
    width = left_margin + plot_width + right_margin
    height = top_margin + plot_height + bottom_margin

    image = Image.new("L", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(font_path), 24)
    label_font = ImageFont.truetype(str(font_path), 16)

    max_value = int(values.max()) if len(values) else 0
    max_value = max(max_value, 1)

    x0 = left_margin
    y0 = top_margin
    x1 = width - right_margin
    y1 = height - bottom_margin

    draw.rectangle((x0, y0, x1, y1), outline=BORDER_COLOR, width=1)

    for tick in collect_ticks(max_value):
        x = x0 + int(round((tick / max_value) * plot_width))
        draw.line((x, y0, x, y1), fill=GRID_COLOR, width=1)
        label = str(tick)
        label_width, label_height = measure_text(draw, label, label_font)
        draw.text((x - label_width // 2, y1 + 10), label, fill=0, font=label_font)

    last_index = len(values) - 1
    for tick in collect_ticks(last_index if last_index >= 0 else 0):
        y = y0 + int(round((tick + 0.5) * BAR_SIZE))
        draw.line((x0 - 5, y, x0, y), fill=BORDER_COLOR, width=1)
        label = str(tick)
        label_width, label_height = measure_text(draw, label, label_font)
        draw.text((x0 - 10 - label_width, y - label_height // 2), label, fill=0, font=label_font)

    for index, value in enumerate(values):
        bar_top = y0 + index * BAR_SIZE + 1
        bar_bottom = y0 + (index + 1) * BAR_SIZE - 1
        bar_width = int(round((int(value) / max_value) * plot_width))
        bar_right = x0 + bar_width
        if value > 0:
            draw.rectangle((x0 + 1, bar_top, bar_right, bar_bottom), fill=CHART_COLOR, outline=CHART_COLOR)

    title = f"Профиль Y: {symbol}"
    title_width, _ = measure_text(draw, title, title_font)
    draw.text(((width - title_width) // 2, 12), title, fill=0, font=title_font)
    draw.text((x1 - 10, y1 + 32), "Вес", fill=0, font=label_font)
    draw.text((x0 - 24, y0 - 30), "y", fill=0, font=label_font)

    image.save(out_path)


def save_symbol_gallery(items: list[tuple[str, str, np.ndarray]], out_path: Path, font_path: Path) -> None:
    columns = 6
    rows = math.ceil(len(items) / columns)
    label_font = ImageFont.truetype(str(font_path), 20)

    max_height = max(binary.shape[0] for _, _, binary in items)
    max_width = max(binary.shape[1] for _, _, binary in items)
    cell_width = max_width + 50
    cell_height = max_height + 70

    image = Image.new(
        "L",
        (columns * cell_width + 20, rows * cell_height + 20),
        BACKGROUND_COLOR,
    )
    draw = ImageDraw.Draw(image)

    for index, (name, symbol, binary) in enumerate(items):
        row = index // columns
        col = index % columns
        x0 = 10 + col * cell_width
        y0 = 10 + row * cell_height
        x_center = x0 + cell_width // 2

        symbol_image = Image.fromarray(binary_to_uint8(binary), mode="L")
        image.paste(
            symbol_image,
            (x_center - symbol_image.width // 2, y0 + 8),
        )

        caption = f"{symbol} ({name})"
        caption_width, _ = measure_text(draw, caption, label_font)
        draw.text(
            (x_center - caption_width // 2, y0 + max_height + 20),
            caption,
            fill=0,
            font=label_font,
        )

    image.save(out_path)


def process_symbols(output_dir: Path, font_path: Path, font_size: int) -> None:
    symbols_dir = output_dir / "symbols"
    profile_x_dir = output_dir / "profiles" / "x"
    profile_y_dir = output_dir / "profiles" / "y"

    for directory in (symbols_dir, profile_x_dir, profile_y_dir):
        directory.mkdir(parents=True, exist_ok=True)

    font = load_font(font_path, font_size)
    rows: list[dict[str, str]] = []
    gallery_items: list[tuple[str, str, np.ndarray]] = []

    for index, (name, symbol) in enumerate(GREEK_SYMBOLS, start=1):
        binary = render_symbol(symbol, font)
        scalar_features = calculate_scalar_features(binary)
        profile_x = calculate_profile_x(binary)
        profile_y = calculate_profile_y(binary)

        file_stem = f"{index:02d}_{name}"
        symbol_path = symbols_dir / f"{file_stem}.png"
        profile_x_path = profile_x_dir / f"{file_stem}_profile_x.png"
        profile_y_path = profile_y_dir / f"{file_stem}_profile_y.png"

        save_binary_image(binary, symbol_path)
        save_profile_x(profile_x, symbol, profile_x_path, font_path)
        save_profile_y(profile_y, symbol, profile_y_path, font_path)

        row = {
            "file": symbol_path.name,
            "symbol_name": name,
            "symbol": symbol,
            "font_name": font_path.stem,
            "font_size": str(font_size),
            "width": str(scalar_features["width"]),
            "height": str(scalar_features["height"]),
            "weight_q1": str(scalar_features["weight_q1"]),
            "weight_q2": str(scalar_features["weight_q2"]),
            "weight_q3": str(scalar_features["weight_q3"]),
            "weight_q4": str(scalar_features["weight_q4"]),
            "specific_weight_q1": format_float(float(scalar_features["specific_weight_q1"])),
            "specific_weight_q2": format_float(float(scalar_features["specific_weight_q2"])),
            "specific_weight_q3": format_float(float(scalar_features["specific_weight_q3"])),
            "specific_weight_q4": format_float(float(scalar_features["specific_weight_q4"])),
            "center_x": format_float(float(scalar_features["center_x"])),
            "center_y": format_float(float(scalar_features["center_y"])),
            "center_x_norm": format_float(float(scalar_features["center_x_norm"])),
            "center_y_norm": format_float(float(scalar_features["center_y_norm"])),
            "inertia_x": format_float(float(scalar_features["inertia_x"])),
            "inertia_y": format_float(float(scalar_features["inertia_y"])),
            "inertia_x_norm": format_float(float(scalar_features["inertia_x_norm"])),
            "inertia_y_norm": format_float(float(scalar_features["inertia_y_norm"])),
        }
        rows.append(row)
        gallery_items.append((name, symbol, binary))

        print(f"OK: {symbol} -> {symbol_path}")

    write_features_csv(rows, output_dir / "features.csv")
    save_symbol_gallery(gallery_items, output_dir / "symbols_gallery.png", font_path)
    print(f"CSV: {output_dir / 'features.csv'}")
    print(f"Gallery: {output_dir / 'symbols_gallery.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ЛР5: генерация эталонных символов и выделение признаков",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Папка для результатов",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=DEFAULT_FONT_PATH,
        help="Путь к файлу шрифта",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=DEFAULT_FONT_SIZE,
        help="Размер шрифта",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.font_size <= 0:
        raise ValueError("Размер шрифта должен быть положительным")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_symbols(
        output_dir=args.output_dir,
        font_path=args.font_path,
        font_size=args.font_size,
    )
    print("Готово")


if __name__ == "__main__":
    main()
