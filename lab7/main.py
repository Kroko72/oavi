from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
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
DEFAULT_EXPERIMENT_FONT_DELTA = 6
DEFAULT_SAMPLE_TEXT = "σε αγαπω"
DEFAULT_EXTRA_SAMPLE_TEXT = "σε αγαπω πολυ και σε θελω"
DEFAULT_BINARY_THRESHOLD = 128
DEFAULT_PROFILE_THRESHOLD = 1
DEFAULT_MIN_SYMBOL_WIDTH = 5

BACKGROUND_COLOR = 255
BORDER_COLOR = 0
GRID_COLOR = 210
CHART_COLOR = 30
BAR_SIZE = 12

FEATURE_CSV_FIELDS = [
    "file",
    "symbol_name",
    "symbol",
    "weight_rel",
    "center_x_rel",
    "center_y_rel",
    "inertia_x_rel",
    "inertia_y_rel",
]


@dataclass
class SymbolFeatures:
    weight_rel: float
    center_x_rel: float
    center_y_rel: float
    inertia_x_rel: float
    inertia_y_rel: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.weight_rel,
                self.center_x_rel,
                self.center_y_rel,
                self.inertia_x_rel,
                self.inertia_y_rel,
            ],
            dtype=np.float64,
        )


@dataclass
class AlphabetTemplate:
    name: str
    symbol: str
    binary: np.ndarray
    features: SymbolFeatures


@dataclass
class CaseResult:
    case_name: str
    image_path: Path
    expected_text: str
    recognized_text: str
    symbol_count: int
    errors: int
    accuracy_percent: float
    hypotheses_path: Path
    summary_path: Path


@dataclass
class PhraseRunGroup:
    title: str
    source_result: CaseResult
    experiment_result: CaseResult


def binary_to_uint8(binary: np.ndarray) -> np.ndarray:
    return np.where(binary == 1, 0, 255).astype(np.uint8)


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    gray = 0.3 * r + 0.59 * g + 0.11 * b
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def load_grayscale_image(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.array(image)

    if array.ndim == 2:
        return array.astype(np.uint8)

    if array.ndim == 3 and array.shape[2] >= 3:
        return rgb_to_grayscale(array[:, :, :3].astype(np.uint8))

    raise ValueError(f"{path.name}: неподдерживаемый формат изображения")


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


def render_text_binary(text: str, font: ImageFont.FreeTypeFont, threshold: int) -> np.ndarray:
    left, top, right, bottom = font.getbbox(text)
    margin = max(10, font.size // 2)
    width = right - left + 2 * margin
    height = bottom - top + 2 * margin

    image = Image.new("L", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    draw.text((margin - left, margin - top), text, font=font, fill=0)

    gray = np.array(image, dtype=np.uint8)
    binary = (gray < threshold).astype(np.uint8)
    return crop_binary_image(binary)


def save_binary_image(binary: np.ndarray, out_path: Path) -> None:
    Image.fromarray(binary_to_uint8(binary), mode="L").save(out_path)


def calculate_vertical_profile(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=0).astype(int)


def calculate_horizontal_profile(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=1).astype(int)


def first_large_index(profile: np.ndarray, threshold: int) -> int | None:
    for index, value in enumerate(profile):
        if int(value) > threshold:
            return index
    return None


def last_large_index(profile: np.ndarray, threshold: int) -> int | None:
    for index in range(len(profile) - 1, -1, -1):
        if int(profile[index]) > threshold:
            return index
    return None


def find_intervals(profile: np.ndarray, threshold: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None

    for index, value in enumerate(profile):
        if int(value) > threshold:
            if start is None:
                start = index
        elif start is not None:
            intervals.append((start, index - 1))
            start = None

    if start is not None:
        intervals.append((start, len(profile) - 1))

    return intervals


def merge_small_symbol_intervals(
    intervals: list[tuple[int, int]],
    min_symbol_width: int,
) -> list[tuple[int, int]]:
    if not intervals:
        return []

    merged: list[tuple[int, int]] = []
    index = 0

    while index < len(intervals):
        left, right = intervals[index]
        width = right - left + 1

        if width < min_symbol_width and index + 1 < len(intervals):
            next_left, next_right = intervals[index + 1]
            merged.append((left, next_right))
            index += 2
            continue

        merged.append((left, right))
        index += 1

    return merged


def split_wide_symbol_intervals(
    profile: np.ndarray,
    intervals: list[tuple[int, int]],
    profile_threshold: int,
    min_symbol_width: int,
) -> list[tuple[int, int]]:
    if not intervals:
        return []

    widths = [right - left + 1 for left, right in intervals]
    typical_width = int(np.median(np.array(widths, dtype=np.int32)))
    split_width_threshold = max(typical_width + 10, int(math.ceil(typical_width * 1.6)))

    result: list[tuple[int, int]] = []
    for left, right in intervals:
        width = right - left + 1
        subprofile = profile[left : right + 1]

        if width < split_width_threshold or len(subprofile) < 2 * min_symbol_width:
            result.append((left, right))
            continue

        edge_margin = max(2, min_symbol_width // 2)
        if len(subprofile) <= 2 * edge_margin:
            result.append((left, right))
            continue

        inner = subprofile[edge_margin : len(subprofile) - edge_margin]
        min_value = int(inner.min())
        if min_value > profile_threshold + 3:
            result.append((left, right))
            continue

        min_index = edge_margin + int(np.argmin(inner))
        gap_limit = min_value + 1
        gap_left = min_index
        gap_right = min_index

        while gap_left > 0 and int(subprofile[gap_left - 1]) <= gap_limit:
            gap_left -= 1
        while gap_right + 1 < len(subprofile) and int(subprofile[gap_right + 1]) <= gap_limit:
            gap_right += 1

        left_interval = (left, left + gap_left - 1)
        right_interval = (left + gap_right + 1, right)

        left_width = left_interval[1] - left_interval[0] + 1
        right_width = right_interval[1] - right_interval[0] + 1
        if left_width >= min_symbol_width and right_width >= min_symbol_width:
            result.append(left_interval)
            result.append(right_interval)
            continue

        result.append((left, right))

    return result


def find_text_area(binary: np.ndarray, threshold: int) -> tuple[int, int, int, int]:
    vertical_profile = calculate_vertical_profile(binary)
    left = first_large_index(vertical_profile, threshold)
    right = last_large_index(vertical_profile, threshold)
    if left is None or right is None:
        raise ValueError("Не удалось найти текстовую область по вертикальному профилю")

    cropped_by_x = binary[:, left : right + 1]
    horizontal_profile = calculate_horizontal_profile(cropped_by_x)
    top = first_large_index(horizontal_profile, threshold)
    bottom = last_large_index(horizontal_profile, threshold)
    if top is None or bottom is None:
        raise ValueError("Не удалось найти текстовую область по горизонтальному профилю")

    return left, top, right, bottom


def segment_symbols(
    binary: np.ndarray,
    profile_threshold: int,
    min_symbol_width: int,
) -> tuple[tuple[int, int, int, int], list[dict[str, int]]]:
    text_left, text_top, text_right, text_bottom = find_text_area(binary, profile_threshold)
    text_area = binary[text_top : text_bottom + 1, text_left : text_right + 1]

    line_intervals = find_intervals(calculate_horizontal_profile(text_area), profile_threshold)
    rectangles: list[dict[str, int]] = []
    symbol_index = 1

    for line_index, (line_top_local, line_bottom_local) in enumerate(line_intervals, start=1):
        line_binary = text_area[line_top_local : line_bottom_local + 1, :]
        profile_x = calculate_vertical_profile(line_binary)
        symbol_intervals = find_intervals(profile_x, profile_threshold)
        symbol_intervals = merge_small_symbol_intervals(symbol_intervals, min_symbol_width)
        symbol_intervals = split_wide_symbol_intervals(
            profile_x,
            symbol_intervals,
            profile_threshold,
            min_symbol_width,
        )

        symbol_index_in_line = 1
        for symbol_left_local, symbol_right_local in symbol_intervals:
            symbol_binary = line_binary[:, symbol_left_local : symbol_right_local + 1]
            symbol_profile_y = calculate_horizontal_profile(symbol_binary)
            symbol_top_local = first_large_index(symbol_profile_y, profile_threshold)
            symbol_bottom_local = last_large_index(symbol_profile_y, profile_threshold)

            if symbol_top_local is None or symbol_bottom_local is None:
                continue

            left = text_left + symbol_left_local
            right = text_left + symbol_right_local
            top = text_top + line_top_local + symbol_top_local
            bottom = text_top + line_top_local + symbol_bottom_local

            rectangles.append(
                {
                    "index": symbol_index,
                    "line_index": line_index,
                    "symbol_index_in_line": symbol_index_in_line,
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": right - left + 1,
                    "height": bottom - top + 1,
                }
            )

            symbol_index += 1
            symbol_index_in_line += 1

    return (text_left, text_top, text_right, text_bottom), rectangles


def save_segmented_symbols(
    binary: np.ndarray,
    rectangles: list[dict[str, int]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in rectangles:
        crop = binary[row["top"] : row["bottom"] + 1, row["left"] : row["right"] + 1]
        crop = crop_binary_image(crop)
        out_path = out_dir / f"{row['index']:02d}.png"
        Image.fromarray(binary_to_uint8(crop), mode="L").save(out_path)


def draw_segmentation_result(
    binary: np.ndarray,
    text_area: tuple[int, int, int, int],
    rectangles: list[dict[str, int]],
    out_path: Path,
) -> None:
    rgb = np.stack([binary_to_uint8(binary)] * 3, axis=2)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)

    text_left, text_top, text_right, text_bottom = text_area
    draw.rectangle((text_left, text_top, text_right, text_bottom), outline=(0, 0, 255), width=1)

    for row in rectangles:
        draw.rectangle(
            (row["left"], row["top"], row["right"], row["bottom"]),
            outline=(255, 0, 0),
            width=1,
        )

    image.save(out_path)


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


def measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def save_profile_x(values: np.ndarray, title: str, out_path: Path, font_path: Path) -> None:
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

    title_width, _ = measure_text(draw, title, title_font)
    draw.text(((width - title_width) // 2, 12), title, fill=0, font=title_font)
    draw.text((x0, y0 + 40), "x", fill=0, font=label_font)
    draw.text((x0 - 42, y1 - 24), "Вес", fill=0, font=label_font)

    image.save(out_path)


def save_profile_y(values: np.ndarray, title: str, out_path: Path, font_path: Path) -> None:
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

    title_width, _ = measure_text(draw, title, title_font)
    draw.text(((width - title_width) // 2, 12), title, fill=0, font=title_font)
    draw.text((x1 - 10, y1 + 32), "Вес", fill=0, font=label_font)
    draw.text((x0 - 24, y0 - 30), "y", fill=0, font=label_font)

    image.save(out_path)


def calculate_features(binary: np.ndarray) -> SymbolFeatures:
    height, width = binary.shape
    weight = float(binary.sum())
    if weight == 0:
        raise ValueError("Невозможно вычислить признаки: вес символа равен нулю")

    area = float(width * height)
    y_indices, x_indices = np.indices(binary.shape)

    center_x = float((x_indices * binary).sum() / weight)
    center_y = float((y_indices * binary).sum() / weight)

    weight_rel = weight / area if area else 0.0
    center_x_rel = center_x / (width - 1) if width > 1 else 0.0
    center_y_rel = center_y / (height - 1) if height > 1 else 0.0

    inertia_x = float((((y_indices - center_y) ** 2) * binary).sum())
    inertia_y = float((((x_indices - center_x) ** 2) * binary).sum())
    normalization = float((width**2) * (height**2))

    inertia_x_rel = inertia_x / normalization if normalization else 0.0
    inertia_y_rel = inertia_y / normalization if normalization else 0.0

    return SymbolFeatures(
        weight_rel=weight_rel,
        center_x_rel=center_x_rel,
        center_y_rel=center_y_rel,
        inertia_x_rel=inertia_x_rel,
        inertia_y_rel=inertia_y_rel,
    )


def format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def write_features_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FEATURE_CSV_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def normalize_symbol_text(text: str) -> str:
    return "".join(char for char in text if not char.isspace())


def build_alphabet_templates(
    output_dir: Path,
    font_path: Path,
    font_size: int,
    threshold: int,
) -> dict[str, AlphabetTemplate]:
    symbols_dir = output_dir / "symbols"
    symbols_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(font_path, font_size)
    templates: dict[str, AlphabetTemplate] = {}
    rows: list[dict[str, str]] = []

    for index, (name, symbol) in enumerate(GREEK_SYMBOLS, start=1):
        binary = render_text_binary(symbol, font, threshold)
        features = calculate_features(binary)
        file_stem = f"{index:02d}_{name}"
        symbol_path = symbols_dir / f"{file_stem}.png"
        save_binary_image(binary, symbol_path)

        templates[symbol] = AlphabetTemplate(
            name=name,
            symbol=symbol,
            binary=binary,
            features=features,
        )

        rows.append(
            {
                "file": symbol_path.name,
                "symbol_name": name,
                "symbol": symbol,
                "weight_rel": format_float(features.weight_rel),
                "center_x_rel": format_float(features.center_x_rel),
                "center_y_rel": format_float(features.center_y_rel),
                "inertia_x_rel": format_float(features.inertia_x_rel),
                "inertia_y_rel": format_float(features.inertia_y_rel),
            }
        )

    write_features_csv(rows, output_dir / "features.csv")
    return templates


def calculate_similarity(distance: float, feature_count: int) -> float:
    max_distance = math.sqrt(float(feature_count))
    similarity = 1.0 - distance / max_distance
    return max(0.0, min(1.0, similarity))


def classify_symbol(
    binary: np.ndarray,
    templates: dict[str, AlphabetTemplate],
) -> tuple[SymbolFeatures, list[tuple[str, float]]]:
    features = calculate_features(binary)
    feature_vector = features.as_vector()
    ranked: list[tuple[str, float, float]] = []

    for template in templates.values():
        distance = float(np.linalg.norm(feature_vector - template.features.as_vector()))
        similarity = calculate_similarity(distance, len(feature_vector))
        ranked.append((template.symbol, similarity, distance))

    ranked.sort(key=lambda item: (item[2], item[0]))
    hypotheses = [(symbol, similarity) for symbol, similarity, _ in ranked]
    return features, hypotheses


def format_hypotheses_line(hypotheses: list[tuple[str, float]]) -> str:
    items = [f'("{symbol}", {similarity:.4f})' for symbol, similarity in hypotheses]
    return "[" + ", ".join(items) + "]"


def save_hypotheses(hypotheses_by_symbol: list[list[tuple[str, float]]], out_path: Path) -> None:
    lines = [f"{index}: {format_hypotheses_line(hypotheses)}" for index, hypotheses in enumerate(hypotheses_by_symbol, start=1)]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_texts(expected: str, recognized: str) -> tuple[int, int, float]:
    matches = sum(1 for left, right in zip(expected, recognized, strict=False) if left == right)
    total = max(len(expected), len(recognized))
    errors = total - matches
    accuracy = (matches / total * 100.0) if total else 0.0
    return matches, errors, accuracy


def save_case_summary(
    out_path: Path,
    case_name: str,
    image_path: Path,
    expected_text: str,
    recognized_text: str,
    symbol_count: int,
    errors: int,
    accuracy_percent: float,
) -> None:
    lines = [
        f"Случай: {case_name}",
        f"Файл: {image_path.name}",
        f"Ожидаемая строка: {expected_text}",
        f"Распознанная строка: {recognized_text}",
        f"Количество найденных символов: {symbol_count}",
        f"Количество ошибок: {errors}",
        f"Доля верно распознанных символов: {accuracy_percent:.2f}%",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_final_report(
    out_path: Path,
    groups: list[PhraseRunGroup],
    source_font_size: int,
    experiment_font_size: int,
) -> None:
    lines = [
        "Лабораторная работа №7",
        "",
        f"Исходный размер шрифта: {source_font_size}",
        f"Экспериментальный размер шрифта: {experiment_font_size}",
    ]

    for group in groups:
        lines.extend(
            [
                "",
                group.title,
                f"Исходная строка: {group.source_result.expected_text}",
                f"Распознавание исходной строки: {group.source_result.recognized_text}",
                f"Ошибки для исходной строки: {group.source_result.errors}",
                f"Точность для исходной строки: {group.source_result.accuracy_percent:.2f}%",
                f"Распознавание строки другого размера: {group.experiment_result.recognized_text}",
                f"Ошибки для экспериментальной строки: {group.experiment_result.errors}",
                f"Точность для экспериментальной строки: {group.experiment_result.accuracy_percent:.2f}%",
            ]
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_sample_input(
    input_dir: Path,
    font_path: Path,
    font_size: int,
    text: str,
    threshold: int,
    file_name: str,
) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(font_path, font_size)
    binary = render_text_binary(text, font, threshold)
    out_path = input_dir / file_name
    save_binary_image(binary, out_path)
    return out_path


def run_phrase_group(
    title: str,
    file_prefix: str,
    text: str,
    input_dir: Path,
    output_dir: Path,
    font_path: Path,
    source_font_size: int,
    experiment_font_size: int,
    binary_threshold: int,
    profile_threshold: int,
    min_symbol_width: int,
    templates: dict[str, AlphabetTemplate],
) -> PhraseRunGroup:
    source_image_path = create_sample_input(
        input_dir=input_dir,
        font_path=font_path,
        font_size=source_font_size,
        text=text,
        threshold=binary_threshold,
        file_name=f"{file_prefix}_{source_font_size}.bmp",
    )
    experiment_image_path = create_sample_input(
        input_dir=input_dir,
        font_path=font_path,
        font_size=experiment_font_size,
        text=text,
        threshold=binary_threshold,
        file_name=f"{file_prefix}_{experiment_font_size}.bmp",
    )

    source_result = process_case(
        image_path=source_image_path,
        expected_text=text,
        output_dir=output_dir,
        font_path=font_path,
        binary_threshold=binary_threshold,
        profile_threshold=profile_threshold,
        min_symbol_width=min_symbol_width,
        templates=templates,
    )
    experiment_result = process_case(
        image_path=experiment_image_path,
        expected_text=text,
        output_dir=output_dir,
        font_path=font_path,
        binary_threshold=binary_threshold,
        profile_threshold=profile_threshold,
        min_symbol_width=min_symbol_width,
        templates=templates,
    )

    return PhraseRunGroup(
        title=title,
        source_result=source_result,
        experiment_result=experiment_result,
    )


def process_case(
    image_path: Path,
    expected_text: str,
    output_dir: Path,
    font_path: Path,
    binary_threshold: int,
    profile_threshold: int,
    min_symbol_width: int,
    templates: dict[str, AlphabetTemplate],
) -> CaseResult:
    gray = load_grayscale_image(image_path)
    binary = (gray < binary_threshold).astype(np.uint8)

    binary_dir = output_dir / "binary"
    profile_dir = output_dir / "profiles" / "text"
    segmentation_dir = output_dir / "segmentation"
    symbols_dir = output_dir / "segmented_symbols" / image_path.stem
    classification_dir = output_dir / "classification"

    for directory in [binary_dir, profile_dir, segmentation_dir, symbols_dir, classification_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    binary_path = binary_dir / f"{image_path.stem}_binary.bmp"
    save_binary_image(binary, binary_path)

    profile_x = calculate_vertical_profile(binary)
    profile_y = calculate_horizontal_profile(binary)
    save_profile_x(profile_x, f"Вертикальный профиль: {image_path.stem}", profile_dir / f"{image_path.stem}_profile_x.png", font_path)
    save_profile_y(profile_y, f"Горизонтальный профиль: {image_path.stem}", profile_dir / f"{image_path.stem}_profile_y.png", font_path)

    text_area, rectangles = segment_symbols(binary, profile_threshold, min_symbol_width)
    draw_segmentation_result(binary, text_area, rectangles, segmentation_dir / f"{image_path.stem}_boxes.png")
    save_segmented_symbols(binary, rectangles, symbols_dir)

    hypotheses_by_symbol: list[list[tuple[str, float]]] = []
    recognized_symbols: list[str] = []

    for row in rectangles:
        symbol_binary = binary[row["top"] : row["bottom"] + 1, row["left"] : row["right"] + 1]
        symbol_binary = crop_binary_image(symbol_binary)
        _, hypotheses = classify_symbol(symbol_binary, templates)
        hypotheses_by_symbol.append(hypotheses)
        if hypotheses:
            recognized_symbols.append(hypotheses[0][0])

    recognized_text = "".join(recognized_symbols)
    expected_symbols = normalize_symbol_text(expected_text)
    _, errors, accuracy_percent = compare_texts(expected_symbols, recognized_text)

    hypotheses_path = classification_dir / f"{image_path.stem}_hypotheses.txt"
    summary_path = classification_dir / f"{image_path.stem}_summary.txt"
    save_hypotheses(hypotheses_by_symbol, hypotheses_path)
    save_case_summary(
        out_path=summary_path,
        case_name=image_path.stem,
        image_path=image_path,
        expected_text=expected_symbols,
        recognized_text=recognized_text,
        symbol_count=len(rectangles),
        errors=errors,
        accuracy_percent=accuracy_percent,
    )

    return CaseResult(
        case_name=image_path.stem,
        image_path=image_path,
        expected_text=expected_symbols,
        recognized_text=recognized_text,
        symbol_count=len(rectangles),
        errors=errors,
        accuracy_percent=accuracy_percent,
        hypotheses_path=hypotheses_path,
        summary_path=summary_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ЛР7: классификация символов по признакам и анализ профилей",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input_images"),
        help="Папка для сгенерированных входных изображений",
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
        help="Исходный размер шрифта",
    )
    parser.add_argument(
        "--experiment-font-size",
        type=int,
        default=None,
        help="Размер шрифта для эксперимента",
    )
    parser.add_argument(
        "--binary-threshold",
        type=int,
        default=DEFAULT_BINARY_THRESHOLD,
        help="Порог бинаризации",
    )
    parser.add_argument(
        "--profile-threshold",
        type=int,
        default=DEFAULT_PROFILE_THRESHOLD,
        help="Малое значение профиля по лекции",
    )
    parser.add_argument(
        "--min-symbol-width",
        type=int,
        default=DEFAULT_MIN_SYMBOL_WIDTH,
        help="Минимальная ширина символа",
    )
    parser.add_argument(
        "--sample-text",
        type=str,
        default=DEFAULT_SAMPLE_TEXT,
        help="Строка для распознавания",
    )
    parser.add_argument(
        "--extra-sample-text",
        type=str,
        default=DEFAULT_EXTRA_SAMPLE_TEXT,
        help="Дополнительная более длинная строка; пустая строка отключает этот прогон",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment_font_size = args.experiment_font_size
    if experiment_font_size is None:
        experiment_font_size = args.font_size + DEFAULT_EXPERIMENT_FONT_DELTA

    if not 0 <= args.binary_threshold <= 255:
        raise ValueError("Порог бинаризации должен быть в диапазоне 0..255")
    if args.profile_threshold < 0:
        raise ValueError("Порог профиля не может быть отрицательным")
    if args.min_symbol_width <= 0:
        raise ValueError("Минимальная ширина символа должна быть положительной")
    if args.font_size <= 0 or experiment_font_size <= 0:
        raise ValueError("Размер шрифта должен быть положительным")
    if experiment_font_size == args.font_size:
        raise ValueError("Экспериментальный размер шрифта должен отличаться от исходного")

    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    templates = build_alphabet_templates(
        output_dir=args.output_dir / "alphabet",
        font_path=args.font_path,
        font_size=args.font_size,
        threshold=args.binary_threshold,
    )

    groups: list[PhraseRunGroup] = []
    groups.append(
        run_phrase_group(
            title="Основная фраза",
            file_prefix="sample_phrase",
            text=args.sample_text,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            font_path=args.font_path,
            source_font_size=args.font_size,
            experiment_font_size=experiment_font_size,
            binary_threshold=args.binary_threshold,
            profile_threshold=args.profile_threshold,
            min_symbol_width=args.min_symbol_width,
            templates=templates,
        )
    )

    extra_text = args.extra_sample_text.strip()
    if extra_text:
        groups.append(
            run_phrase_group(
                title="Дополнительная длинная фраза",
                file_prefix="extra_phrase",
                text=extra_text,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                font_path=args.font_path,
                source_font_size=args.font_size,
                experiment_font_size=experiment_font_size,
                binary_threshold=args.binary_threshold,
                profile_threshold=args.profile_threshold,
                min_symbol_width=args.min_symbol_width,
                templates=templates,
            )
        )

    final_report_path = args.output_dir / "final_report.txt"
    save_final_report(
        out_path=final_report_path,
        groups=groups,
        source_font_size=args.font_size,
        experiment_font_size=experiment_font_size,
    )

    print(f"Эталоны: {args.output_dir / 'alphabet'}")
    print(f"Итоговый отчет: {final_report_path}")
    for group in groups:
        print(f"{group.title}:")
        print(f"  исходная строка -> {group.source_result.image_path}")
        print(f"  экспериментальная строка -> {group.experiment_result.image_path}")
        print(f"  гипотезы для исходной строки -> {group.source_result.hypotheses_path}")
        print(f"  гипотезы для экспериментальной строки -> {group.experiment_result.hypotheses_path}")
        print(
            f"  распознавание исходной строки -> {group.source_result.recognized_text} "
            f"({group.source_result.accuracy_percent:.2f}%)"
        )
        print(
            f"  распознавание экспериментальной строки -> {group.experiment_result.recognized_text} "
            f"({group.experiment_result.accuracy_percent:.2f}%)"
        )


if __name__ == "__main__":
    main()
