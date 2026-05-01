import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

Path("output/.matplotlib").mkdir(parents=True, exist_ok=True)
Path("output/.cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("output") / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("output") / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft


DEFAULT_SAMPLES_DIR = Path("samples")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_EXPECTED = "+89177425210"

ALPHABET = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "plus"]
SYMBOLS = {str(i): str(i) for i in range(10)}
SYMBOLS["plus"] = "+"

WINDOW_SECONDS = 0.03
OVERLAP_PART = 2 / 3
FRAME_SECONDS = 0.025
HOP_SECONDS = 0.01

SEGMENT_RELATIVE_THRESHOLD = 0.10
SEGMENT_MIN_THRESHOLD = 0.004
MIN_SEGMENT_SECONDS = 0.12
MAX_PAUSE_SECONDS = 0.16
SEGMENT_PAD_SECONDS = 0.06

MIN_FREQ = 80.0
MAX_FREQ = 5500.0
LOG_BANDS = 30
EPS = 1e-10


@dataclass
class Segment:
    index: int
    start: int
    end: int

    def start_seconds(self, sample_rate: int) -> float:
        return self.start / sample_rate

    def end_seconds(self, sample_rate: int) -> float:
        return self.end / sample_rate


@dataclass
class SegmentationInfo:
    segments: list[Segment]
    rms_times: np.ndarray
    rms_values: np.ndarray
    threshold: float


@dataclass
class RecognitionResult:
    segment: Segment
    symbol: str
    best_distance: float
    second_symbol: str
    second_distance: float
    confidence: float


def ensure_directories(base_dir: Path) -> dict[str, Path]:
    directories = {
        "spectrograms": base_dir / "spectrograms",
        "segmentation": base_dir / "segmentation",
        "segments": base_dir / "segments",
        "tables": base_dir / "tables",
        "reports": base_dir / "reports",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def pcm_to_float(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / np.iinfo(data.dtype).max
    return data.astype(np.float32)


def float_to_int16(data: np.ndarray) -> np.ndarray:
    clipped = np.clip(data, -1.0, 1.0)
    return np.rint(clipped * np.iinfo(np.int16).max).astype(np.int16)


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    return sample_rate, to_mono(pcm_to_float(data))


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(values, kernel, mode="same")


def calculate_rms(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    frame_size = int(round(FRAME_SECONDS * sample_rate))
    hop_size = int(round(HOP_SECONDS * sample_rate))
    starts = np.arange(0, max(1, signal.shape[0] - frame_size + 1), hop_size)

    values = []
    for start in starts:
        frame = signal[start : start + frame_size]
        values.append(float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0)

    times = (starts + frame_size / 2) / sample_rate
    return times, moving_average(np.array(values), 5), frame_size, hop_size


def fill_short_pauses(active: np.ndarray) -> np.ndarray:
    result = active.copy()
    max_pause_frames = int(round(MAX_PAUSE_SECONDS / HOP_SECONDS))

    index = 0
    while index < result.shape[0]:
        if result[index]:
            index += 1
            continue

        pause_start = index
        while index < result.shape[0] and not result[index]:
            index += 1
        pause_end = index

        left_sound = pause_start > 0
        right_sound = pause_end < result.shape[0]
        if left_sound and right_sound and pause_end - pause_start <= max_pause_frames:
            result[pause_start:pause_end] = True

    return result


def segment_audio(signal: np.ndarray, sample_rate: int) -> SegmentationInfo:
    times, rms_values, frame_size, hop_size = calculate_rms(signal, sample_rate)
    noise_level = float(np.percentile(rms_values, 15))
    threshold = max(
        SEGMENT_MIN_THRESHOLD,
        noise_level + SEGMENT_RELATIVE_THRESHOLD * (float(rms_values.max()) - noise_level),
    )

    active = fill_short_pauses(rms_values > threshold)
    min_frames = int(round(MIN_SEGMENT_SECONDS / HOP_SECONDS))
    pad = int(round(SEGMENT_PAD_SECONDS * sample_rate))

    segments = []
    index = 0
    while index < active.shape[0]:
        if not active[index]:
            index += 1
            continue

        start_frame = index
        while index < active.shape[0] and active[index]:
            index += 1
        end_frame = index

        if end_frame - start_frame >= min_frames:
            start = max(0, start_frame * hop_size - pad)
            end = min(signal.shape[0], (end_frame - 1) * hop_size + frame_size + pad)
            segments.append(Segment(len(segments) + 1, start, end))

    return SegmentationInfo(segments, times, rms_values, threshold)


def trim_word(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    info = segment_audio(signal, sample_rate)
    if not info.segments:
        return signal

    segment = info.segments[0]
    return signal[segment.start : segment.end]


def calculate_stft(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = int(round(WINDOW_SECONDS * sample_rate))
    noverlap = int(round(nperseg * OVERLAP_PART))
    return stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )


def save_spectrogram(signal: np.ndarray, sample_rate: int, out_path: Path) -> None:
    frequencies, times, spectrum = calculate_stft(signal, sample_rate)
    magnitude_db = 20.0 * np.log10(np.abs(spectrum) + EPS)

    positive = frequencies >= 20.0
    frequencies = frequencies[positive]
    magnitude_db = magnitude_db[positive, :]

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, frequencies, magnitude_db, shading="auto", cmap="magma")
    plt.yscale("log")
    plt.ylim(20, sample_rate / 2)
    plt.xlabel("Время, с")
    plt.ylabel("Частота, Гц")
    plt.title("Спектрограмма записи телефонного номера")
    plt.colorbar(label="Амплитуда, дБ")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def log_spectral_sequence(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    signal = signal.astype(np.float32)
    signal = signal - float(np.mean(signal))

    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0:
        signal = signal / peak

    frequencies, _, spectrum = calculate_stft(signal, sample_rate)
    power = np.abs(spectrum) ** 2
    max_freq = min(MAX_FREQ, sample_rate / 2)
    edges = np.geomspace(MIN_FREQ, max_freq, LOG_BANDS + 1)

    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        if np.any(mask):
            bands.append(power[mask].mean(axis=0))
        else:
            bands.append(np.zeros(power.shape[1], dtype=np.float32))

    features = np.log(np.array(bands).T + EPS)
    features = features - features.mean(axis=1, keepdims=True)
    features = features / (features.std(axis=1, keepdims=True) + EPS)
    return features.astype(np.float32)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + EPS)


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = normalize_rows(first)
    second = normalize_rows(second)

    rows, cols = first.shape[0], second.shape[0]
    distances = np.full((rows + 1, cols + 1), np.inf, dtype=np.float32)
    distances[0, 0] = 0.0

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            cost = 1.0 - float(np.dot(first[row - 1], second[col - 1]))
            distances[row, col] = cost + min(
                distances[row - 1, col],
                distances[row, col - 1],
                distances[row - 1, col - 1],
            )

    return float(distances[rows, cols] / (rows + cols))


def load_templates(samples_dir: Path) -> dict[str, np.ndarray]:
    templates = {}
    for label in ALPHABET:
        sample_rate, signal = read_wav(samples_dir / f"{label}.wav")
        trimmed = trim_word(signal, sample_rate)
        templates[label] = log_spectral_sequence(trimmed, sample_rate)
    return templates


def recognize_segments(
    signal: np.ndarray,
    sample_rate: int,
    segments: list[Segment],
    templates: dict[str, np.ndarray],
) -> list[RecognitionResult]:
    results = []

    for segment in segments:
        segment_signal = signal[segment.start : segment.end]
        features = log_spectral_sequence(segment_signal, sample_rate)

        distances = []
        for label, template in templates.items():
            distance = dtw_distance(features, template)
            distances.append((label, distance))

        distances.sort(key=lambda item: item[1])
        best_label, best_distance = distances[0]
        second_label, second_distance = distances[1]
        confidence = (second_distance - best_distance) / max(second_distance, EPS)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        results.append(
            RecognitionResult(
                segment=segment,
                symbol=SYMBOLS[best_label],
                best_distance=best_distance,
                second_symbol=SYMBOLS[second_label],
                second_distance=second_distance,
                confidence=confidence,
            )
        )

    return results


def levenshtein(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, first_char in enumerate(first, start=1):
        current = [i]
        for j, second_char in enumerate(second, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (first_char != second_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def save_segmentation_plot(
    signal: np.ndarray,
    sample_rate: int,
    info: SegmentationInfo,
    out_path: Path,
) -> None:
    times = np.arange(signal.shape[0]) / sample_rate

    plt.figure(figsize=(12, 6))
    ax_signal = plt.subplot(2, 1, 1)
    ax_signal.plot(times, signal, linewidth=0.7)
    ax_signal.set_title("Сегментация записи телефонного номера")
    ax_signal.set_ylabel("Амплитуда")

    for segment in info.segments:
        ax_signal.axvspan(
            segment.start_seconds(sample_rate),
            segment.end_seconds(sample_rate),
            alpha=0.2,
            color="tab:green",
        )
        ax_signal.text(
            segment.start_seconds(sample_rate),
            0.9,
            str(segment.index),
            transform=ax_signal.get_xaxis_transform(),
            fontsize=9,
        )

    ax_rms = plt.subplot(2, 1, 2, sharex=ax_signal)
    ax_rms.plot(info.rms_times, info.rms_values, linewidth=1.0)
    ax_rms.axhline(info.threshold, color="tab:red", linestyle="--", label="порог")
    ax_rms.set_xlabel("Время, с")
    ax_rms.set_ylabel("RMS")
    ax_rms.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_segments(signal: np.ndarray, sample_rate: int, results: list[RecognitionResult], out_dir: Path) -> None:
    for result in results:
        segment_signal = signal[result.segment.start : result.segment.end]
        safe_symbol = "plus" if result.symbol == "+" else result.symbol
        out_path = out_dir / f"{result.segment.index:02d}_{safe_symbol}.wav"
        wavfile.write(out_path, sample_rate, float_to_int16(segment_signal))


def write_results_csv(results: list[RecognitionResult], sample_rate: int, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            [
                "index",
                "start_s",
                "end_s",
                "recognized",
                "best_distance",
                "second",
                "second_distance",
                "confidence",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.segment.index,
                    f"{result.segment.start_seconds(sample_rate):.3f}",
                    f"{result.segment.end_seconds(sample_rate):.3f}",
                    result.symbol,
                    f"{result.best_distance:.6f}",
                    result.second_symbol,
                    f"{result.second_distance:.6f}",
                    f"{result.confidence:.6f}",
                ]
            )


def write_report(
    sample_rate: int,
    duration: float,
    info: SegmentationInfo,
    results: list[RecognitionResult],
    recognized: str,
    expected: str,
    out_path: Path,
) -> None:
    error_count = levenshtein(expected, recognized) if expected else None
    reliability = float(np.mean([result.confidence for result in results])) if results else 0.0

    lines = [
        "Лабораторная работа №10",
        "Тема: обработка голоса. Вариант 3: анализатор речи",
        "",
        "Использованные положения лекции:",
        "  речь рассматривается как почти непрерывный звуковой поток;",
        "  сегментация упрощается тем, что словарь заранее ограничен цифрами и словом плюс;",
        "  для сравнения используются спектральные признаки и формантные области;",
        "  для спектрограммы используется оконное преобразование Фурье с окном Ханна.",
        "",
        f"Файл телефонного номера: samples/phone.wav",
        f"Частота дискретизации: {sample_rate} Гц",
        f"Длительность: {duration:.2f} с",
        "",
        "Параметры спектрограммы:",
        f"  окно: Ханна",
        f"  длина окна: {WINDOW_SECONDS:.3f} с",
        f"  перекрытие: {OVERLAP_PART * 100:.0f}%",
        "  шкала частот: логарифмическая",
        "",
        "Параметры сегментации:",
        f"  шаг RMS: {HOP_SECONDS:.3f} с",
        f"  окно RMS: {FRAME_SECONDS:.3f} с",
        f"  порог RMS: {info.threshold:.6f}",
        f"  найдено сегментов: {len(info.segments)}",
        "",
        "Результаты распознавания:",
    ]

    for result in results:
        lines.append(
            "  "
            f"{result.segment.index:02d}. "
            f"{result.segment.start_seconds(sample_rate):.2f}-"
            f"{result.segment.end_seconds(sample_rate):.2f} c: "
            f"{result.symbol}, "
            f"d={result.best_distance:.4f}, "
            f"достоверность={result.confidence * 100:.1f}%"
        )

    lines.extend(
        [
            "",
            f"Распознанная цепочка: {recognized}",
            f"Эталонная цепочка: {expected if expected else 'не задана'}",
        ]
    )

    if error_count is None:
        lines.append("Число ошибок: не посчитано, потому что эталон не задан")
    else:
        lines.append(f"Число ошибок по расстоянию Левенштейна: {error_count}")

    lines.append(f"Оценка достоверности: {reliability * 100:.1f}%")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Лабораторная работа №10: анализатор речи")
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected", default=DEFAULT_EXPECTED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dirs = ensure_directories(args.output_dir)

    sample_rate, phone_signal = read_wav(args.samples_dir / "phone.wav")
    duration = phone_signal.shape[0] / sample_rate

    save_spectrogram(
        phone_signal,
        sample_rate,
        output_dirs["spectrograms"] / "phone_spectrogram.png",
    )

    segmentation_info = segment_audio(phone_signal, sample_rate)
    save_segmentation_plot(
        phone_signal,
        sample_rate,
        segmentation_info,
        output_dirs["segmentation"] / "phone_segments.png",
    )

    templates = load_templates(args.samples_dir)
    results = recognize_segments(phone_signal, sample_rate, segmentation_info.segments, templates)
    recognized = "".join(result.symbol for result in results)

    save_segments(phone_signal, sample_rate, results, output_dirs["segments"])
    write_results_csv(results, sample_rate, output_dirs["tables"] / "recognition.csv")
    write_report(
        sample_rate,
        duration,
        segmentation_info,
        results,
        recognized,
        args.expected,
        output_dirs["reports"] / "report.txt",
    )


if __name__ == "__main__":
    main()
