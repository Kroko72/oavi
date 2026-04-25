import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import istft, stft


DEFAULT_INPUT = Path("sweden.wav")
DEFAULT_OUTPUT_DIR = Path("output")
WINDOW_SECONDS = 0.05
OVERLAP_PART = 0.75
NOISE_PERCENTILE = 15
SUPPRESSION_K = 1.6
MIN_GAIN = 0.08
TIME_STEP = 0.1
FREQ_STEP = 50.0
TOP_ENERGY_COUNT = 12
EPS = 1e-12

CSV_FIELDS = [
    "place",
    "time_start_s",
    "time_end_s",
    "freq_start_hz",
    "freq_end_hz",
    "energy",
]


@dataclass
class AudioInfo:
    sample_rate: int
    channels: int
    duration: float
    dtype: str


@dataclass
class NoiseInfo:
    rms_signal: float
    rms_noise_before: float
    rms_noise_after: float
    snr_before_db: float
    snr_after_db: float


def ensure_directories(base_dir: Path) -> dict[str, Path]:
    directories = {
        "spectrograms": base_dir / "spectrograms",
        "audio": base_dir / "audio",
        "tables": base_dir / "tables",
        "reports": base_dir / "reports",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def pcm_to_float(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        limit = float(np.iinfo(data.dtype).max)
        return data.astype(np.float32) / limit
    return data.astype(np.float32)


def float_to_int16(data: np.ndarray) -> np.ndarray:
    clipped = np.clip(data, -1.0, 1.0)
    return np.rint(clipped * np.iinfo(np.int16).max).astype(np.int16)


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def stft_params(sample_rate: int) -> tuple[int, int]:
    nperseg = int(round(sample_rate * WINDOW_SECONDS))
    noverlap = int(round(nperseg * OVERLAP_PART))
    return nperseg, noverlap


def calculate_stft(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg, noverlap = stft_params(sample_rate)
    frequencies, times, spectrum = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    return frequencies, times, spectrum


def spectral_subtraction(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    nperseg, noverlap = stft_params(sample_rate)
    _, _, spectrum = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )

    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    noise_spectrum = np.percentile(magnitude, NOISE_PERCENTILE, axis=1, keepdims=True)
    gain = 1.0 - SUPPRESSION_K * noise_spectrum / np.maximum(magnitude, EPS)
    gain = np.clip(gain, MIN_GAIN, 1.0)

    cleaned_spectrum = magnitude * gain * np.exp(1j * phase)
    _, cleaned = istft(
        cleaned_spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    return cleaned[: signal.shape[0]].astype(np.float32)


def denoise_audio(data: np.ndarray, sample_rate: int) -> np.ndarray:
    if data.ndim == 1:
        return spectral_subtraction(data, sample_rate)

    channels = []
    for channel_index in range(data.shape[1]):
        channels.append(spectral_subtraction(data[:, channel_index], sample_rate))
    return np.column_stack(channels).astype(np.float32)


def save_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    title: str,
    out_path: Path,
) -> None:
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
    plt.title(title)
    plt.colorbar(label="Амплитуда, дБ")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def find_quiet_mask(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_size = max(1, int(round(TIME_STEP * sample_rate)))
    frame_count = int(np.ceil(signal.shape[0] / frame_size))
    rms_values = np.zeros(frame_count, dtype=np.float64)

    for frame_index in range(frame_count):
        start = frame_index * frame_size
        end = min(start + frame_size, signal.shape[0])
        frame = signal[start:end]
        rms_values[frame_index] = np.sqrt(np.mean(frame * frame)) if frame.size else 0.0

    threshold = np.percentile(rms_values, NOISE_PERCENTILE)
    return rms_values <= threshold


def estimate_noise(original: np.ndarray, cleaned: np.ndarray, sample_rate: int) -> NoiseInfo:
    frame_size = max(1, int(round(TIME_STEP * sample_rate)))
    quiet_mask = find_quiet_mask(original, sample_rate)

    def quiet_rms(signal: np.ndarray) -> float:
        values = []
        for frame_index, is_quiet in enumerate(quiet_mask):
            if not is_quiet:
                continue
            start = frame_index * frame_size
            end = min(start + frame_size, signal.shape[0])
            frame = signal[start:end]
            if frame.size:
                values.append(float(np.mean(frame * frame)))
        return float(np.sqrt(np.mean(values))) if values else 0.0

    rms_signal = float(np.sqrt(np.mean(original * original)))
    rms_noise_before = quiet_rms(original)
    rms_noise_after = quiet_rms(cleaned)
    snr_before = 20.0 * np.log10(rms_signal / max(rms_noise_before, EPS))
    snr_after = 20.0 * np.log10(rms_signal / max(rms_noise_after, EPS))

    return NoiseInfo(
        rms_signal=rms_signal,
        rms_noise_before=rms_noise_before,
        rms_noise_after=rms_noise_after,
        snr_before_db=float(snr_before),
        snr_after_db=float(snr_after),
    )


def find_energy_maxima(signal: np.ndarray, sample_rate: int) -> list[dict[str, float]]:
    frequencies, times, spectrum = calculate_stft(signal, sample_rate)
    power = np.abs(spectrum) ** 2

    results = []
    max_time_bin = int(np.ceil(times.max() / TIME_STEP)) + 1
    max_freq_bin = int(np.ceil(frequencies.max() / FREQ_STEP)) + 1

    for time_bin in range(max_time_bin):
        time_start = time_bin * TIME_STEP
        time_end = time_start + TIME_STEP
        time_mask = (times >= time_start) & (times < time_end)
        if not np.any(time_mask):
            continue

        for freq_bin in range(max_freq_bin):
            freq_start = freq_bin * FREQ_STEP
            freq_end = freq_start + FREQ_STEP
            freq_mask = (frequencies >= freq_start) & (frequencies < freq_end)
            if not np.any(freq_mask):
                continue

            energy = float(power[np.ix_(freq_mask, time_mask)].sum())
            if energy <= 0:
                continue

            results.append(
                {
                    "time_start_s": time_start,
                    "time_end_s": time_end,
                    "freq_start_hz": freq_start,
                    "freq_end_hz": freq_end,
                    "energy": energy,
                }
            )

    results.sort(key=lambda item: item["energy"], reverse=True)
    return results[:TOP_ENERGY_COUNT]


def write_energy_csv(maxima: list[dict[str, float]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, delimiter=";")
        writer.writeheader()
        for place, item in enumerate(maxima, start=1):
            writer.writerow(
                {
                    "place": place,
                    "time_start_s": f"{item['time_start_s']:.1f}",
                    "time_end_s": f"{item['time_end_s']:.1f}",
                    "freq_start_hz": f"{item['freq_start_hz']:.0f}",
                    "freq_end_hz": f"{item['freq_end_hz']:.0f}",
                    "energy": f"{item['energy']:.6f}",
                }
            )


def write_report(
    audio_info: AudioInfo,
    noise_info: NoiseInfo,
    maxima: list[dict[str, float]],
    out_path: Path,
) -> None:
    lines = [
        "Лабораторная работа №9",
        "Тема: анализ шума",
        "",
        "Исходный файл: sweden.wav",
        f"Частота дискретизации: {audio_info.sample_rate} Гц",
        f"Каналов: {audio_info.channels}",
        f"Длительность: {audio_info.duration:.2f} с",
        f"Тип исходных отсчетов: {audio_info.dtype}",
        "",
        "Параметры STFT по лекции:",
        f"  окно: Ханна",
        f"  длина окна: {WINDOW_SECONDS:.3f} с",
        f"  перекрытие: {OVERLAP_PART * 100:.0f}%",
        "",
        "Шумопонижение:",
        "  метод: спектральное вычитание",
        "  оценка шума: частотный спектр тихих участков по 15-му процентилю",
        f"  формула: Y[f,t] = max(X[f,t] - kW[f,t], 0), k = {SUPPRESSION_K}",
        f"  нижнее ограничение усиления для уменьшения музыкального шума: {MIN_GAIN}",
        "",
        "Оценка уровня шума:",
        f"  RMS всего сигнала: {noise_info.rms_signal:.6f}",
        f"  RMS шума до обработки: {noise_info.rms_noise_before:.6f}",
        f"  RMS шума после обработки: {noise_info.rms_noise_after:.6f}",
        f"  ОСШ до обработки: {noise_info.snr_before_db:.2f} дБ",
        f"  ОСШ после обработки: {noise_info.snr_after_db:.2f} дБ",
        "",
        f"Наибольшая энергия при delta t = {TIME_STEP:.1f} с и delta f = {FREQ_STEP:.0f} Гц:",
    ]

    for place, item in enumerate(maxima, start=1):
        lines.append(
            "  "
            f"{place}. t={item['time_start_s']:.1f}-{item['time_end_s']:.1f} c, "
            f"f={item['freq_start_hz']:.0f}-{item['freq_end_hz']:.0f} Гц, "
            f"E={item['energy']:.6f}"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Лабораторная работа №9: анализ шума")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Входной WAV-файл")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Папка для результатов")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dirs = ensure_directories(args.output_dir)

    sample_rate, source_data = wavfile.read(args.input)
    source_float = pcm_to_float(source_data)
    original_mono = to_mono(source_float)

    cleaned_float = denoise_audio(source_float, sample_rate)
    cleaned_mono = to_mono(cleaned_float)

    audio_info = AudioInfo(
        sample_rate=sample_rate,
        channels=1 if source_data.ndim == 1 else source_data.shape[1],
        duration=source_data.shape[0] / sample_rate,
        dtype=str(source_data.dtype),
    )

    wavfile.write(output_dirs["audio"] / "sweden_denoised.wav", sample_rate, float_to_int16(cleaned_float))

    save_spectrogram(
        original_mono,
        sample_rate,
        "Спектрограмма исходного сигнала",
        output_dirs["spectrograms"] / "sweden_before.png",
    )
    save_spectrogram(
        cleaned_mono,
        sample_rate,
        "Спектрограмма после спектрального вычитания",
        output_dirs["spectrograms"] / "sweden_after.png",
    )

    noise_info = estimate_noise(original_mono, cleaned_mono, sample_rate)
    maxima = find_energy_maxima(cleaned_mono, sample_rate)

    write_energy_csv(maxima, output_dirs["tables"] / "energy_maxima.csv")
    write_report(audio_info, noise_info, maxima, output_dirs["reports"] / "report.txt")


if __name__ == "__main__":
    main()
