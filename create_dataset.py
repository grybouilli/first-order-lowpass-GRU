import numpy as np
from scipy.signal import lfilter, butter, square
import colorednoise as cn
from random import uniform
from sklearn.preprocessing import minmax_scale


def lowpass_filter(
    input: np.ndarray, cutoff_freq: float, sample_rate: int
) -> np.ndarray:
    """Filter input with a first-order digital lowpass filter.
    Args:
        input (np.ndarray): The input signal
        cutoff_freq (float): The cutoff frequency in Hz
        sample_rate (int): The sample rate of the signal in Hz
    Returns:
        np.ndarray: The filtered signal
    """
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist  # in range (0, 1)
    b, a = butter(1, normalized_cutoff, btype="low", analog=False)
    return lfilter(b, a, input)


def generate_white_noise(
    length: float, sample_rate: int, cutoff_freq: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a some white noise and its filtered version.

    Args:
        length (float): The length of the sample to generate in seconds
        sample_rate (int): The sample rate in Hz of the signal
        cutoff_freq (int): The cutoff frequency of the lowpass filter in Hertz
    Returns:
        tuple[np.ndarray, np.ndarray]: The white noise and the filtered white noise
    """

    N = int(length * sample_rate)
    white_noise = np.random.uniform(-1, 1, N)
    filtered_noise = lowpass_filter(white_noise, cutoff_freq, sample_rate)
    return white_noise, filtered_noise


def generate_pink_noise(
    length: float, sample_rate: int, cutoff_freq: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a some pink noise and its filtered version.

    Args:
        length (float): The length of the sample to generate in seconds
        sample_rate (int): The sample rate in Hz of the signal
        cutoff_freq (int): The cutoff frequency of the lowpass filter in Hertz
    Returns:
        tuple[np.ndarray, np.ndarray]: The pink noise and the filtered pink noise
    """

    N = int(length * sample_rate)
    beta = 1  # the exponent: 0=white noite; 1=pink noise;  2=red noise (also "brownian noise")

    pink_noise = cn.powerlaw_psd_gaussian(beta, N)
    pink_noise = minmax_scale(pink_noise, feature_range=(-1, 1))
    filtered_noise = lowpass_filter(pink_noise, cutoff_freq, sample_rate)
    return pink_noise, filtered_noise


def generate_log_sweep(
    sample_rate: int,
    start_freq: float,
    end_freq: float,
    factor: float,
    cutoff_freq: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Build list of frequencies: start_freq * factor * i until we exceed end_freq
    freqs = []
    f = start_freq
    while True:
        freqs.append(f)
        if f > end_freq:
            break
        f *= factor
    segments = []
    for idx, f in enumerate(freqs):
        half_period_samples = int(sample_rate / (f))
        t = np.linspace(0, 1 / (2 * f), half_period_samples, endpoint=False)
        sign = (-1) ** idx
        segments.append(sign * np.sin(2 * np.pi * f * t))

    sweep = np.concatenate(segments)
    return sweep, lowpass_filter(sweep, cutoff_freq, sample_rate)


def generate_log_sweep_optimal(
    sample_rate: int,
    start_freq: float,
    end_freq: float,
    max_samples: int,
    cutoff_freq: float,
) -> tuple[np.ndarray, np.ndarray]:

    def compute_freqs(factor):
        freqs = []
        f = start_freq
        while f <= end_freq:
            freqs.append(f)
            f *= factor
        return freqs

    def total_samples(factor):
        return sum(int(sample_rate / (2 * f)) for f in compute_freqs(factor))

    # Binary search on factor in range (1.0001, 2.0)
    # smaller factor = more frequencies = more samples consumed
    # larger factor = fewer frequencies = fewer samples consumed
    lo, hi = 1.0001, 2.0
    for _ in range(64):  # 64 iterations is more than enough precision
        mid = (lo + hi) / 2
        if total_samples(mid) > max_samples:
            lo = mid  # too many samples, increase factor to skip more frequencies
        else:
            hi = mid

    best_factor = hi
    freqs = compute_freqs(best_factor)

    segments = []
    for idx, f in enumerate(freqs):
        half_period_samples = int(sample_rate / (2 * f))
        t = np.linspace(0, 1 / (2 * f), half_period_samples, endpoint=False)
        sign = (-1) ** idx
        segments.append(sign * np.sin(2 * np.pi * f * t))

    sweep = np.concatenate(segments)
    # Trim or pad to exactly max_samples
    if len(sweep) < max_samples:
        sweep = np.pad(sweep, (0, max_samples - len(sweep)))
    else:
        sweep = sweep[:max_samples]

    return sweep, lowpass_filter(sweep, cutoff_freq, sample_rate)


def cut_to_nearest_multiple(input: np.ndarray, buffer_size: int) -> np.ndarray:
    r = len(input) // buffer_size
    return input[: r * buffer_size]


def cut_to_size(input: np.ndarray, size: int) -> np.ndarray:
    return input[:size]


def create_data_set(
    sample_rate: int,
    buffer_size: int,
    min_length: float,
    max_length: float,
    min_co_f: int,
    max_co_f: int,
    fstep: int,
    max_buffer_amount: int = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []

    # We use logarithmic spacing to avoid class imbalance problem in the frequency domain, which occurs when using uniform spacing in Hz, meaning the range 50–500 Hz has far fewer samples than 500–5000 Hz relative to how wide each octave is.
    cutoffs = np.geomspace(min_co_f, max_co_f, fstep)
    print(f"Will compute cutoff frequencies in {cutoffs}")

    # white noise
    print(f"Generating white noises...")
    for f in cutoffs:
        newx, newy = generate_white_noise(
            uniform(min_length, max_length), sample_rate, f
        )
        newx = np.append(
            cut_to_nearest_multiple(newx, buffer_size), np.log2(2 * f / sample_rate)
        )
        newy = cut_to_nearest_multiple(newy, buffer_size)
        x.append(newx)
        y.append(newy)
    print(f"Generated {len(cutoffs)} sample noises")

    # pink noise
    print(f"Generating pink noises...")
    for f in cutoffs:
        newx, newy = generate_pink_noise(
            uniform(min_length, max_length), sample_rate, f
        )
        newx = np.append(
            cut_to_nearest_multiple(newx, buffer_size), np.log2(2 * f / sample_rate)
        )
        newy = cut_to_nearest_multiple(newy, buffer_size)
        x.append(newx)
        y.append(newy)
    print(f"Generated {len(cutoffs)} sample noises")

    # sine sweeps

    min_freq = 5
    max_freq = 200
    print(
        f"Generating sine sweeps in very low frequencies ([{min_freq}, {max_freq}])..."
    )
    for f in cutoffs:
        if f > max_freq * 1.5:
            break
        newx, newy = generate_log_sweep_optimal(
            sample_rate, min_freq, max_freq, max_buffer_amount * buffer_size, f
        )
        newx = np.append(
            cut_to_nearest_multiple(newx, buffer_size), np.log2(2 * f / sample_rate)
        )
        newy = cut_to_nearest_multiple(newy, buffer_size)
        x.append(newx)
        y.append(newy)

    min_freq = 500
    max_freq = 1000
    print(f"Generating sine sweeps in low frequencies ([{min_freq}, {max_freq}])...")
    for f in cutoffs:
        if f > max_freq * 1.5:
            break
        newx, newy = generate_log_sweep_optimal(
            sample_rate, min_freq, max_freq, max_buffer_amount * buffer_size, f
        )
        newx = np.append(
            cut_to_nearest_multiple(newx, buffer_size), np.log2(2 * f / sample_rate)
        )
        newy = cut_to_nearest_multiple(newy, buffer_size)
        x.append(newx)
        y.append(newy)

    min_freq = 1000
    max_freq = 7500
    print(f"Generating sine sweeps in mid frequencies ([{min_freq}, {max_freq}])...")
    for f in cutoffs:
        if f > max_freq * 1.5:
            break
        if f < 0.7 * min_freq:
            break
        newx, newy = generate_log_sweep_optimal(
            sample_rate, min_freq, max_freq, max_buffer_amount * buffer_size, f
        )
        newx = np.append(
            cut_to_nearest_multiple(newx, buffer_size), np.log2(2 * f / sample_rate)
        )
        newy = cut_to_nearest_multiple(newy, buffer_size)
        x.append(newx)
        y.append(newy)

    min_freq = 7500
    max_freq = 17000
    print(f"Generating sine sweeps in mid frequencies ([{min_freq}, {max_freq}])...")
    for f in cutoffs:
        if f > max_freq * 1.5:
            break
        if f < 0.7 * min_freq:
            break
        newx, newy = generate_log_sweep_optimal(
            sample_rate, min_freq, max_freq, max_buffer_amount * buffer_size, f
        )
        newx = np.append(
            cut_to_nearest_multiple(newx, buffer_size), np.log2(2 * f / sample_rate)
        )
        newy = cut_to_nearest_multiple(newy, buffer_size)
        x.append(newx)
        y.append(newy)
    print(f"Generated sample sine sweeps")

    return normalize_dataset_lengths(x, y, buffer_size, max_buffer_amount)


def normalize_dataset_lengths(
    x: np.ndarray, y: np.ndarray, buffer_size: int, max_buffer_amount=np.inf
) -> tuple[np.ndarray, np.ndarray]:
    print("Normalizing dataset")

    inputs, outputs = [], []
    min_len = max(buffer_size, len(min(y, key=len)))
    if max_buffer_amount < np.inf:
        min_len = min(max_buffer_amount * buffer_size, min_len)

    print("Size will be {}".format(min_len))

    for xseq, yseq in zip(x, y):
        new_xseq = []
        new_yseq = []
        if len(yseq) >= min_len:
            new_xseq = cut_to_size(xseq[:-1], min_len)
            new_yseq = cut_to_size(yseq, min_len)
        else:
            print("Warning, padding a sample")
            new_xseq = np.pad(xseq[:-1], min_len - len(xseq))
            new_yseq = np.pad(yseq, min_len - len(yseq))
        new_xseq = np.append(new_xseq, xseq[-1])

        inputs.append(new_xseq)
        outputs.append(new_yseq)
    print("Normalization done")
    return inputs, outputs


if __name__ == "__main__":
    import argparse
    import soundfile as sf
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--buffer_size", type=int, default=1024)
    parser.add_argument("--cutoff_freq_n", type=int, default=200)
    parser.add_argument("--max_buffer_amount", type=int, default=-1)

    args = parser.parse_args()

    if args.max_buffer_amount < 0:
        args.max_buffer_amount = np.inf

    x, y = create_data_set(
        args.sample_rate,
        args.buffer_size,
        1,
        5,
        20,
        7500,
        args.cutoff_freq_n,
        args.max_buffer_amount,
    )

    run_id = 0
    sample_folder = ""
    while True:
        try:
            sample_folder = f"dataset-{run_id}"
            os.makedirs(sample_folder)
            break
        except OSError:
            if Path(sample_folder).is_dir():
                run_id = run_id + 1
                continue
            raise

    xpath = os.path.join(sample_folder, "inputs")
    ypath = os.path.join(sample_folder, "expected")
    os.makedirs(xpath)
    os.makedirs(ypath)

    for i in range(len(x)):
        np.save(os.path.join(xpath, f"input-{i}.npy"), x[i])

    for i in range(len(y)):
        np.save(os.path.join(ypath, f"expected-{i}.npy"), y[i])
