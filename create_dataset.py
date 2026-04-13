"""
Dataset generation for training of first order IRR GRU
=======================================================================
Architecture: GRU (recurrent) + Linear (output projection), no nonlinearity
              (first order lowpass).

Dataset:      Exponential sine sweep + white noise, passed through a target
              first-order lowpass filter.
"""

import numpy as np
from scipy import signal as sp_signal
from random import uniform
from joblib import Parallel, delayed

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────


def make_lowpass_coeffs(cutoff_hz: float, sample_rate: float):
    """
    First-order IIR lowpass via bilinear transform.
    Returns (b, a) in scipy convention: b = [b0, b1], a = [1, a1]
    """
    wc = 2 * np.pi * cutoff_hz / sample_rate  # digital angular frequency
    alpha = np.exp(-wc)  # pole location
    b = np.array([(1 - alpha) / 2, (1 - alpha) / 2])  # numerator
    a = np.array([1.0, -alpha])  # denominator
    return b, a


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────


def exponential_sweep(
    n_samples: int, f1: float, f2: float, sample_rate: float, fade_samples: int = 256
) -> np.ndarray:
    """
    Exponential (log) sine sweep from f1 to f2.
    x(t) = sin[ 2π·f1·L·(exp(t/L) - 1) ],  L = T / ln(f2/f1)
    """
    T = n_samples / sample_rate
    L = T / np.log(f2 / f1)
    t = np.arange(n_samples) / sample_rate
    sweep = np.sin(2 * np.pi * f1 * L * (np.exp(t / L) - 1))
    # Fade in/out to reduce spectral leakage
    fade = np.ones(n_samples)
    fade[:fade_samples] = np.linspace(0, 1, fade_samples)
    fade[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return (sweep * fade).astype(np.float32)


class RampTypes:
    growing_linear = "growing_linear"
    decreasing_linear = "decreasing_linear"
    gaussian_distrib = "gaussian_distrib"
    no_ramp = "no_ramp"


def bandlimited_white_noise(
    n_samples: int,
    sample_rate: float,
    f_low: float = 20.0,
    f_high: float | None = None,
    amplitude_ramp: bool = True,
    ramp_type: str = RampTypes.growing_linear,
    fade_samples: int = 256,
    rng: np.random.Generator | None = None,
    fir: np.ndarray | None = None,  # pre-computed FIR coefficients
) -> np.ndarray:
    """
    Bandlimited white noise for IIR filter training.
    ...
    Parameters
    ----------
    ...
    rng           : optional numpy Generator for thread-safe random state.
                    If None, falls back to np.random.randn (not thread-safe).
    fir           : optional pre-computed FIR coefficients (float32 ndarray).
                    If provided, the internal firwin() call is skipped entirely,
                    which is a significant speedup when generating many signals
                    with the same f_low / f_high / sample_rate.
    """
    if f_high is None:
        f_high = sample_rate / 2.0 * 0.99

    # ── 1. Generate white Gaussian noise ────────────────────────────────────
    if rng is not None:
        noise = rng.standard_normal(n_samples).astype(np.float32)
    else:
        noise = np.random.randn(n_samples).astype(np.float32)

    # ── 2. Bandlimit with a linear-phase FIR (Kaiser window) ────────────────
    if fir is None:
        numtaps = min(1025, 4 * int(sample_rate / max(f_low, 1.0)) + 1)
        if numtaps % 2 == 0:
            numtaps += 1

        fir = sp_signal.firwin(
            numtaps,
            cutoff=[f_low, f_high],
            pass_zero=False,
            window="kaiser",
            width=fade_samples,
            fs=sample_rate,
        ).astype(np.float32)

    noise = sp_signal.filtfilt(fir, [1.0], noise).astype(np.float32)

    # ── 3. Peak-normalise to ±1 ──────────────────────────────────────────────
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise /= peak

    # ── 4. Amplitude ramp ────────────────────────────────────────────────────
    if amplitude_ramp:
        match ramp_type:
            case RampTypes.growing_linear:
                ramp = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
                noise = noise * ramp
            case RampTypes.decreasing_linear:
                ramp = np.linspace(1.0, 0.0, n_samples, dtype=np.float32)
                noise = noise * ramp
            case RampTypes.gaussian_distrib:
                mu, sigma = 0, 0.1
                x = np.linspace(-0.5, 0.5, n_samples, dtype=np.float32)
                ramp = np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / np.sqrt(
                    2 * np.pi * sigma**2
                )
                ramp /= ramp.max()
                noise = noise * ramp
            case RampTypes.no_ramp:
                pass

    # ── 5. Fade-in / fade-out taper ──────────────────────────────────────────
    if fade_samples > 0:
        t = np.linspace(0.0, np.pi / 2, fade_samples, dtype=np.float32)
        taper = np.sin(t) ** 2
        fade = np.ones(n_samples, dtype=np.float32)
        fade[:fade_samples] = taper
        fade[-fade_samples:] = taper[::-1]
        noise = noise * fade

    return noise


def normalize_freq(freq: float, sample_rate: float) -> float:
    return -np.log2(2 * freq / sample_rate) / np.log2(1 / sample_rate)


def make_dataset_signal(
    buffer_size: int,
    max_buffer_count: int,
    sample_rate: float,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    """
    Concatenate an exponential sine sweep and bandlimited white noise.
    Both components cover the full amplitude range; together they give
    uniform spectral AND amplitude coverage.

    - The sweep exercises all frequencies sequentially at fixed amplitude.
    - The noise exercises all frequencies simultaneously across all amplitudes.

    Input signals (sweeps, noises) are regenerated independently for each
    cutoff frequency to maximise dataset diversity. All per-fc work is
    parallelised with joblib.
    """

    total_samples = buffer_size * max_buffer_count

    # ------------------------------------------------------------------
    # Helper: generate all signals + apply filter for ONE cutoff frequency
    # Returns a list of (x_row, y_row) pairs.
    # Everything is self-contained so joblib can ship it to any worker.
    # ------------------------------------------------------------------
    rng = np.random.default_rng()

    f_high_fir = sample_rate / 2.0 * 0.99
    numtaps = min(1025, 4 * int(sample_rate / max(f_low, 1.0)) + 1)
    if numtaps % 2 == 0:
        numtaps += 1
    fir = sp_signal.firwin(
        numtaps,
        cutoff=[f_low, f_high_fir],
        pass_zero=False,
        window="kaiser",
        width=256,  # matches fade_samples default
        fs=sample_rate,
    ).astype(np.float32)

    sweep_lo = exponential_sweep(
        total_samples, f1=f_low, f2=f_high / 2, sample_rate=sample_rate
    )
    sweep_hi = exponential_sweep(
        total_samples, f1=f_high / 2, f2=f_high, sample_rate=sample_rate
    )

    noises = [
        bandlimited_white_noise(
            total_samples,
            sample_rate,
            f_low=f_low,
            f_high=f_high,
            amplitude_ramp=True,
            ramp_type=ramp,
            rng=rng,
            fir=fir,  # ← skip firwin() inside the function
        )
        for ramp in (
            RampTypes.growing_linear,
            RampTypes.decreasing_linear,
            RampTypes.gaussian_distrib,
            RampTypes.no_ramp,
        )
    ]
    signals = np.vstack([sweep_lo, sweep_hi, noises])
    print(signals)
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample_rate", type=int, default=44100, help="Generated signal sample-rate"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1024,
        help="Generated signal will have buffer_size * max_buffer_amount samples",
    )

    parser.add_argument(
        "--max_buffer_amount",
        type=int,
        default=-1,
        help="Generated signal will have buffer_size * max_buffer_amount samples",
    )

    args = parser.parse_args()

    if args.max_buffer_amount < 0:
        args.max_buffer_amount = (args.sample_rate * 2) // args.buffer_size

    inputs = make_dataset_signal(
        args.buffer_size,
        args.max_buffer_amount,
        args.sample_rate,
        f_low=20,
        f_high=args.sample_rate / 2 - 1,
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

    idx = 0
    print("Saving samples...")
    for x in inputs:
        np.save(os.path.join(sample_folder, f"input-{idx}.npy"), x.astype(np.float32))
        idx += 1
    print("Samples saved to {}".format(sample_folder))
