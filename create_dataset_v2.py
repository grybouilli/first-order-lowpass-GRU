"""
Dataset generation for training of first order IRR GRU
=======================================================================
Architecture: GRU (recurrent) + Linear (output projection), no nonlinearity
              on the output — this is intentional for modeling a linear IIR
              (first order lowpass).

Dataset:      Exponential sine sweep + white noise, passed through a target
              first-order lowpass filter.
"""

import numpy as np
from scipy import signal as sp_signal
from random import uniform
from joblib import Parallel, delayed

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TARGET FILTER  (first-order IIR lowpass — the "ground truth" to learn)
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
# 2.  DATASET  — generates input/target pairs on-the-fly
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
) -> np.ndarray:
    """
    Bandlimited white noise for IIR filter training.

    Design rationale
    ----------------
    - White Gaussian noise has a flat power spectral density, so every
      frequency bin receives equal energy. This uniformly exercises the
      filter's gain and phase response across the whole band — ideal for
      learning filter coefficients that must be accurate at all frequencies.

    - The amplitude ramp (0 → 1) ensures the full input amplitude range is
      covered. Without it, the model is only trained at a single RMS level
      and may not generalise to louder or quieter inputs (critical for
      nonlinear circuits; harmless-but-good-practice for linear IIR).

    - Bandlimiting via a linear-phase FIR (Kaiser window) removes energy
      above Nyquist and below f_low. For a first-order lowpass whose cutoff
      is well above f_low, this makes no practical difference; for filters
      with very low cutoffs, it prevents the DC component from dominating
      the loss.

    - Fade-in / fade-out tapers suppress spectral leakage at the segment
      boundaries, which matters when the segment is later used in an FFT-
      based evaluation (e.g. frequency-domain loss or spectral visualisation).

    Parameters
    ----------
    n_samples     : number of output samples
    sample_rate   : Hz
    f_low         : lower bandlimit in Hz  (default 20 Hz — sub-bass cutoff)
    f_high        : upper bandlimit in Hz  (default: 0.99 × Nyquist)
    amplitude_ramp: if True, multiply by a linear ramp 0 → 1
    fade_samples  : length of cos² fade-in / fade-out taper

    Returns
    -------
    noise : float32 ndarray of shape (n_samples,), peak-normalised to ±1
    """
    if f_high is None:
        f_high = sample_rate / 2.0 * 0.99  # stay just below Nyquist

    # ── 1. Generate white Gaussian noise ────────────────────────────────────
    noise = np.random.randn(n_samples).astype(np.float32)

    # ── 2. Bandlimit with a linear-phase FIR (Kaiser window) ────────────────
    #   numtaps chosen so the transition band is narrow but the filter is
    #   short enough to be fast. Rule of thumb: ~4 × sample_rate / f_low,
    #   capped at 1025 to avoid excessive latency artefacts in the signal.
    numtaps = min(1025, 4 * int(sample_rate / max(f_low, 1.0)) + 1)
    if numtaps % 2 == 0:
        numtaps += 1  # Kaiser FIR must have odd length for linear phase

    fir = sp_signal.firwin(
        numtaps,
        cutoff=[f_low, f_high],
        pass_zero=False,  # bandpass (not low/highpass)
        window="kaiser",
        width=fade_samples,
        fs=sample_rate,
    ).astype(np.float32)

    # Apply FIR with zero-phase filtering (lfilter would introduce group delay)
    noise = sp_signal.filtfilt(fir, [1.0], noise).astype(np.float32)

    # ── 3. Peak-normalise to ±1 ──────────────────────────────────────────────
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise /= peak

    # ── 4. Amplitude ramp  (0 → 1 linear) ───────────────────────────────────
    if amplitude_ramp:
        match ramp_type:
            case RampTypes.growing_linear:
                ramp = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
                noise = noise * ramp
            case RampTypes.decreasing_linear:
                ramp = np.linspace(1.0, 0.0, n_samples, dtype=np.float32)
                noise = noise * ramp
            case RampTypes.gaussian_distrib:
                mu = 0
                sigma = 0.1
                gaussian_distrib = lambda x: np.exp(
                    -((x - mu) ** 2) / (2 * sigma**2)
                ) / np.sqrt(2 * np.pi * sigma**2)
                x = np.linspace(-0.5, 0.5, n_samples, dtype=np.float32)
                ramp = gaussian_distrib(x)
                ramp /= max(ramp)  # keep in range [0,1]
                noise = noise * ramp
            case RampTypes.no_ramp:
                pass

    # ── 5. Fade-in / fade-out taper (cos² shape, smoother than linear) ──────
    if fade_samples > 0:
        t = np.linspace(0.0, np.pi / 2, fade_samples, dtype=np.float32)
        taper = np.sin(t) ** 2  # 0 → 1  (cos² fade-in)
        fade = np.ones(n_samples, dtype=np.float32)
        fade[:fade_samples] = taper
        fade[-fade_samples:] = taper[::-1]
        noise = noise * fade

    return noise


def normalize_freq(freq: float, sample_rate: float) -> float:
    return -np.log2(2 * freq / sample_rate) / np.log2(1 / sample_rate)


import scipy.signal as ss


class Filters:
    butter = "butter"
    cheby1 = "cheby1"
    algo = {
        "butter": ss.butter,
        "cheby1": ss.cheby1,
    }
    supported_algo = [butter, cheby1]

    lowpass = "lowpass"
    highpass = "highpass"
    bandpass = "bandpass"

    supported_types = [lowpass, highpass]


def make_dataset_signal(
    filter_algo: str,
    filter_type: str,
    filter_order: int,
    buffer_size: int,
    max_buffer_count: int,
    sample_rate: float,
    f_low: float,
    f_high: float,
    min_fc: float,
    max_fc: float,
    amount_of_fc: int,
    cheby_ripple: float = 0.5,
) -> np.ndarray:
    """
    Concatenate an exponential sine sweep and bandlimited white noise.
    Both components cover the full amplitude range; together they give
    uniform spectral AND amplitude coverage.

    - The sweep exercises all frequencies sequentially at fixed amplitude.
    - The noise exercises all frequencies simultaneously across all amplitudes.
    """

    total_samples = buffer_size * max_buffer_count
    if filter_algo not in Filters.algo.keys():
        raise Exception(f"Unknown filter algorithm: {filter_algo}")

    cutoffs = np.geomspace(min_fc, max_fc, amount_of_fc)
    x, y = [], []

    print("Generating impulse...")
    from scipy.signal import unit_impulse

    imp = unit_impulse(buffer_size * max_buffer_count, ifx="mid")
    print("Impulse sample amount = {}".format(len(imp)))

    print("Generating sine sweeps...")
    sweeps = []

    sweeps.append(
        exponential_sweep(
            total_samples, f1=f_low, f2=f_high / 2, sample_rate=sample_rate
        )
    )
    sweeps.append(
        exponential_sweep(
            total_samples, f1=f_high / 2, f2=f_high, sample_rate=sample_rate
        )
    )
    noises = []

    print(f"Sine sweep sample amount = {len(sweeps[0])}")
    print(f"Sine sweep sample amount = {len(sweeps[1])}")
    print("Generating white noises...")

    def gen_noise(ramp_type: str):
        noises.append(
            bandlimited_white_noise(
                total_samples,
                sample_rate,
                f_low=f_low,
                f_high=f_high,
                amplitude_ramp=True,
                ramp_type=ramp_type,
            )
        )

    for ramp in [
        RampTypes.growing_linear,
        RampTypes.decreasing_linear,
        RampTypes.gaussian_distrib,
        RampTypes.no_ramp,
    ]:
        gen_noise(ramp)

    print(f"Noise sample amount = {len(noises[0])}")
    print(f"Noise sample amount = {len(noises[1])}")
    print(f"Noise sample amount = {len(noises[2])}")

    print("Filtering signals...")
    for fc in cutoffs:
        # print(f"FC = {fc}")
        b, a = 0, 0
        match filter_algo:
            case Filters.butter:
                b, a = Filters.algo[Filters.butter](
                    filter_order, fc, btype=filter_type, fs=sample_rate
                )
            case Filters.cheby1:
                b, a = Filters.algo[Filters.butter](
                    filter_order, cheby_ripple, fc, btype=filter_type, fs=sample_rate
                )

        y.append(sp_signal.lfilter(b, a, imp).astype(np.float32))
        imp_plus_freq = np.append(imp, normalize_freq(fc, sample_rate))
        x.append(imp_plus_freq)

        y.append(sp_signal.lfilter(b, a, sweeps[0]).astype(np.float32))
        sweep_plus_freq = np.append(sweeps[0], normalize_freq(fc, sample_rate))
        x.append(sweep_plus_freq)

        y.append(sp_signal.lfilter(b, a, sweeps[1]).astype(np.float32))
        sweep_plus_freq = np.append(sweeps[1], normalize_freq(fc, sample_rate))
        x.append(sweep_plus_freq)

        for noise in noises:
            y.append(sp_signal.lfilter(b, a, noise).astype(np.float32))
            noise_plus_freq = np.append(noise, normalize_freq(fc, sample_rate))
            x.append(noise_plus_freq)

    print("Done generating samples")

    return x, y


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
        "--amount_of_fc",
        type=int,
        default=200,
        help="Expected signals will be filtered with frequencies ranging from 50 to 7500 Hz on a logarithmic scale. This option gives the amount of cut-off frequencies to use.",
    )
    parser.add_argument(
        "--max_buffer_amount",
        type=int,
        default=-1,
        help="Generated signal will have buffer_size * max_buffer_amount samples",
    )

    parser.add_argument(
        "--filter_algo",
        type=str,
        default="butter",
        help=f"Filter's equation-based definition. Supported : {Filters.supported_algo}",
        choices=Filters.supported_algo,
    )

    parser.add_argument(
        "--filter_type",
        type=str,
        default="lowpass",
        help=f"Filter's type. Supported : {Filters.supported_types}",
        choices=Filters.supported_types,
    )

    parser.add_argument(
        "--filter_order",
        type=int,
        default=1,
        help="Filter's order",
    )

    parser.add_argument(
        "--cheby_ripple",
        type=float,
        default=0.5,
        help="Chebyshev passband or stopband ripple value ; is ignored if specified for a filter that is not cheby1 or cheby2",
    )

    args = parser.parse_args()

    if args.max_buffer_amount < 0:
        args.max_buffer_amount = (args.sample_rate * 2) // args.buffer_size

    inputs, outputs = make_dataset_signal(
        args.filter_algo,
        args.filter_type,
        args.filter_order,
        args.buffer_size,
        args.max_buffer_amount,
        args.sample_rate,
        f_low=20,
        f_high=args.sample_rate / 2 - 1,
        min_fc=50,
        max_fc=7500,
        amount_of_fc=args.amount_of_fc,
        cheby_ripple=args.cheby_ripple,
    )

    run_id = 0
    sample_folder = ""
    while True:
        try:
            sample_folder = f"dataset-{args.filter_algo}-{args.filter_type}-{args.filter_order}-{run_id}"
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

    idx = 0
    print("Saving samples...")
    for x, y in zip(inputs, outputs):
        np.save(os.path.join(xpath, f"input-{idx}.npy"), x.astype(np.float32))
        np.save(os.path.join(ypath, f"expected-{idx}.npy"), y.astype(np.float32))
        idx += 1
    print("Samples saved to {}".format(sample_folder))
