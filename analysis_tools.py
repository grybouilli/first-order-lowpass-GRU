import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from joblib import Parallel, delayed
from torch.nn import Module
from create_dataset import normalize_freq
from model_tools import run_inference


# method from https://dsp.stackexchange.com/a/73993/91311
# the idea of the method is to produce a gain that will be time-independant
def demod_signal(time: np.ndarray, signal: np.ndarray, frequency: float) -> np.ndarray:
    x = np.cos(2 * np.pi * frequency * time) - 1j * np.sin(2 * np.pi * frequency * time)
    return signal * x


def signal_gain_at_f(time: np.ndarray, signal: np.ndarray, frequency: float) -> float:
    z = demod_signal(time, signal, frequency)
    avg_z = np.average(z)  # avg_z is independant from time
    return 2 * np.absolute(avg_z)


def signal_phase_at_f(time: np.ndarray, signal: np.ndarray, frequency: float) -> float:
    z = demod_signal(time, signal, frequency)
    avg_z = np.average(z)  # avg_z is independant from time
    return np.angle(avg_z)


def filter_gains(
    frequencies: np.ndarray,
    buffer_size: int,
    buffer_count: int,
    filter_for_fc: callable,
    fc_hertz: float,
    sample_rate: float = 48000,
):
    def process_freq(freq):
        p = sample_rate / freq
        m = (buffer_count * buffer_size) // p
        t = np.linspace(0, m / freq, buffer_size * buffer_count, endpoint=False)
        input_signal = np.cos(2 * np.pi * freq * t)
        signal = filter_for_fc(input_signal, fc_hertz)
        return signal_gain_at_f(t, signal, freq)

    return Parallel(n_jobs=-1)(delayed(process_freq)(freq) for freq in frequencies)


def filter_phases(
    frequencies: np.ndarray,
    buffer_size: int,
    buffer_count: int,
    filter_for_fc: callable,
    fc_hertz: float,
    sample_rate: float = 48000,
):
    def process_freq(freq):
        p = sample_rate / freq
        m = (buffer_count * buffer_size) // p
        t = np.linspace(0, m / freq, buffer_size * buffer_count, endpoint=False)
        input_signal = np.cos(2 * np.pi * freq * t)
        signal = filter_for_fc(input_signal, fc_hertz)
        return signal_phase_at_f(t, signal, freq)

    return Parallel(n_jobs=-1)(delayed(process_freq)(freq) for freq in frequencies)


def plot_bode_GRU_into(
    axes: Axes,
    model: Module,
    cutoff_freq: float,
    buffer_size: int,
    sample_rate: int,
    n_freqs: int = 100,
    buffer_count: int = 10,
    fmt: str = "-",
    label: str = "GRU Filter",
    plot_type: str = "gain",
):

    def iir_gru(input_signal: np.ndarray, fc_hertz: float):
        output = run_inference(
            model,
            input_signal,
            normalize_freq(fc_hertz, sample_rate),
            buffer_size,
        )
        return output

    freqs = np.geomspace(20, 20000, n_freqs)  # log-spaced, 20Hz to Nyquist
    gains = None
    phases = None
    if plot_type == "gain":
        gains = filter_gains(
            freqs,
            buffer_size,
            buffer_count,
            iir_gru,
            cutoff_freq,
            sample_rate=sample_rate,
        )
        magnitudes_db = 20 * np.log10(np.array(gains) + 1e-8)
        axes.semilogx(freqs, magnitudes_db, fmt, label=label)
    else:
        phases = filter_phases(
            freqs,
            buffer_size,
            buffer_count,
            iir_gru,
            cutoff_freq,
            sample_rate=sample_rate,
        )
        axes.semilogx(freqs, np.unwrap(phases), fmt, label=label)


def plot_bode_GRU(
    model: Module,
    cutoff_freq: float,
    buffer_size: int,
    sample_rate: int,
    n_freqs: int = 100,
    buffer_count: int = 10,
    fmt: str = "-",
    label: str = "GRU Filter",
    show=True,
    plot_type: str = "gain",
) -> tuple[plt.figure.Figure, Axes]:
    fig = plt.figure(figsize=(10, 5))
    axes = fig.add_axes(rect=[0.125, 0.11, 0.775, 0.77])
    plot_bode_GRU_into(
        axes,
        model,
        cutoff_freq,
        buffer_size,
        sample_rate,
        n_freqs,
        buffer_count,
        fmt,
        label,
        plot_type=plot_type,
    )
    axes.axvline(cutoff_freq, color="r", linestyle=":", label=f"fc = {cutoff_freq} Hz")
    axes.axhline(-3, color="gray", linestyle=":", label="-3 dB")
    axes.set_xlabel("Frequency (Hz)")
    if plot_type == "gain":
        axes.set_ylabel("Magnitude (dB)")
        axes.set_title(f"Bode Magnitude Plot (steady-state) — fc = {cutoff_freq} Hz")
    else:
        axes.set_ylabel("Phase (rad)")
        axes.set_title(f"Bode Phase Plot (steady-state) — fc = {cutoff_freq} Hz")
    axes.legend()
    axes.grid(True, which="both")
    fig.tight_layout()

    if show:
        fig.show()

    return fig, axes


def plot_bode_ref_filter_into(
    axes: Axes,
    filt: callable,
    sample_rate: int,
    n_freqs: int = 100,
    fmt: str = "--",
    label: str = "Reference Filter",
    plot_type: str = "gain",
):
    from scipy.signal import freqz

    freqs = np.geomspace(20, 20000, n_freqs)  # log-spaced, 20Hz to Nyquist

    b, a = filt()
    w, h = freqz(b, a, worN=freqs, fs=sample_rate)

    if plot_type == "gain":
        reference_db = 20 * np.log10(np.abs(h) + 1e-8)
        axes.semilogx(w, reference_db, fmt, label=label)
    else:
        reference_db = np.angle(h)
        axes.semilogx(w, reference_db, fmt, label=label)
    return


def plot_bode_ref_filter(
    filt: callable,
    cutoff_freq: float,
    sample_rate: int,
    n_freqs: int = 100,
    fmt: str = "--",
    label: str = "Reference Filter",
    show=True,
    plot_type: str = "gain",
) -> tuple[plt.figure.Figure, Axes]:
    fig = plt.figure(figsize=(10, 5))
    axes = fig.add_axes(rect=[0.125, 0.11, 0.775, 0.77])
    plot_bode_ref_filter_into(
        axes, filt, sample_rate, n_freqs, fmt, label, plot_type=plot_type
    )
    axes.axvline(cutoff_freq, color="r", linestyle=":", label=f"fc = {cutoff_freq} Hz")
    axes.axhline(-3, color="gray", linestyle=":", label="-3 dB")
    axes.set_xlabel("Frequency (Hz)")
    if plot_type == "gain":
        axes.set_ylabel("Magnitude (dB)")
        axes.set_title(f"Bode Magnitude Plot (steady-state) — fc = {cutoff_freq} Hz")
    else:
        axes.set_ylabel("Phase (rad)")
        axes.set_title(f"Bode Phase Plot (steady-state) — fc = {cutoff_freq} Hz")
    axes.legend()
    axes.grid(True, which="both")
    fig.tight_layout()

    if show:
        fig.show()

    return fig, axes


def plot_cheby_into(
    axes: Axes,
    cutoff_freq: float,
    order: int,
    ripple: float,
    sample_rate: int,
    n_freqs: int = 100,
    fmt: str = "--",
    label: str = "Reference Filter",
    show=True,
    plot_type: str = "gain",
    filter_type: str = "low",
):
    from scipy.signal import cheby1

    filt = lambda: cheby1(
        order, ripple, 2 * cutoff_freq / sample_rate, btype=filter_type, analog=False
    )
    return plot_bode_ref_filter_into(
        axes, filt, sample_rate, n_freqs, fmt, label, plot_type=plot_type
    )


def plot_cheby(
    axes: Axes,
    cutoff_freq: float,
    order: int,
    ripple: float,
    sample_rate: int,
    n_freqs: int = 100,
    fmt: str = "--",
    label: str = "Reference Filter",
    show=True,
    plot_type: str = "gain",
    filter_type: str = "low",
):
    from scipy.signal import cheby1

    filt = lambda: cheby1(
        order, ripple, 2 * cutoff_freq / sample_rate, btype=filter_type, analog=False
    )
    return plot_bode_ref_filter(
        filt, cutoff_freq, sample_rate, n_freqs, fmt, label, show, plot_type=plot_type
    )


def plot_butter_worth_into(
    axes: Axes,
    cutoff_freq: float,
    order: int,
    sample_rate: int,
    n_freqs: int = 100,
    fmt: str = "--",
    label: str = "Reference Filter",
    show=True,
    plot_type: str = "gain",
    filter_type: str = "low",
):
    from scipy.signal import butter

    filt = lambda: butter(
        order, 2 * cutoff_freq / sample_rate, btype=filter_type, analog=False
    )
    return plot_bode_ref_filter_into(
        axes, filt, sample_rate, n_freqs, fmt, label, plot_type=plot_type
    )


def plot_butter_worth(
    cutoff_freq: float,
    order: int,
    sample_rate: int,
    n_freqs: int = 100,
    fmt: str = "--",
    label: str = "Reference Filter",
    show=True,
    plot_type: str = "gain",
):
    from scipy.signal import butter

    filt = lambda: butter(
        order, 2 * cutoff_freq / sample_rate, btype="low", analog=False
    )
    return plot_bode_ref_filter(
        filt, cutoff_freq, sample_rate, n_freqs, fmt, label, show, plot_type=plot_type
    )


def plot_bode_so(
    model: Module,
    cutoff_freq: float,
    buffer_size: int,
    sample_rate: int,
    n_freqs: int = 100,
    buffer_count: int = 10,
    plot_type: str = "gain",
) -> None:
    _, axes = plot_bode_GRU(
        model, cutoff_freq, buffer_size, sample_rate, n_freqs, buffer_count
    )

    plot_bode_ref_filter_into(
        axes, cutoff_freq, sample_rate, n_freqs, plot_type=plot_type
    )
