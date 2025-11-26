import os
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from psf_functions import compute_psd, smooth_boxcar


def load_frame_counters(folder_data: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load standard frame counters from a data folder."""
    filenames = [
        "cred3_frame_counter.fits",
        "hrtc_counter.fits",
        "pyramid_counter.fits",
    ]
    return tuple(fits.getdata(os.path.join(folder_data, name)) for name in filenames)  # type: ignore[return-value]


def plot_frame_counters(counters: Sequence[np.ndarray], title_prefix: str = "") -> Tuple[plt.Figure, np.ndarray]:
    """Plot stacked frame counters."""
    labels = ["CRED3 frame counter", "HRTC counter", "Pyramid counter"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for ax, data, label in zip(axes, counters, labels):
        ax.plot(data)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Index")
    if title_prefix:
        fig.suptitle(f"{title_prefix} counters")
    return fig, axes


def load_mode_timestamps(folder_data: str, filename: str = "modes_in_ts.fits") -> np.ndarray:
    """Load mode timestamps from disk."""
    return fits.getdata(os.path.join(folder_data, filename))


def plot_timestamp_diffs(
    timestamps: np.ndarray,
    title: str,
    missed_threshold: float = 0.0015,
    ylim: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Plot timestamp deltas and report duplicate/missed frames."""
    diffs = np.diff(np.asarray(timestamps.squeeze()))
    duplicate_indices = np.where(diffs == 0)[0]
    missed_indices = np.where(diffs > missed_threshold)[0]

    plt.figure()
    plt.plot(diffs)
    if ylim:
        plt.ylim(*ylim)
    plt.title(title)
    plt.show()

    print("duplicate frames count:", len(duplicate_indices))
    missed_durations = np.round(diffs[missed_indices], 4) - 0.001
    missed_count = np.sum(missed_durations) / 0.001 if missed_indices.size else 0
    print("missed frames count:", missed_count)
    loop_rate = float(np.median(diffs))
    print(loop_rate)
    return diffs, duplicate_indices, missed_indices, loop_rate


def load_modes(folder_data: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load common mode and voltage files from a data folder."""
    modes_in = fits.getdata(os.path.join(folder_data, "modes_in.fits"))
    modes_out = fits.getdata(os.path.join(folder_data, "modes_out.fits"))
    voltages = fits.getdata(os.path.join(folder_data, "voltages.fits"))
    modes_in_bis = fits.getdata(os.path.join(folder_data, "modes_in_bis.fits"))
    return modes_in, modes_out, voltages, modes_in_bis


def load_dm_maps(folder_data: str) -> Tuple[np.ndarray, ...]:
    """Load deformable mirror command maps."""
    maps = []
    for name in ["dm0.fits", "dm1.fits", "dm2.fits", "dm3.fits"]:
        maps.append(fits.getdata(os.path.join(folder_data, name)).astype(float))
    return tuple(maps)


def plot_dm_maps(
    dm_maps: Sequence[np.ndarray],
    snapshot_idx: int = 0,
    title: str | None = None,
    include_sum: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot snapshot of DM channels with optional summed axis."""
    extra = 1 if include_sum else 0
    fig, axes = plt.subplots(1, len(dm_maps) + extra, figsize=(14, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, dm, name in zip(axes, dm_maps, [f"dm{i}" for i in range(len(dm_maps))]):
        ax.plot(dm[snapshot_idx, :])
        ax.set_title(name)
        ax.set_xlabel("Actuator index")
        ax.set_ylabel("Voltage")
        ax.grid(True, alpha=0.3)

    if include_sum:
        summed = np.sum(dm_maps, axis=0)
        axes[-1].plot(summed[snapshot_idx, :])
        axes[-1].set_title("sum dm")
        axes[-1].set_xlabel("Actuator index")
        axes[-1].set_ylabel("Voltage")
        axes[-1].grid(True, alpha=0.3)

    return fig, axes


def plot_dm_time_series(
    dm_maps: Sequence[np.ndarray],
    n: int = 5,
    title: str | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot first ``n`` actuators over time for each DM map."""
    fig, axes = plt.subplots(1, len(dm_maps), figsize=(14, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, dm, name in zip(axes, dm_maps, [f"dm{i}" for i in range(len(dm_maps))]):
        for i in range(n):
            ax.plot(dm[:, i], label=f"Act {i}")
        ax.set_title(name)
        ax.set_xlabel("Time index")
        ax.set_ylabel("Voltage")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    return fig, axes


def plot_modes_psd(
    modes: np.ndarray,
    fs: float,
    title: str,
    num_modes: int = 5,
    ylim: Tuple[float, float] = (1e-11, 1e-3),
) -> Tuple[np.ndarray, np.ndarray]:
    """Plot PSD for the first ``num_modes`` KL modes."""
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(*ylim)
    plt.xlim(0, fs / 2)

    freqs = None
    psd_smooth = None
    for i in range(min(num_modes, modes.shape[1])):
        data = modes[:, i]
        freqs, psd = compute_psd(data, fs)
        psd_smooth = smooth_boxcar(psd, N=9)
        plt.plot(freqs, psd_smooth, label=f"Mode {i+1}")
    plt.legend()
    plt.grid()
    return freqs if freqs is not None else np.array([]), psd_smooth if psd_smooth is not None else np.array([])


def plot_summed_modes_psd(
    summed_modes: np.ndarray,
    fs: float,
    title: str,
    ylim: Tuple[float, float] = (1e-11, 1e-3),
) -> Tuple[np.ndarray, np.ndarray]:
    """Plot PSD of the combined modal energy."""
    freqs, psd = compute_psd(summed_modes, fs)
    psd_smooth = smooth_boxcar(psd, N=9)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(*ylim)
    plt.xlim(0, fs / 2)
    plt.plot(freqs, psd_smooth)
    plt.grid()
    return freqs, psd_smooth
