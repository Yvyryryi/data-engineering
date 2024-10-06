from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt
from numpy._typing import NDArray
from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import numpy as np
from matplotlib import cm
from scipy.signal import stft

def get_metadata() -> pd.DataFrame:
    return pd.read_csv("data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv")

def preprocess_data_sample(sample_ind: int, metadata) -> Tuple[pd.DataFrame, float]:
    df: pd.DataFrame = pd.read_csv(f"data/lunar/training/data/S12_GradeA/{metadata['filename'].iloc[sample_ind]}.csv", parse_dates =['time_abs(%Y-%m-%dT%H:%M:%S.%f)'] ,index_col = ['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
    df.columns = ['time', 'v']
    return df, metadata['time_rel(sec)'].iloc[sample_ind]

def total_variation_forward(ax, idx, time, velocity, arrive) -> None:
    forward_diff = np.diff(velocity)
    tv = np.abs(forward_diff)
    ax[idx].set(xlabel = 'time', ylabel = 'TV forward')
    ax[idx].plot(
        time[:-1],
        tv,
    )
    ax[idx].axvline(arrive, c = 'orange', linestyle = ':')

def total_variation_backward(ax, idx, time, velocity, arrive) -> None:
    backward_diff = np.diff(velocity[::-1])[::-1]
    tv = np.abs(backward_diff)
    ax[idx].set(xlabel = 'time', ylabel = 'TV backward')
    ax[idx].plot(
        time[1:],
        tv,
    )
    ax[idx].axvline(arrive, c = 'orange', linestyle = ':')

def plot_wavelet(ax, idx, velocity: NDArray, wavelet: str, arrive: float | int) -> None:
    if wavelet == 'morl' or wavelet.startswith('gaus'): # using continuous analysis
        scales = np.arange(1, 128)
        coeffs, _ = pywt.cwt(velocity, scales, wavelet)
        ax[idx].imshow(abs(coeffs), extent = [0, len(velocity), 1, 128], aspect = 'auto',
                   cmap = 'viridis', interpolation = 'nearest')
        ax[idx].set(ylabel = 'amplitude')
        plt.tight_layout()
        plt.show()
    else:
        coeffs = pywt.wavedec(velocity, wavelet, level = 10)
        ax[idx].plot(pywt.waverec(coeffs, wavelet))
        ax[idx].axvline(arrive, c = 'red', linestyle = ':')
        ax[idx].set(ylabel='time')
        plt.tight_layout()
        plt.show()

def smooth_decays(signal: NDArray, sampling_rate=500, wavelet='morl',
                  widths=np.arange(1, 31), decay_length=0.1, smoothness_threshold=0.8) -> List[Tuple[float|int, ...]]:
    coef, _ = pywt.cwt(signal, widths, wavelet)
    scalogram = np.sum(np.abs(coef), axis=0)
    peaks, _ = find_peaks(scalogram, distance = 500)
    smooth_decays = []
    for peak in peaks:
        decay_samples = int(decay_length * sampling_rate)
        if peak + decay_samples >= len(signal):
            continue
        decay = np.abs(signal[peak:peak+decay_samples])
        diffs = np.diff(decay)
        smoothness = np.sum(diffs < 0) / len(diffs)
        avg_slope = (decay[-1] - decay[0]) / len(decay)
        combined_metrics = smoothness * (1 - np.abs(avg_slope))
        if combined_metrics >= smoothness_threshold:
            smooth_decays.append((peak, peak + decay_samples, combined_metrics))
    return smooth_decays

def dx_dt(ax, idx, time, velocity, arrive) -> None:
    ax[idx].set(xlabel = 'time', ylabel = 'Velocity')
    ax[idx].plot(time, velocity, color = 'orange')
    ax[idx].axvline(arrive, c = 'red', linestyle = ':')

def dx2_dt2(ax, idx, time, velocity, arrive) -> None:
    ax[idx].set(xlabel = 'time', ylabel = 'Acceleration')
    ax[idx].plot(
        time[:-1],
        (velocity[:-1] - velocity[1:]) / (time[:-1] - time[1:]),
    )
    ax[idx].axvline(arrive, c = 'orange', linestyle = ':')

def spectogram_sfft(ax, idx, vel, sampling_rate, nperseg, arrive) -> None:
    frequencies, times, Zxx = stft(vel, fs=sampling_rate, nperseg=nperseg)
    ax[idx].set(ylabel = "freq", xlabel = "Time [s]")
    ax[idx].axvline(arrive, color = 'red')
    ax[idx].pcolormesh(times, frequencies, np.abs(Zxx), cmap = cm.jet)

def gaussian(ax, idx, time, velocity, arrive) -> None:
    filtered: NDArray = gaussian_filter(velocity, velocity.std())
    ax[idx].set(xlabel = 'time', ylabel = 'Gaussian')
    ax[idx].plot(time, filtered, color = 'orange')
    ax[idx].axvline(arrive, c = 'red', linestyle = ':')

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butterworth_filter(ax, idx, time, velocity, arrive) -> None:
    out = bandpass_filter(velocity, 0.1, 0.5, fs = 60) ##this is the border of the image set
    ax[idx].set(xlabel = 'time', ylabel = 'Butterworth bandpass filtering')
    ax[idx].plot(time, out, color = 'orange')
    ax[idx].axvline(arrive, c = 'red', linestyle = ':')

def create_graph(*args) -> None:
    _, ax = plt.subplots(len(args), 1, figsize = (12, 5 * len(args)), dpi = 100)
    for idx, arg in enumerate(args):
        arg(ax, idx)
    plt.show()
