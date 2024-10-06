import pandas as pd
from numpy._typing import NDArray
from typing import List, Tuple
import matplotlib.pyplot as plt
import pywt
import numpy as np
from scipy.signal import find_peaks


def get_metadata() -> pd.DataFrame:
    return pd.read_csv("data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv")

def preprocess_data_sample(sample_ind: int, metadata) -> Tuple[pd.DataFrame, float]:
    df: pd.DataFrame = pd.read_csv(f"data/lunar/training/data/S12_GradeA/{metadata['filename'].iloc[sample_ind]}.csv", parse_dates =['time_abs(%Y-%m-%dT%H:%M:%S.%f)'] ,index_col = ['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
    df.columns = ['time', 'v']
    return df, metadata['time_rel(sec)'].iloc[sample_ind]

def tv_analysis(chunk_size: int, v: NDArray) -> None:
    out = v
    for _ in range(chunk_size):
        out = out[1:] - out[:-1]
    return out

def plot_wavelet(velocity: NDArray, wavelet: str | List[str], arrive: float | int) -> None:
    if isinstance(wavelet, str):
        if wavelet == 'morl' or wavelet.startswith('gaus'): # using continuous analysis
            scales = np.arange(1, 128)
            coeffs, _ = pywt.cwt(velocity, scales, wavelet)
            plt.imshow(abs(coeffs), extent = [0, len(velocity), 1, 128], aspect = 'auto',
                       cmap = 'viridis', interpolation = 'nearest')
            plt.ylabel('amplitude')
        else:
            coeffs = pywt.wavedec(velocity, wavelet, level = 10)
            plt.plot(pywt.waverec(coeffs, wavelet))
            plt.axvline(arrive, c = 'red')
            plt.ylabel('time')
            plt.tight_layout()
            plt.show()
    elif isinstance(wavelet, list):
        for wave in wavelet:
            plot_wavelet(velocity, wave, arrive)

def smooth_decays(time: NDArray, signal: NDArray, arrive: int | float, sampling_rate=1000, wavelet='morl',
                  widths=np.arange(1, 31), prominence=0.1, decay_length=0.1, smoothness_threshold=0.8) -> None:
    coef, _ = pywt.cwt(signal, widths, wavelet)
    scalogram = np.sum(np.abs(coef), axis=0)
    peaks, _ = find_peaks(scalogram, prominence=prominence*np.max(scalogram))
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
    _, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=100)
    ax[0].plot(time, signal)
    ax[0].axvline(arrive, color='r', linestyle=':', label='Target')
    for peak, _, _ in smooth_decays:
        ax[0].axvline(time[peak], color='b', linestyle='-', label='Peak detection')
    ax[0].set_title('Signal and peak detection')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Velocity')
    ax[0].legend()
    metrics = [metric for _, _, metric in smooth_decays]
    ax[1].plot(range(len(metrics)), metrics)
    ax[1].axvline(arrive, color='r', linestyle=':', label='Target')
    for i, (peak, _, _) in enumerate(smooth_decays):
        ax[1].axvline(i, color='b', linestyle='-', label='Peak detection' if i == 0 else '')
    ax[1].set_title('Combined Metrics for Detected Decays')
    ax[1].set_xlabel('Decay Number')
    ax[1].set_ylabel('Combined Metric')
    ax[1].legend()

    plt.tight_layout()
    plt.show()
