"""
Preprocesamiento de amplitud (pipeline publicado 04, usado por E1a) y variante causal por ventana (E5).

Copias literales:
  hampel_filter, disenar_butterworth, pipeline_amplitud, pipeline_fase_ispd
      <- auditoria_fase_A/A1_auditoria_top20_subportadoras.py L38-80 (copias de 04_preprocesamiento.py L78-158)
  window_process
      <- experimentos_fase_C/E5_causal.py L29-37
E1a: detrend lineal -> Hampel (k=10, t=3) -> Butterworth pasa-banda orden 5, 0.8-2.17 Hz, sosfiltfilt -> min-max [-1, 1].
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from .config import BUTTER_HIGH_HZ, BUTTER_LOW_HZ, BUTTER_ORDER, HAMPEL_K, HAMPEL_THRESHOLD


def hampel_filter(x, k=HAMPEL_K, t0=HAMPEL_THRESHOLD):
    x = np.asarray(x, dtype=float); n = len(x); y = x.copy()
    if n <= 2 * k:
        return y
    pad = np.pad(x, k, mode='edge')
    windows = sliding_window_view(pad, 2 * k + 1)
    med = np.median(windows, axis=1)
    mad = 1.4826 * np.median(np.abs(windows - med[:, None]), axis=1)
    mask = (mad > 0) & (np.abs(x - med) > t0 * mad)
    y[mask] = med[mask]
    return y


def disenar_butterworth(fs):
    return signal.butter(BUTTER_ORDER, [BUTTER_LOW_HZ, BUTTER_HIGH_HZ], btype='bandpass', fs=fs, output='sos')


def pipeline_amplitud(amp_col, sos_bp, minmax=True):
    x = amp_col.astype(float)
    x = signal.detrend(x, type='linear')
    x = hampel_filter(x)
    try:
        x = signal.sosfiltfilt(sos_bp, x)
    except Exception:
        pass
    if minmax:
        rng = x.max() - x.min()
        if rng > 1e-10:
            x = 2.0 * (x - x.min()) / rng - 1.0
    return x.astype(np.float32)


def pipeline_fase_ispd(fase_k, fase_k1, sos_bp):
    """Fase ISPD de 04 (solo para reproducir experimentos historicos; E1a no la usa)."""
    diff = np.unwrap(fase_k - fase_k1)
    x = signal.detrend(diff, type='linear'); x = hampel_filter(x)
    try:
        x = signal.sosfiltfilt(sos_bp, x)
    except Exception:
        pass
    rng = x.max() - x.min()
    if rng > 1e-10:
        x = 2.0 * (x - x.min()) / rng - 1.0
    return x.astype(np.float32)


def process_matrix(A, fs, minmax=True):
    """Aplica pipeline_amplitud columna a columna (como E1_rf.features_capture L93)."""
    sos = disenar_butterworth(fs)
    return np.column_stack([pipeline_amplitud(A[:, j], sos, minmax=minmax) for j in range(A.shape[1])])


def window_process(Aw, sos):
    """Variante causal de E5: todo el procesamiento depende solo de la ventana (E5_causal.py L29-37)."""
    out = np.empty_like(Aw, dtype=float)
    for j in range(Aw.shape[1]):
        x = signal.detrend(Aw[:, j].astype(float), type="linear")
        x = hampel_filter(x)
        x = signal.sosfiltfilt(sos, x)
        q75, q25 = np.percentile(x, [75, 25]); iqr = q75 - q25
        out[:, j] = (x - np.median(x)) / (iqr if iqr > 1e-12 else 1.0)
    return out
