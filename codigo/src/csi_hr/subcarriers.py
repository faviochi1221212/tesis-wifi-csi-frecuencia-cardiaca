"""
Subportadoras: mapeo columna -> buffer, bloque activo (54) y seleccion top-20 por energia en banda.

Copias literales:
  seleccionar_subportadoras <- A1_auditoria_top20_subportadoras.py L84-91 (copia de 12 L116-124)
  arm_matrix                <- experimentos_fase_C/E1_rf.py L77-83
"""
import numpy as np

from .config import ACTIVE_SUBCARRIERS, BUTTER_HIGH_HZ, BUTTER_LOW_HZ, COL2BUF, N_TOP_SUBCARRIERS

ARMS = ("BASE", "E1b", "E1a")


def seleccionar_subportadoras(datos, fs, n=N_TOP_SUBCARRIERS):
    freqs = np.fft.rfftfreq(datos.shape[0], d=1.0 / fs)
    mask = (freqs >= BUTTER_LOW_HZ) & (freqs <= BUTTER_HIGH_HZ)
    energias = []
    for i in range(datos.shape[1]):
        mag = np.abs(np.fft.rfft(datos[:, i]))
        energias.append(np.sum(mag[mask] ** 2) if np.any(mask) else 0.0)
    return np.argsort(energias)[::-1][:n], np.array(energias)


def arm_matrix(H, arm):
    """H: (n, 256) complejo del buffer.
    BASE: 234 columnas, |H|/||H_256|| (escalado CSIKit scaled=True) -> reproduce el dataset publicado.
    E1b : 54 activas, |H|/||H_256||.
    E1a : 54 activas, |H| crudo (sin escalado)  -> brazo del MAE historico 8.400."""
    scaled = H / np.linalg.norm(H, axis=1, keepdims=True)
    if arm == "BASE":
        return np.abs(scaled[:, COL2BUF])
    if arm == "E1b":
        return np.abs(scaled[:, COL2BUF[ACTIVE_SUBCARRIERS]])
    if arm == "E1a":
        return np.abs(H[:, COL2BUF[ACTIVE_SUBCARRIERS]])
    raise ValueError(arm)


def selected_columns_234(arm, idx):
    """Indices de columna (0..233) de las subportadoras elegidas (E1_rf L95)."""
    return np.arange(234)[idx] if arm == "BASE" else ACTIVE_SUBCARRIERS[idx]
