"""
Ventaneo + etiqueta del reloj + sync_ok.

features_capture es copia literal de experimentos_fase_C/E1_rf.py L86-111 (logica de 12: ventanas de 231 muestras,
paso 57, segmentos por huecos > 1 s, seg_id = pos*100 + gap_id, etiqueta = media de np.interp(ts_csi, ts_reloj,
hr_reloj) sobre la ventana, filtro 40-180 BPM, sync_ok = lectura del reloj a <= 15 s del centro).
Unico cambio: parametro `arms` (por defecto los tres brazos de E1, en el mismo orden) para poder calcular solo E1a.
"""
import numpy as np

from .config import FS, GAP_SEC, HR_MAX_BPM, HR_MIN_BPM, MAX_DIST_WATCH_S, STEP_SAMPLES, WINDOW_SAMPLES
from .features import extraer_features_ventana, senal_agregada
from .preprocessing import disenar_butterworth, pipeline_amplitud
from .subcarriers import ARMS, arm_matrix, seleccionar_subportadoras, selected_columns_234

WIN, STEP = WINDOW_SAMPLES, STEP_SAMPLES


def segment_starts(ts, win=WIN, step=STEP, gap_sec=GAP_SEC):
    """(gap_id, ini) de cada ventana: los huecos > gap_sec parten la captura; ventanas completas dentro de cada tramo."""
    cortes = np.where(np.diff(ts) > gap_sec)[0] + 1
    out = []
    for gap_id, seg in enumerate(np.split(np.arange(len(ts)), cortes)):
        a0, b0 = int(seg[0]), int(seg[-1]) + 1
        if b0 - a0 < win:
            continue
        ini = a0
        while ini + win <= b0:
            out.append((gap_id, ini)); ini += step
    return out


def features_capture(part, pos, ts, H, tsw, hrw, arms=ARMS):
    sos = disenar_butterworth(FS)
    interp = np.interp(ts, tsw, hrw, left=hrw[0], right=hrw[-1])
    cortes = np.where(np.diff(ts) > GAP_SEC)[0] + 1
    rows, sel = [], {}
    for arm in arms:
        A = arm_matrix(H, arm)
        datos = np.column_stack([pipeline_amplitud(A[:, j], sos) for j in range(A.shape[1])])
        idx, _ = seleccionar_subportadoras(datos, FS)
        sel[arm] = selected_columns_234(arm, idx).tolist()
        senal = senal_agregada(datos, idx)
        for gap_id, seg in enumerate(np.split(np.arange(len(ts)), cortes)):
            a0, b0 = int(seg[0]), int(seg[-1]) + 1
            if b0 - a0 < WIN:
                continue
            ini = a0
            while ini + WIN <= b0:
                tc = ts[min(ini + WIN // 2, len(ts) - 1)]
                bpm_v = float(np.mean(interp[ini:ini + WIN]))
                if HR_MIN_BPM <= bpm_v <= HR_MAX_BPM:
                    rows.append({"arm": arm, "participante": part, "pos_id": pos, "seg_id": pos * 100 + gap_id, "win_start": ini,
                                 "bpm_watch": bpm_v, "sync_ok": bool(np.min(np.abs(tsw - tc)) <= MAX_DIST_WATCH_S),
                                 **extraer_features_ventana(senal[ini:ini + WIN], FS)})
                ini += STEP
    return rows, sel
