"""
04_preprocesamiento.py
Tesis: Monitoreo no invasivo de la frecuencia cardiaca mediante senales Wi-Fi CSI
Universidad de Lima - Ingenieria de Sistemas
Autores: Yadhira Sarmiento Escobar (20214190) y Favio Chavarry Minaya (20214680)

Pipeline de filtrado mejorado (v3) — cambios respecto a v2:
  v2: RunningMean(5) -> Hampel(10, 3σ) -> Butterworth BP(0.8-2.17, ord=3)
  v3: Detrend -> Hampel(10, 3σ) -> Butterworth BP(0.8-2.17, ord=5, sosfiltfilt)
       + ISPD (diferencia de fase inter-subportadora, nueva columna "ipd_XXX")

Por que mejora:
  - Orden 5 vs 3: rolloff mas nitido (38 dB/octava vs 18), menos ruido en banda
  - sosfiltfilt: fase cero (no desplaza el pico cardiaco en el tiempo)
  - Detrend lineal: elimina deriva lenta por temperatura/movimiento ambiente
  - ISPD: cancela ruido comun a todas las subportadoras (CFO, SFO), mejora SNR

Entrada : csi_csv/  (numeros complejos a+bj en columnas sub_XXX + timestamp)
Salida  : csi_filtrado_v3/  (amplitud filtrada sub_XXX + fase ISPD ipd_XXX)

Si csi_csv/ no existe, aplica filtrado adicional sobre csi_filtrado_v2/ (fallback).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path

# ─── RUTAS ────────────────────────────────────────────────────────────────────
INPUT_AMP_DIR    = r"C:\Users\LENOVO\Desktop\csi_csv"          # complejo (preferido)
INPUT_FILT_DIR   = r"C:\Users\LENOVO\Desktop\csi_filtrado_v2"  # fallback
OUTPUT_DIR       = r"C:\Users\LENOVO\Desktop\csi_filtrado_v3"

# ─── PARAMETROS ───────────────────────────────────────────────────────────────
FS          = 7.7
F_LOW       = 0.8
F_HIGH      = 2.17
BUTTER_ORD  = 5       # v3: orden 5 (vs 3 en v2)
HAMPEL_K    = 10      # mismo que v2
HAMPEL_T    = 3.0     # mismo que v2 (3 MAD)
MIN_SEG_S   = 10      # descartar archivos con menos de 10 s

# ─── FILTRO BUTTERWORTH (calculado una sola vez) ───────────────────────────────
# 'sos' (second-order sections) en vez de coeficientes (b,a) directos: con
# orden 5 los coeficientes (b,a) de un filtro IIR pueden volverse numericamente
# inestables (errores de redondeo se amplifican); sos factoriza el filtro en
# secciones de 2do orden encadenadas, mucho mas estable numericamente.
sos_bp = signal.butter(BUTTER_ORD, [F_LOW, F_HIGH],
                       btype='bandpass', fs=FS, output='sos')


def hampel_filter(x, k=HAMPEL_K, t0=HAMPEL_T):
    """
    Reemplaza spikes por la mediana local (Hampel identifier) — vectorizado.
}
    Para cada muestra se mira una ventana de 2k+1 vecinos (k antes, k despues) y
    se calcula su mediana y su MAD (Median Absolute Deviation). Si la muestra se
    aleja de la mediana local mas de t0 desviaciones (en unidades de MAD escalado
    a "equivalente std" con el factor 1.4826, valido para datos gaussianos), se
    considera un outlier/spike y se reemplaza por la mediana local. Se usa
    mediana/MAD en vez de media/std porque son robustas a outliers — la propia
    media y std se distorsionarian con el spike que se quiere detectar.

    t0=3.0 (3 "sigmas" equivalentes) es mas tolerante que un t0 mas chico:
    deja pasar mas variacion real de la señal y solo recorta los picos mas
    extremos, en vez de aplanar tambien la dinamica cardiaca legitima.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = x.copy()
    if n <= 2 * k:
        return y
    pad = np.pad(x, k, mode='edge')
    windows = sliding_window_view(pad, 2 * k + 1)   # (n, 2k+1)
    med = np.median(windows, axis=1)
    mad = 1.4826 * np.median(np.abs(windows - med[:, None]), axis=1)
    mask = (mad > 0) & (np.abs(x - med) > t0 * mad)
    y[mask] = med[mask]
    return y


def pipeline_amplitud(amp_col):
    """
    Pipeline completo para una columna de amplitud CSI.
    Detrend -> Hampel -> Butterworth ord-5 sosfiltfilt -> normalizar.
    """
    x = amp_col.astype(float)
    x = signal.detrend(x, type='linear')
    x = hampel_filter(x)
    try:
        x = signal.sosfiltfilt(sos_bp, x)   # fase cero (forward-backward)
    except Exception:
        pass
    rng = x.max() - x.min()
    if rng > 1e-10:
        x = 2.0 * (x - x.min()) / rng - 1.0
    return x.astype(np.float32)


def pipeline_fase_ispd(fase_k, fase_k1):
    """
    Inter-Subcarrier Phase Difference (ISPD) entre subportadoras k y k+1.
    Cancela CFO y ruido comun; preserva variaciones de canal por cuerpo.
    ISPD = angulo( H_k * conj(H_{k+1}) ) = fase_k - fase_{k+1} (mod 2pi)
    """
    diff = fase_k - fase_k1
    diff = np.unwrap(diff)               # evitar saltos de 2pi
    x = signal.detrend(diff, type='linear')
    x = hampel_filter(x)
    try:
        x = signal.sosfiltfilt(sos_bp, x)
    except Exception:
        pass
    rng = x.max() - x.min()
    if rng > 1e-10:
        x = 2.0 * (x - x.min()) / rng - 1.0
    return x.astype(np.float32)


def parsear_complejo(s):
    try:
        return complex(str(s).strip().replace(" ", "").lstrip("(").rstrip(")"))
    except Exception:
        return complex(0, 0)


def procesar_desde_csv_complejo(path_in):
    """
    Lee CSV con numeros complejos, devuelve (amp_filtrada, ispd_filtrada, timestamps).
    amp_filtrada : array (N, n_sub)
    ispd_filtrada: array (N, n_sub-1)
    """
    df = pd.read_csv(path_in)
    sub_cols = [c for c in df.columns if c.startswith("sub_")]
    if not sub_cols or "timestamp" not in df.columns:
        return None, None, None

    # Eliminar filas con valores nulos o timestamp nulo
    df = df.dropna(subset=["timestamp"] + sub_cols)
    df = df.reset_index(drop=True)

    N = len(df)
    n_sub = len(sub_cols)
    amp   = np.zeros((N, n_sub), dtype=np.float32)
    fase  = np.zeros((N, n_sub), dtype=np.float32)

    for j, col in enumerate(sub_cols):
        vals = df[col].apply(parsear_complejo).values
        amp[:, j]  = np.abs(vals).astype(np.float32)
        fase[:, j] = np.angle(vals).astype(np.float32)

    # Eliminar filas donde la amplitud es NaN o infinita tras parsear
    mask_valido = np.all(np.isfinite(amp), axis=1)
    amp  = amp[mask_valido]
    fase = fase[mask_valido]
    timestamps_validos = df["timestamp"].values[mask_valido]

    N = amp.shape[0]
    if N < int(FS * MIN_SEG_S):
        return None, None, None

    amp_filt  = np.zeros_like(amp)
    for j in range(n_sub):
        amp_filt[:, j] = pipeline_amplitud(amp[:, j])

    ispd_filt = np.zeros((N, n_sub - 1), dtype=np.float32)
    for j in range(n_sub - 1):
        ispd_filt[:, j] = pipeline_fase_ispd(fase[:, j], fase[:, j + 1])

    return amp_filt, ispd_filt, timestamps_validos


def procesar_desde_csv_real(path_in):
    """
    Lee CSV con amplitudes ya filtradas (v2) y aplica refinamiento adicional.
    No produce ISPD (no hay fase disponible).
    """
    df = pd.read_csv(path_in)
    sub_cols = [c for c in df.columns if c.startswith("sub_")]
    if not sub_cols or "timestamp" not in df.columns:
        return None, None, None

    # Eliminar filas con valores nulos
    df = df.dropna(subset=["timestamp"] + sub_cols)
    df = df.reset_index(drop=True)

    amp = df[sub_cols].values.astype(np.float32)
    # Eliminar filas con NaN o infinitos tras conversion
    mask_valido = np.all(np.isfinite(amp), axis=1)
    amp = amp[mask_valido]
    ts  = df["timestamp"].values[mask_valido]

    N   = amp.shape[0]
    if N < int(FS * MIN_SEG_S):
        return None, None, None

    amp_filt = np.zeros_like(amp)
    for j in range(amp.shape[1]):
        x = signal.detrend(amp[:, j].astype(float), type='linear')
        x = hampel_filter(x)
        try:
            x = signal.sosfiltfilt(sos_bp, x)
        except Exception:
            pass
        rng = x.max() - x.min()
        if rng > 1e-10:
            x = 2.0 * (x - x.min()) / rng - 1.0
        amp_filt[:, j] = x.astype(np.float32)

    return amp_filt, None, ts


# ─── SELECCION DE FUENTE ──────────────────────────────────────────────────────
usa_complejo = os.path.isdir(INPUT_AMP_DIR)
INPUT_DIR    = INPUT_AMP_DIR if usa_complejo else INPUT_FILT_DIR

print("=" * 60)
print("  04 - PREPROCESAMIENTO v3 (Pipeline Mejorado)")
print("=" * 60)
print(f"  Modo           : {'Complejo (ISPD disponible)' if usa_complejo else 'Real (solo amplitud)'}")
print(f"  Entrada        : {INPUT_DIR}")
print(f"  Salida         : {OUTPUT_DIR}")
print(f"  Butterworth    : orden {BUTTER_ORD} (v2=3)")
print(f"  Hampel         : k={HAMPEL_K}, t={HAMPEL_T}")
print(f"  Banda          : {F_LOW}–{F_HIGH} Hz")
if not usa_complejo:
    print(f"  [AVISO] csi_csv/ no encontrado — usando csi_filtrado_v2/ como entrada")
    print(f"          ISPD no disponible con este modo")

os.makedirs(OUTPUT_DIR, exist_ok=True)

procesados  = 0
con_ispd    = 0
errores     = 0

participantes = sorted(os.listdir(INPUT_DIR))
total = len(participantes)

for idx_p, participante in enumerate(participantes):
    dir_in  = os.path.join(INPUT_DIR,  participante)
    dir_out = os.path.join(OUTPUT_DIR, participante)
    if not os.path.isdir(dir_in):
        continue
    os.makedirs(dir_out, exist_ok=True)

    archivos = sorted([f for f in os.listdir(dir_in) if f.endswith(".csv")])

    for archivo in archivos:
        path_in  = os.path.join(dir_in,  archivo)
        base     = archivo.replace(".csv", "")

        try:
            if usa_complejo:
                amp_filt, ispd_filt, timestamps = procesar_desde_csv_complejo(path_in)
            else:
                amp_filt, ispd_filt, timestamps = procesar_desde_csv_real(path_in)

            if amp_filt is None:
                errores += 1
                continue

            n_sub = amp_filt.shape[1]
            sub_cols  = [f"sub_{j:03d}" for j in range(n_sub)]

            df_out = pd.DataFrame(amp_filt, columns=sub_cols)
            df_out.insert(0, "timestamp", timestamps)

            if ispd_filt is not None:
                n_ipd = ispd_filt.shape[1]
                ipd_cols = [f"ipd_{j:03d}" for j in range(n_ipd)]
                df_ipd = pd.DataFrame(ispd_filt, columns=ipd_cols)
                df_out = pd.concat([df_out, df_ipd], axis=1)
                con_ispd += 1

            path_out = os.path.join(dir_out, archivo)
            df_out.to_csv(path_out, index=False)
            procesados += 1

        except Exception as e:
            errores += 1
            print(f"  [ERR] {participante}/{archivo}: {e}")

    if (idx_p + 1) % 20 == 0 or (idx_p + 1) == total:
        print(f"  Progreso: {idx_p + 1}/{total} participantes | "
              f"OK={procesados} | ERR={errores}")

print(f"\n  Archivos procesados : {procesados}")
print(f"  Con ISPD            : {con_ispd}")
print(f"  Errores             : {errores}")
print(f"\n  Datos en : {OUTPUT_DIR}")
print("\n  Siguiente paso:")
print("    python 07_features_y_modelos_v3.py")
print("\nScript 04 v3 completado.\n")
