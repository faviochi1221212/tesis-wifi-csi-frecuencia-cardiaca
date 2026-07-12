"""
12_features_modelos_v3_synced.py  — version corregida y mejorada
Tesis: Monitoreo no invasivo de la frecuencia cardiaca mediante senales Wi-Fi CSI
Universidad de Lima - Ingenieria de Sistemas
Autores: Yadhira Sarmiento Escobar (20214190) y Favio Chavarry Minaya (20214680)

Cambios respecto a la version anterior:
  - Se procesa cada pos_N.csv individualmente (no se concatenan primero),
    lo que permite agregar pos_id a cada registro para filtrar escenarios
    sedentarios en 24_optuna_sedentario_lstm.py.
  - seg_id = pos_id * 100 + gap_id (unico dentro de cada participante).
  - Se agrega sync_ok (bool) y min_dist_watch_s por ventana:
      sync_ok = True si hay una lectura del smartwatch dentro de
      MAX_DIST_BPM_S segundos del centro de la ventana.
  - Se genera diagnostico_sincronizacion.csv por participante.
  - Se genera fig_sync_alineamiento.png mostrando solapamiento CSI vs SW.

TZ_OFFSET = -7200: corregido (antes +10800, causaba 0% solapamiento).
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import signal
from scipy.stats import pearsonr, kurtosis as scipy_kurtosis, skew as scipy_skew
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pywt
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [AVISO] XGBoost no instalado. pip install xgboost")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
CSI_V3     = r"C:\Users\LENOVO\Desktop\csi_filtrado_v3"
CSI_V2     = r"C:\Users\LENOVO\Desktop\csi_filtrado_v2"
CSI_DIR    = CSI_V3 if os.path.isdir(CSI_V3) else CSI_V2
SW_DIR     = r"C:\Users\LENOVO\Desktop\Data_DS1_smartwatch-main\Data"
OUTPUT_DIR = r"C:\Users\LENOVO\Desktop\tesis-wifi-csi-frecuencia-cardiaca-main\resultados"

# ─── PARAMETROS ───────────────────────────────────────────────────────────────
FS             = 7.7
F_LOW          = 0.8
F_HIGH         = 2.17
WIN_SEC        = 30.0
OVERLAP        = 0.75
N_SUBCAR       = 20
N_IPD_SUBCAR   = 10
WAVELET        = 'db4'
RANDOM_STATE   = 42
TZ_OFFSET      = -7200    # CORREGIDO (antes +10800, bug de sync de ~5h)
GAP_SEC        = 1.0
MAX_DIST_BPM_S = WIN_SEC / 2   # 15 s: distancia maxima al SW mas cercano

WIN_SAMPLES    = int(WIN_SEC * FS)    # ~231 muestras
STEP_SAMPLES   = int(WIN_SAMPLES * (1 - OVERLAP))   # ~57 muestras

RF_PARAMS = {
    "n_estimators": 500, "max_depth": 10,
    "min_samples_leaf": 4, "random_state": RANDOM_STATE, "n_jobs": -1,
}
SVR_PARAMS  = {"kernel": "rbf", "C": 10, "epsilon": 0.5, "gamma": "scale"}
XGB_PARAMS  = {
    "n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0,
}

print("=" * 66)
print("  12 - FEATURES + MODELOS v3 SYNCED (corregido con pos_id y diagnostico)")
print("=" * 66)
print(f"  CSI_DIR        : {CSI_DIR}")
print(f"  TZ_OFFSET      : {TZ_OFFSET}s  (corregido, antes +10800)")
print(f"  WIN_SEC        : {WIN_SEC} s")
print(f"  MAX_DIST_BPM_S : {MAX_DIST_BPM_S} s  (umbral sync estricta)")

# ─── FUNCIONES AUXILIARES (identicas a la version anterior) ──────────────────

def seleccionar_subportadoras(datos, n=N_SUBCAR):
    freqs = np.fft.rfftfreq(datos.shape[0], d=1.0 / FS)
    mask  = (freqs >= F_LOW) & (freqs <= F_HIGH)
    energias = []
    for i in range(datos.shape[1]):
        mag = np.abs(np.fft.rfft(datos[:, i]))
        e   = np.sum(mag[mask] ** 2) if np.any(mask) else 0.0
        energias.append(e)
    return np.argsort(energias)[::-1][:n]


def senal_agregada(datos, indices):
    segs = []
    for i in indices:
        s   = datos[:, i].astype(float)
        rng = s.max() - s.min()
        if rng > 1e-10:
            s = (s - s.mean()) / rng
        segs.append(s)
    return np.mean(segs, axis=0)


def bpm_parabolico(freqs, fft_mag, mask):
    freqs_b = freqs[mask]; fft_b = fft_mag[mask]
    if len(fft_b) == 0: return 0.0
    idx = int(np.argmax(fft_b))
    if 0 < idx < len(fft_b) - 1:
        y0, y1, y2 = fft_b[idx-1], fft_b[idx], fft_b[idx+1]
        denom = y0 - 2*y1 + y2
        delta = 0.5*(y0 - y2) / (denom + 1e-10) if abs(denom) > 1e-10 else 0.0
        df    = freqs_b[1] - freqs_b[0] if len(freqs_b) > 1 else 0.0
        return (freqs_b[idx] + delta * df) * 60.0
    return freqs_b[idx] * 60.0


def bpm_harmonico(freqs, fft_mag, f_low, f_high):
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask): return 0.0
    candidates = freqs[mask]
    scores     = np.zeros(len(candidates))
    for k, f0 in enumerate(candidates):
        idx0  = np.argmin(np.abs(freqs - f0))
        score = fft_mag[idx0]
        f2    = 2.0 * f0
        if f2 <= freqs[-1]:
            idx2  = np.argmin(np.abs(freqs - f2))
            score = score + 0.3 * fft_mag[idx2]
        scores[k] = score
    return candidates[int(np.argmax(scores))] * 60.0


def bpm_ac_parabolico(ac, lag_min, lag_max, fs):
    seg = ac[lag_min:lag_max]
    if len(seg) == 0: return 0.0
    peak = int(np.argmax(seg))
    if 0 < peak < len(seg) - 1:
        y0, y1, y2 = seg[peak-1], seg[peak], seg[peak+1]
        denom = y0 - 2*y1 + y2
        delta = 0.5*(y0 - y2) / (denom + 1e-10) if abs(denom) > 1e-10 else 0.0
        lag_exact = (peak + delta) + lag_min
    else:
        lag_exact = float(peak + lag_min)
    return (fs / (lag_exact + 1e-10)) * 60.0


def extraer_features_ventana(senal):
    n = len(senal)
    if n < 10: return None
    media     = float(np.mean(senal))
    std       = float(np.std(senal))
    varianza  = float(np.var(senal))
    energia   = float(np.sum(senal ** 2))
    rms       = float(np.sqrt(np.mean(senal ** 2)))
    pico_pico = float(senal.max() - senal.min())
    kurt      = float(scipy_kurtosis(senal))
    skewn     = float(scipy_skew(senal))

    ventana_hann = np.hanning(n)
    freqs        = np.fft.rfftfreq(n, d=1.0 / FS)
    fft_mag      = np.abs(np.fft.rfft(senal * ventana_hann))
    mask         = (freqs >= F_LOW) & (freqs <= F_HIGH)
    if not np.any(mask): return None

    e_banda  = float(np.sum(fft_mag[mask] ** 2))
    e_total  = float(np.sum(fft_mag ** 2)) + 1e-10
    ratio_e  = e_banda / e_total
    snr_db   = float(10.0 * np.log10(e_banda / (e_total - e_banda + 1e-10)))

    bpm_fft_par = bpm_parabolico(freqs, fft_mag, mask)
    freq_dom    = bpm_fft_par / 60.0
    bpm_est     = bpm_fft_par
    bpm_harm    = bpm_harmonico(freqs, fft_mag, F_LOW, F_HIGH)

    pesos     = fft_mag[mask]
    centroide = float(np.sum(freqs[mask] * pesos) / (np.sum(pesos) + 1e-10))
    umbral    = fft_mag[mask].max() / np.sqrt(2)
    denom_ab  = np.sum(fft_mag[mask] >= umbral) + 1e-10
    ancho     = float(np.sum(freqs[mask][fft_mag[mask] >= umbral]) / denom_ab)

    sc  = senal - np.mean(senal)
    ac  = np.correlate(sc, sc, mode='full')
    ac  = ac[len(ac) // 2:]
    ac  = ac / (ac[0] + 1e-10)
    lag_min = max(1, int(FS / F_HIGH))
    lag_max = min(n - 1, int(FS / F_LOW))
    if lag_max > lag_min and lag_max < len(ac):
        seg      = ac[lag_min:lag_max]
        pico_ac  = float(np.max(seg))
        lag_p    = int(np.argmax(seg)) + lag_min
        bpm_ac   = float((FS / (lag_p + 1e-10)) * 60.0)
        bpm_ac_p = bpm_ac_parabolico(ac, lag_min, lag_max, FS)
    else:
        pico_ac = bpm_ac = bpm_ac_p = 0.0

    try:
        coeffs      = pywt.wavedec(senal, WAVELET, level=3)
        e_wav       = [float(np.sum(c ** 2)) for c in coeffs]
        e_wav_total = sum(e_wav) + 1e-10
        ratio_wav   = [e / e_wav_total for e in e_wav]
        d1          = coeffs[-1]
        d1_norm     = d1 ** 2 / (np.sum(d1 ** 2) + 1e-10)
        entropia    = float(-np.sum(d1_norm * np.log(d1_norm + 1e-10)))
    except Exception:
        ratio_wav = [0.0] * 4
        entropia  = 0.0

    return {
        "media": media, "std": std, "varianza": varianza,
        "energia": energia, "rms": rms, "pico_pico": pico_pico,
        "e_banda": e_banda, "ratio_e": ratio_e,
        "freq_dom": freq_dom, "bpm_est": bpm_est,
        "centroide": centroide, "ancho_banda": ancho,
        "pico_ac": pico_ac, "bpm_ac": bpm_ac,
        "wav_e0": ratio_wav[0], "wav_e1": ratio_wav[1],
        "wav_e2": ratio_wav[2], "wav_e3": ratio_wav[3],
        "entropia": entropia, "bpm_harm": bpm_harm,
        "bpm_ac_parab": bpm_ac_p, "snr_cardiac": snr_db,
        "kurtosis_s": kurt, "skewness_s": skewn,
    }


def extraer_features_ispd(senal_ipd):
    n = len(senal_ipd)
    if n < 10: return {}
    ventana_hann = np.hanning(n)
    freqs        = np.fft.rfftfreq(n, d=1.0 / FS)
    fft_mag      = np.abs(np.fft.rfft(senal_ipd * ventana_hann))
    mask         = (freqs >= F_LOW) & (freqs <= F_HIGH)
    if not np.any(mask): return {}
    bpm_ipd   = bpm_parabolico(freqs, fft_mag, mask)
    bpm_ipd_h = bpm_harmonico(freqs, fft_mag, F_LOW, F_HIGH)
    e_ipd     = float(np.sum(fft_mag[mask] ** 2))
    e_tot_ipd = float(np.sum(fft_mag ** 2)) + 1e-10
    ratio_ipd = e_ipd / e_tot_ipd
    snr_ipd   = float(10.0 * np.log10(e_ipd / (e_tot_ipd - e_ipd + 1e-10)))
    sc  = senal_ipd - np.mean(senal_ipd)
    ac  = np.correlate(sc, sc, mode='full')
    ac  = ac[len(ac) // 2:]
    ac  = ac / (ac[0] + 1e-10)
    lag_min = max(1, int(FS / F_HIGH))
    lag_max = min(n - 1, int(FS / F_LOW))
    bpm_ac_ipd = bpm_ac_parabolico(ac, lag_min, lag_max, FS) if lag_max > lag_min and lag_max < len(ac) else 0.0
    return {
        "ipd_bpm_fft": bpm_ipd, "ipd_bpm_harm": bpm_ipd_h,
        "ipd_bpm_ac": bpm_ac_ipd, "ipd_ratio_e": ratio_ipd, "ipd_snr": snr_ipd,
    }


def metricas(y_true, y_pred, nombre=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        r, _ = pearsonr(y_true, y_pred)
    except Exception:
        r = np.nan
    pct = np.mean(np.abs(y_true - y_pred) < 1.5) * 100
    print(f"\n  {nombre}")
    print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.4f}  %<1.5bpm={pct:.1f}%")
    return {"modelo": nombre, "MAE": mae, "RMSE": rmse, "r": r, "PCT_1.5": pct}


def leer_watch_completo(sw_dir):
    ts_all, hr_all = [], []
    for archivo in os.listdir(sw_dir):
        if not archivo.endswith("HeartRateData.json"):
            continue
        try:
            with open(os.path.join(sw_dir, archivo), "r") as f:
                data = json.load(f)
            hr    = np.array(data.get("heart_rate", []), dtype=float)
            ts_raw = data.get("start_time", [])
            ts_ep  = np.array([
                datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() + TZ_OFFSET
                for t in ts_raw
            ])
            mask = (hr >= 40) & (hr <= 180)
            ts_all.extend(ts_ep[mask].tolist())
            hr_all.extend(hr[mask].tolist())
        except Exception:
            continue
    if len(ts_all) < 2:
        return None, None
    orden = np.argsort(ts_all)
    return np.array(ts_all)[orden], np.array(hr_all)[orden]


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────
# Cambio clave: se procesa cada pos_N.csv individualmente para conservar pos_id.
# La concatenacion de la version anterior perdia la informacion de posicion.

registros     = []
sync_diag     = []
part_sin_sw   = 0
fig_guardada  = False   # solo genera la figura de alineamiento para 1 participante

os.makedirs(OUTPUT_DIR, exist_ok=True)

for part in sorted(os.listdir(CSI_DIR)):
    dir_csi = os.path.join(CSI_DIR, part)
    if not os.path.isdir(dir_csi):
        continue

    dir_sw = os.path.join(SW_DIR, part)
    if not os.path.isdir(dir_sw):
        part_sin_sw += 1
        continue

    ts_watch, hr_watch = leer_watch_completo(dir_sw)
    if ts_watch is None:
        part_sin_sw += 1
        continue

    ts_csi_global     = []   # todos los timestamps CSI de este participante
    n_ventanas_part   = 0
    n_ventanas_ok_part = 0

    archivos_pos = sorted([f for f in os.listdir(dir_csi) if f.endswith(".csv")])

    for archivo in archivos_pos:
        m = re.match(r'^pos_(\d+)\.csv$', archivo)
        if not m:
            continue
        pos_id = int(m.group(1))

        try:
            df_pos = pd.read_csv(os.path.join(dir_csi, archivo))
        except Exception:
            continue

        cols_sub = [c for c in df_pos.columns if c.startswith("sub_")]
        cols_ipd = [c for c in df_pos.columns if c.startswith("ipd_")]
        if not cols_sub or "timestamp" not in df_pos.columns:
            continue

        df_pos = (df_pos.dropna(subset=["timestamp"])
                        .sort_values("timestamp")
                        .reset_index(drop=True))
        ts_csi = df_pos["timestamp"].values.astype(float)
        if len(ts_csi) < WIN_SAMPLES:
            continue

        ts_csi_global.extend(ts_csi.tolist())

        datos     = df_pos[cols_sub].values.astype(float)
        datos_ipd = df_pos[cols_ipd].values.astype(float) if cols_ipd else None

        bpm_interp = np.interp(ts_csi, ts_watch, hr_watch,
                               left=hr_watch[0], right=hr_watch[-1])

        indices   = seleccionar_subportadoras(datos)
        senal     = senal_agregada(datos, indices)
        senal_ipd = None
        if datos_ipd is not None and datos_ipd.shape[1] > 0:
            indices_ipd = seleccionar_subportadoras(datos_ipd, n=N_IPD_SUBCAR)
            senal_ipd   = senal_agregada(datos_ipd, indices_ipd)

        # seg_id unico dentro de cada participante: pos_id*100 + indice de brecha
        cortes    = np.where(np.diff(ts_csi) > GAP_SEC)[0] + 1
        segmentos = np.split(np.arange(len(ts_csi)), cortes)

        for gap_id, seg in enumerate(segmentos):
            t_ini, t_fin = int(seg[0]), int(seg[-1]) + 1
            if t_fin - t_ini < WIN_SAMPLES:
                continue

            seg_id_unico = pos_id * 100 + gap_id

            inicio = t_ini
            while inicio + WIN_SAMPLES <= t_fin:
                idx_centro   = inicio + WIN_SAMPLES // 2
                t_centro     = ts_csi[min(idx_centro, len(ts_csi) - 1)]
                dist_watch   = np.abs(ts_watch - t_centro)
                min_dist     = float(np.min(dist_watch))
                sync_ok      = min_dist <= MAX_DIST_BPM_S

                bpm_v = float(np.mean(bpm_interp[inicio: inicio + WIN_SAMPLES]))
                if not (40 <= bpm_v <= 180):
                    inicio += STEP_SAMPLES
                    continue

                feats = extraer_features_ventana(senal[inicio: inicio + WIN_SAMPLES])
                if feats is None:
                    inicio += STEP_SAMPLES
                    continue

                if senal_ipd is not None:
                    feats.update(extraer_features_ispd(senal_ipd[inicio: inicio + WIN_SAMPLES]))

                feats["pos_id"]          = pos_id
                feats["seg_id"]          = seg_id_unico
                feats["win_start"]       = inicio
                feats["bpm_watch"]       = bpm_v
                feats["participante"]    = part
                feats["sync_ok"]         = sync_ok
                feats["min_dist_watch_s"] = min_dist
                registros.append(feats)
                n_ventanas_part += 1
                if sync_ok:
                    n_ventanas_ok_part += 1

                inicio += STEP_SAMPLES

    # ── Diagnostico de sincronizacion por participante ────────────────────────
    if ts_csi_global:
        ts_arr    = np.array(ts_csi_global)
        csi_ini   = float(ts_arr.min())
        csi_fin   = float(ts_arr.max())
        sw_ini    = float(ts_watch.min())
        sw_fin    = float(ts_watch.max())
        overlap_s = max(0.0, min(csi_fin, sw_fin) - max(csi_ini, sw_ini))
        watch_dt  = float(np.mean(np.diff(ts_watch))) if len(ts_watch) > 1 else np.nan

        sync_diag.append({
            "participante":       part,
            "csi_start_epoch":    csi_ini,
            "csi_end_epoch":      csi_fin,
            "csi_dur_s":          csi_fin - csi_ini,
            "watch_start_epoch":  sw_ini,
            "watch_end_epoch":    sw_fin,
            "watch_dur_s":        sw_fin - sw_ini,
            "overlap_s":          overlap_s,
            "watch_delta_mean_s": watch_dt,
            "n_ventanas_total":   n_ventanas_part,
            "n_ventanas_sync_ok": n_ventanas_ok_part,
        })

        # Genera la figura de alineamiento temporal para el primer participante
        # con solapamiento real (overlap_s > 0)
        if not fig_guardada and overlap_s > 0:
            fig, ax = plt.subplots(figsize=(13, 3))
            # Franja CSI
            ax.broken_barh([(csi_ini, csi_fin - csi_ini)], (1.5, 0.8),
                           facecolors="#2196F3", alpha=0.7, label="CSI")
            # Franja Smartwatch
            ax.broken_barh([(sw_ini, sw_fin - sw_ini)], (0.5, 0.8),
                           facecolors="#4CAF50", alpha=0.7, label="Smartwatch")
            # Solapamiento
            ol_ini = max(csi_ini, sw_ini)
            ol_fin = min(csi_fin, sw_fin)
            if ol_fin > ol_ini:
                ax.broken_barh([(ol_ini, ol_fin - ol_ini)], (0.5, 1.8),
                               facecolors="#FF9800", alpha=0.4, label=f"Solapamiento ({overlap_s:.0f} s)")
            ax.set_yticks([0.9, 1.9])
            ax.set_yticklabels(["Smartwatch", "CSI"])
            ax.set_xlabel("Tiempo (epoch s)")
            ax.set_title(f"Alineamiento temporal CSI vs Smartwatch — Participante: {part}\n"
                         f"TZ_OFFSET={TZ_OFFSET}s | Solapamiento={overlap_s:.0f}s | "
                         f"Ventanas sync_ok={n_ventanas_ok_part}/{n_ventanas_part}",
                         fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            fig_path = os.path.join(OUTPUT_DIR, "fig_sync_alineamiento.png")
            plt.savefig(fig_path, dpi=130, bbox_inches="tight")
            plt.close()
            print(f"  [FIGURA] fig_sync_alineamiento.png guardada ({part})")
            fig_guardada = True

print(f"\n  Registros validos   : {len(registros)}")
print(f"  Participantes sin SW: {part_sin_sw}")

# ─── GUARDAR DIAGNOSTICO DE SINCRONIZACION ────────────────────────────────────
if sync_diag:
    df_diag = pd.DataFrame(sync_diag)
    path_diag = os.path.join(OUTPUT_DIR, "diagnostico_sincronizacion.csv")
    df_diag.to_csv(path_diag, index=False)
    n_ok  = int((df_diag["overlap_s"] > 0).sum())
    n_tot = len(df_diag)
    print(f"  Sync OK (overlap>0) : {n_ok}/{n_tot} participantes")
    print(f"  diagnostico_sincronizacion.csv -> {OUTPUT_DIR}")

if not registros:
    print("\n  [ERROR] No se generaron registros. Verifica las rutas y los datos.")
    raise SystemExit(1)

df_dataset = pd.DataFrame(registros)
print(f"  BPM promedio        : {df_dataset['bpm_watch'].mean():.1f}")
print(f"  BPM std             : {df_dataset['bpm_watch'].std():.1f}")
print(f"  Posiciones detectadas: {sorted(df_dataset['pos_id'].unique())}")
sync_pct = df_dataset["sync_ok"].mean() * 100 if "sync_ok" in df_dataset.columns else 0
print(f"  Ventanas con sync_ok: {sync_pct:.1f}%")

# ─── SPLIT POR PARTICIPANTE (70/10/20) ────────────────────────────────────────
participantes = sorted(df_dataset["participante"].unique())
n = len(participantes)
np.random.seed(RANDOM_STATE)
idx_perm = np.random.permutation(n)
participantes = [participantes[i] for i in idx_perm]
n_train = int(n * 0.70)
n_val   = int(n * 0.20)
p_train = participantes[:n_train]
p_val   = participantes[n_train: n_train + n_val]
p_test  = participantes[n_train + n_val:]

df_train = df_dataset[df_dataset["participante"].isin(p_train)]
df_val   = df_dataset[df_dataset["participante"].isin(p_val)]
df_test  = df_dataset[df_dataset["participante"].isin(p_test)]

print(f"\n  Train : {len(p_train)} partic., {len(df_train)} registros")
print(f"  Val   : {len(p_val)} partic.,  {len(df_val)} registros")
print(f"  Test  : {len(p_test)} partic.,  {len(df_test)} registros")

# ─── FEATURES ─────────────────────────────────────────────────────────────────
FEAT_BASE = [
    "media", "std", "varianza", "energia", "rms", "pico_pico",
    "e_banda", "ratio_e", "freq_dom", "bpm_est",
    "centroide", "ancho_banda", "pico_ac", "bpm_ac",
    "wav_e0", "wav_e1", "wav_e2", "wav_e3", "entropia",
    "bpm_harm", "bpm_ac_parab", "snr_cardiac", "kurtosis_s", "skewness_s",
]
FEAT_IPD = ["ipd_bpm_fft", "ipd_bpm_harm", "ipd_bpm_ac", "ipd_ratio_e", "ipd_snr"]
FEAT_COLS = FEAT_BASE + [c for c in FEAT_IPD if c in df_dataset.columns]
META_COLS = {"seg_id", "win_start", "bpm_watch", "participante",
             "pos_id", "sync_ok", "min_dist_watch_s"}
print(f"\n  Features totales  : {len(FEAT_COLS)}")
print(f"  ISPD disponible   : {'SI' if any(c in df_dataset.columns for c in FEAT_IPD) else 'NO'}")

X_train = df_train[FEAT_COLS].fillna(0).values
y_train = df_train["bpm_watch"].values
X_val   = df_val[FEAT_COLS].fillna(0).values
y_val   = df_val["bpm_watch"].values
X_test  = df_test[FEAT_COLS].fillna(0).values
y_test  = df_test["bpm_watch"].values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ─── ENTRENAMIENTO ────────────────────────────────────────────────────────────
print("\n  Entrenando Random Forest...")
rf = RandomForestRegressor(**RF_PARAMS)
rf.fit(X_train, y_train)

print("  Entrenando SVR...")
svr = SVR(**SVR_PARAMS)
svr.fit(X_train, y_train)

if HAS_XGB:
    print("  Entrenando XGBoost...")
    xgb_model = XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# ─── EVALUACION ───────────────────────────────────────────────────────────────
resultados = []
print("\n>>> VALIDACION")
resultados.append(metricas(y_val, rf.predict(X_val),   "RF v3-synced   - Val"))
resultados.append(metricas(y_val, svr.predict(X_val),  "SVR v3-synced  - Val"))
if HAS_XGB:
    pred_xgb_val = xgb_model.predict(X_val)
    resultados.append(metricas(y_val, pred_xgb_val, "XGB v3-synced  - Val"))
    resultados.append(metricas(y_val, (rf.predict(X_val) + pred_xgb_val) / 2, "Ens RF+XGB - Val"))

print("\n>>> TEST")
pred_rf_test  = rf.predict(X_test)
pred_svr_test = svr.predict(X_test)
resultados.append(metricas(y_test, pred_rf_test,  "RF v3-synced   - Test"))
resultados.append(metricas(y_test, pred_svr_test, "SVR v3-synced  - Test"))
if HAS_XGB:
    pred_xgb_test = xgb_model.predict(X_test)
    resultados.append(metricas(y_test, pred_xgb_test, "XGB v3-synced  - Test"))
    resultados.append(metricas(y_test, (pred_rf_test + pred_xgb_test) / 2, "Ens RF+XGB - Test"))

# ─── IMPORTANCIA DE FEATURES ──────────────────────────────────────────────────
importancias = pd.DataFrame({
    "feature":     FEAT_COLS,
    "importancia": rf.feature_importances_
}).sort_values("importancia", ascending=False)
print("\n  Top-10 features mas importantes (RF):")
print(importancias.head(10).to_string(index=False))

# ─── GUARDAR ──────────────────────────────────────────────────────────────────
pd.DataFrame(resultados).to_csv(
    os.path.join(OUTPUT_DIR, "resultados_modelos_v3_synced.csv"), index=False)

df_dataset.to_csv(
    os.path.join(OUTPUT_DIR, "dataset_completo_v3_synced.csv"), index=False)

df_test_pred = df_test.copy()
df_test_pred["pred_rf_v3"]  = pred_rf_test
df_test_pred["pred_svr_v3"] = pred_svr_test
if HAS_XGB:
    df_test_pred["pred_xgb_v3"] = pred_xgb_test
df_test_pred.to_csv(
    os.path.join(OUTPUT_DIR, "predicciones_test_v3_synced.csv"), index=False)

importancias.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance_v3_synced.csv"), index=False)

print(f"\n  dataset_completo_v3_synced.csv  ({len(df_dataset)} registros, "
      f"{len(FEAT_COLS)} features + pos_id + sync_ok)")
print(f"  diagnostico_sincronizacion.csv")
print(f"  feature_importance_v3_synced.csv")
print(f"  resultados_modelos_v3_synced.csv")
print("\nScript 12 completado.")
print("  Siguiente paso: python 23_seleccion_mi.py")
