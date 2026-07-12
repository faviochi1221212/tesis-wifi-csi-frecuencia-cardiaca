"""
25_corridas_repetidas.py
Tesis: Monitoreo no invasivo de la frecuencia cardiaca mediante senales Wi-Fi CSI
Universidad de Lima - Ingenieria de Sistemas
Autores: Yadhira Sarmiento Escobar (20214190) y Favio Chavarry Minaya (20214680)

Pedido del asesor (audio de asesoria 2026-07): en vez de reportar el MAE de
un unico split, repetir el entrenamiento/evaluacion del modelo final
N_CORRIDAS veces con distintos splits aleatorios y reportar la media +/-
intervalo de confianza de MAE, RMSE y MAPE para RF y SVR -- igual en
espiritu a como reportan sus metricas los papers de referencia (media +/-
IC sobre multiples muestreos aleatorios, en vez de un numero puntual).

El split en cada corrida es POR GRABACION dentro de cada persona (igual que
19_combinado_residual.py, solo cambia la semilla aleatoria) -- NO por
persona completa. Se preserva asi la calibracion por media_persona, que es
la base del metodo que da el MAE bajo; excluir personas enteras del
entrenamiento roto ese esquema de calibracion (no habria como calcular su
media_persona) y no es lo que se implementa aqui.

Los hiperparametros de RF y SVR se mantienen FIJOS (ya tuneados por
GridSearchCV en 19_combinado_residual.py) -- solo se repiten el split y el
entrenamiento, no la busqueda de hiperparametros, para no mezclar la
varianza de resampling con la varianza de tuneo de hiperparametros.

Ademas de "todas las posiciones" (Escenario A, el modelo final), se repite el
mismo experimento sobre "sedentario + sincronizacion estricta" (Escenario C
de 24_optuna_sedentario_lstm.py, que con un unico split daba MAE=7.63 --
mejor que el modelo general). El objetivo es verificar si esa mejora es
real y se sostiene bajo remuestreo, o si tambien fue un split con suerte
(como paso con el 8.78 del modelo general, cuya media real resulto 9.51).

Prerequisito: haber ejecutado 12_features_modelos_v3_synced.py y
23_seleccion_mi.py (usa el mismo dataset_completo_v3_synced.csv y el mismo
top-10 de features de permutation_importance_features.csv).

Salidas (en resultados/):
  resultados_33_corridas.csv  - una fila por corrida x escenario x modelo
  resumen_33_corridas.csv     - media, std e IC95% por escenario x modelo x metrica
  fig_33_corridas_barras.png  - barras con error bars, Escenario A vs C (estilo figura del asesor)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR  = r"C:\Users\LENOVO\Desktop\tesis-wifi-csi-frecuencia-cardiaca-main\resultados"
DATASET_CSV = os.path.join(OUTPUT_DIR, "dataset_completo_v3_synced.csv")
PI_CSV      = os.path.join(OUTPUT_DIR, "permutation_importance_features.csv")

N_TOP_FEAT  = 10
N_CORRIDAS  = 33
W_SUAVIZADO = 9
TEST_FRAC   = 0.20

RF_BEST = {
    "max_depth": 6, "max_features": "sqrt", "min_samples_leaf": 29,
    "n_estimators": 387, "random_state": 42, "n_jobs": -1,
}
# Mejores hiperparametros de SVR ya encontrados por GridSearchCV en
# 19_combinado_residual.py -- fijos aqui (ver docstring).
SVR_BEST = {"kernel": "rbf", "C": 1, "epsilon": 0.5, "gamma": "scale"}

POSICIONES_SEDENTARIAS = {1, 2, 3, 4, 5, 6, 11, 12, 13, 14}

C_RF, C_SVR = "#2196F3", "#FF5722"   # mismos colores que 08_figuras_tesis.py

print("=" * 66)
print(f"  25 - {N_CORRIDAS} CORRIDAS REPETIDAS (RF vs SVR, media +/- IC95%)")
print("=" * 66)

df = pd.read_csv(DATASET_CSV)
pi_rank = pd.read_csv(PI_CSV)
feat_cols = pi_rank["feature"].tolist()[:N_TOP_FEAT]
print(f"  Features (top-{N_TOP_FEAT}): {feat_cols}\n")

df_A = df.copy()
df_C = df[df["pos_id"].isin(POSICIONES_SEDENTARIAS) & df["sync_ok"].astype(bool)].copy()
print(f"  Escenario A (todas las posiciones): {len(df_A)} registros")
print(f"  Escenario C (sedentario + sync estricta): {len(df_C)} registros\n")
ESCENARIOS = {"A: Todas las posiciones": df_A, "C: Sedentario + sync estricta": df_C}


def split_por_grabacion(df, seed, test_frac=TEST_FRAC):
    """Split por grabacion (seg_id) dentro de cada participante, sin fuga."""
    rng = np.random.RandomState(seed)
    idx_tr, idx_te = [], []
    for part in sorted(df["participante"].unique()):
        dfp = df[df["participante"] == part]
        segs = sorted(dfp["seg_id"].unique())
        n = len(segs)
        if n < 2:
            idx_tr.extend(dfp.index.tolist())
            continue
        perm = rng.permutation(n)
        n_test = max(1, int(round(n * test_frac)))
        te_segs = {segs[i] for i in perm[:n_test]}
        tr_segs = {segs[i] for i in perm[n_test:]}
        idx_tr.extend(dfp[dfp["seg_id"].isin(tr_segs)].index.tolist())
        idx_te.extend(dfp[dfp["seg_id"].isin(te_segs)].index.tolist())
    return idx_tr, idx_te


def suavizar(df_te, pred_raw, w=W_SUAVIZADO):
    """Media movil centrada agrupada por (participante, seg_id).

    Reindexado a df_te.index antes de devolver -- ver el mismo bug ya
    corregido en 19_combinado_residual.py y 24_optuna_sedentario_lstm.py.
    """
    tmp = df_te[["participante", "seg_id", "win_start"]].copy()
    tmp["pred"] = pd.Series(pred_raw, index=df_te.index).values
    tmp = tmp.sort_values(["participante", "seg_id", "win_start"])
    tmp["pred_suave"] = (
        tmp.groupby(["participante", "seg_id"])["pred"]
        .transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    )
    return tmp["pred_suave"].reindex(df_te.index).values


def metricas(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    try:
        r, _ = pearsonr(y_true, y_pred)
    except Exception:
        r = np.nan
    return mae, rmse, mape, r


resultados = []
for escenario, df_esc in ESCENARIOS.items():
    print(f"\n  --- {escenario} ---")
    for corrida in range(N_CORRIDAS):
        seed = corrida
        idx_tr, idx_te = split_por_grabacion(df_esc, seed)
        df_tr, df_te = df_esc.loc[idx_tr].copy(), df_esc.loc[idx_te].copy()

        medias = df_tr.groupby("participante")["bpm_watch"].mean()
        df_tr["media_persona"] = df_tr["participante"].map(medias)
        df_te["media_persona"] = df_te["participante"].map(medias).fillna(df_tr["bpm_watch"].mean())
        df_tr["residual"] = df_tr["bpm_watch"] - df_tr["media_persona"]

        sc   = StandardScaler().fit(df_tr[feat_cols].fillna(0))
        X_tr = sc.transform(df_tr[feat_cols].fillna(0))
        X_te = sc.transform(df_te[feat_cols].fillna(0))
        y_tr_res = df_tr["residual"].values
        y_te     = df_te["bpm_watch"].values

        mae_bl, rmse_bl, mape_bl, r_bl = metricas(y_te, df_te["media_persona"].values)
        resultados.append({"escenario": escenario, "corrida": corrida, "seed": seed, "modelo": "Baseline",
                            "MAE": mae_bl, "RMSE": rmse_bl, "MAPE": mape_bl, "r": r_bl})

        rf = RandomForestRegressor(**RF_BEST).fit(X_tr, y_tr_res)
        pred_rf_raw   = df_te["media_persona"].values + rf.predict(X_te)
        pred_rf_suave = suavizar(df_te, pred_rf_raw)
        mae_rf, rmse_rf, mape_rf, r_rf = metricas(y_te, pred_rf_suave)
        resultados.append({"escenario": escenario, "corrida": corrida, "seed": seed, "modelo": "RF",
                            "MAE": mae_rf, "RMSE": rmse_rf, "MAPE": mape_rf, "r": r_rf})

        svr = SVR(**SVR_BEST).fit(X_tr, y_tr_res)
        pred_svr_raw   = df_te["media_persona"].values + svr.predict(X_te)
        pred_svr_suave = suavizar(df_te, pred_svr_raw)
        mae_svr, rmse_svr, mape_svr, r_svr = metricas(y_te, pred_svr_suave)
        resultados.append({"escenario": escenario, "corrida": corrida, "seed": seed, "modelo": "SVR",
                            "MAE": mae_svr, "RMSE": rmse_svr, "MAPE": mape_svr, "r": r_svr})

        print(f"  Corrida {corrida+1:2d}/{N_CORRIDAS}  Baseline: MAE={mae_bl:.3f}  |  "
              f"RF: MAE={mae_rf:.3f}  |  SVR: MAE={mae_svr:.3f}")

df_res = pd.DataFrame(resultados)
df_res.to_csv(os.path.join(OUTPUT_DIR, "resultados_33_corridas.csv"), index=False)
print(f"\n  resultados_33_corridas.csv -> {OUTPUT_DIR}")


# ─── RESUMEN: media, std, IC95% (por escenario x modelo) ──────────────────────
def ic95(s):
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s))


resumen = df_res.groupby(["escenario", "modelo"])[["MAE", "RMSE", "MAPE", "r"]].agg(["mean", "std"])
resumen.columns = ["_".join(c) for c in resumen.columns]
for m in ["MAE", "RMSE", "MAPE", "r"]:
    resumen[f"{m}_ic95"] = df_res.groupby(["escenario", "modelo"])[m].apply(ic95)
resumen = resumen.reset_index()
resumen.to_csv(os.path.join(OUTPUT_DIR, "resumen_33_corridas.csv"), index=False)

print(f"\n  resumen_33_corridas.csv -> {OUTPUT_DIR}\n")
print("=" * 66)
print("  RESUMEN (media +/- IC95% sobre {} corridas)".format(N_CORRIDAS))
print("=" * 66)
print(resumen.to_string(index=False))


# ─── FIGURA: barras con error bars (estilo figura del asesor) ─────────────────
# Filas = escenario (A: todas las posiciones / C: sedentario + sync estricta)
# Columnas = metrica (MAE / RMSE / MAPE). Un solo eje por subgrafico (nunca
# doble eje) porque MAE/RMSE (lat/min) y MAPE (%) tienen escalas distintas.
metricas_fig = ["MAE", "RMSE", "MAPE"]
modelos      = ["Baseline", "RF", "SVR"]
colores      = {"Baseline": "#9E9E9E", "RF": C_RF, "SVR": C_SVR}
escenarios_fig = list(ESCENARIOS.keys())

fig, axes = plt.subplots(len(escenarios_fig), len(metricas_fig), figsize=(13, 8.5))
for fila, escenario in enumerate(escenarios_fig):
    res_esc = resumen[resumen["escenario"] == escenario]
    for col, met in enumerate(metricas_fig):
        ax = axes[fila, col]
        medias_m = [res_esc.loc[res_esc["modelo"] == mod, f"{met}_mean"].values[0] for mod in modelos]
        ics_m    = [res_esc.loc[res_esc["modelo"] == mod, f"{met}_ic95"].values[0] for mod in modelos]
        bars = ax.bar(modelos, medias_m, yerr=ics_m, capsize=6,
                       color=[colores[m] for m in modelos], edgecolor="black", linewidth=0.8,
                       error_kw={"linewidth": 1.3, "ecolor": "black"})
        unidad = "%" if met == "MAPE" else "lat/min"
        ax.set_ylabel(f"{met} ({unidad})")
        if fila == 0:
            ax.set_title(met, fontweight="bold")
        if col == 0:
            ax.text(-0.35, 0.5, escenario, transform=ax.transAxes, rotation=90,
                     va="center", ha="center", fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, medias_m):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                     f"{val:.2f}", ha="center", va="center", fontsize=9,
                     color="white", fontweight="bold")

fig.suptitle(f"Comparacion RF vs SVR — Media ± IC95% sobre {N_CORRIDAS} corridas\n"
             f"(splits aleatorios por grabación dentro de cada persona)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_33_corridas_barras.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  fig_33_corridas_barras.png -> {OUTPUT_DIR}")

print("\nScript 25 completado.")
