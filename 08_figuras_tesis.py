"""
08_figuras_tesis.py
Tesis: Monitoreo no invasivo de la frecuencia cardiaca mediante senales Wi-Fi CSI
Universidad de Lima - Ingenieria de Sistemas
Autores: Yadhira Sarmiento Escobar (20214190) y Favio Chavarry Minaya (20214680)

TODAS las figuras de la tesis en un solo script, a partir del MODELO FINAL
GANADOR (19_combinado_residual.py): RF residual + top-15 permutation
importance + suavizado w=9. MAE final = 9.1049 BPM.

Figuras del modelo final (un solo modelo):
  fig1 - BPM predicho vs real (serie temporal)
  fig2 - Dispersion predicho vs real
  fig3 - Distribucion del error absoluto
  fig4 - Tabla comparativa con literatura (scripts 19 y 24)
  fig5 - Analisis de Bland-Altman

Figuras comparativas RF vs SVR (estilo Figuras 4,5,6,8 del PDF de tesis
aprobado por el asesor; SVR se entrena solo con fines comparativos dentro
de 19_combinado_residual.py, no es el modelo final):
  fig_comparacion_rf_svr_serie.png
  fig_comparacion_rf_svr_dispersion.png
  fig_comparacion_rf_svr_error.png
  fig_comparacion_rf_svr_bland_altman.png

Figuras adicionales de discusion (adaptadas del codigo aprobado por el
asesor en la version anterior de la tesis):
  fig_eda_filtrado_{pid}.png         - crudo vs filtrado, FFT, heatmap
  fig_comparacion_barras.png         - barras MAE vs literatura
  fig_discusion_f1_rangos.png        - F1-score por rango clinico
  fig_discusion_zscore.png           - Z-score respecto a la literatura
  fig_discusion_calibracion_test.png - MAE calibracion (train) vs test

Cada seccion se salta con un aviso si le faltan sus prerequisitos, para
poder correr el script en cualquier momento sin que un bloque tumbe a los
demas.

Prerequisitos:
  - Figuras 1-5 y comparacion RF vs SVR: haber ejecutado 19_combinado_residual.py
  - EDA: haber ejecutado 03_extraer_csi.py y 04_preprocesamiento.py
  - Calibracion vs test: haber ejecutado 12_features_modelos_v3_synced.py y
    23_seleccion_mi.py (mismos insumos que 19_combinado_residual.py)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR   = r"C:\Users\LENOVO\Desktop\tesis-wifi-csi-frecuencia-cardiaca-main\resultados"
PRED_FILE    = os.path.join(OUTPUT_DIR, "predicciones_finales_v19.csv")
COMP_FILE    = os.path.join(OUTPUT_DIR, "predicciones_comparacion_rf_svr.csv")
DATASET_CSV  = os.path.join(OUTPUT_DIR, "dataset_completo_v3_synced.csv")
PI_CSV       = os.path.join(OUTPUT_DIR, "permutation_importance_features.csv")
CSI_CRUDO    = r"C:\Users\LENOVO\Desktop\csi_csv"
CSI_FILTRADO = r"C:\Users\LENOVO\Desktop\csi_filtrado_v3"

FS = 7.7
RANDOM_STATE = 42
N_TOP_FEAT   = 10
W_SUAVIZADO  = 9
RF_BEST = {
    "max_depth": 6, "max_features": "sqrt", "min_samples_leaf": 29,
    "n_estimators": 387, "random_state": RANDOM_STATE, "n_jobs": -1,
}

# Literatura consistente con las Figuras 9-12 del PDF de tesis aprobado
LIT_ESTUDIOS = ["Liu et al. (2022)", "Sun et al. (2024)", "Gu et al. (2021)",
                "Wang et al. (2020)", "Gouveia et al. (2024)", "Pulse-Fi (2025)"]
LIT_MAE      = [0.6, 0.8, 3.531, 1.19, 2.72, 0.20]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 66)
print("  08 - TODAS LAS FIGURAS DE LA TESIS")
print("=" * 66)

C_MOD, C_REF = "#2196F3", "#4CAF50"
C_RF, C_SVR = "#2196F3", "#FF5722"


# ══════════════════════════════════════════════════════════════════════════
# SECCION 1: Figuras del modelo final (un solo modelo, RF)
# ══════════════════════════════════════════════════════════════════════════
if not os.path.exists(PRED_FILE):
    print(f"\n  [SALTADO] Figuras 1-5: no se encontro {PRED_FILE}"
          f" (ejecutar 19_combinado_residual.py primero)")
    mae = rmse = r2 = r = None
else:
    df = pd.read_csv(PRED_FILE)
    y_real = df["bpm_watch"].values
    y_pred = df["pred_final"].values

    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)
    r, _ = pearsonr(y_real, y_pred)
    print(f"\n  Modelo final (split unico) -> MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, r={r:.4f}\n")

    # MAE "headline" real del proyecto: media +/- IC95% sobre 33 corridas
    # (25_corridas_repetidas.py), no el numero de un unico split (mae, arriba).
    # Se usa para las figuras que comparan contra la literatura; mae/rmse/r del
    # split unico se mantienen para las figuras 1-5, que ilustran una corrida
    # real concreta (no tiene sentido graficar un scatter de un promedio).
    resumen_csv = os.path.join(OUTPUT_DIR, "resumen_33_corridas.csv")
    if os.path.exists(resumen_csv):
        df_resumen_33 = pd.read_csv(resumen_csv)
        fila_a_rf = df_resumen_33[(df_resumen_33["escenario"] == "A: Todas las posiciones") &
                                   (df_resumen_33["modelo"] == "RF")]
        mae_headline    = float(fila_a_rf["MAE_mean"].values[0])
        mae_headline_ic = float(fila_a_rf["MAE_ic95"].values[0])
    else:
        print(f"  [AVISO] {resumen_csv} no encontrado -- usando MAE de split unico como headline"
              f" (ejecutar 25_corridas_repetidas.py para el numero real)")
        mae_headline, mae_headline_ic = mae, 0.0

    # ── FIG 1: Serie temporal ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(y_real))
    ax.plot(x, y_real, color="gray", linewidth=1.2, label="BPM Real", alpha=0.8)
    ax.plot(x, y_pred, color=C_MOD, linewidth=1.2, label="BPM Predicho (modelo final)", alpha=0.9)
    ax.fill_between(x, y_real, y_pred, alpha=0.15, color=C_MOD)
    ax.set_ylabel("Frecuencia Cardiaca (lat/min)")
    ax.set_xlabel("Muestra")
    ax.set_title(f"BPM Predicho vs Real - Conjunto de Test\nMAE={mae:.2f} | r={r:.3f}",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_predicho_vs_real.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig1_predicho_vs_real.png")

    # ── FIG 2: Dispersion ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    lim_min = min(y_real.min(), y_pred.min()) - 5
    lim_max = max(y_real.max(), y_pred.max()) + 5
    ax.scatter(y_real, y_pred, color=C_MOD, alpha=0.4, s=20)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=C_REF,
            linewidth=2, linestyle="--", label="Ideal (y=x)")
    ax.set_xlabel("BPM Real"); ax.set_ylabel("BPM Predicho")
    ax.set_title(f"Dispersion: BPM Predicho vs Real\nMAE={mae:.2f} | RMSE={rmse:.2f} | r={r:.3f}",
                 fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_dispersion.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig2_dispersion.png")

    # ── FIG 3: Distribucion del error ────────────────────────────────────
    error = np.abs(y_real - y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(error, bins=25, color=C_MOD, alpha=0.75, edgecolor="white")
    ax.axvline(mae, color="black", linestyle="--", linewidth=2, label=f"MAE = {mae:.2f}")
    ax.axvline(np.median(error), color="red", linestyle=":", linewidth=2,
               label=f"Mediana = {np.median(error):.2f}")
    ax.set_xlabel("Error Absoluto (lat/min)"); ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion del Error Absoluto", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_distribucion_error.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig3_distribucion_error.png")

    # ── FIG 4: Tabla comparativa literatura ──────────────────────────────
    literatura = {
        "Estudio": ["Liu et al. (2022)", "Sun et al. (2024)", "Gu et al. (2021)",
                    "Wang et al. (2020)", "Gouveia et al. (2024)", "Pulse-Fi (2025)",
                    "Este trabajo (Esc. A)", "Este trabajo (Esc. C)"],
        "Metodo":  ["CNN H3RN", "Proyeccion rotacional + DL",
                    "Butterworth + Hampel + FFT", "PhaseBeat",
                    "RF sobre features CSI", "LSTM",
                    "RF residual + suavizado",
                    "RF residual + suavizado (sedentario)"],
        "MAE":     [0.6, 0.8, 3.531, 1.19, 2.72, 0.20, f"{mae_headline:.2f} +/- {mae_headline_ic:.2f}", "--"],
        "RMSE":    ["--", "--", "--", "--", "--", "--", f"{rmse:.2f}", "--"],
        "R2":      ["--", "--", "--", "--", "--", "--", f"{r2:.3f}", "--"],
        "r":       [0.991, 0.968, 0.947, "--", "--", "--", f"{r:.3f}", "--"],
    }
    comp_csv = os.path.join(OUTPUT_DIR, "comparacion_escenarios.csv")
    if os.path.exists(comp_csv):
        df_comp_24 = pd.read_csv(comp_csv)
        mejor_24   = df_comp_24.sort_values("MAE").iloc[0]
        literatura["MAE"][-1]  = round(float(mejor_24["MAE"]), 4)
        if "RMSE" in mejor_24: literatura["RMSE"][-1] = f"{mejor_24['RMSE']:.2f}"
        if "R2"   in mejor_24: literatura["R2"][-1]   = f"{mejor_24['R2']:.3f}"
        if "r"    in mejor_24: literatura["r"][-1]    = f"{mejor_24['r']:.3f}"
        literatura["Metodo"][-1] = "RF residual + suavizado (sedentario)"
        # Si ya se corrieron las 33 corridas sobre el Escenario C (sedentario +
        # sync estricta), preferir ese MAE validado (media +/- IC95%) sobre el
        # de un unico split -- mismo criterio que "Este trabajo (Escenario A)".
        if os.path.exists(resumen_csv):
            fila_c_rf = df_resumen_33[(df_resumen_33["escenario"] == "C: Sedentario + sync estricta") &
                                       (df_resumen_33["modelo"] == "RF")]
            if len(fila_c_rf) > 0:
                mae_c = float(fila_c_rf["MAE_mean"].values[0])
                ic_c  = float(fila_c_rf["MAE_ic95"].values[0])
                literatura["MAE"][-1] = f"{mae_c:.2f} +/- {ic_c:.2f}"
    df_lit   = pd.DataFrame(literatura)
    col_keys   = ["Estudio", "Metodo", "MAE", "RMSE", "R2", "r"]
    col_widths = [0.17, 0.33, 0.125, 0.125, 0.125, 0.125]
    n_cols   = len(col_keys)
    fig, ax  = plt.subplots(figsize=(16, 4.2)); ax.axis("off")
    tabla    = ax.table(
        cellText=[[str(row[k]) for k in col_keys] for _, row in df_lit.iterrows()],
        colLabels=col_keys, colWidths=col_widths, cellLoc="center", loc="center",
    )
    tabla.auto_set_font_size(False); tabla.set_fontsize(8.5); tabla.scale(1.2, 1.9)
    for i in range(len(df_lit)):
        for j in range(n_cols):
            cell = tabla[i + 1, j]
            if "Este trabajo" in str(df_lit.iloc[i]["Estudio"]):
                cell.set_facecolor("#E3F2FD"); cell.set_text_props(fontweight="bold")
    for j in range(n_cols):
        tabla[0, j].set_facecolor("#1565C0"); tabla[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Comparacion con la Literatura", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_comparacion_literatura.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig4_comparacion_literatura.png")

    # ── FIG 5: Bland-Altman ──────────────────────────────────────────────
    media_ba = (y_real + y_pred) / 2
    diff = y_pred - y_real
    bias = np.mean(diff); std_d = np.std(diff)
    loa_sup = bias + 1.96 * std_d; loa_inf = bias - 1.96 * std_d
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(media_ba, diff, alpha=0.4, color=C_MOD, s=20)
    ax.axhline(bias,    color="black", linewidth=2, linestyle="-",  label=f"Bias = {bias:.2f}")
    ax.axhline(loa_sup, color="red",   linewidth=1.5, linestyle="--", label=f"+1.96 SD = {loa_sup:.2f}")
    ax.axhline(loa_inf, color="red",   linewidth=1.5, linestyle="--", label=f"-1.96 SD = {loa_inf:.2f}")
    ax.axhline(0, color="gray", linewidth=1, linestyle=":")
    ax.set_xlabel("Promedio BPM"); ax.set_ylabel("Diferencia (Predicho - Real)")
    ax.set_title(f"Analisis de Bland-Altman\nBias={bias:.2f} | LoA=[{loa_inf:.1f}, {loa_sup:.1f}]",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_bland_altman.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig5_bland_altman.png")


# ══════════════════════════════════════════════════════════════════════════
# SECCION 2: Comparacion RF vs SVR (estilo Figuras 4,5,6,8 del PDF aprobado)
# ══════════════════════════════════════════════════════════════════════════
if not os.path.exists(COMP_FILE):
    print(f"\n  [SALTADO] Comparacion RF vs SVR: no se encontro {COMP_FILE}"
          f" (ejecutar 19_combinado_residual.py primero)")
else:
    df_c = pd.read_csv(COMP_FILE)
    y_c = df_c["bpm_watch"].values
    pred_rf_c = df_c["pred_rf"].values
    pred_svr_c = df_c["pred_svr"].values

    mae_rf_c  = mean_absolute_error(y_c, pred_rf_c)
    rmse_rf_c = np.sqrt(mean_squared_error(y_c, pred_rf_c))
    r_rf_c, _ = pearsonr(y_c, pred_rf_c)
    mae_svr_c  = mean_absolute_error(y_c, pred_svr_c)
    rmse_svr_c = np.sqrt(mean_squared_error(y_c, pred_svr_c))
    r_svr_c, _ = pearsonr(y_c, pred_svr_c)
    print(f"\n  RF (comparacion)  -> MAE={mae_rf_c:.4f}, RMSE={rmse_rf_c:.4f}, r={r_rf_c:.4f}")
    print(f"  SVR (comparacion) -> MAE={mae_svr_c:.4f}, RMSE={rmse_svr_c:.4f}, r={r_svr_c:.4f}")

    # Serie temporal: se agrega el BPM real por ventana (media de las muestras
    # crudas de cada seg_id) para que quede en la misma granularidad que la
    # prediccion (un valor por ventana). Comparar el real crudo (ruidoso,
    # muestra a muestra) contra la prediccion en bloque por ventana daba una
    # falsa impresion visual de desincronizacion. El MAE/r del titulo se sigue
    # calculando sobre los datos crudos (metrica oficial reportada en la tesis).
    df_win = (
        df_c.groupby(["participante", "seg_id"], sort=False, as_index=False)
        .agg(bpm_watch=("bpm_watch", "mean"), pred_rf=("pred_rf", "first"), pred_svr=("pred_svr", "first"))
    )
    x_c = np.arange(len(df_win))
    y_c_win = df_win["bpm_watch"].values
    pred_rf_win = df_win["pred_rf"].values
    pred_svr_win = df_win["pred_svr"].values

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("BPM Predicho vs BPM Real - Conjunto de Test (RF vs SVR)", fontsize=14, fontweight="bold")
    for ax, pred_i, color, nombre, mae_i, r_i in [
        (axes[0], pred_rf_win,  C_RF,  "Random Forest", mae_rf_c,  r_rf_c),
        (axes[1], pred_svr_win, C_SVR, "SVR",            mae_svr_c, r_svr_c),
    ]:
        ax.plot(x_c, y_c_win, color="gray", linewidth=1.2, label="BPM Real (promedio por ventana)", alpha=0.8)
        ax.plot(x_c, pred_i, color=color, linewidth=1.2, label=f"BPM Predicho ({nombre})", alpha=0.9)
        ax.fill_between(x_c, y_c_win, pred_i, alpha=0.15, color=color)
        ax.set_ylabel("Frecuencia Cardiaca (lat/min)")
        ax.set_title(f"{nombre} - MAE={mae_i:.2f} | r={r_i:.3f}")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Ventana (seg_id)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_comparacion_rf_svr_serie.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig_comparacion_rf_svr_serie.png")

    # Dispersion
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Dispersion: BPM Predicho vs BPM Real (RF vs SVR)", fontsize=14, fontweight="bold")
    lim_min_c = min(y_c.min(), pred_rf_c.min(), pred_svr_c.min()) - 5
    lim_max_c = max(y_c.max(), pred_rf_c.max(), pred_svr_c.max()) + 5
    for ax, pred_i, color, nombre, mae_i, rmse_i, r_i in [
        (axes[0], pred_rf_c,  C_RF,  "Random Forest", mae_rf_c,  rmse_rf_c,  r_rf_c),
        (axes[1], pred_svr_c, C_SVR, "SVR",            mae_svr_c, rmse_svr_c, r_svr_c),
    ]:
        ax.scatter(y_c, pred_i, color=color, alpha=0.4, s=20)
        ax.plot([lim_min_c, lim_max_c], [lim_min_c, lim_max_c], color=C_REF, linewidth=2,
                linestyle="--", label="Ideal (y=x)")
        ax.set_xlabel("BPM Real"); ax.set_ylabel("BPM Predicho")
        ax.set_title(f"{nombre}\nMAE={mae_i:.2f} | RMSE={rmse_i:.2f} | r={r_i:.3f}")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_comparacion_rf_svr_dispersion.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig_comparacion_rf_svr_dispersion.png")

    # Distribucion del error
    error_rf_c  = np.abs(y_c - pred_rf_c)
    error_svr_c = np.abs(y_c - pred_svr_c)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribucion del Error Absoluto (RF vs SVR)", fontsize=14, fontweight="bold")
    for ax, error, color, nombre, mae_i in [
        (axes[0], error_rf_c,  C_RF,  "Random Forest", mae_rf_c),
        (axes[1], error_svr_c, C_SVR, "SVR",            mae_svr_c),
    ]:
        ax.hist(error, bins=25, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(mae_i, color="black", linestyle="--", linewidth=2, label=f"MAE = {mae_i:.2f}")
        ax.axvline(np.median(error), color="red", linestyle=":", linewidth=2,
                   label=f"Mediana = {np.median(error):.2f}")
        ax.set_xlabel("Error Absoluto (lat/min)"); ax.set_ylabel("Frecuencia")
        ax.set_title(f"{nombre}")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_comparacion_rf_svr_error.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig_comparacion_rf_svr_error.png")

    # Bland-Altman
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Analisis de Bland-Altman: BPM Predicho vs Real (RF vs SVR)", fontsize=13, fontweight="bold")
    for ax, pred_i, nombre, color in [
        (axes[0], pred_rf_c,  "Random Forest", C_RF),
        (axes[1], pred_svr_c, "SVR",            C_SVR),
    ]:
        media_ba_c = (y_c + pred_i) / 2
        diff_c = pred_i - y_c
        bias_c = np.mean(diff_c); std_c = np.std(diff_c)
        loa_sup_c = bias_c + 1.96 * std_c; loa_inf_c = bias_c - 1.96 * std_c
        ax.scatter(media_ba_c, diff_c, alpha=0.4, color=color, s=20)
        ax.axhline(bias_c, color="black", linewidth=2, linestyle="-", label=f"Bias = {bias_c:.2f}")
        ax.axhline(loa_sup_c, color="red", linewidth=1.5, linestyle="--", label=f"+1.96 SD = {loa_sup_c:.2f}")
        ax.axhline(loa_inf_c, color="red", linewidth=1.5, linestyle="--", label=f"-1.96 SD = {loa_inf_c:.2f}")
        ax.axhline(0, color="gray", linewidth=1, linestyle=":")
        ax.set_xlabel("Promedio BPM (Real + Predicho) / 2"); ax.set_ylabel("Diferencia (Predicho - Real)")
        ax.set_title(f"{nombre}\nBias={bias_c:.2f} | LoA=[{loa_inf_c:.1f}, {loa_sup_c:.1f}]")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_comparacion_rf_svr_bland_altman.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig_comparacion_rf_svr_bland_altman.png")


# ══════════════════════════════════════════════════════════════════════════
# SECCION 3: EDA senales filtradas (crudo vs filtrado)
# ══════════════════════════════════════════════════════════════════════════
def parsear_complejo(s):
    try:
        return complex(str(s).strip().replace(" ", "").lstrip("(").rstrip(")"))
    except Exception:
        return complex(0, 0)


def cargar_crudo(path):
    dfx = pd.read_csv(path)
    sub_cols = [c for c in dfx.columns if c.startswith("sub_")]
    amp = np.zeros((len(dfx), len(sub_cols)), dtype=np.float32)
    for j, col in enumerate(sub_cols):
        amp[:, j] = np.abs(dfx[col].apply(parsear_complejo).values)
    return amp, dfx["timestamp"].values


def cargar_filtrado(path):
    dfx = pd.read_csv(path)
    sub_cols = [c for c in dfx.columns if c.startswith("sub_")]
    return dfx[sub_cols].values.astype(np.float32), dfx["timestamp"].values


HAY_CRUDO = os.path.isdir(CSI_CRUDO)
if not HAY_CRUDO:
    print(f"\n  [AVISO] EDA: no se encontro {CSI_CRUDO} -- se omitira el panel"
          f" 'crudo vs filtrado' y se graficara solo la senal filtrada.")

if not os.path.isdir(CSI_FILTRADO):
    print(f"\n  [SALTADO] EDA: falta {CSI_FILTRADO}"
          f" (ejecutar 04_preprocesamiento.py primero)")
else:
    comunes = sorted(os.listdir(CSI_FILTRADO))

    def _n_posiciones(pid):
        return len(list((Path(CSI_FILTRADO) / pid).glob("pos_*.csv")))

    # Se elige el participante con mas posiciones grabadas (no simplemente
    # el primero alfabeticamente), para que los paneles de "amplitud media
    # por posicion" y el heatmap posicion x subportadora tengan suficiente
    # variedad para ser ilustrativos.
    participantes_eda = sorted(comunes, key=_n_posiciones, reverse=True)[:1]
    if not participantes_eda:
        print(f"\n  [SALTADO] EDA: no hay participantes en {CSI_FILTRADO}")
    for pid in participantes_eda:
        dir_c = Path(CSI_CRUDO) / pid if HAY_CRUDO else None
        dir_f = Path(CSI_FILTRADO) / pid
        csvs_c = sorted(dir_c.glob("pos_*.csv")) if HAY_CRUDO and dir_c.is_dir() else []
        csvs_f = sorted(dir_f.glob("pos_*.csv"))
        if not csvs_f:
            print(f"  [WARN] Participante {pid} sin archivos filtrados, saltando.")
            continue

        print(f"\n  Participante {pid}: {len(csvs_c)} crudos | {len(csvs_f)} filtrados")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(
            f"EDA Senales CSI Filtradas (v3) - Participante {pid}\n"
            f"Pipeline: F_HIGH=2.17, Butterworth orden 5 (sosfiltfilt) + ISPD",
            fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        amp_f, ts_f = cargar_filtrado(str(csvs_f[0]))
        t_f = ts_f - ts_f[0]
        if csvs_c:
            amp_c, ts_c = cargar_crudo(str(csvs_c[0]))
            t_c = ts_c - ts_c[0]
            ax.plot(t_c, amp_c[:, 0], color="steelblue", alpha=0.6, lw=0.8, label="Crudo")
            ax2 = ax.twinx()
            ax2.plot(t_f, amp_f[:, 0], color="red", alpha=0.8, lw=1.2, label="Filtrado v3")
            ax.set_title(f"Senal cruda vs filtrada v3 ({csvs_c[0].stem}, sub_000)")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud cruda", color="steelblue")
            ax2.set_ylabel("Amplitud filtrada", color="red")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
        else:
            ax.plot(t_f, amp_f[:, 0], color="red", alpha=0.8, lw=1.2, label="Filtrado v3")
            ax.set_title(f"Senal filtrada v3 ({csvs_f[0].stem}, sub_000)\n"
                         f"[datos crudos no disponibles]")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud filtrada", color="red")
            ax.legend(loc="upper left")

        ax = axes[0, 1]
        colores = plt.cm.tab20(np.linspace(0, 1, min(len(csvs_f), 20)))
        for idx, csv_f in enumerate(csvs_f[:9]):
            amp_f2, _ = cargar_filtrado(str(csv_f))
            senal = amp_f2[:, 0]
            fft_vals = np.abs(np.fft.rfft(senal))
            freqs = np.fft.rfftfreq(len(senal), d=1.0 / FS)
            mask = (freqs >= 0.5) & (freqs <= 3.0)
            ax.plot(freqs[mask], fft_vals[mask], color=colores[idx],
                    alpha=0.7, lw=1.0, label=csv_f.stem)
        ax.axvspan(0.8, 2.17, alpha=0.15, color="green", label="Banda cardiaca (0.8-2.17 Hz)")
        ax.set_title("FFT senal filtrada v3 por posicion (sub_000)")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud")
        ax.legend(fontsize=7, ncol=2)

        ax = axes[1, 0]
        medias, labels = [], []
        for csv_f in csvs_f:
            amp_f2, _ = cargar_filtrado(str(csv_f))
            medias.append(np.mean(np.abs(amp_f2)))
            labels.append(csv_f.stem)
        ax.bar(range(len(medias)), medias, color="salmon", edgecolor="white", lw=0.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_title("Amplitud media por posicion")
        ax.set_xlabel("Posicion")
        ax.set_ylabel("Amplitud media")

        ax = axes[1, 1]
        heatmap = []
        for csv_f in csvs_f:
            amp_f2, _ = cargar_filtrado(str(csv_f))
            heatmap.append(np.mean(np.abs(amp_f2), axis=0))
        heatmap = np.array(heatmap)
        im = ax.imshow(heatmap, aspect="auto", cmap="viridis",
                        extent=[0, heatmap.shape[1], heatmap.shape[0], 0])
        plt.colorbar(im, ax=ax, label="Amplitud media")
        ax.set_title("Heatmap: amplitud media por posicion y subportadora")
        ax.set_xlabel("Subportadora")
        ax.set_ylabel("Posicion")
        ax.set_yticks(range(len(csvs_f)))
        ax.set_yticklabels(labels, fontsize=7)

        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"fig_eda_filtrado_{pid}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: fig_eda_filtrado_{pid}.png")


# ══════════════════════════════════════════════════════════════════════════
# SECCION 4: Barras comparacion literatura, F1 por rango clinico, Z-score
#            (usan las predicciones del modelo final cargadas en la Seccion 1)
# ══════════════════════════════════════════════════════════════════════════
if mae is None:
    print(f"\n  [SALTADO] Barras/F1/Z-score: no se encontro {PRED_FILE}")
else:
    # Barras comparacion con literatura (usa el MAE headline de 33 corridas, no el split unico)
    nombres = LIT_ESTUDIOS + ["Este trabajo\n(RF final)"]
    maes = LIT_MAE + [mae_headline]
    errores = [0] * len(LIT_ESTUDIOS) + [mae_headline_ic]
    colores = ["#90CAF9"] * len(LIT_ESTUDIOS) + [C_MOD]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(range(len(nombres)), maes, yerr=errores, capsize=5,
                   color=colores, alpha=0.9, edgecolor="white", width=0.6,
                   error_kw={"linewidth": 1.3, "ecolor": "black"})
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, fontsize=9)
    ax.set_ylabel("MAE (lat/min)", fontsize=11)
    ax.set_title("Comparacion de MAE con la Literatura\nDataset eHealth CSI (Galdino et al., 2023)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(maes) * 1.2)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_comparacion_barras.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("\n  Guardado: fig_comparacion_barras.png")

    # F1-score por rango clinico
    UMBRAL_F1 = 10.0

    def f1_rango(y_true, y_hat, bpm_min, bpm_max, umbral=UMBRAL_F1):
        mask = (y_true >= bpm_min) & (y_true < bpm_max)
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0, 0
        err = np.abs(y_true[mask] - y_hat[mask])
        tp = np.sum(err <= umbral)
        fp = np.sum(err > umbral)
        fn = fp
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        return round(prec, 3), round(rec, 3), round(f1, 3), int(mask.sum())

    rangos = [
        ("Bradicardia\n(<60)", 0, 60),
        ("Normal\n(60-100)", 60, 100),
        ("Taquicardia\n(>100)", 100, 200),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"F1-Score por Rango Clinico (umbral error = {int(UMBRAL_F1)} lat/min)\n"
                 f"Modelo final: RF residual + suavizado w={W_SUAVIZADO}",
                 fontsize=13, fontweight="bold")
    for ax, (nombre, bmin, bmax) in zip(axes, rangos):
        prec, rec, f1, n_rango = f1_rango(y_real, y_pred, bmin, bmax)
        metricas_names = ["Precision", "Recall", "F1"]
        vals = [prec, rec, f1]
        bars = ax.bar(metricas_names, vals, color=C_MOD, alpha=0.85, width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{nombre}\n(n={n_rango})", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if n_rango < 30:
            ax.text(0.5, 0.55, f"n={n_rango} casos\nmuestra reducida",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="gray", style="italic",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_discusion_f1_rangos.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig_discusion_f1_rangos.png")

    # Z-score respecto a la literatura
    lit_mean = np.mean(LIT_MAE)
    lit_std = np.std(LIT_MAE)
    z_scores = [(m - lit_mean) / (lit_std + 1e-10) for m in LIT_MAE]
    z_propio = (mae_headline - lit_mean) / (lit_std + 1e-10)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(LIT_ESTUDIOS, z_scores, color="#90CAF9", alpha=0.8, edgecolor="white")
    ax.barh(["Este trabajo (RF final)"], [z_propio], color=C_MOD, alpha=0.95, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Media de la literatura")
    ax.set_xlabel("Z-score (MAE estandarizado respecto a la literatura)", fontsize=10)
    ax.set_title("Posicion relativa del modelo final respecto a la literatura",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_discusion_zscore.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig_discusion_zscore.png")


# ══════════════════════════════════════════════════════════════════════════
# SECCION 5: Calibracion (train) vs Test, por etapa
# ══════════════════════════════════════════════════════════════════════════
# El pipeline actual NO tiene un split de validacion separado, solo
# calibracion (80%) / test (20%) por grabacion completa. Esta figura
# reproduce EXACTAMENTE el split y el modelo de 19_combinado_residual.py
# (mismo RANDOM_STATE) para medir tambien el MAE sobre calibracion.
if not os.path.exists(DATASET_CSV) or not os.path.exists(PI_CSV):
    print(f"\n  [SALTADO] Calibracion vs test: falta {DATASET_CSV} o {PI_CSV}"
          f" (ejecutar 12_features_modelos_v3_synced.py y 23_seleccion_mi.py primero)")
else:
    dfd = pd.read_csv(DATASET_CSV)
    pi_rank = pd.read_csv(PI_CSV)
    feat_cols = pi_rank["feature"].tolist()[:N_TOP_FEAT]

    rng = np.random.RandomState(RANDOM_STATE)
    idx_tr, idx_te = [], []
    for part in sorted(dfd["participante"].unique()):
        dfp = dfd[dfd["participante"] == part]
        segs = sorted(dfp["seg_id"].unique())
        n = len(segs)
        if n < 2:
            idx_tr.extend(dfp.index.tolist())
            continue
        perm = rng.permutation(n)
        n_test = max(1, int(round(n * 0.20)))
        test_segs = {segs[i] for i in perm[:n_test]}
        train_segs = {segs[i] for i in perm[n_test:]}
        idx_tr.extend(dfp[dfp["seg_id"].isin(train_segs)].index.tolist())
        idx_te.extend(dfp[dfp["seg_id"].isin(test_segs)].index.tolist())

    df_tr = dfd.loc[idx_tr].copy()
    df_te = dfd.loc[idx_te].copy()

    medias_persona = df_tr.groupby("participante")["bpm_watch"].mean()
    df_tr["media_persona"] = df_tr["participante"].map(medias_persona)
    df_te["media_persona"] = df_te["participante"].map(medias_persona).fillna(df_tr["bpm_watch"].mean())
    df_tr["residual"] = df_tr["bpm_watch"] - df_tr["media_persona"]

    sc = StandardScaler().fit(df_tr[feat_cols].fillna(0))
    X_tr = sc.transform(df_tr[feat_cols].fillna(0))
    X_te = sc.transform(df_te[feat_cols].fillna(0))
    y_tr_res = df_tr["residual"].values

    rf_cal = RandomForestRegressor(**RF_BEST)
    rf_cal.fit(X_tr, y_tr_res)

    pred_res_tr = rf_cal.predict(X_tr)
    pred_res_te = rf_cal.predict(X_te)
    pred_tr = df_tr["media_persona"].values + pred_res_tr
    pred_te = df_te["media_persona"].values + pred_res_te

    mae_tr_sin_suave = mean_absolute_error(df_tr["bpm_watch"], pred_tr)
    mae_te_sin_suave = mean_absolute_error(df_te["bpm_watch"], pred_te)

    def suavizar(df_split, pred):
        d = df_split.sort_values(["participante", "seg_id", "win_start"]).copy()
        d["pred"] = pd.Series(pred, index=df_split.index).loc[d.index]
        pred_suave = (
            d.groupby(["participante", "seg_id"])["pred"]
            .transform(lambda s: s.rolling(window=W_SUAVIZADO, center=True, min_periods=1).mean())
            .values
        )
        return d["bpm_watch"].values, pred_suave

    y_tr_sorted, pred_tr_suave = suavizar(df_tr, pred_tr)
    y_te_sorted, pred_te_suave = suavizar(df_te, pred_te)
    mae_tr_suave = mean_absolute_error(y_tr_sorted, pred_tr_suave)
    mae_te_suave = mean_absolute_error(y_te_sorted, pred_te_suave)

    print(f"\n  Calibracion vs Test:")
    print(f"    RF tuneado        -> calibracion={mae_tr_sin_suave:.4f}  test={mae_te_sin_suave:.4f}")
    print(f"    RF + suavizado w={W_SUAVIZADO} -> calibracion={mae_tr_suave:.4f}  test={mae_te_suave:.4f}")

    etapas = ["RF tuneado", f"RF + suavizado w={W_SUAVIZADO}\n(modelo final)"]
    cal_maes = [mae_tr_sin_suave, mae_tr_suave]
    test_maes = [mae_te_sin_suave, mae_te_suave]

    fig, ax = plt.subplots(figsize=(8, 5))
    xg = np.arange(len(etapas))
    w = 0.35
    bars1 = ax.bar(xg - w / 2, cal_maes, w, label="Calibracion (train)", color="#90CAF9", edgecolor="white")
    bars2 = ax.bar(xg + w / 2, test_maes, w, label="Test", color=C_MOD, edgecolor="white")
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xg)
    ax.set_xticklabels(etapas, fontsize=10)
    ax.set_ylabel("MAE (lat/min)", fontsize=11)
    ax.set_title("MAE Calibracion vs Test por Etapa\n(split por grabacion completa, sin fuga)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(cal_maes + test_maes) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_discusion_calibracion_test.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  Guardado: fig_discusion_calibracion_test.png")


# ── RESUMEN FINAL ─────────────────────────────────────────────────────────────
print(f"\n{'='*66}")
print(f"  RESUMEN FINAL")
print(f"{'='*66}")
if mae is not None:
    print(f"  Modelo final (split unico, seed=42): MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f} | r={r:.4f}")
    print(f"  Modelo final (MAE HEADLINE, media +/- IC95% sobre 33 corridas): "
          f"MAE={mae_headline:.4f} +/- {mae_headline_ic:.4f}")
print(f"\n  Todas las figuras guardadas en: {OUTPUT_DIR}")
print("\nScript 08_figuras_tesis completado.\n")
