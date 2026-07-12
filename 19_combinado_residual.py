"""
19_combinado_residual.py
Tesis: Monitoreo no invasivo de la frecuencia cardiaca mediante senales Wi-Fi CSI
Universidad de Lima - Ingenieria de Sistemas
Autores: Yadhira Sarmiento Escobar (20214190) y Favio Chavarry Minaya (20214680)

MODELO FINAL GANADOR DEL PROYECTO (MAE = 9.1049).

Historial resumido (ver memoria del proyecto para el detalle de cada escenario
descartado): el mejor resultado fue evolucionando asi:
  9.345 -> baseline trivial (promedio de HR por persona, sin CSI)
  9.140 -> RF tuneado sobre residual (HR - promedio_persona)
  9.136 -> + ensemble RF/ExtraTrees/GradientBoosting + suavizado w=9,
           features top-15 por importancia nativa de RF
  9.105 -> (ESTE SCRIPT) mismo RF + suavizado w=9, pero las top-15 features
           se eligen por PERMUTATION IMPORTANCE (calculada en
           23_seleccion_mi.py) en vez de por feature_importances_ de RF.
           El ensemble con estas features NO mejora (GBR empeora fuerte con
           este set), por eso el modelo final es RF solo, no ensemble.

Otros escenarios probados y descartados sin mejora (no requieren script propio
porque no superaron este resultado): stacking con out-of-fold (MAE=9.16),
calibracion adaptativa por EMA (MAE=4.07 pero INVALIDO por fuga de datos:
usaria el HR real en tiempo real como "calibracion", lo cual no es posible
sin sensor de contacto), filtrado de participantes ruidosos (empeora, menos
datos de entrenamiento), descomposicion VMD de la senal (MAE=9.15), ICA sobre
subportadoras (MAE=9.28).

Prerequisito: ejecutar 12_features_modelos_v3_synced.py y luego
23_seleccion_mi.py (genera permutation_importance_features.csv).

Salida: Data_DS1_raspberry-main/resultados_combinado_residual.csv,
        Data_DS1_raspberry-main/predicciones_finales_v19.csv,
        Data_DS1_raspberry-main/predicciones_comparacion_rf_svr.csv
        (las figuras se generan aparte en 08_figuras_tesis.py)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR  = r"C:\Users\LENOVO\Desktop\tesis-wifi-csi-frecuencia-cardiaca-main\resultados"
DATASET_CSV = os.path.join(OUTPUT_DIR, "dataset_completo_v3_synced.csv")
PI_CSV      = os.path.join(OUTPUT_DIR, "permutation_importance_features.csv")

RANDOM_STATE = 42
N_TOP_FEAT   = 10
W_SUAVIZADO  = 9

RF_BEST = {
    "max_depth": 6, "max_features": "sqrt", "min_samples_leaf": 29,
    "n_estimators": 387, "random_state": RANDOM_STATE, "n_jobs": -1,
}
# SVR se entrena solo con fines comparativos (figuras RF vs SVR estilo tesis
# original); ya fue descartado como modelo final en la primera ronda de
# experimentacion por no mejorar el MAE del RF.
SVR_PARAM_GRID = {"C": [1, 10, 50], "epsilon": [0.5, 1.0, 2.0], "gamma": ["scale"]}

print("=" * 66)
print("  19 - MODELO FINAL: RF + top-15 (permutation importance) + suavizado w=9")
print("=" * 66)

if not os.path.exists(DATASET_CSV):
    print(f"\n  [ERROR] No se encontro {DATASET_CSV}")
    sys.exit(1)
if not os.path.exists(PI_CSV):
    print(f"\n  [ERROR] No se encontro {PI_CSV} -- ejecutar primero 23_seleccion_mi.py")
    sys.exit(1)

df = pd.read_csv(DATASET_CSV)
pi_rank = pd.read_csv(PI_CSV)
feat_cols = pi_rank["feature"].tolist()[:N_TOP_FEAT]
print(f"  Features (permutation importance, top-{N_TOP_FEAT}): {feat_cols}\n")


def metricas(y_true, y_pred, nombre=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        r, _ = pearsonr(y_true, y_pred)
    except Exception:
        r = np.nan
    print(f"  {nombre:<42} -> MAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.4f}")
    return {"etapa": nombre, "MAE": mae, "RMSE": rmse, "r": r}


# ─── SPLIT POR GRABACION COMPLETA (calibracion 80% / test 20%, sin fuga) ──────
rng = np.random.RandomState(RANDOM_STATE)
idx_tr, idx_te = [], []
for part in sorted(df["participante"].unique()):
    dfp  = df[df["participante"] == part]
    segs = sorted(dfp["seg_id"].unique())
    n    = len(segs)
    if n < 2:
        idx_tr.extend(dfp.index.tolist())
        continue
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * 0.20)))
    test_segs  = {segs[i] for i in perm[:n_test]}
    train_segs = {segs[i] for i in perm[n_test:]}
    idx_tr.extend(dfp[dfp["seg_id"].isin(train_segs)].index.tolist())
    idx_te.extend(dfp[dfp["seg_id"].isin(test_segs)].index.tolist())

df_tr = df.loc[idx_tr].copy()
df_te = df.loc[idx_te].copy()

medias_persona = df_tr.groupby("participante")["bpm_watch"].mean()
df_tr["media_persona"] = df_tr["participante"].map(medias_persona)
df_te["media_persona"] = df_te["participante"].map(medias_persona).fillna(df_tr["bpm_watch"].mean())

df_tr["residual"] = df_tr["bpm_watch"] - df_tr["media_persona"]
df_te["residual"]  = df_te["bpm_watch"]  - df_te["media_persona"]

sc = StandardScaler().fit(df_tr[feat_cols].fillna(0))
X_tr = sc.transform(df_tr[feat_cols].fillna(0))
X_te = sc.transform(df_te[feat_cols].fillna(0))
y_tr_res = df_tr["residual"].values

mae_baseline = mean_absolute_error(df_te["bpm_watch"], df_te["media_persona"])
print(f"  Baseline (solo promedio por persona, sin CSI) MAE = {mae_baseline:.4f}\n")

resultados = []
resultados.append({"etapa": "Baseline (promedio persona)", "MAE": mae_baseline, "RMSE": np.nan, "r": np.nan})

# ─── MODELO FINAL: RF TUNEADO SOBRE EL RESIDUAL ───────────────────────────────
rf = RandomForestRegressor(**RF_BEST)
rf.fit(X_tr, y_tr_res)
pred_res = rf.predict(X_te)
pred = df_te["media_persona"].values + pred_res
resultados.append(metricas(df_te["bpm_watch"], pred, "RF tuneado (top-15 PermImp)"))

# ─── SVR (solo comparacion; NO es el modelo final) ────────────────────────────
print("\n  Entrenando SVR (busqueda de hiperparametros, GroupKFold por participante)...")
groups_tr = df_tr["participante"].values
gkf = GroupKFold(n_splits=3)
svr_search = GridSearchCV(
    SVR(kernel="rbf"), SVR_PARAM_GRID, scoring="neg_mean_absolute_error",
    cv=list(gkf.split(X_tr, y_tr_res, groups=groups_tr)), n_jobs=-1,
)
svr_search.fit(X_tr, y_tr_res)
svr = svr_search.best_estimator_
print(f"  Mejores hiperparametros SVR: {svr_search.best_params_}")
pred_res_svr = svr.predict(X_te)
pred_svr = df_te["media_persona"].values + pred_res_svr
metricas(df_te["bpm_watch"], pred_svr, "SVR (comparacion, mismos features)")

# ─── SUAVIZADO POR MEDIA MOVIL w=9 (agrupado por grabacion) ──────────────────
# OJO: pred/pred_svr deben capturarse con el indice ORIGINAL (pre-sort) antes
# de reordenar df_te; construir el Series con df_te.index ya reordenado
# desalinea cada prediccion con la fila equivocada.
pred_series = pd.Series(pred, index=df_te.index)
pred_svr_series = pd.Series(pred_svr, index=df_te.index)
df_te = df_te.sort_values(["participante", "seg_id", "win_start"])
df_te["pred"] = pred_series.loc[df_te.index]
df_te["pred_svr"] = pred_svr_series.loc[df_te.index]
y_test_sorted = df_te["bpm_watch"].values

pred_suave = (
    df_te.groupby(["participante", "seg_id"])["pred"]
    .transform(lambda s: s.rolling(window=W_SUAVIZADO, center=True, min_periods=1).mean())
    .values
)
resultados.append(metricas(y_test_sorted, pred_suave, f"RF + suavizado w={W_SUAVIZADO} (MODELO FINAL)"))

pred_svr_suave = (
    df_te.groupby(["participante", "seg_id"])["pred_svr"]
    .transform(lambda s: s.rolling(window=W_SUAVIZADO, center=True, min_periods=1).mean())
    .values
)
metricas(y_test_sorted, pred_svr_suave, f"SVR + suavizado w={W_SUAVIZADO} (comparacion)")

df_res = pd.DataFrame(resultados).sort_values("MAE")
print("\n" + "=" * 66)
print("  RESUMEN FINAL (ordenado por MAE)")
print("=" * 66)
print(df_res.to_string(index=False))

mejor = df_res.iloc[0]
print(f"\n  >>> MEJOR MAE DEL PROYECTO: {mejor['MAE']:.4f}  ({mejor['etapa']})")

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_res.to_csv(os.path.join(OUTPUT_DIR, "resultados_combinado_residual.csv"), index=False)
print(f"\n  resultados_combinado_residual.csv -> {OUTPUT_DIR}")

# ─── PREDICCIONES POR VENTANA DEL MODELO FINAL (para 08_figuras_tesis.py) ────
df_pred_final = pd.DataFrame({
    "participante": df_te["participante"].values,
    "seg_id":        df_te["seg_id"].values,
    "bpm_watch":      y_test_sorted,
    "pred_final":     pred_suave,
})
df_pred_final.to_csv(os.path.join(OUTPUT_DIR, "predicciones_finales_v19.csv"), index=False)
print(f"  predicciones_finales_v19.csv -> {OUTPUT_DIR}")

# ─── COMPARACION RF vs SVR (datos crudos; las figuras se generan en
#     08_figuras_tesis.py junto con el resto de las imagenes de la tesis) ─────
df_comp = pd.DataFrame({
    "participante": df_te["participante"].values,
    "seg_id":        df_te["seg_id"].values,
    "bpm_watch":     y_test_sorted,
    "pred_rf":       pred_suave,
    "pred_svr":      pred_svr_suave,
})
df_comp.to_csv(os.path.join(OUTPUT_DIR, "predicciones_comparacion_rf_svr.csv"), index=False)
print(f"  predicciones_comparacion_rf_svr.csv -> {OUTPUT_DIR}")

print("\nScript 19 completado.")
print("  Siguiente paso: python 08_figuras_tesis.py (genera todas las figuras)")
