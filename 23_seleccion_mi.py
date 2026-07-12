"""
23_seleccion_mi.py
Tesis: Monitoreo no invasivo de la frecuencia cardiaca mediante senales Wi-Fi CSI
Universidad de Lima - Ingenieria de Sistemas
Autores: Yadhira Sarmiento Escobar (20214190) y Favio Chavarry Minaya (20214680)

Escenario nuevo: 16_seleccion_features.py (ya descartado) eligio el top-N
features SOLO por feature_importances_ de un RF, que esta sesgado hacia
variables continuas de alta cardinalidad. Aqui se repite la busqueda de
N optimo pero con dos criterios alternativos:
  1. mutual_info_regression (no depende de un modelo, mide dependencia
     no lineal feature-target directamente).
  2. permutation_importance sobre el set de validacion interna (mide la
     caida real de desempeno al barajar cada feature, no solo el uso
     interno del arbol).

Se evalua top-N (N=5..29) con cada criterio sobre el modelo ganador
(RF tuneado + suavizado w=9, igual que en 19) y se compara contra el
top-15 por RF importance ya usado.

Salida: Data_DS1_raspberry-main/resultados_seleccion_mi.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR  = r"C:\Users\LENOVO\Desktop\tesis-wifi-csi-frecuencia-cardiaca-main\resultados"
DATASET_CSV = os.path.join(OUTPUT_DIR, "dataset_completo_v3_synced.csv")

RANDOM_STATE = 42
W_SUAVIZADO  = 9
N_LIST       = [5, 8, 10, 12, 15, 20, 25, 29]

RF_BEST = {
    "max_depth": 6, "max_features": "sqrt", "min_samples_leaf": 29,
    "n_estimators": 387, "random_state": RANDOM_STATE, "n_jobs": -1,
}

print("=" * 66)
print("  23 - SELECCION DE FEATURES por Mutual Information y Permutation Importance")
print("=" * 66)

if not os.path.exists(DATASET_CSV):
    print(f"\n  [ERROR] No se encontro {DATASET_CSV}")
    sys.exit(1)

df = pd.read_csv(DATASET_CSV)

ID_COLS  = {"seg_id", "win_start", "bpm_watch", "participante",
            "pos_id", "sync_ok", "min_dist_watch_s"}  # metadatos (12 ya los
            # trata como META_COLS, no como features predictoras); sync_ok y
            # min_dist_watch_s ademas no existirian en un despliegue real sin
            # smartwatch de referencia -> excluir del pool de seleccion
ALL_FEAT = [c for c in df.columns if c not in ID_COLS]
print(f"  Features disponibles: {len(ALL_FEAT)}\n")


def metricas(y_true, y_pred, nombre=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        r, _ = pearsonr(y_true, y_pred)
    except Exception:
        r = np.nan
    return {"etapa": nombre, "MAE": mae, "RMSE": rmse, "r": r}


# ─── SPLIT POR GRABACION COMPLETA (idéntico a 19) ─────────────────────────────
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

X_tr_full = df_tr[ALL_FEAT].fillna(0).values
X_te_full = df_te[ALL_FEAT].fillna(0).values
y_tr_res  = df_tr["residual"].values
y_te_res  = df_te["residual"].values
groups_tr = df_tr["participante"].values

sc_full = StandardScaler().fit(X_tr_full)
X_tr_full_sc = sc_full.transform(X_tr_full)
X_te_full_sc = sc_full.transform(X_te_full)

# ─── CRITERIO 1: MUTUAL INFORMATION ────────────────────────────────────────────
print("  Calculando mutual information (feature vs residual)...")
mi_scores = mutual_info_regression(X_tr_full_sc, y_tr_res, random_state=RANDOM_STATE)
mi_rank = pd.DataFrame({"feature": ALL_FEAT, "mi": mi_scores}).sort_values("mi", ascending=False)
print(mi_rank.head(10).to_string(index=False))

# ─── CRITERIO 2: PERMUTATION IMPORTANCE (sobre 1 fold de validacion interna) ──
print("\n  Calculando permutation importance (1 fold GroupKFold de validacion)...")
gkf = GroupKFold(n_splits=5)
idx_fit, idx_val = next(gkf.split(X_tr_full_sc, y_tr_res, groups=groups_tr))
rf_pi = RandomForestRegressor(**RF_BEST).fit(X_tr_full_sc[idx_fit], y_tr_res[idx_fit])
pi = permutation_importance(rf_pi, X_tr_full_sc[idx_val], y_tr_res[idx_val],
                             n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
pi_rank = pd.DataFrame({"feature": ALL_FEAT, "pi": pi.importances_mean}).sort_values("pi", ascending=False)
print(pi_rank.head(10).to_string(index=False))


def evaluar_top_n(feat_cols, etiqueta):
    sc = StandardScaler().fit(df_tr[feat_cols].fillna(0))
    X_tr = sc.transform(df_tr[feat_cols].fillna(0))
    X_te = sc.transform(df_te[feat_cols].fillna(0))
    rf = RandomForestRegressor(**RF_BEST).fit(X_tr, y_tr_res)
    pred_res = rf.predict(X_te)
    pred = df_te["media_persona"].values + pred_res

    df_tmp = df_te[["participante", "seg_id", "win_start"]].copy()
    df_tmp["bpm_watch"] = df_te["bpm_watch"].values
    df_tmp["pred"] = pred
    df_tmp = df_tmp.sort_values(["participante", "seg_id", "win_start"])
    pred_suave = (
        df_tmp.groupby(["participante", "seg_id"])["pred"]
        .transform(lambda s: s.rolling(window=W_SUAVIZADO, center=True, min_periods=1).mean())
        .values
    )
    return metricas(df_tmp["bpm_watch"].values, pred_suave, etiqueta)


resultados = []
print("\n  Evaluando top-N por cada criterio (RF + suavizado w=9)...\n")
for n in N_LIST:
    feat_mi = mi_rank["feature"].tolist()[:n]
    feat_pi = pi_rank["feature"].tolist()[:n]
    r_mi = evaluar_top_n(feat_mi, f"MI top-{n}")
    r_pi = evaluar_top_n(feat_pi, f"PermImp top-{n}")
    print(f"  {r_mi['etapa']:<16} MAE={r_mi['MAE']:.4f}   |   {r_pi['etapa']:<18} MAE={r_pi['MAE']:.4f}")
    resultados.append(r_mi)
    resultados.append(r_pi)

# Referencia: top-15 por RF importance (igual a script 19)
importancias = pd.read_csv(os.path.join(OUTPUT_DIR, "feature_importance_v3_synced.csv")).sort_values("importancia", ascending=False)
feat_rf15 = [f for f in importancias["feature"].tolist() if f in df.columns][:15]
r_ref = evaluar_top_n(feat_rf15, "RF-importance top-15 (referencia=19)")
resultados.append(r_ref)
print(f"\n  {r_ref['etapa']:<40} MAE={r_ref['MAE']:.4f}")

df_res = pd.DataFrame(resultados).sort_values("MAE")
print("\n" + "=" * 66)
print("  RESUMEN FINAL (ordenado por MAE)")
print("=" * 66)
print(df_res.to_string(index=False))

mejor = df_res.iloc[0]
print(f"\n  >>> MEJOR MAE: {mejor['MAE']:.4f}  ({mejor['etapa']})")
print(f"  >>> Comparado con el mejor previo (script 19): 9.1364")

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_res.to_csv(os.path.join(OUTPUT_DIR, "resultados_seleccion_mi.csv"), index=False)
mi_rank.to_csv(os.path.join(OUTPUT_DIR, "mi_ranking_features.csv"), index=False)
pi_rank.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance_features.csv"), index=False)
print(f"\n  resultados_seleccion_mi.csv -> {OUTPUT_DIR}")
print("\nScript 23 completado.")
