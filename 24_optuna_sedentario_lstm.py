"""
24_optuna_sedentario_lstm.py
Tesis: Monitoreo no invasivo de la frecuencia cardiaca mediante senales Wi-Fi CSI
Universidad de Lima - Ingenieria de Sistemas
Autores: Yadhira Sarmiento Escobar (20214190) y Favio Chavarry Minaya (20214680)

PREREQUISITO: haber ejecutado (en orden):
  python 12_features_modelos_v3_synced.py   (genera dataset con pos_id y sync_ok)
  python 23_seleccion_mi.py                 (genera permutation_importance_features.csv)

Implementa 5 escenarios experimentales:
  A - Todas las posiciones (linea base para comparacion)
  B - Solo posiciones sedentarias (sin ejercicio ni alta actividad)
  C - Sedentario + sincronizacion estricta (sync_ok=True)
  D - Sedentario + optimizacion de hiperparametros con Optuna (RF, XGB, LGB)
  E - Sedentario + LSTM sobre secuencias de features

Metricas reportadas: MAE, RMSE, R2, r (Pearson)

Salidas generadas:
  dataset_sedentario_synced.csv
  diagnostico_sincronizacion.csv          (ya generado por 12, se usa aqui)
  resultados_optuna.csv
  resultados_lstm.csv
  comparacion_escenarios.csv
  predicciones_finales_mejoradas.csv
  fig_comparacion_escenarios.png
  fig_lstm_predicho_vs_real.png

Dependencias opcionales (el script corre aunque no esten):
  pip install optuna xgboost lightgbm torch
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Dependencias opcionales ────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  [AVISO] Optuna no instalado — Escenario D omitido. pip install optuna")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [AVISO] XGBoost no instalado. pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  [AVISO] LightGBM no instalado. pip install lightgbm")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except (ImportError, OSError) as e:
    HAS_TORCH = False
    print(f"  [AVISO] PyTorch no disponible ({type(e).__name__}: {e}) — Escenario E omitido.")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = r"C:\Users\LENOVO\Desktop\tesis-wifi-csi-frecuencia-cardiaca-main\resultados"
DATASET_CSV = os.path.join(OUTPUT_DIR, "dataset_completo_v3_synced.csv")
PI_CSV      = os.path.join(OUTPUT_DIR, "permutation_importance_features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── PARAMETROS ───────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_TOP_FEAT   = 10  # alineado con el modelo final vigente (19_combinado_residual.py)
W_SUAVIZADO  = 9

# Posiciones consideradas sedentarias en el protocolo eHealth CSI.
# Ajustar segun el dataset real. Valores tipicos (Galdino et al. 2023):
#   1-6:   sentado en diferentes orientaciones
#   11-14: acostado (supino, prono, lateral)
# Las posiciones 7-10 (de pie), 15 (transicion), 16-17 (activas) se excluyen.
POSICIONES_SEDENTARIAS = {1, 2, 3, 4, 5, 6, 11, 12, 13, 14}

# Optuna
OPTUNA_TRIALS    = 50    # reducir a 20 si el tiempo es limitado
OPTUNA_CV_SPLITS = 5

# LSTM
LSTM_SEQ_LEN  = 5      # numero de ventanas consecutivas como contexto
LSTM_HIDDEN   = 64
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.3
LSTM_EPOCHS   = 80
LSTM_PATIENCE = 15
LSTM_LR       = 1e-3
LSTM_BATCH    = 64

print("=" * 70)
print("  24 - ESCENARIOS SEDENTARIOS + OPTUNA + LSTM")
print("=" * 70)

# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────
if not os.path.exists(DATASET_CSV):
    print(f"\n  [ERROR] No se encontro {DATASET_CSV}")
    print("  Ejecuta primero: python 12_features_modelos_v3_synced.py")
    sys.exit(1)

df_base = pd.read_csv(DATASET_CSV)
print(f"  Dataset cargado: {len(df_base)} registros, {len(df_base.columns)} columnas")

# Verificar columna pos_id (solo disponible si se uso la version corregida de 12)
tiene_pos_id = "pos_id" in df_base.columns
tiene_sync_ok = "sync_ok" in df_base.columns
if not tiene_pos_id:
    print("  [AVISO] Columna pos_id no encontrada. Re-ejecuta 12_features_modelos_v3_synced.py")
    print("          Escenarios B/C usaran todos los registros (igual que A)")
if not tiene_sync_ok:
    print("  [AVISO] Columna sync_ok no encontrada.")
    print("          Escenario C usara todos los registros sedentarios sin filtro de sync")

# Cargar top-N features por permutation importance
META_COLS = {"seg_id", "win_start", "bpm_watch", "participante",
             "pos_id", "sync_ok", "min_dist_watch_s"}
ALL_FEAT  = [c for c in df_base.columns if c not in META_COLS]

feat_cols_base = ALL_FEAT   # fallback: todas las features
if os.path.exists(PI_CSV):
    pi_rank    = pd.read_csv(PI_CSV)
    feat_cols_base = [f for f in pi_rank["feature"].tolist()[:N_TOP_FEAT]
                      if f in df_base.columns]
    print(f"  Features (permutation importance top-{N_TOP_FEAT}): {feat_cols_base}")
else:
    print(f"  [AVISO] {PI_CSV} no encontrado — usando todas ({len(ALL_FEAT)}) las features")

# ─── FUNCIONES COMUNES ────────────────────────────────────────────────────────

def metricas_completas(y_true, y_pred, nombre=""):
    """MAE, RMSE, R2 y correlacion de Pearson."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae    = float(mean_absolute_error(y_true, y_pred))
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2     = float(r2_score(y_true, y_pred))
    try:
        r, _ = pearsonr(y_true, y_pred)
        r    = float(r)
    except Exception:
        r = float("nan")
    if nombre:
        print(f"  {nombre:<46} MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  r={r:.4f}")
    return {"escenario": nombre, "MAE": mae, "RMSE": rmse, "R2": r2, "r": r}


def split_por_grabacion(df, test_frac=0.20, seed=RANDOM_STATE):
    """Split 80/20 por grabacion (seg_id) dentro de cada participante, sin fuga."""
    rng = np.random.RandomState(seed)
    idx_tr, idx_te = [], []
    for part in sorted(df["participante"].unique()):
        dfp  = df[df["participante"] == part]
        segs = sorted(dfp["seg_id"].unique())
        n    = len(segs)
        if n < 2:
            idx_tr.extend(dfp.index.tolist())
            continue
        perm   = rng.permutation(n)
        n_test = max(1, int(round(n * test_frac)))
        te_segs = {segs[i] for i in perm[:n_test]}
        tr_segs = {segs[i] for i in perm[n_test:]}
        idx_tr.extend(dfp[dfp["seg_id"].isin(tr_segs)].index.tolist())
        idx_te.extend(dfp[dfp["seg_id"].isin(te_segs)].index.tolist())
    return idx_tr, idx_te


def suavizar(df_te, pred_raw, w=W_SUAVIZADO):
    """Media movil centrada agrupada por (participante, seg_id).

    OJO: el resultado se re-indexa a df_te.index antes de devolverlo. El
    rolling se calcula sobre las filas ordenadas por win_start (necesario
    para que la ventana movil respete el orden temporal), pero el caller
    siempre compara el resultado contra columnas de df_te en su orden
    ORIGINAL (ej. df_te["bpm_watch"].values) -- sin este reindex, cada
    prediccion quedaba emparejada con la fila equivocada.
    """
    tmp = df_te[["participante", "seg_id", "win_start"]].copy()
    tmp["pred"] = pd.Series(pred_raw, index=df_te.index).values
    tmp = tmp.sort_values(["participante", "seg_id", "win_start"])
    tmp["pred_suave"] = (
        tmp.groupby(["participante", "seg_id"])["pred"]
        .transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    )
    return tmp["pred_suave"].reindex(df_te.index).values


def entrenar_evaluar_rf(df_tr, df_te, feat_cols, params_rf=None, w=W_SUAVIZADO):
    """Entrena RF sobre residual, evalua con suavizado y devuelve metricas + predicciones."""
    if params_rf is None:
        params_rf = {"max_depth": 6, "max_features": "sqrt", "min_samples_leaf": 29,
                     "n_estimators": 387, "random_state": RANDOM_STATE, "n_jobs": -1}

    medias      = df_tr.groupby("participante")["bpm_watch"].mean()
    df_tr       = df_tr.copy()
    df_te       = df_te.copy()
    df_tr["media_p"] = df_tr["participante"].map(medias)
    df_te["media_p"] = df_te["participante"].map(medias).fillna(df_tr["bpm_watch"].mean())
    df_tr["residual"] = df_tr["bpm_watch"] - df_tr["media_p"]
    df_te["residual"] = df_te["bpm_watch"] - df_te["media_p"]

    sc    = StandardScaler().fit(df_tr[feat_cols].fillna(0))
    X_tr  = sc.transform(df_tr[feat_cols].fillna(0))
    X_te  = sc.transform(df_te[feat_cols].fillna(0))
    y_tr  = df_tr["residual"].values

    rf = RandomForestRegressor(**params_rf)
    rf.fit(X_tr, y_tr)

    pred_res  = rf.predict(X_te)
    pred_raw  = df_te["media_p"].values + pred_res
    pred_suav = suavizar(df_te, pred_raw, w)
    y_te      = df_te["bpm_watch"].values

    return pred_suav, y_te, rf, sc


# ═══════════════════════════════════════════════════════════════════════════════
#  ESCENARIO A — Todas las posiciones (referencia)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"  ESCENARIO A — Todas las posiciones (referencia con top-{N_TOP_FEAT} features)")
print("="*70)

df_A = df_base.copy()
idx_tr_A, idx_te_A = split_por_grabacion(df_A)
df_tr_A, df_te_A   = df_A.loc[idx_tr_A].copy(), df_A.loc[idx_te_A].copy()

pred_A, y_A, _, _ = entrenar_evaluar_rf(df_tr_A, df_te_A, feat_cols_base)
res_A = metricas_completas(y_A, pred_A, "A: Todas posiciones")

# ═══════════════════════════════════════════════════════════════════════════════
#  ESCENARIO B — Solo posiciones sedentarias
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"  ESCENARIO B — Posiciones sedentarias {sorted(POSICIONES_SEDENTARIAS)}")
print("="*70)

if tiene_pos_id:
    df_B = df_base[df_base["pos_id"].isin(POSICIONES_SEDENTARIAS)].copy()
    print(f"  Registros sedentarios: {len(df_B)}/{len(df_base)} ({100*len(df_B)/len(df_base):.1f}%)")
else:
    df_B = df_base.copy()
    print("  [AVISO] pos_id no disponible — Escenario B == Escenario A")

idx_tr_B, idx_te_B = split_por_grabacion(df_B)
df_tr_B, df_te_B   = df_B.loc[idx_tr_B].copy(), df_B.loc[idx_te_B].copy()

# Guardar dataset sedentario
df_B.to_csv(os.path.join(OUTPUT_DIR, "dataset_sedentario_synced.csv"), index=False)
print(f"  dataset_sedentario_synced.csv guardado")

pred_B, y_B, _, _ = entrenar_evaluar_rf(df_tr_B, df_te_B, feat_cols_base)
res_B = metricas_completas(y_B, pred_B, "B: Solo sedentarias")

# ═══════════════════════════════════════════════════════════════════════════════
#  ESCENARIO C — Sedentario + sincronizacion estricta (sync_ok=True)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  ESCENARIO C — Sedentario + sync estricta (solo ventanas con SW cercano)")
print("="*70)

if tiene_sync_ok and tiene_pos_id:
    df_C = df_base[
        df_base["pos_id"].isin(POSICIONES_SEDENTARIAS) &
        df_base["sync_ok"].astype(bool)
    ].copy()
elif tiene_sync_ok:
    df_C = df_base[df_base["sync_ok"].astype(bool)].copy()
elif tiene_pos_id:
    df_C = df_base[df_base["pos_id"].isin(POSICIONES_SEDENTARIAS)].copy()
    print("  [AVISO] sync_ok no disponible — Escenario C == Escenario B")
else:
    df_C = df_base.copy()
    print("  [AVISO] pos_id ni sync_ok disponibles — Escenario C == Escenario A")

print(f"  Registros tras sync estricta: {len(df_C)}")

if len(df_C) < 100:
    print("  [AVISO] Muy pocos registros tras filtro sync. Usando df_B como fallback.")
    df_C = df_B.copy()

idx_tr_C, idx_te_C = split_por_grabacion(df_C)
df_tr_C, df_te_C   = df_C.loc[idx_tr_C].copy(), df_C.loc[idx_te_C].copy()

pred_C, y_C, _, _ = entrenar_evaluar_rf(df_tr_C, df_te_C, feat_cols_base)
res_C = metricas_completas(y_C, pred_C, "C: Sedentario + sync estricta")

# ═══════════════════════════════════════════════════════════════════════════════
#  ESCENARIO D — Sedentario + Optuna (RF, XGB, LGB)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  ESCENARIO D — Sedentario + Optuna")
print("="*70)

resultados_optuna = []

# Preparar datos de D (mismo df_B, sin sync estricta para tener suficientes muestras)
medias_D  = df_tr_B.groupby("participante")["bpm_watch"].mean()
df_tr_D   = df_tr_B.copy()
df_te_D   = df_te_B.copy()
df_tr_D["media_p"]  = df_tr_D["participante"].map(medias_D)
df_te_D["media_p"]  = df_te_D["participante"].map(medias_D).fillna(df_tr_D["bpm_watch"].mean())
df_tr_D["residual"] = df_tr_D["bpm_watch"] - df_tr_D["media_p"]
df_te_D["residual"] = df_te_D["bpm_watch"] - df_te_D["media_p"]

sc_D    = StandardScaler().fit(df_tr_D[feat_cols_base].fillna(0))
X_tr_D  = sc_D.transform(df_tr_D[feat_cols_base].fillna(0))
X_te_D  = sc_D.transform(df_te_D[feat_cols_base].fillna(0))
y_tr_D  = df_tr_D["residual"].values
groups_D = df_tr_D["participante"].values


def cv_mae_rf(params, X, y, groups, n_splits=OPTUNA_CV_SPLITS):
    """GroupKFold CV (por participante) para evaluar hiperparametros sin fuga."""
    gkf  = GroupKFold(n_splits=n_splits)
    maes = []
    for idx_fit, idx_val in gkf.split(X, y, groups):
        m = RandomForestRegressor(**params)
        m.fit(X[idx_fit], y[idx_fit])
        maes.append(mean_absolute_error(y[idx_val], m.predict(X[idx_val])))
    return float(np.mean(maes))


if HAS_OPTUNA:
    # ── D.1: Optuna para RF ───────────────────────────────────────────────────
    print(f"  [D1] Optimizando RF con Optuna ({OPTUNA_TRIALS} trials)...")

    def objetivo_rf(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 700),
            "max_depth":       trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
            "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "random_state":    RANDOM_STATE,
            "n_jobs":          -1,
        }
        return cv_mae_rf(params, X_tr_D, y_tr_D, groups_D)

    study_rf = optuna.create_study(direction="minimize",
                                   sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(objetivo_rf, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_rf_params = study_rf.best_params
    best_rf_params.update({"random_state": RANDOM_STATE, "n_jobs": -1})
    print(f"  Mejores params RF: {best_rf_params}  (CV-MAE={study_rf.best_value:.4f})")

    rf_opt = RandomForestRegressor(**best_rf_params)
    rf_opt.fit(X_tr_D, y_tr_D)
    pred_raw_rf_opt  = df_te_D["media_p"].values + rf_opt.predict(X_te_D)
    pred_suav_rf_opt = suavizar(df_te_D, pred_raw_rf_opt)
    res_D_rf = metricas_completas(df_te_D["bpm_watch"].values, pred_suav_rf_opt,
                                   "D: RF + Optuna + suavizado")
    resultados_optuna.append({**res_D_rf, "modelo": "RF",
                               "best_params": str(best_rf_params),
                               "cv_mae": study_rf.best_value})

    # ── D.2: Optuna para XGBoost ──────────────────────────────────────────────
    if HAS_XGB:
        print(f"  [D2] Optimizando XGBoost con Optuna ({OPTUNA_TRIALS} trials)...")

        def objetivo_xgb(trial):
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 100, 700),
                "max_depth":       trial.suggest_int("max_depth", 3, 10),
                "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0,
            }
            gkf  = GroupKFold(n_splits=OPTUNA_CV_SPLITS)
            maes = []
            for idx_fit, idx_val in gkf.split(X_tr_D, y_tr_D, groups_D):
                m = XGBRegressor(**params)
                m.fit(X_tr_D[idx_fit], y_tr_D[idx_fit])
                maes.append(mean_absolute_error(y_tr_D[idx_val], m.predict(X_tr_D[idx_val])))
            return float(np.mean(maes))

        study_xgb = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study_xgb.optimize(objetivo_xgb, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        best_xgb_params = study_xgb.best_params
        best_xgb_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0})
        print(f"  Mejores params XGB: {best_xgb_params}  (CV-MAE={study_xgb.best_value:.4f})")

        xgb_opt = XGBRegressor(**best_xgb_params)
        xgb_opt.fit(X_tr_D, y_tr_D)
        pred_raw_xgb  = df_te_D["media_p"].values + xgb_opt.predict(X_te_D)
        pred_suav_xgb = suavizar(df_te_D, pred_raw_xgb)
        res_D_xgb = metricas_completas(df_te_D["bpm_watch"].values, pred_suav_xgb,
                                        "D: XGB + Optuna + suavizado")
        resultados_optuna.append({**res_D_xgb, "modelo": "XGBoost",
                                   "best_params": str(best_xgb_params),
                                   "cv_mae": study_xgb.best_value})

    # ── D.3: Optuna para LightGBM ─────────────────────────────────────────────
    if HAS_LGB:
        print(f"  [D3] Optimizando LightGBM con Optuna ({OPTUNA_TRIALS} trials)...")

        def objetivo_lgb(trial):
            params = {
                "n_estimators":   trial.suggest_int("n_estimators", 100, 700),
                "max_depth":      trial.suggest_int("max_depth", 3, 12),
                "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves":     trial.suggest_int("num_leaves", 10, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
                "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state":   RANDOM_STATE, "n_jobs": -1, "verbose": -1,
            }
            gkf  = GroupKFold(n_splits=OPTUNA_CV_SPLITS)
            maes = []
            for idx_fit, idx_val in gkf.split(X_tr_D, y_tr_D, groups_D):
                m = lgb.LGBMRegressor(**params)
                m.fit(X_tr_D[idx_fit], y_tr_D[idx_fit])
                maes.append(mean_absolute_error(y_tr_D[idx_val], m.predict(X_tr_D[idx_val])))
            return float(np.mean(maes))

        study_lgb = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study_lgb.optimize(objetivo_lgb, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        best_lgb_params = study_lgb.best_params
        best_lgb_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1})
        print(f"  Mejores params LGB: {best_lgb_params}  (CV-MAE={study_lgb.best_value:.4f})")

        lgb_opt = lgb.LGBMRegressor(**best_lgb_params)
        lgb_opt.fit(X_tr_D, y_tr_D)
        pred_raw_lgb  = df_te_D["media_p"].values + lgb_opt.predict(X_te_D)
        pred_suav_lgb = suavizar(df_te_D, pred_raw_lgb)
        res_D_lgb = metricas_completas(df_te_D["bpm_watch"].values, pred_suav_lgb,
                                        "D: LGB + Optuna + suavizado")
        resultados_optuna.append({**res_D_lgb, "modelo": "LightGBM",
                                   "best_params": str(best_lgb_params),
                                   "cv_mae": study_lgb.best_value})

    if resultados_optuna:
        df_optuna_res = pd.DataFrame(resultados_optuna)
        df_optuna_res.to_csv(os.path.join(OUTPUT_DIR, "resultados_optuna.csv"), index=False)
        print(f"  resultados_optuna.csv guardado")
        res_D = min(resultados_optuna, key=lambda x: x["MAE"])
    else:
        res_D = res_B.copy()
        res_D["escenario"] = "D: RF+Optuna (sin optuna instalado)"

else:
    print("  [OMITIDO] Optuna no disponible. Instala: pip install optuna")
    res_D = res_B.copy()
    res_D["escenario"] = "D: Optuna no disponible"

# ═══════════════════════════════════════════════════════════════════════════════
#  ESCENARIO E — Sedentario + LSTM
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  ESCENARIO E — Sedentario + LSTM (secuencias de features)")
print("="*70)

res_E = None
resultados_lstm = []

if not HAS_TORCH:
    print("  [OMITIDO] PyTorch no instalado. Instala: pip install torch")
    res_E = res_B.copy()
    res_E["escenario"] = "E: LSTM no disponible (sin PyTorch)"
else:
    # ── Preparar datos para LSTM (mismo df_B, split identico a B) ────────────
    medias_E  = df_tr_B.groupby("participante")["bpm_watch"].mean()
    df_tr_E   = df_tr_B.copy()
    df_te_E   = df_te_B.copy()
    df_tr_E["media_p"]  = df_tr_E["participante"].map(medias_E)
    df_te_E["media_p"]  = df_te_E["participante"].map(medias_E).fillna(df_tr_E["bpm_watch"].mean())
    df_tr_E["residual"] = df_tr_E["bpm_watch"] - df_tr_E["media_p"]
    df_te_E["residual"] = df_te_E["bpm_watch"] - df_te_E["media_p"]

    sc_E    = StandardScaler().fit(df_tr_E[feat_cols_base].fillna(0))
    arr_tr  = sc_E.transform(df_tr_E[feat_cols_base].fillna(0))
    arr_te  = sc_E.transform(df_te_E[feat_cols_base].fillna(0))
    n_feats = arr_tr.shape[1]

    df_tr_E["_feat_idx"] = np.arange(len(df_tr_E))
    df_te_E["_feat_idx"] = np.arange(len(df_te_E))

    def construir_secuencias(df_in, arr_feats, seq_len=LSTM_SEQ_LEN):
        """Secuencias de seq_len ventanas consecutivas del mismo segmento."""
        X_seqs, y_seqs = [], []
        cols_sort = ["participante", "seg_id", "win_start"]
        for part in df_in["participante"].unique():
            for seg in df_in[df_in["participante"] == part]["seg_id"].unique():
                rows = (df_in[(df_in["participante"] == part) &
                               (df_in["seg_id"] == seg)]
                        .sort_values("win_start"))
                fidxs = rows["_feat_idx"].values
                ys    = rows["residual"].values
                if len(fidxs) < seq_len:
                    continue
                for i in range(len(fidxs) - seq_len + 1):
                    X_seqs.append(arr_feats[fidxs[i: i + seq_len]])
                    y_seqs.append(ys[i + seq_len - 1])
        if not X_seqs:
            return np.empty((0, seq_len, n_feats)), np.empty(0)
        return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

    print("  Construyendo secuencias LSTM...")
    X_tr_seq, y_tr_seq = construir_secuencias(df_tr_E, arr_tr)
    X_te_seq, y_te_seq = construir_secuencias(df_te_E, arr_te)
    print(f"  Secuencias train: {X_tr_seq.shape}  |  test: {X_te_seq.shape}")

    if X_tr_seq.shape[0] < 32:
        print("  [AVISO] Muy pocas secuencias para entrenar LSTM. Saltando escenario E.")
        res_E = res_B.copy()
        res_E["escenario"] = "E: LSTM omitido (pocas secuencias)"
    else:
        # ── Definicion del modelo LSTM ─────────────────────────────────────────
        class LSTMHRModel(nn.Module):
            def __init__(self, input_size, hidden=LSTM_HIDDEN,
                         n_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, n_layers,
                                    batch_first=True,
                                    dropout=dropout if n_layers > 1 else 0.0)
                self.head = nn.Sequential(
                    nn.Linear(hidden, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        # ── Train/val interno (80/20 de las secuencias de entrenamiento) ──────
        n_tr = X_tr_seq.shape[0]
        n_val_lstm = max(1, int(n_tr * 0.2))
        perm_lstm  = np.random.RandomState(RANDOM_STATE).permutation(n_tr)
        idx_val_l  = perm_lstm[:n_val_lstm]
        idx_fit_l  = perm_lstm[n_val_lstm:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Dispositivo LSTM: {device}")

        def make_loader(X, y, batch=LSTM_BATCH, shuffle=True):
            ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                               torch.tensor(y, dtype=torch.float32))
            return DataLoader(ds, batch_size=batch, shuffle=shuffle)

        train_loader = make_loader(X_tr_seq[idx_fit_l], y_tr_seq[idx_fit_l])
        val_loader   = make_loader(X_tr_seq[idx_val_l], y_tr_seq[idx_val_l], shuffle=False)

        model     = LSTMHRModel(n_feats).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        criterion = nn.L1Loss()   # MAE loss, mas robusto a outliers de BPM

        best_val    = float("inf")
        patience_ct = 0
        best_state  = None
        history     = {"train": [], "val": []}

        print(f"  Entrenando LSTM ({LSTM_EPOCHS} epochs max, patience={LSTM_PATIENCE})...")
        for epoch in range(1, LSTM_EPOCHS + 1):
            # ── entrenamiento ──────────────────────────────────────────────────
            model.train()
            losses = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(loss.item())
            tr_loss = float(np.mean(losses))

            # ── validacion ────────────────────────────────────────────────────
            model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    val_losses.append(criterion(model(Xb), yb).item())
            vl_loss = float(np.mean(val_losses))
            history["train"].append(tr_loss)
            history["val"].append(vl_loss)
            scheduler.step(vl_loss)

            if vl_loss < best_val:
                best_val    = vl_loss
                patience_ct = 0
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ct += 1

            if epoch % 10 == 0:
                print(f"    Epoch {epoch:3d}/{LSTM_EPOCHS}  "
                      f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}")

            if patience_ct >= LSTM_PATIENCE:
                print(f"  Early stopping en epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # ── Prediccion en test ────────────────────────────────────────────────
        model.eval()
        X_te_t = torch.tensor(X_te_seq, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_res_lstm = model(X_te_t).cpu().numpy()

        # Mapear predicciones de secuencias de vuelta a ventanas de df_te_E
        # Cada secuencia predice el residual de la ultima ventana de la seq
        n_te = X_te_seq.shape[0]
        win_indices = []
        for part in df_te_E["participante"].unique():
            for seg in df_te_E[df_te_E["participante"] == part]["seg_id"].unique():
                rows = (df_te_E[(df_te_E["participante"] == part) &
                                 (df_te_E["seg_id"] == seg)]
                        .sort_values("win_start"))
                if len(rows) >= LSTM_SEQ_LEN:
                    for i in range(len(rows) - LSTM_SEQ_LEN + 1):
                        win_indices.append(rows.index[i + LSTM_SEQ_LEN - 1])

        df_lstm_pred = df_te_E.loc[win_indices].copy()
        df_lstm_pred["pred_res_lstm"]  = pred_res_lstm[:len(win_indices)]
        df_lstm_pred["pred_bpm_lstm"]  = (df_lstm_pred["media_p"].values +
                                           df_lstm_pred["pred_res_lstm"].values)
        df_lstm_pred = df_lstm_pred.sort_values(["participante", "seg_id", "win_start"])

        pred_suav_lstm = (
            df_lstm_pred.groupby(["participante", "seg_id"])["pred_bpm_lstm"]
            .transform(lambda s: s.rolling(W_SUAVIZADO, center=True, min_periods=1).mean())
            .values
        )
        y_lstm = df_lstm_pred["bpm_watch"].values
        res_E  = metricas_completas(y_lstm, pred_suav_lstm, "E: LSTM + suavizado")

        resultados_lstm.append({**res_E, "best_val_loss": best_val,
                                 "epochs_run": len(history["train"])})
        pd.DataFrame(resultados_lstm).to_csv(
            os.path.join(OUTPUT_DIR, "resultados_lstm.csv"), index=False)

        # ── Figura LSTM ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Curvas de aprendizaje
        axes[0].plot(history["train"], label="Train loss", color="#2196F3")
        axes[0].plot(history["val"],   label="Val loss",   color="#F44336")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MAE loss (residual BPM)")
        axes[0].set_title("Curvas de aprendizaje LSTM"); axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Predicho vs real (muestra de 300 puntos)
        sample_n = min(300, len(y_lstm))
        idx_s    = np.linspace(0, len(y_lstm) - 1, sample_n, dtype=int)
        axes[1].plot(y_lstm[idx_s],          color="gray", lw=1.2, label="BPM Real", alpha=0.8)
        axes[1].plot(pred_suav_lstm[idx_s],  color="#2196F3", lw=1.2,
                     label="BPM Predicho (LSTM)", alpha=0.9)
        mae_lstm = res_E["MAE"]
        r_lstm   = res_E["r"]
        axes[1].set_title(f"LSTM: BPM Predicho vs Real\nMAE={mae_lstm:.2f} | r={r_lstm:.3f}")
        axes[1].set_xlabel("Muestra"); axes[1].set_ylabel("BPM")
        axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fig_lstm_predicho_vs_real.png"),
                    dpi=130, bbox_inches="tight")
        plt.close()
        print("  fig_lstm_predicho_vs_real.png guardada")

# ═══════════════════════════════════════════════════════════════════════════════
#  RESUMEN COMPARATIVO
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  COMPARACION DE ESCENARIOS")
print("="*70)

comparacion = [res_A, res_B, res_C]

if HAS_OPTUNA and resultados_optuna:
    comparacion.extend(resultados_optuna)
else:
    comparacion.append(res_D)

if res_E is not None:
    comparacion.append(res_E)

df_comp = pd.DataFrame(comparacion)
df_comp = df_comp.sort_values("MAE").reset_index(drop=True)

# Columnas a mostrar
cols_show = [c for c in ["escenario", "MAE", "RMSE", "R2", "r"] if c in df_comp.columns]
print(df_comp[cols_show].to_string(index=False))

mejor = df_comp.iloc[0]
print(f"\n  >>> MEJOR MAE: {mejor['MAE']:.4f}  ({mejor['escenario']})")

# Referencia: modelo final de 19_combinado_residual.py
print(f"  >>> Referencia (script 19, todas posiciones): 9.1049 BPM")

df_comp.to_csv(os.path.join(OUTPUT_DIR, "comparacion_escenarios.csv"), index=False)
print(f"\n  comparacion_escenarios.csv guardado")

# ─── GUARDAR PREDICCIONES FINALES (mejor escenario encontrado) ────────────────
# Identificar mejor escenario entre A, B, C
mejor_esc = df_comp.iloc[0]["escenario"]
pred_final_map = {
    res_A["escenario"]: (pred_A, y_A, df_te_A),
    res_B["escenario"]: (pred_B, y_B, df_te_B),
    res_C["escenario"]: (pred_C, y_C, df_te_C),
}
if mejor_esc in pred_final_map:
    pred_final, y_final, df_te_final = pred_final_map[mejor_esc]
    df_pred_out = pd.DataFrame({
        "participante": df_te_final["participante"].values,
        "seg_id":       df_te_final["seg_id"].values,
        "bpm_watch":    y_final,
        "pred_final":   pred_final,
    })
    df_pred_out.to_csv(
        os.path.join(OUTPUT_DIR, "predicciones_finales_mejoradas.csv"), index=False)
    print(f"  predicciones_finales_mejoradas.csv guardado ({len(df_pred_out)} ventanas)")

# ─── FIGURA COMPARACION DE ESCENARIOS ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
esc_labels = [e.split(":")[0].strip() for e in df_comp["escenario"].tolist()]
maes       = df_comp["MAE"].values
colores    = ["#2196F3" if m == maes.min() else "#90CAF9" for m in maes]
bars       = ax.barh(esc_labels, maes, color=colores, edgecolor="white", height=0.6)

for bar, val in zip(bars, maes):
    ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9, fontweight="bold")

ax.axvline(9.1049, color="black", linestyle="--", linewidth=1.5,
           label="Referencia (script 19): 9.1049")
ax.set_xlabel("MAE (BPM)", fontsize=11)
ax.set_title("Comparacion de Escenarios — MAE en conjunto de test", fontweight="bold")
ax.invert_yaxis()
ax.legend(fontsize=9); ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_comparacion_escenarios.png"),
            dpi=130, bbox_inches="tight")
plt.close()
print("  fig_comparacion_escenarios.png guardada")

print(f"\n{'='*70}")
print("  Archivos generados en:", OUTPUT_DIR)
print("    dataset_sedentario_synced.csv")
if HAS_OPTUNA: print("    resultados_optuna.csv")
if HAS_TORCH:  print("    resultados_lstm.csv")
print("    comparacion_escenarios.csv")
print("    predicciones_finales_mejoradas.csv")
print("    fig_comparacion_escenarios.png")
if HAS_TORCH:  print("    fig_lstm_predicho_vs_real.png")
print("\nScript 24 completado.")
