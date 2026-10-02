"""
Modelos HISTORICOS (sin tuning nuevo) y post-procesado.

  fit_predict_residual <- cuerpo de experimentos_fase_C/E1_rf.run_33 (L164-171): residual sobre media_persona,
                          StandardScaler solo con train, fillna(0), suavizado centrado w=9.
  suavizar             <- E1_rf.py L151-156 (= 25_corridas_repetidas.py)
  causal_smooth, centered_smooth <- experimentos_fase_C/E5_generalizacion.py L48-55
Parametros: config.RF_PARAMS_HISTORICAL y config.SVR_PARAMS_HISTORICAL (dos variantes historicas; hay que elegir
una explicitamente). No son "best" ni "final".
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import RF_PARAMS_HISTORICAL, SMOOTH_WINDOW, SVR_PARAMS_HISTORICAL


def make_rf_historical():
    return RandomForestRegressor(**RF_PARAMS_HISTORICAL)


def make_svr_historical(variant):
    """variant: 'script25_33corridas' o 'script12_dataset' (sin valor por defecto a proposito)."""
    return SVR(**SVR_PARAMS_HISTORICAL[variant])


def suavizar(df_te, pred_raw, w=SMOOTH_WINDOW):
    tmp = df_te[["participante", "seg_id", "win_start"]].copy()
    tmp["pred"] = pd.Series(pred_raw, index=df_te.index).values
    tmp = tmp.sort_values(["participante", "seg_id", "win_start"])
    tmp["ps"] = tmp.groupby(["participante", "seg_id"])["pred"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    return tmp["ps"].reindex(df_te.index).values


def fit_predict_residual(df_tr, df_te, feat_cols, model, smooth=True):
    """Protocolo personalizado de E1/25. Devuelve (pred, media_persona_test)."""
    df_tr, df_te = df_tr.copy(), df_te.copy()
    medias = df_tr.groupby("participante")["bpm_watch"].mean()
    df_tr["media_persona"] = df_tr["participante"].map(medias)
    df_te["media_persona"] = df_te["participante"].map(medias).fillna(df_tr["bpm_watch"].mean())
    df_tr["residual"] = df_tr["bpm_watch"] - df_tr["media_persona"]
    sc = StandardScaler().fit(df_tr[feat_cols].fillna(0))
    model.fit(sc.transform(df_tr[feat_cols].fillna(0)), df_tr["residual"].values)
    raw = df_te["media_persona"].values + model.predict(sc.transform(df_te[feat_cols].fillna(0)))
    return (suavizar(df_te, raw) if smooth else raw), df_te["media_persona"].values


def fit_predict_absolute(D, tr, te, feat_cols, model):
    """Protocolo sujeto-independiente de E5 (L111-114): objetivo HR absoluto, scaler solo con train."""
    sc = StandardScaler().fit(D.loc[tr, feat_cols].fillna(0))
    model.fit(sc.transform(D.loc[tr, feat_cols].fillna(0)), D.loc[tr, "bpm_watch"].values)
    return model.predict(sc.transform(D.loc[te, feat_cols].fillna(0)))


def causal_smooth(df, col):
    d = df.sort_values(["captura", "win_start"])
    return d.groupby("captura")[col].transform(lambda s: s.expanding().mean()).reindex(df.index)


def centered_smooth(df, col, w=SMOOTH_WINDOW):
    d = df.sort_values(["participante", "seg_id", "win_start"])
    return d.groupby(["participante", "seg_id"])[col].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean()).reindex(df.index)
