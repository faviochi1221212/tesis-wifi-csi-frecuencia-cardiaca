"""
Validacion ANIDADA limpia para el escenario PERSONALIZADO (calibracion individual).

Todo lo que se decide (selector de features, N, hiperparametros RF/SVR, ventana de suavizado) se decide con
datos del OUTER TRAIN unicamente, mediante INNER CV agrupada por captura. El OUTER TEST solo se usa al final
para predecir y medir. El protocolo y los espacios de busqueda se fijan en SEARCH antes de cualquier ejecucion.

Estrategia (definida ANTES de ejecutar, por coste computacional; ver PROTOCOL_NOTES):
  Etapa 1 (features + N + suavizado): para cada familia (RF, SVR) se evaluan 13 subconjuntos
          {MI, PI} x N in {5,8,10,12,15,20} + ALL con un modelo BASE fijo de esa familia, en las 5 inner folds,
          y con las 5 ventanas de suavizado (post-proceso barato). Se elige el subconjunto con menor MAE inner.
  Etapa 2 (hiperparametros + suavizado): con el subconjunto elegido se evaluan 40 (RF) / 50 (SVR) candidatos
          muestreados del espacio + configuraciones historicas como controles; se elige (params, w) con menor
          MAE inner. Criterio unico: MAE global de las ventanas de inner validation (pooled out-of-fold).
  Refit: ranking final = media de los scores de las 5 inner folds (solo datos permitidos); top-N; modelo con los
         params elegidos entrenado en TODO el outer train; prediccion del outer test; suavizado w dentro de captura.
"""
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import RF_PARAMS_HISTORICAL, STATIC_POSITIONS, SVR_PARAMS_HISTORICAL
from .features import FEAT_NAMES
from .models import suavizar

# ------------------------------------------------------------------ protocolo (fijado antes de ejecutar)
SEARCH = {
    "inner_folds": 5,
    "selectors": ["MI", "PI"],
    "n_features": [5, 8, 10, 12, 15, 20],          # + "ALL"
    "smooth_windows": [1, 3, 5, 7, 9],              # 1 = NO_SMOOTHING; >1 = NONCAUSAL_CENTERED dentro de captura
    "rf_space": {"n_estimators": [200, 400, 600, 800], "max_depth": [None, 4, 6, 8, 12],
                 "min_samples_leaf": [2, 4, 8, 16, 24, 32], "min_samples_split": [2, 4, 8, 16],
                 "max_features": ["sqrt", 0.5, 0.75, 1.0], "bootstrap": [True]},
    "n_random_rf": 40,
    "svr_space": {"C": [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100], "epsilon": [0.1, 0.25, 0.5, 0.75, 1.0, 2.0],
                  "gamma": ["scale", "auto", 0.001, 0.005, 0.01, 0.05, 0.1]},
    "n_random_svr": 50,
    # modelos BASE de la etapa 1 (neutros, no historicos) y del ranking PI
    "rf_base": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 8, "min_samples_split": 2,
                "max_features": "sqrt", "bootstrap": True},
    "svr_base": {"C": 1.0, "epsilon": 0.1, "gamma": "scale"},
    "pi_repeats": 5,
    "target": "residual = HR - media_persona(train)",
    "selection_metric": "MAE global (pooled) de las ventanas de inner validation",
}
PROTOCOL_NOTES = [
    "Busqueda en dos etapas (features/N/suavizado con modelo base fijo; luego hiperparametros + suavizado con el "
    "subconjunto elegido) en lugar del producto completo, por coste: ~2700 ajustes RF + ~3400 SVR por seed exterior "
    "(>40 h para 33 seeds). Decidido antes de ejecutar ningun outer test.",
    "Los rankings MI y PI se calculan una vez por inner fold y se comparten entre RF y SVR. PI usa el RF base "
    "(entrenado en inner train, importancia sobre inner validation), tambien para la familia SVR.",
    "Ranking final para el refit exterior: media de los scores de las 5 inner folds del metodo elegido.",
    "Historicos (RF_PARAMS_HISTORICAL; SVR C=1/eps=0.5 y C=10/eps=0.5) se anaden como candidatos de control sin privilegio.",
    "Empates: se elige el primero en el orden predefinido de candidatos.",
    "RF y SVR se reportan ambos; no hay seleccion entre familias.",
]
META_COLUMNS = {"participante", "pos_id", "seg_id", "win_start", "bpm_watch", "sync_ok", "media_persona", "pred",
                "corrida", "arm", "row_index", "captura", "group_id", "min_dist_watch_s", "fs_real", "timestamp"}
FORBIDDEN_TOKENS = ("watch", "heart", "target", "label", "media_persona", "residual", "pred", "sync", "hr_real")


def smoothing_label(w):
    return "NO_SMOOTHING" if w == 1 else f"NONCAUSAL_CENTERED_w{w}"


# ------------------------------------------------------------------ datos
def candidate_features(columns):
    """ALL_E1A_FEATURES: las 24 features de extraer_features_ventana presentes y que no son metadatos."""
    return [c for c in FEAT_NAMES if c in columns and c not in META_COLUMNS]


def audit_features(d, feats, max_abs_r=0.95):
    """Auditoria de leakage de las features candidatas. Devuelve dict; lanza si detecta leakage.
    (1) nombres: ningun token de etiqueta/identificador; (2) procedencia: extraer_features_ventana(senal, fs) solo
    recibe la senal CSI; (3) numerica: |r| con bpm_watch < 0.95 y ninguna columna identica a bpm_watch.
    La correlacion es solo un chequeo de auditoria (no se usa para seleccionar)."""
    rep = {"n_candidate_features": len(feats), "candidate_features": list(feats), "sospechosas": []}
    for f in feats:
        r = float(np.corrcoef(d[f].astype(float), d.bpm_watch.astype(float))[0, 1]) if d[f].std() > 0 else 0.0
        same = bool(np.allclose(d[f].values, d.bpm_watch.values))
        bad_name = any(t in f.lower() for t in FORBIDDEN_TOKENS) or f in META_COLUMNS
        rep.setdefault("pearson_con_bpm_watch", {})[f] = r
        if bad_name or same or abs(r) >= max_abs_r:
            rep["sospechosas"].append({"feature": f, "r": r, "identica_a_label": same, "nombre_prohibido": bad_name})
    rep["procedencia"] = "extraer_features_ventana(senal, fs): solo senal CSI agregada y fs; no recibe HR ni timestamps del reloj"
    rep["PASS"] = not rep["sospechosas"]
    if not rep["PASS"]:
        raise RuntimeError(f"DETENIDO: features sospechosas de leakage: {rep['sospechosas']}")
    return rep


# ------------------------------------------------------------------ folds internos (agrupados por captura)
def inner_folds(dtr, k, seed):
    """Asigna CAPTURAS completas a k folds, repartiendo las capturas de cada participante entre folds
    (simula dentro del train la calibracion personalizada del outer test). Devuelve [(idx_train, idx_val)]."""
    rng = np.random.default_rng(seed)
    caps = dtr[["participante", "seg_id"]].drop_duplicates().sort_values(["participante", "seg_id"])
    assign = {}
    for part, g in caps.groupby("participante", sort=True):
        segs = g.seg_id.values[rng.permutation(len(g))]; off = int(rng.integers(k))
        for i, s in enumerate(segs):
            assign[(part, s)] = (off + i) % k
    fold = np.array([assign[(p, s)] for p, s in zip(dtr.participante, dtr.seg_id)])
    out = []
    for f in range(k):
        tr, va = dtr.index[fold != f], dtr.index[fold == f]
        ct = set(zip(dtr.loc[tr, "participante"], dtr.loc[tr, "seg_id"])); cv = set(zip(dtr.loc[va, "participante"], dtr.loc[va, "seg_id"]))
        assert not (ct & cv), "captura compartida inner train/val"
        assert not (set(tr) & set(va)), "ventana compartida inner train/val"
        out.append((tr, va))
    assert sorted(np.concatenate([va for _, va in out])) == sorted(dtr.index), "las inner val no cubren el outer train"
    return out


def capture_ids(df):
    return sorted({f"{p}_{s}" for p, s in zip(df.participante, df.seg_id)})


# ------------------------------------------------------------------ media personal (solo labels de train)
def personal_mean(d_train):
    return d_train.groupby("participante")["bpm_watch"].mean(), float(d_train["bpm_watch"].mean())


def personal_base(d_apply, means, glob, d_train=None):
    if d_train is not None:   # guard: el conjunto donde se aplica no comparte capturas con el que define la media
        assert not (set(zip(d_train.participante, d_train.seg_id)) & set(zip(d_apply.participante, d_apply.seg_id)))
    return d_apply["participante"].map(means).fillna(glob).values


# ------------------------------------------------------------------ modelos
def make_model(family, params, seed):
    if family == "RF":
        p = {k: v for k, v in params.items() if k not in ("random_state", "n_jobs")}
        return make_pipeline(StandardScaler(), RandomForestRegressor(**p, random_state=seed, n_jobs=1))
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", **{k: v for k, v in params.items() if k != "kernel"}))


def candidates(family, seed):
    if family == "RF":
        c = [dict(p) for p in ParameterSampler(SEARCH["rf_space"], n_iter=SEARCH["n_random_rf"], random_state=seed)]
        c.append({k: v for k, v in RF_PARAMS_HISTORICAL.items() if k not in ("random_state", "n_jobs")} | {"_control": "RF_PARAMS_HISTORICAL"})
    else:
        c = [dict(p) for p in ParameterSampler(SEARCH["svr_space"], n_iter=SEARCH["n_random_svr"], random_state=seed)]
        for name, p in SVR_PARAMS_HISTORICAL.items():
            c.append({k: v for k, v in p.items() if k != "kernel"} | {"_control": f"SVR_HISTORICAL_{name}"})
    return c


def _clean(params):
    return {k: v for k, v in params.items() if not k.startswith("_")}


# ------------------------------------------------------------------ preparacion de folds
def prepare_folds(dtr, feats, k, seed):
    folds = []
    for f, (tr, va) in enumerate(inner_folds(dtr, k, seed)):
        a, b = dtr.loc[tr], dtr.loc[va]
        means, glob = personal_mean(a)                              # SOLO inner train
        base_tr = personal_base(a, means, glob)
        base_va = personal_base(b, means, glob, d_train=a)
        folds.append({"fold": f, "idx_tr": tr, "idx_va": va, "Xtr": a[feats].fillna(0).values, "Xva": b[feats].fillna(0).values,
                      "ytr_res": a.bpm_watch.values - base_tr, "base_va": base_va, "yva": b.bpm_watch.values,
                      "meta_va": b[["participante", "seg_id", "win_start"]].copy(),
                      "captures_tr": capture_ids(a), "captures_va": capture_ids(b)})
    return folds


def fold_rankings(fold, seed, rf_base, pi_repeats):
    """Scores MI (inner train) y PI (RF base en inner train, importancia en inner val). Datos permitidos unicamente."""
    sc = StandardScaler().fit(fold["Xtr"])
    mi = mutual_info_regression(sc.transform(fold["Xtr"]), fold["ytr_res"], random_state=seed)
    m = make_model("RF", rf_base, seed).fit(fold["Xtr"], fold["ytr_res"])
    pi = permutation_importance(m, fold["Xva"], fold["yva"] - fold["base_va"], scoring="neg_mean_absolute_error",
                                n_repeats=pi_repeats, random_state=seed, n_jobs=1).importances_mean
    return {"MI": np.asarray(mi, float), "PI": np.asarray(pi, float)}


def all_fold_rankings(folds, seed, n_jobs):
    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fold_rankings)(f, seed, SEARCH["rf_base"], SEARCH["pi_repeats"]) for f in folds)


def top_n(scores, n):
    order = np.argsort(-scores, kind="stable")
    return order if n == "ALL" else order[:n]


def subsets_list(n_feats):
    ns = [n for n in SEARCH["n_features"] if n < n_feats]
    return [(m, n) for m in SEARCH["selectors"] for n in ns] + [("ALL", "ALL")]


def _fit_predict_fold(family, params, seed, fold, cols):
    m = make_model(family, _clean(params), seed).fit(fold["Xtr"][:, cols], fold["ytr_res"])
    return fold["base_va"] + m.predict(fold["Xva"][:, cols])


def _pooled_mae_by_w(folds, preds):
    out = {}
    for w in SEARCH["smooth_windows"]:
        err = [np.abs(f["yva"] - (p if w == 1 else suavizar(f["meta_va"], p, w))) for f, p in zip(folds, preds)]
        out[w] = float(np.mean(np.concatenate(err)))
    return out


def _subset_cols(fold_rank, method, n, n_feats):
    return np.arange(n_feats) if method == "ALL" else top_n(fold_rank[method], n)


def select_and_tune(dtr, feats, family, seed, n_jobs, folds=None, ranks=None, log=print):
    """Toda la seleccion dentro del OUTER TRAIN (dtr). Devuelve la configuracion congelada y las tablas inner."""
    t0 = time.time()
    k = SEARCH["inner_folds"]
    folds = folds if folds is not None else prepare_folds(dtr, feats, k, seed)
    ranks = ranks if ranks is not None else all_fold_rankings(folds, seed, n_jobs)
    nf = len(feats)
    # ---- etapa 1: subconjunto de features (modelo base) x suavizado
    subs = subsets_list(nf)
    base = SEARCH["rf_base"] if family == "RF" else SEARCH["svr_base"]
    jobs = [(si, fi) for si in range(len(subs)) for fi in range(len(folds))]
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_predict_fold)(family, base, seed, folds[fi], _subset_cols(ranks[fi], subs[si][0], subs[si][1], nf)) for si, fi in jobs)
    stage1 = []
    for si, (m, n) in enumerate(subs):
        preds = [res[jobs.index((si, fi))] for fi in range(len(folds))]
        for w, mae in _pooled_mae_by_w(folds, preds).items():
            stage1.append({"selector": m, "N": n, "w": w, "inner_mae": mae})
    s1 = pd.DataFrame(stage1)
    best1 = s1.iloc[int(np.argmin(s1.inner_mae.values))]
    method, n = best1.selector, best1.N
    log(f"    [{family}] etapa 1: selector={method} N={n} (w={best1.w}) inner MAE={best1.inner_mae:.4f} ({time.time()-t0:.0f}s)")
    # ---- etapa 2: hiperparametros x suavizado sobre el subconjunto elegido
    cands = candidates(family, seed)
    cols_f = [_subset_cols(ranks[fi], method, n, nf) for fi in range(len(folds))]
    jobs2 = [(ci, fi) for ci in range(len(cands)) for fi in range(len(folds))]
    res2 = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_predict_fold)(family, cands[ci], seed, folds[fi], cols_f[fi]) for ci, fi in jobs2)
    stage2 = []
    for ci, c in enumerate(cands):
        preds = [res2[ci * len(folds) + fi] for fi in range(len(folds))]
        for w, mae in _pooled_mae_by_w(folds, preds).items():
            stage2.append({"cand": ci, "params": {kk: (str(v) if v is None else v) for kk, v in c.items()}, "w": w, "inner_mae": mae})
    s2 = pd.DataFrame(stage2)
    b2 = s2.iloc[int(np.argmin(s2.inner_mae.values))]
    chosen_params = cands[int(b2.cand)]
    # ---- ranking final para el refit exterior (media de scores inner del metodo elegido)
    if method == "ALL":
        final_idx = list(range(nf)); final_scores = None
    else:
        final_scores = np.mean([r[method] for r in ranks], axis=0)
        final_idx = [int(i) for i in top_n(final_scores, n)]
    sel = {"family": family, "selector": method, "N": n if n == "ALL" else int(n), "features": [feats[i] for i in final_idx],
           "params": _clean(chosen_params), "params_control": chosen_params.get("_control"),
           "smoothing_w": int(b2.w), "smoothing": smoothing_label(int(b2.w)), "inner_mae": float(b2.inner_mae),
           "final_ranking_scores": None if final_scores is None else dict(zip(feats, map(float, final_scores))),
           "stage1": stage1, "stage2": stage2, "t_seleccion_s": round(time.time() - t0, 1)}
    log(f"    [{family}] etapa 2: params={sel['params']} w={sel['smoothing_w']} inner MAE={sel['inner_mae']:.4f} ({time.time()-t0:.0f}s)")
    return sel


def fit_outer_and_predict(dtr, dte, feats_sel, family, params, w, seed):
    """Refit en TODO el outer train y prediccion UNICA del outer test (sus labels no se usan aqui)."""
    means, glob = personal_mean(dtr)
    base_tr = personal_base(dtr, means, glob)
    base_te = personal_base(dte, means, glob, d_train=dtr)
    m = make_model(family, params, seed).fit(dtr[feats_sel].fillna(0).values, dtr.bpm_watch.values - base_tr)
    raw = base_te + m.predict(dte[feats_sel].fillna(0).values)
    pred = raw if w == 1 else suavizar(dte[["participante", "seg_id", "win_start"]], raw, w)
    scaler = m.named_steps["standardscaler"]
    return pred, base_te, {"scaler_mean_equals_train_mean": bool(np.allclose(scaler.mean_, dtr[feats_sel].fillna(0).values.mean(0)))}


# ------------------------------------------------------------------ metricas de evaluacion (solo reporte)
def eval_metrics(y, p, subj):
    y, p = np.asarray(y, float), np.asarray(p, float); a = np.abs(p - y)
    df = pd.DataFrame({"y": y, "p": p, "s": np.asarray(subj)})
    yc = df.y - df.groupby("s").y.transform("mean"); pc = df.p - df.groupby("s").p.transform("mean")
    return {"MAE": float(a.mean()), "MAE_macro_sujeto": float(df.assign(a=a).groupby("s").a.mean().mean()),
            "RMSE": float(np.sqrt(np.mean((p - y) ** 2))), "MedAE": float(np.median(a)),
            "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else float("nan"),
            "pearson_dentro_sujeto": float(np.corrcoef(yc, pc)[0, 1]) if np.std(pc) > 0 else float("nan"), "n": int(len(y))}


def check_static(d):
    assert set(d.pos_id.astype(int).unique()) <= set(STATIC_POSITIONS)


def n_jobs_default():
    return max(1, (os.cpu_count() or 2) - 1)
