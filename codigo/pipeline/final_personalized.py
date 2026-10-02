"""
final_personalized.py -- entrypoint del pipeline oficial PERSONALIZADO (calibracion individual).

Protocolo (el mismo sujeto puede estar en train y test; nunca la misma captura):

    datos (dataset publicado + features E1a)
      -> STATIC_POSITIONS {1,2,4,5,6,7,8,9,10,11,12,13}
      -> split por captura dentro de cada sujeto (TEST_FRAC 0.20, semillas 0..32)   [assert shared_captures == 0]
      -> TRAIN: seleccion de features / N / hiperparametros / suavizado SOLO con train (inner CV por captura)
      -> TEST (una sola prediccion por seed y familia)
      -> RF / SVR sobre residual (HR - media_persona de train), StandardScaler solo train
      -> MAE

Modos:
  --historical-baseline   configuracion HISTORICA (features N=10, RF_PARAMS_HISTORICAL, SVR_PARAMS_HISTORICAL, w=9)
                          [--dry-run valida datos, posiciones y splits sin entrenar]
  --clean                 EXPERIMENTO FINAL LIMPIO con validacion anidada (csi_hr.nested); guardado incremental por
                          seed en resultados/final_personalized/, --resume, --preflight (seeds 0 y 1)
  --summarize             agrega las seeds completas: comparison.csv, mae_por_respiracion.csv, mae_por_posicion.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # para 'import pipeline.reproduce_840'

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from csi_hr import config as C, metrics as M, models, splits  # noqa: E402

WARNING = ("Historical baseline configuration.\n"
           "Feature selection and hyperparameter tuning have not yet been rebuilt with\n"
           "nested/train-only validation. Do not report this run as the final clean thesis result.")
SAVED_FEATURES = C.E1_DIR / "features_rf_3brazos.csv.gz"
OUT_DIR = C.RESULTS_ROOT / "pipeline_runs" / "final_personalized"


# ------------------------------------------------------------------ pipeline limpio (validacion anidada, csi_hr.nested)
# La seleccion de features y el tuning se hacen conjuntamente en dos etapas dentro del outer train
# (ver csi_hr.nested.SEARCH / PROTOCOL_NOTES). Estas tres funciones son la interfaz explicita.
def select_features_train_only(df_train, candidate_features, family="RF", seed=0, n_jobs=None):
    """Etapa 1 (+2) dentro de df_train: selector, N y suavizado. Devuelve la seleccion congelada."""
    from csi_hr import nested as N
    return N.select_and_tune(df_train, list(candidate_features), family, seed, n_jobs or N.n_jobs_default())


def tune_rf_train_only(df_train, features, seed=0, n_jobs=None):
    return select_features_train_only(df_train, features, "RF", seed, n_jobs)


def tune_svr_train_only(df_train, features, seed=0, n_jobs=None):
    return select_features_train_only(df_train, features, "SVR", seed, n_jobs)


CLEAN_DIR = C.RESULTS_ROOT / "final_personalized"
FAMILIES = ("RF", "SVR")


def load_clean_data():
    """df_C (STATIC_POSITIONS + sync_ok, indice del dataset publicado) con TODAS las features E1a candidatas."""
    from csi_hr import nested as N
    dC = splits.load_df_C()
    F = pd.read_csv(SAVED_FEATURES, dtype={"participante": str}); F = F[F.arm == "E1a"].drop(columns=["arm"])
    F["participante"] = F.participante.astype(str).str.zfill(3)
    feats = N.candidate_features(F.columns)
    m = dC[splits.KEYS].reset_index().merge(F[splits.KEYS + feats], on=splits.KEYS, how="left").set_index("index")
    if m[feats].isna().any().any():
        raise RuntimeError("ventanas sin features E1a")
    d = dC[splits.KEYS + ["bpm_watch", "sync_ok"]].copy(); d[feats] = m.loc[d.index, feats].values
    N.check_static(d)
    return d, feats


def outer_splits(d, n_runs=C.N_RUNS):
    """Splits externos HISTORICOS (splits_rf.json), verificados contra split_por_grabacion y sin capturas compartidas."""
    saved = json.load(open(C.E1_DIR / "splits_rf.json"))["splits"]
    out = {}
    for r in range(n_runs):
        tr, te = saved[str(r)]["train"], saved[str(r)]["test"]
        assert (tr, te) == splits.split_por_grabacion(d, r), f"split {r} distinto al historico"
        chk = splits.check_personalized(d, tr, te)
        out[r] = (tr, te, chk)
    return out


def _git_commit():
    import subprocess
    root = str(C.PROJECT_ROOT)
    h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "pipeline"], cwd=root, capture_output=True, text=True).stdout.strip())
    return {"commit": h, "src_o_pipeline_con_cambios_sin_commit": dirty}


class _RamMonitor:
    """Pico de memoria fisica usada del SISTEMA (incluye workers loky) durante la ejecucion, via GlobalMemoryStatusEx."""
    def __init__(self):
        import threading
        self.peak_gb, self.base_gb, self._stop = 0.0, self._used(), threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _used():
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        try:
            s = MS(); s.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return (s.ullTotalPhys - s.ullAvailPhys) / 1e9
        except Exception:
            return float("nan")

    def _run(self):
        while not self._stop.wait(1.0):
            self.peak_gb = max(self.peak_gb, self._used())

    def __enter__(self):
        self._t.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join()


def _json_dump(obj, path):
    """Escritura atomica (tmp + os.replace): un corte a mitad nunca deja un JSON truncado con el nombre final."""
    import os
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    os.replace(tmp, path)


def _csv_gz_dump(df, path):
    import os
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def _json_load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def seed_complete(out_dir, seed, families=FAMILIES):
    return all((Path(out_dir) / fam.lower() / f"seed_{seed:02d}.json").exists() for fam in families)


def run_clean(seeds, out_dir=CLEAN_DIR, resume=False, n_jobs=None, data=None, log=print, outer=None):
    """Ejecuta el protocolo limpio para las seeds dadas. Cada seed se guarda completa antes de pasar a la siguiente."""
    from csi_hr import nested as N
    out_dir = Path(out_dir); n_jobs = n_jobs or N.n_jobs_default()
    d, feats = data if data is not None else load_clean_data()
    audit, outer = init_clean_out_dir(d, feats, out_dir, n_jobs, outer=outer, log=log)
    for seed in seeds:
        if resume and seed_complete(out_dir, seed):
            log(f"seed {seed:02d}: ya completa, se omite (--resume)"); continue
        audit = run_seed(d, feats, seed, outer[seed], out_dir, n_jobs, FAMILIES, log=log, audit=audit)
    return audit


def init_clean_out_dir(d, feats, out_dir, n_jobs, outer=None, log=print):
    """Prepara out_dir: subcarpetas, config.json (rechaza un cambio del espacio de busqueda a mitad de ejecucion),
    outer_splits.json y la auditoria existente. Devuelve (audit, outer)."""
    from csi_hr import nested as N
    out_dir = Path(out_dir)
    for sub in ("inner_splits", "rf", "svr", "selections", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    audit_feats = N.audit_features(d, feats)
    log(f"n_candidate_features = {len(feats)}\ncandidate_features = {feats}")
    cfg = {"escenario": "PERSONALIZADO / CALIBRACION INDIVIDUAL", "STATIC_POSITIONS": sorted(C.STATIC_POSITIONS),
           "representacion": "E1a (54 activas, |H| crudo), preprocesamiento validado, ventana 231 / paso 57",
           "outer": "splits historicos resultados/fase_C/E1/splits_rf.json (split por captura dentro de participante, TEST_FRAC 0.20)",
           "search": N.SEARCH, "protocol_notes": N.PROTOCOL_NOTES, "candidate_features": feats, "git": _git_commit(),
           "n_jobs": n_jobs, "historico_referencia": {"MAE_RF_E1a": C.MAE_E1A_HISTORICAL, "etiqueta": "HISTORICAL, NOT CLEANLY SELECTED"}}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        prev = json.loads(cfg_path.read_text(encoding="utf-8"))
        if json.dumps(prev["search"], sort_keys=True, default=str) != json.dumps(json.loads(json.dumps(N.SEARCH, default=str)), sort_keys=True, default=str):
            raise RuntimeError("El espacio de busqueda difiere del config.json existente: no se permite cambiar el protocolo a mitad de ejecucion")
        cfg["git_ejecuciones_previas"] = prev.get("git_ejecuciones_previas", []) + [prev["git"]]
    _json_dump(cfg, cfg_path)
    outer = outer if outer is not None else outer_splits(d)
    _json_dump({str(r): {"train_captures": N.capture_ids(d.loc[tr]), "test_captures": N.capture_ids(d.loc[te]), **chk}
                for r, (tr, te, chk) in outer.items()}, out_dir / "outer_splits.json")
    audit_path = out_dir / "leakage_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"features": audit_feats, "seeds": {}}
    return audit, outer


def run_seed(d, feats, seed, split, out_dir, n_jobs, families=FAMILIES, log=print, audit=None, timings=None, n_jobs_for=None):
    """Una seed exterior del protocolo limpio para las familias pedidas (misma secuencia que run_clean):
    folds inner -> rankings -> seleccion+tuning de CADA familia pedida -> seleccion congelada en disco -> refit y
    prediccion unica del outer test -> resultado de cada familia guardado en cuanto termina.
    Con un subconjunto de familias (p. ej. solo SVR tras un corte) las selecciones/tablas/auditoria de las familias
    ya guardadas se conservan (merge), no se sobrescriben. timings (dict) recibe duraciones por etapa;
    n_jobs_for(etapa) permite al llamador bajar el paralelismo antes de cada etapa (control de RAM)."""
    from csi_hr import nested as N
    out_dir = Path(out_dir); families = tuple(families)
    timings = timings if timings is not None else {}
    nj = n_jobs_for or (lambda stage: n_jobs)
    audit_path = out_dir / "leakage_audit.json"
    if audit is None:
        audit = _json_load(audit_path) or {"features": N.audit_features(d, feats), "seeds": {}}
    t0 = time.time()
    tr, te, chk = split
    dtr, dte = d.loc[tr], d.loc[te]
    assert not (set(N.capture_ids(dtr)) & set(N.capture_ids(dte))) and not (set(tr) & set(te))
    log(f"=== seed {seed:02d}: outer train {len(dtr)} ventanas / test {len(dte)} ventanas")
    with _RamMonitor() as ram:
        folds = N.prepare_folds(dtr, feats, N.SEARCH["inner_folds"], seed)
        _json_dump({"seed": seed, "folds": [{"fold": f["fold"], "train_captures": f["captures_tr"], "val_captures": f["captures_va"],
                                             "n_train": len(f["idx_tr"]), "n_val": len(f["idx_va"]),
                                             "shared_captures": len(set(f["captures_tr"]) & set(f["captures_va"]))} for f in folds]},
                   out_dir / "inner_splits" / f"seed_{seed:02d}.json")
        ranks = N.all_fold_rankings(folds, seed, nj("rankings"))
        timings["rankings_s"] = round(time.time() - t0, 1)
        sels = {}
        for fam in families:
            t1 = time.time()
            sels[fam] = N.select_and_tune(dtr, feats, fam, seed, nj(fam), folds=folds, ranks=ranks, log=log)
            timings[f"{fam}_seleccion_s"] = round(time.time() - t1, 1)
        t_frozen = time.time()
        # --- seleccion CONGELADA y guardada ANTES de tocar el outer test (merge con familias ya guardadas)
        fold_rankings = [{m: dict(zip(feats, map(float, r[m]))) for m in r} for r in ranks]
        sel_path = out_dir / "selections" / f"seed_{seed:02d}_features.json"
        prev = _json_load(sel_path, {})
        if prev.get("fold_rankings") is not None and prev["fold_rankings"] != fold_rankings:
            raise RuntimeError(f"seed {seed}: los rankings inner recalculados difieren de los guardados (no determinista)")
        _json_dump({**prev, "seed": seed, "congelada_antes_de_outer_test": True, "t_congelada": t_frozen, "fold_rankings": fold_rankings,
                    **{fam: {k: v for k, v in s.items() if k not in ("stage1", "stage2")} for fam, s in sels.items()}},
                   sel_path)
        tab_path = out_dir / "logs" / f"seed_{seed:02d}_inner_tables.json"
        _json_dump({**_json_load(tab_path, {}), **{fam: {"stage1": s["stage1"], "stage2": s["stage2"]} for fam, s in sels.items()}}, tab_path)
        seed_audit = {**audit["seeds"].get(str(seed), {}),
                      "outer_shared_captures": chk["capturas_compartidas"], "outer_shared_windows": chk["ventanas_compartidas"],
                      "outer_shared_subjects": chk["sujetos_compartidos"],
                      "inner_shared_captures_max": max(len(set(f["captures_tr"]) & set(f["captures_va"])) for f in folds),
                      "inner_folds_solo_outer_train": bool(set(np.concatenate([f["idx_va"] for f in folds])) == set(tr)),
                      "outer_test_en_inner": bool(set(np.concatenate([f["idx_va"] for f in folds])) & set(te)),
                      "seleccion_guardada_antes_de_predecir_test": True}
        for fam, s in sels.items():
            t1 = time.time()
            pred, base_te, sc = N.fit_outer_and_predict(dtr, dte, s["features"], fam, s["params"], s["smoothing_w"], seed)
            out = dte[splits.KEYS + ["bpm_watch"]].copy()
            out["pred"] = pred; out["B1_media_personal_train"] = base_te; out["B0_media_global_train"] = float(dtr.bpm_watch.mean())
            out["seed"] = seed; out["row_index"] = dte.index
            met = N.eval_metrics(out.bpm_watch, out.pred, out.participante)
            res = {"seed": seed, "family": fam, "complete": True, "metrics_outer_test": met,
                   "MAE_B0_global": float(np.mean(np.abs(out.bpm_watch - out.B0_media_global_train))),
                   "MAE_B1_personal": float(np.mean(np.abs(out.bpm_watch - out.B1_media_personal_train))),
                   "mae_por_sujeto": (out.pred - out.bpm_watch).abs().groupby(out.participante).mean().to_dict(),
                   "seleccion": {k: v for k, v in s.items() if k not in ("stage1", "stage2")}, "scaler_check": sc,
                   "delta_vs_historico_840": met["MAE"] - C.MAE_E1A_HISTORICAL}
            # predicciones primero, JSON despues: el JSON es la marca de "familia/seed completa"
            _csv_gz_dump(out, out_dir / fam.lower() / f"seed_{seed:02d}_predictions.csv.gz")
            _json_dump(res, out_dir / fam.lower() / f"seed_{seed:02d}.json")
            timings[f"{fam}_refit_s"] = round(time.time() - t1, 1)
            timings[f"{fam}_s"] = round(timings[f"{fam}_seleccion_s"] + timings[f"{fam}_refit_s"], 1)
            seed_audit[f"{fam}_scaler_fit_solo_outer_train"] = sc["scaler_mean_equals_train_mean"]
            log(f"    [{fam}] OUTER TEST seed {seed:02d}: MAE={met['MAE']:.4f}  B1={res['MAE_B1_personal']:.4f}  B0={res['MAE_B0_global']:.4f}")
    seed_audit.update({"runtime_s": round(time.time() - t0, 1), "ram_pico_sistema_gb": round(ram.peak_gb, 2), "ram_base_gb": round(ram.base_gb, 2)})
    timings.update({"seed_s": seed_audit["runtime_s"], "ram_pico_sistema_gb": seed_audit["ram_pico_sistema_gb"]})
    audit["seeds"][str(seed)] = seed_audit
    audit["PASS"] = all(v["outer_shared_captures"] == 0 and v["outer_shared_windows"] == 0 and v["inner_shared_captures_max"] == 0
                        and v["inner_folds_solo_outer_train"] and not v["outer_test_en_inner"] for v in audit["seeds"].values())
    _json_dump(audit, audit_path)
    log(f"=== seed {seed:02d} guardada ({seed_audit['runtime_s']:.0f}s, RAM pico sistema {seed_audit['ram_pico_sistema_gb']} GB)")
    return audit


def summarize(out_dir=CLEAN_DIR):
    """Agrega SOLO seeds completas (descriptivo; no se usa para decidir nada)."""
    out_dir = Path(out_dir)
    rows, preds = [], {}
    for fam in FAMILIES:
        P = [pd.read_csv(f, dtype={"participante": str}) for f in sorted((out_dir / fam.lower()).glob("seed_*_predictions.csv.gz"))]
        if not P:
            continue
        P = pd.concat(P); preds[fam] = P
        per = P.groupby("seed").apply(lambda x: M.mae(x.bpm_watch, x.pred), include_groups=False)
        from csi_hr import nested as N
        g = N.eval_metrics(P.bpm_watch, P.pred, P.participante); ci = M.ic95_runs(per.values) if len(per) > 1 else float("nan")
        rows.append({"model": f"{fam}_clean", "n_seeds": len(per), "mae_global": g["MAE"], "mae_run_mean": per.mean(), "mae_run_std": per.std(ddof=1),
                     "ci95_low": per.mean() - ci, "ci95_high": per.mean() + ci, "macro_subject_mae": g["MAE_macro_sujeto"], "rmse": g["RMSE"],
                     "baseline_global": M.mae(P.bpm_watch, P.B0_media_global_train), "baseline_personal": M.mae(P.bpm_watch, P.B1_media_personal_train),
                     "delta_vs_historical_840": g["MAE"] - C.MAE_E1A_HISTORICAL})
    R = pd.read_csv(C.E1_DIR / "predicciones_rf_todas.csv.gz", dtype={"participante": str}); R = R[R.arm == "E1a"]
    from csi_hr import nested as N
    per = R.groupby("corrida").apply(lambda x: M.mae(x.bpm_watch, x.pred), include_groups=False); g = N.eval_metrics(R.bpm_watch, R.pred, R.participante)
    ci = M.ic95_runs(per.values)
    rows.append({"model": "RF_historical (HISTORICAL, NOT CLEANLY SELECTED)", "n_seeds": 33, "mae_global": g["MAE"], "mae_run_mean": per.mean(),
                 "mae_run_std": per.std(ddof=1), "ci95_low": per.mean() - ci, "ci95_high": per.mean() + ci, "macro_subject_mae": g["MAE_macro_sujeto"],
                 "rmse": g["RMSE"], "baseline_global": float("nan"), "baseline_personal": M.mae(R.bpm_watch, R.media_persona),
                 "delta_vs_historical_840": g["MAE"] - C.MAE_E1A_HISTORICAL})
    comp = pd.DataFrame(rows); comp.to_csv(out_dir / "comparison.csv", index=False)
    # descriptivo por respiracion y posicion (despues de congelar predicciones)
    resp, pos = [], []
    for fam, P in preds.items():
        for name, S in (("normal", C.NORMAL_BREATHING_POSITIONS), ("alternada", C.ALTERNATE_BREATHING_POSITIONS)):
            q = P[P.pos_id.isin(S)]
            resp.append({"model": fam, "respiracion": name, "MAE": M.mae(q.bpm_watch, q.pred), "MAE_B1": M.mae(q.bpm_watch, q.B1_media_personal_train), "n": len(q)})
        for p_, q in P.groupby("pos_id"):
            pos.append({"model": fam, "pos_id": int(p_), "MAE": M.mae(q.bpm_watch, q.pred), "MAE_B1": M.mae(q.bpm_watch, q.B1_media_personal_train), "n": len(q)})
    pd.DataFrame(resp).to_csv(out_dir / "mae_por_respiracion.csv", index=False)
    pd.DataFrame(pos).to_csv(out_dir / "mae_por_posicion.csv", index=False)
    # frecuencias de seleccion
    freq = []
    for fam in FAMILIES:
        for f in sorted((out_dir / fam.lower()).glob("seed_[0-9][0-9].json")):
            s = json.loads(f.read_text(encoding="utf-8"))["seleccion"]
            freq.append({"family": fam, "seed": int(f.stem[-2:]), "selector": s["selector"], "N": s["N"], "features": "|".join(s["features"]),
                         "smoothing_w": s["smoothing_w"], "params": json.dumps(s["params"], sort_keys=True), "control": s.get("params_control"),
                         "inner_mae": s["inner_mae"]})
    pd.DataFrame(freq).to_csv(out_dir / "selecciones_por_seed.csv", index=False)
    return comp


# ------------------------------------------------------------------ datos
def load_data(features_source="saved"):
    """dataset publicado filtrado a STATIC_POSITIONS + sync_ok, con las features E1a.
    features_source='saved': features E1a guardadas por E1 (equivalencia con src/ demostrada en tests);
    'rebuild': recalculadas desde los PCAP (lento)."""
    from pipeline.reproduce_840 import assemble, build_e1a_features
    dC = splits.load_df_C()
    if features_source == "saved":
        F = pd.read_csv(SAVED_FEATURES, dtype={"participante": str}); F = F[F.arm == "E1a"].drop(columns=["arm"])
    else:
        F = build_e1a_features(sorted(dC.participante.unique()))
    d = assemble(dC, F)
    assert set(d.pos_id.astype(int).unique()) <= set(C.STATIC_POSITIONS)
    return d


def check_splits(d, n_runs=C.N_RUNS):
    out = []
    for r in range(n_runs):
        tr, te = splits.split_por_grabacion(d, r)
        chk = splits.check_personalized(d, tr, te)          # assert shared_captures == 0
        out.append({"corrida": r, "n_train": len(tr), "n_test": len(te), **chk})
    return pd.DataFrame(out)


def make_model(model, svr_variant=None):
    if model == "rf":
        return models.make_rf_historical
    if svr_variant is None:
        raise ValueError("SVR: indicar --svr-variant (script25_33corridas | script12_dataset); no hay valor final decidido")
    return lambda: models.make_svr_historical(svr_variant)


def run_historical_baseline(model="rf", svr_variant=None, features_source="saved", n_runs=C.N_RUNS, out_dir=OUT_DIR):
    from pipeline.reproduce_840 import FEATS, run_33
    print(WARNING + "\n", flush=True)
    d = load_data(features_source)
    P = run_33(d, FEATS, n_runs=n_runs, make_model=make_model(model, svr_variant))
    per_run = P.groupby("corrida").apply(lambda x: M.mae(x.bpm_watch, x.pred), include_groups=False)
    res = {"modelo": model, "svr_variant": svr_variant, "features": FEATS, "MAE": M.mae(P.bpm_watch, P.pred),
           "MAE_media_corridas": float(per_run.mean()), "MAE_ic95": float(M.ic95_runs(per_run.values)) if len(per_run) > 1 else None,
           "n_corridas": n_runs, "estado": "BASELINE HISTORICO (no es el resultado final limpio)"}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = model if model == "rf" else f"svr_{svr_variant}"
    P.to_csv(out_dir / f"predicciones_{tag}.csv.gz", index=False, compression="gzip")
    (out_dir / f"resumen_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pipeline personalizado")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--historical-baseline", action="store_true")
    mode.add_argument("--clean", action="store_true", help="experimento final limpio (validacion anidada)")
    mode.add_argument("--summarize", action="store_true", help="agrega las seeds completas del experimento limpio")
    ap.add_argument("--model", choices=["rf", "svr"], default="rf")
    ap.add_argument("--svr-variant", choices=sorted(C.SVR_PARAMS_HISTORICAL), default=None)
    ap.add_argument("--features-source", choices=["saved", "rebuild"], default="saved")
    ap.add_argument("--n-runs", type=int, default=C.N_RUNS)
    ap.add_argument("--dry-run", action="store_true", help="valida datos y splits sin entrenar")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="--clean: seeds exteriores (por defecto 0..32)")
    ap.add_argument("--preflight", action="store_true", help="--clean: seeds 0 y 1 en resultados/final_personalized/preflight/")
    ap.add_argument("--resume", action="store_true", help="--clean: omite seeds ya completas")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    if a.clean:
        out = Path(a.out_dir) if a.out_dir else (CLEAN_DIR / "preflight" if a.preflight else CLEAN_DIR)
        seeds = a.seeds if a.seeds else ([0, 1] if a.preflight else list(range(C.N_RUNS)))
        res = run_clean(seeds, out, resume=a.resume, n_jobs=a.n_jobs)
        print(json.dumps(res, indent=1, default=str)[:4000])
        return res
    if a.summarize:
        comp = summarize(Path(a.out_dir) if a.out_dir else CLEAN_DIR)
        print(comp.to_string(index=False))
        return comp
    if a.dry_run:
        print(WARNING + "\n", flush=True)
        d = load_data(a.features_source); S = check_splits(d, a.n_runs)
        res = {"n_ventanas": len(d), "posiciones": sorted(int(p) for p in d.pos_id.unique()), "corridas": len(S),
               "max_capturas_compartidas": int(S.capturas_compartidas.max()), "max_ventanas_compartidas": int(S.ventanas_compartidas.max())}
    else:
        res = run_historical_baseline(a.model, a.svr_variant, a.features_source, a.n_runs)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
