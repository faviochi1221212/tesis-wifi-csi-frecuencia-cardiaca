"""
Splits de evaluacion.

  split_por_grabacion  <- experimentos_fase_C/E1_rf.py L139-148 (= 25_corridas_repetidas.py): split PERSONALIZADO,
                          dentro de cada participante, por seg_id (captura), 20 % test, semilla = corrida.
  group_kfold_subject  <- experimentos_fase_C/E5_generalizacion.py L69-77: GroupKFold(5) por participante
                          (SUJETO-INDEPENDIENTE), con los chequeos de fuga de E5.
  load_df_C / load_df_E5 <- E1_rf.py L186-188 y E5_generalizacion.base_df (L34-38).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import DATASET_PUBLISHED, MACRO_POSTURE, N_FOLDS_SUBJECT, STATIC_POSITIONS, TEST_FRAC

KEYS = ["participante", "pos_id", "seg_id", "win_start"]


def load_published(path=DATASET_PUBLISHED):
    pub = pd.read_csv(path, dtype={"participante": str})
    pub["participante"] = pub.participante.str.zfill(3)
    return pub


def load_df_C(pub=None):
    """Ventanas publicadas de STATIC_POSITIONS con sync_ok, CONSERVANDO el indice original (como E1_rf L188;
    los indices de splits_rf.json se refieren a este indice)."""
    pub = load_published() if pub is None else pub
    return pub[pub["pos_id"].isin(STATIC_POSITIONS) & pub["sync_ok"].astype(bool)].copy()


def load_df_E5(pub=None):
    """Igual filtro, con indice reiniciado y columnas macro/captura (E5_generalizacion.base_df)."""
    pub = load_published() if pub is None else pub
    dC = pub[pub.pos_id.isin(STATIC_POSITIONS) & pub.sync_ok.astype(bool)][KEYS + ["bpm_watch"]].reset_index(drop=True)
    dC["macro"] = dC.pos_id.map(MACRO_POSTURE); dC["captura"] = dC.participante + "_" + dC.seg_id.astype(str)
    return dC


def split_por_grabacion(df, seed, test_frac=TEST_FRAC):
    rng = np.random.RandomState(seed); idx_tr, idx_te = [], []
    for part in sorted(df["participante"].unique()):
        dfp = df[df["participante"] == part]; segs = sorted(dfp["seg_id"].unique()); n = len(segs)
        if n < 2:
            idx_tr.extend(dfp.index.tolist()); continue
        perm = rng.permutation(n); n_test = max(1, int(round(n * test_frac)))
        te = {segs[i] for i in perm[:n_test]}; tr = {segs[i] for i in perm[n_test:]}
        idx_tr.extend(dfp[dfp["seg_id"].isin(tr)].index.tolist()); idx_te.extend(dfp[dfp["seg_id"].isin(te)].index.tolist())
    return idx_tr, idx_te


def check_personalized(df, idx_tr, idx_te):
    """Asserts del protocolo personalizado: 0 capturas y 0 ventanas compartidas (los sujetos SI se comparten)."""
    tr, te = df.loc[idx_tr], df.loc[idx_te]
    cap = lambda d: set(zip(d.participante, d.pos_id.astype(int)))  # noqa: E731
    seg = lambda d: set(zip(d.participante, d.seg_id))  # noqa: E731
    out = {"capturas_compartidas": len(cap(tr) & cap(te)), "segmentos_compartidos": len(seg(tr) & seg(te)),
           "ventanas_compartidas": len(set(idx_tr) & set(idx_te)),
           "sujetos_compartidos": len(set(tr.participante) & set(te.participante))}
    assert out["capturas_compartidas"] == 0, out
    assert out["segmentos_compartidos"] == 0, out
    assert out["ventanas_compartidas"] == 0, out
    return out


def group_kfold_subject(dC, n_splits=N_FOLDS_SUBJECT, keys=KEYS):
    """Folds de E5 (GroupKFold por participante) + chequeo de fuga: 0 sujetos, capturas y ventanas compartidas."""
    folds, leak = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(dC, groups=dC.participante)):
        folds.append((tr, te))
        cap = "captura" if "captura" in dC.columns else "seg_id"
        leak.append({"fold": k, "n_train_sujetos": int(dC.loc[tr, "participante"].nunique()),
                     "n_test_sujetos": int(dC.loc[te, "participante"].nunique()),
                     "sujetos_compartidos": len(set(dC.loc[tr, "participante"]) & set(dC.loc[te, "participante"])),
                     "capturas_compartidas": len(set(zip(dC.loc[tr, "participante"], dC.loc[tr, cap])) & set(zip(dC.loc[te, "participante"], dC.loc[te, cap]))),
                     "ventanas_compartidas": len(set(map(tuple, dC.loc[tr, keys].values)) & set(map(tuple, dC.loc[te, keys].values)))})
    assert all(l["sujetos_compartidos"] == 0 and l["capturas_compartidas"] == 0 and l["ventanas_compartidas"] == 0 for l in leak), leak
    return folds, leak
