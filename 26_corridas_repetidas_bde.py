"""
26_corridas_repetidas_bde.py
Tesis: Monitoreo no invasivo de la frecuencia cardiaca mediante senales Wi-Fi CSI
Universidad de Lima - Ingenieria de Sistemas
Autores: Yadhira Sarmiento Escobar (20214190) y Favio Chavarry Minaya (20214680)

Extension de 25_corridas_repetidas.py (que valido los Escenarios A y C) a los
Escenarios B, D (Optuna RF/XGB/LGB) y E (LSTM) de 24_optuna_sedentario_lstm.py,
que en ese script solo se habian evaluado con un UNICO split. El objetivo es
el mismo: verificar si sus MAE reportados (7.68-7.77) son representativos o
tambien fueron splits con suerte, y obtener media +/- IC95% comparables entre
si con rigor estadistico.

Los hiperparametros de D (RF/XGB/LGB) se REUTILIZAN fijos de la busqueda de
Optuna ya hecha en 24_optuna_sedentario_lstm.py (no se vuelve a correr Optuna
33 veces -- séria carisimo y mezclaria varianza de resampling con varianza de
tuneo, mismo criterio que en 25_corridas_repetidas.py). La arquitectura del
LSTM (Escenario E) tambien se reutiliza tal cual esta definida en 24 (no fue
tuneada por Optuna ahi, son hiperparametros fijos de por si).

Prerequisito: haber ejecutado 12_features_modelos_v3_synced.py y
23_seleccion_mi.py.

Salidas (en resultados/):
  resultados_33_corridas_bde.csv  - una fila por corrida x escenario
  resumen_33_corridas_bde.csv     - media, std e IC95% por escenario x metrica
  fig_33_corridas_bde_barras.png  - barras con error bars, B/D/E
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [AVISO] XGBoost no instalado -- Escenario D-XGB omitido.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  [AVISO] LightGBM no instalado -- Escenario D-LGB omitido.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except (ImportError, OSError) as e:
    HAS_TORCH = False
    print(f"  [AVISO] PyTorch no disponible ({type(e).__name__}) -- Escenario E omitido.")

OUTPUT_DIR  = r"C:\Users\LENOVO\Desktop\tesis-wifi-csi-frecuencia-cardiaca-main\resultados"
DATASET_CSV = os.path.join(OUTPUT_DIR, "dataset_completo_v3_synced.csv")
PI_CSV      = os.path.join(OUTPUT_DIR, "permutation_importance_features.csv")

N_TOP_FEAT  = 10
N_CORRIDAS  = 33
W_SUAVIZADO = 9
TEST_FRAC   = 0.20
RANDOM_STATE = 42

POSICIONES_SEDENTARIAS = {1, 2, 3, 4, 5, 6, 11, 12, 13, 14}

RF_BASE = {"max_depth": 6, "max_features": "sqrt", "min_samples_leaf": 29,
           "n_estimators": 387, "random_state": RANDOM_STATE, "n_jobs": -1}
# Hiperparametros ya encontrados por Optuna en 24_optuna_sedentario_lstm.py
# (50 trials, GroupKFold) -- fijos aqui, ver docstring.
RF_OPTUNA_BEST = {"n_estimators": 199, "max_depth": 3, "min_samples_leaf": 1,
                  "max_features": "log2", "random_state": RANDOM_STATE, "n_jobs": -1}
XGB_BEST = {"n_estimators": 130, "max_depth": 4, "learning_rate": 0.012305245203060727,
            "subsample": 0.527914920826915, "colsample_bytree": 0.5666466176782431,
            "min_child_weight": 6, "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0}
LGB_BEST = {"n_estimators": 128, "max_depth": 3, "learning_rate": 0.010183896098660642,
            "num_leaves": 43, "min_child_samples": 32, "subsample": 0.6787451976616498,
            "colsample_bytree": 0.9366439820002148, "random_state": RANDOM_STATE,
            "n_jobs": -1, "verbose": -1}

# Hiperparametros LSTM (fijos de entrada, no tuneados por Optuna en 24)
LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT = 5, 64, 2, 0.3
LSTM_EPOCHS, LSTM_PATIENCE, LSTM_LR, LSTM_BATCH = 80, 15, 1e-3, 64

print("=" * 66)
print(f"  26 - {N_CORRIDAS} CORRIDAS REPETIDAS: Escenarios B, D (Optuna) y E (LSTM)")
print("=" * 66)

df = pd.read_csv(DATASET_CSV)
pi_rank = pd.read_csv(PI_CSV)
feat_cols = pi_rank["feature"].tolist()[:N_TOP_FEAT]
print(f"  Features (top-{N_TOP_FEAT}): {feat_cols}")

df_B = df[df["pos_id"].isin(POSICIONES_SEDENTARIAS)].copy()
print(f"  Escenario B/D/E (sedentario, sin filtro sync): {len(df_B)} registros\n")


def split_por_grabacion(df, seed, test_frac=TEST_FRAC):
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


def entrenar_evaluar_arbol(modelo_cls, params, df_tr, df_te, feat_cols):
    medias = df_tr.groupby("participante")["bpm_watch"].mean()
    df_tr = df_tr.copy(); df_te = df_te.copy()
    df_tr["media_p"] = df_tr["participante"].map(medias)
    df_te["media_p"] = df_te["participante"].map(medias).fillna(df_tr["bpm_watch"].mean())
    df_tr["residual"] = df_tr["bpm_watch"] - df_tr["media_p"]

    sc   = StandardScaler().fit(df_tr[feat_cols].fillna(0))
    X_tr = sc.transform(df_tr[feat_cols].fillna(0))
    X_te = sc.transform(df_te[feat_cols].fillna(0))

    modelo = modelo_cls(**params).fit(X_tr, df_tr["residual"].values)
    pred_raw   = df_te["media_p"].values + modelo.predict(X_te)
    pred_suave = suavizar(df_te, pred_raw)
    return metricas(df_te["bpm_watch"].values, pred_suave)


class LSTMHRModel(nn.Module if HAS_TORCH else object):
    def __init__(self, input_size, hidden=LSTM_HIDDEN, n_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, n_layers, batch_first=True,
                             dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def entrenar_evaluar_lstm(df_tr, df_te, feat_cols, seed):
    medias = df_tr.groupby("participante")["bpm_watch"].mean()
    df_tr = df_tr.copy(); df_te = df_te.copy()
    df_tr["media_p"] = df_tr["participante"].map(medias)
    df_te["media_p"] = df_te["participante"].map(medias).fillna(df_tr["bpm_watch"].mean())
    df_tr["residual"] = df_tr["bpm_watch"] - df_tr["media_p"]
    df_te["residual"] = df_te["bpm_watch"] - df_te["media_p"]

    sc     = StandardScaler().fit(df_tr[feat_cols].fillna(0))
    arr_tr = sc.transform(df_tr[feat_cols].fillna(0))
    arr_te = sc.transform(df_te[feat_cols].fillna(0))
    n_feats = arr_tr.shape[1]
    df_tr["_feat_idx"] = np.arange(len(df_tr))
    df_te["_feat_idx"] = np.arange(len(df_te))

    def construir_secuencias(df_in, arr_feats, seq_len=LSTM_SEQ_LEN):
        X_seqs, y_seqs = [], []
        for part in df_in["participante"].unique():
            for seg in df_in[df_in["participante"] == part]["seg_id"].unique():
                rows = (df_in[(df_in["participante"] == part) & (df_in["seg_id"] == seg)]
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

    X_tr_seq, y_tr_seq = construir_secuencias(df_tr, arr_tr)
    X_te_seq, y_te_seq = construir_secuencias(df_te, arr_te)
    if X_tr_seq.shape[0] < 32:
        return None

    torch.manual_seed(seed)
    n_tr = X_tr_seq.shape[0]
    n_val = max(1, int(n_tr * 0.2))
    perm = np.random.RandomState(seed).permutation(n_tr)
    idx_val, idx_fit = perm[:n_val], perm[n_val:]

    device = torch.device("cpu")

    def make_loader(X, y, shuffle=True):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=LSTM_BATCH, shuffle=shuffle)

    train_loader = make_loader(X_tr_seq[idx_fit], y_tr_seq[idx_fit])
    val_loader   = make_loader(X_tr_seq[idx_val], y_tr_seq[idx_val], shuffle=False)

    model     = LSTMHRModel(n_feats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.L1Loss()

    best_val, patience_ct, best_state = float("inf"), 0, None
    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl_loss = float(np.mean([criterion(model(Xb), yb).item() for Xb, yb in val_loader]))
        scheduler.step(vl_loss)
        if vl_loss < best_val:
            best_val, patience_ct = vl_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
        if patience_ct >= LSTM_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_res_lstm = model(torch.tensor(X_te_seq, dtype=torch.float32)).numpy()

    win_indices = []
    for part in df_te["participante"].unique():
        for seg in df_te[df_te["participante"] == part]["seg_id"].unique():
            rows = df_te[(df_te["participante"] == part) & (df_te["seg_id"] == seg)].sort_values("win_start")
            if len(rows) >= LSTM_SEQ_LEN:
                for i in range(len(rows) - LSTM_SEQ_LEN + 1):
                    win_indices.append(rows.index[i + LSTM_SEQ_LEN - 1])

    df_pred = df_te.loc[win_indices].copy()
    df_pred["pred_bpm"] = df_pred["media_p"].values + pred_res_lstm[:len(win_indices)]
    df_pred = df_pred.sort_values(["participante", "seg_id", "win_start"])
    pred_suave = (
        df_pred.groupby(["participante", "seg_id"])["pred_bpm"]
        .transform(lambda s: s.rolling(W_SUAVIZADO, center=True, min_periods=1).mean())
        .values
    )
    return metricas(df_pred["bpm_watch"].values, pred_suave)


resultados = []
for corrida in range(N_CORRIDAS):
    seed = corrida
    idx_tr, idx_te = split_por_grabacion(df_B, seed)
    df_tr, df_te = df_B.loc[idx_tr].copy(), df_B.loc[idx_te].copy()

    mae_b, rmse_b, mape_b, r_b = entrenar_evaluar_arbol(RandomForestRegressor, RF_BASE, df_tr, df_te, feat_cols)
    resultados.append({"escenario": "B: Solo sedentarias", "corrida": corrida, "seed": seed,
                        "MAE": mae_b, "RMSE": rmse_b, "MAPE": mape_b, "r": r_b})

    mae_dr, rmse_dr, mape_dr, r_dr = entrenar_evaluar_arbol(RandomForestRegressor, RF_OPTUNA_BEST, df_tr, df_te, feat_cols)
    resultados.append({"escenario": "D: RF + Optuna", "corrida": corrida, "seed": seed,
                        "MAE": mae_dr, "RMSE": rmse_dr, "MAPE": mape_dr, "r": r_dr})

    linea = f"  Corrida {corrida+1:2d}/{N_CORRIDAS}  B: MAE={mae_b:.3f}  |  D-RF: MAE={mae_dr:.3f}"

    if HAS_XGB:
        mae_dx, rmse_dx, mape_dx, r_dx = entrenar_evaluar_arbol(XGBRegressor, XGB_BEST, df_tr, df_te, feat_cols)
        resultados.append({"escenario": "D: XGBoost + Optuna", "corrida": corrida, "seed": seed,
                            "MAE": mae_dx, "RMSE": rmse_dx, "MAPE": mape_dx, "r": r_dx})
        linea += f"  |  D-XGB: MAE={mae_dx:.3f}"

    if HAS_LGB:
        mae_dl, rmse_dl, mape_dl, r_dl = entrenar_evaluar_arbol(lgb.LGBMRegressor, LGB_BEST, df_tr, df_te, feat_cols)
        resultados.append({"escenario": "D: LightGBM + Optuna", "corrida": corrida, "seed": seed,
                            "MAE": mae_dl, "RMSE": rmse_dl, "MAPE": mape_dl, "r": r_dl})
        linea += f"  |  D-LGB: MAE={mae_dl:.3f}"

    if HAS_TORCH:
        res_lstm = entrenar_evaluar_lstm(df_tr, df_te, feat_cols, seed)
        if res_lstm is not None:
            mae_e, rmse_e, mape_e, r_e = res_lstm
            resultados.append({"escenario": "E: LSTM", "corrida": corrida, "seed": seed,
                                "MAE": mae_e, "RMSE": rmse_e, "MAPE": mape_e, "r": r_e})
            linea += f"  |  E-LSTM: MAE={mae_e:.3f}"

    print(linea)

df_res = pd.DataFrame(resultados)
df_res.to_csv(os.path.join(OUTPUT_DIR, "resultados_33_corridas_bde.csv"), index=False)
print(f"\n  resultados_33_corridas_bde.csv -> {OUTPUT_DIR}")


def ic95(s):
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s))


resumen = df_res.groupby("escenario")[["MAE", "RMSE", "MAPE", "r"]].agg(["mean", "std"])
resumen.columns = ["_".join(c) for c in resumen.columns]
for m in ["MAE", "RMSE", "MAPE", "r"]:
    resumen[f"{m}_ic95"] = df_res.groupby("escenario")[m].apply(ic95)
resumen = resumen.reset_index().sort_values("MAE_mean")
resumen.to_csv(os.path.join(OUTPUT_DIR, "resumen_33_corridas_bde.csv"), index=False)

print(f"\n  resumen_33_corridas_bde.csv -> {OUTPUT_DIR}\n")
print("=" * 66)
print(f"  RESUMEN (media +/- IC95% sobre {N_CORRIDAS} corridas)")
print("=" * 66)
print(resumen.to_string(index=False))

# ─── FIGURA: barras con error bars ─────────────────────────────────────────────
colores_esc = {"B: Solo sedentarias": "#9E9E9E", "D: RF + Optuna": "#2196F3",
               "D: XGBoost + Optuna": "#4CAF50", "D: LightGBM + Optuna": "#FF9800",
               "E: LSTM": "#9C27B0"}
fig, ax = plt.subplots(figsize=(10, 6))
escs = resumen["escenario"].tolist()
medias = resumen["MAE_mean"].tolist()
ics = resumen["MAE_ic95"].tolist()
colores = [colores_esc.get(e, "#90CAF9") for e in escs]
bars = ax.bar(escs, medias, yerr=ics, capsize=6, color=colores, edgecolor="black",
              linewidth=0.8, error_kw={"linewidth": 1.3, "ecolor": "black"})
ax.set_ylabel("MAE (lat/min)")
ax.set_title(f"Escenarios B/D/E — Media ± IC95% sobre {N_CORRIDAS} corridas\n"
             f"(splits aleatorios por grabación, subconjunto sedentario)", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
for bar, val in zip(bars, medias):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
             f"{val:.2f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_33_corridas_bde_barras.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  fig_33_corridas_bde_barras.png -> {OUTPUT_DIR}")

print("\nScript 26 completado.")
