"""
generar_resultados.py -- metricas y figuras del experimento final (escenario personalizado), desde las
predicciones congeladas. NO entrena modelos y NO modifica las predicciones.

Lee (solo lectura):
  resultados/predicciones_rf.csv.gz, resultados/predicciones_svr.csv.gz   (33 semillas x ventanas de prueba)
  resultados/configuraciones_seleccionadas.csv                           (suavizado w elegido por semilla)
Escribe:
  resultados/metricas_por_semilla.csv   MAE, RMSE y Pearson r por semilla y metodo (RF, SVR, B1, B0)
  resultados/metricas_finales.csv       resumen entre las 33 semillas (media, DE, IC95 del MAE)
  figuras/figura_1_comparacion_mae.(png|pdf)
  figuras/figura_2_referencia_vs_prediccion.(png|pdf)
  figuras/figura_3_test_predicho_vs_referencia.(png|pdf)
  material_adicional/figura_4_rf_vs_svr_test.(png|pdf)   (anexo / sustentacion; no es figura principal)

Uso (desde la raiz de la entrega):  python codigo/generar_resultados.py
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codigo" / "src"))
from csi_hr import config as C  # noqa: E402

RES, FIG = ROOT / "resultados", ROOT / "figuras"
SEEDS = list(range(C.N_RUNS))                    # 0..32
METHODS = ("RF", "SVR", "B1", "B0")
KEYS = ["seed", "participante", "pos_id", "seg_id", "win_start"]

COL = {"RF": "#2a78d6", "SVR": "#eb6834", "B1": "#7a7974", "B0": "#a9a8a2", "ref": "#0b0b0b"}
MARK = {"RF": "o", "SVR": "s"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def load():
    a = pd.read_csv(RES / "predicciones_rf.csv.gz", dtype={"participante": str})
    b = pd.read_csv(RES / "predicciones_svr.csv.gz", dtype={"participante": str})
    same = KEYS + ["bpm_watch", "B1_media_personal_train", "B0_media_global_train"]
    if not a[same].equals(b[same]):
        raise RuntimeError("RF y SVR no comparten las mismas ventanas, referencias y lineas base")
    D = a[KEYS + ["bpm_watch"]].rename(columns={"bpm_watch": "y"})
    D["RF"], D["SVR"] = a["pred"].values, b["pred"].values
    D["B1"], D["B0"] = a["B1_media_personal_train"].values, a["B0_media_global_train"].values
    if sorted(D.seed.unique()) != SEEDS:
        raise RuntimeError("faltan semillas")
    if (D.groupby("seed").B0.nunique() != 1).any():
        raise RuntimeError("B0 deberia ser constante dentro de cada semilla")
    return D


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); e = p - y
    r = float(np.corrcoef(y, p)[0, 1]) if np.ptp(p) > 0 else float("nan")   # prediccion constante -> no definido
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))), "pearson_r": r, "n_ventanas": len(y)}


def tables(D):
    M = pd.DataFrame([{"seed": s, "metodo": m, **metrics(g.y, g[m])} for s, g in D.groupby("seed") for m in METHODS])
    rows = []
    for m in METHODS:
        g = M[M.metodo == m]; n = len(g); mae = g.MAE
        ci = 1.96 * mae.std(ddof=1) / np.sqrt(n)
        r_ok = g.pearson_r.notna().all()
        rows.append({"metodo": m, "n_semillas": n, "MAE_media": mae.mean(), "MAE_de": mae.std(ddof=1),
                     "MAE_IC95_inf": mae.mean() - ci, "MAE_IC95_sup": mae.mean() + ci,
                     "RMSE_media": g.RMSE.mean(), "RMSE_de": g.RMSE.std(ddof=1),
                     "Pearson_media": g.pearson_r.mean() if r_ok else np.nan,
                     "Pearson_de": g.pearson_r.std(ddof=1) if r_ok else np.nan,
                     "Pearson_nota": "" if r_ok else "No definido (prediccion constante)"})
    return M, pd.DataFrame(rows)


def figura1(M):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    data = [M[M.metodo == m].sort_values("seed").MAE.values for m in METHODS]
    ax1.boxplot(data, positions=range(len(METHODS)), widths=0.5, showfliers=False, patch_artist=True,
                medianprops={"color": INK, "linewidth": 1.6}, boxprops={"facecolor": "white", "edgecolor": INK2},
                whiskerprops={"color": INK2}, capprops={"color": INK2})
    rng = np.random.default_rng(0)                   # desplazamiento horizontal fijo (reproducible)
    for i, (m, v) in enumerate(zip(METHODS, data)):
        ax1.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=22, color=COL[m], edgecolor="white", linewidth=0.8, zorder=3)
    ax1.set_xticks(range(len(METHODS)))
    ax1.set_xticklabels(["RF", "SVR", "B1\n(media del\nparticipante)", "B0\n(media\nglobal)"])
    ax1.set_ylabel("MAE por corrida (BPM)")
    ax1.set_title("(a) MAE de las 33 corridas", loc="left", color=INK)
    b1 = M[M.metodo == "B1"].sort_values("seed").MAE.values
    ax2.axhline(0, color=INK2, linewidth=1)
    for fam in ("RF", "SVR"):
        d = b1 - M[M.metodo == fam].sort_values("seed").MAE.values
        ax2.scatter(SEEDS, d, s=30, marker=MARK[fam], color=COL[fam], edgecolor="white", linewidth=0.8, zorder=3,
                    label=f"B1 − {fam}  ({fam} < B1 en {int((d > 0).sum())}/33; media {d.mean():.2f})")
    ax2.set_xlabel("Semilla (corrida)"); ax2.set_ylabel("Diferencia de MAE (BPM)")
    ax2.set_xticks(range(0, 33, 4)); ax2.set_xlim(-1, 33)
    ax2.set_title("(b) Diferencias pareadas por semilla (> 0: el modelo mejora a B1)", loc="left", color=INK)
    ax2.legend(loc="upper left", fontsize=9)
    lo, hi = ax2.get_ylim(); ax2.set_ylim(min(lo, -0.1), hi * 1.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figura_1_comparacion_mae.{ext}")
    plt.close(fig)


def semilla_mediana(M):
    """Semilla cuyo MAE medio (RF+SVR)/2 ocupa la posicion mediana de las 33 corridas."""
    S = (M[M.metodo.isin(["RF", "SVR"])].groupby("seed").MAE.mean().reset_index()
         .sort_values(["MAE", "seed"], kind="mergesort").reset_index(drop=True))
    return int(S.seed.iloc[len(S) // 2])


def seleccionar_grabacion(D, seed):
    """En la semilla mediana, la grabacion de prueba cuyo MAE medio (RF+SVR)/2 es el mas cercano a la mediana por
    grabacion (empates: menor participante, luego menor seg_id). No se elige por su error."""
    R = (D[D.seed == seed].groupby(["participante", "seg_id"])
         .apply(lambda g: (np.mean(np.abs(g.RF - g.y)) + np.mean(np.abs(g.SVR - g.y))) / 2, include_groups=False)
         .rename("mae").reset_index())
    R["dist"] = (R.mae - R.mae.median()).abs()
    sel = R.sort_values(["dist", "participante", "seg_id"], kind="mergesort").iloc[0]
    return sel.participante, int(sel.seg_id), float(R.mae.median()), len(R)


def figura2(D, M):
    seed = semilla_mediana(M)
    part, seg, med, n_rec = seleccionar_grabacion(D, seed)
    cfg = pd.read_csv(RES / "configuraciones_seleccionadas.csv")
    w = {fam: int(cfg[(cfg.seed == seed) & (cfg.modelo == fam)].suavizado_w.iloc[0]) for fam in ("RF", "SVR")}
    G = D[(D.seed == seed) & (D.participante == part)].sort_values(["seg_id", "win_start"])
    recs = sorted(G.seg_id.unique())
    vals = G[["y", "RF", "SVR", "B1"]].values; pad = 0.08 * (vals.max() - vals.min())
    ylim = (vals.min() - pad, vals.max() + pad)                          # mismo eje Y en ambos paneles
    fig, axes = plt.subplots(1, len(recs), figsize=(5.2 * len(recs), 4.2), sharey=True, squeeze=False)
    for ax, sg in zip(axes[0], recs):
        g = G[G.seg_id == sg]
        t = (g.win_start.values + C.WINDOW_SAMPLES / 2) / C.FS            # centro nominal de la ventana (s)
        ax.axhline(g.B1.iloc[0], color=COL["B1"], linestyle=(0, (5, 3)), linewidth=1.5, label="B1 (media del participante en entrenamiento)")
        ax.plot(t, g.y, color=COL["ref"], linewidth=2, marker="D", markersize=6, label="Referencia (smartwatch)", zorder=4)
        for fam in ("RF", "SVR"):
            ax.plot(t, g[fam], color=COL[fam], linewidth=2, marker=MARK[fam], markersize=7, markeredgecolor="white",
                    markeredgewidth=0.8, label=f"{fam} (suavizado w={w[fam]})", zorder=3)
        mae = {m: np.mean(np.abs(g[m] - g.y)) for m in ("RF", "SVR", "B1")}
        tag = "  [grabación seleccionada]" if sg == seg else ""
        ax.set_title(f"Posición {int(g.pos_id.iloc[0])} · grabación {sg} · {len(g)} ventanas{tag}\n"
                     f"MAE: RF {mae['RF']:.2f} · SVR {mae['SVR']:.2f} · B1 {mae['B1']:.2f} BPM", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("Tiempo nominal del centro de la ventana (s, FS = 7,7 Hz)")
        ax.set_ylim(*ylim)
    axes[0][0].set_ylabel("Frecuencia cardíaca (BPM)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=9)
    fig.suptitle(f"Semilla {seed} · participante {part}: grabaciones de prueba", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figura_2_referencia_vs_prediccion.{ext}")
    plt.close(fig)
    return {"semilla": seed, "participante": part, "grabacion_seleccionada": seg,
            "grabaciones_mostradas": [int(r) for r in recs], "mediana_mae_por_grabacion": round(med, 4),
            "n_grabaciones_prueba": n_rec, "suavizado_w": w}


def _test_ordenado(D, seed):
    """Ventanas de prueba de una semilla ordenadas por participante, grabacion y tiempo; 'corte' marca el inicio de
    cada grabacion (las lineas se interrumpen ahi: no es una serie temporal continua)."""
    g = D[D.seed == seed].sort_values(["participante", "seg_id", "win_start"]).reset_index(drop=True)
    rec = g.participante + "_" + g.seg_id.astype(str)
    g["corte"] = np.r_[False, rec.values[1:] != rec.values[:-1]]
    return g


def _por_grabacion(g, col):
    """x, y con NaN en cada cambio de grabacion (interrumpe la linea entre grabaciones)."""
    x, y = [], []
    for i, (c, v) in enumerate(zip(g.corte.values, g[col].values)):
        if c:
            x.append(np.nan); y.append(np.nan)
        x.append(i); y.append(v)
    return np.array(x, float), np.array(y, float)


def _panel_test(ax, g, fam, nombre, ylim):
    for i in np.where(g.corte.values)[0]:
        ax.axvline(i - 0.5, color=GRID, linewidth=0.5, zorder=0)
    ax.plot(*_por_grabacion(g, "y"), color=COL["ref"], linewidth=1.0, label="Frecuencia cardíaca de referencia")
    ax.plot(*_por_grabacion(g, fam), color=COL[fam], linewidth=1.4, label=f"Frecuencia cardíaca estimada ({nombre})")
    ax.set_ylim(*ylim); ax.set_xlim(-5, len(g) + 4)
    ax.set_ylabel("Frecuencia cardíaca (BPM)")
    ax.grid(False, axis="x")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9, borderaxespad=0.2)   # fuera del area de datos
    m = metrics(g.y, g[fam])
    return m


NOTA_TEST = ("Ventanas ordenadas por participante, grabación y tiempo. Las líneas grises verticales separan grabaciones; "
             "las curvas se interrumpen entre grabaciones (no es una señal temporal continua).")


def figura3(D, M):
    """Referencia vs SVR sobre todo el conjunto de prueba de la semilla representativa (mediana).
    SVR: menor MAE medio en las 33 corridas (diferencia con RF pequena, no concluyente)."""
    seed = semilla_mediana(M)
    g = _test_ordenado(D, seed)
    v = g[["y", "SVR"]].values; ylim = (v.min() - 3, v.max() + 3)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    m = _panel_test(ax, g, "SVR", "SVR", ylim)
    ax.set_xlabel("Ventanas del conjunto de prueba")
    fig.suptitle("Frecuencia cardíaca de referencia y estimada mediante SVR en el conjunto de prueba", x=0.01, ha="left", color=INK)
    ax.set_title(f"Semilla {seed} — MAE = {m['MAE']:.2f} BPM; r = {m['pearson_r']:.3f}", loc="left", fontsize=10, color=INK2)
    fig.text(0.01, -0.02, f"{NOTA_TEST} {len(g)} ventanas de {g.corte.sum() + 1} grabaciones.", fontsize=8.5, color=INK2, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figura_3_test_predicho_vs_referencia.{ext}")
    plt.close(fig)
    return {"semilla": seed, "MAE_SVR": round(m["MAE"], 4), "r_SVR": round(m["pearson_r"], 4), "ventanas": len(g)}


def figura4(D, M):
    """Material adicional (no es figura principal): RF y SVR frente a la misma referencia, mismas ventanas y ejes."""
    seed = semilla_mediana(M)
    g = _test_ordenado(D, seed)
    v = g[["y", "RF", "SVR"]].values; ylim = (v.min() - 3, v.max() + 3)
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)
    for ax, fam, nombre in zip(axs, ("RF", "SVR"), ("Random Forest", "SVR")):
        m = _panel_test(ax, g, fam, nombre, ylim)
        ax.set_title(f"{nombre} — semilla {seed}: MAE = {m['MAE']:.2f} BPM; r = {m['pearson_r']:.3f}", loc="left", fontsize=10, color=INK)
    axs[1].set_xlabel("Ventanas del conjunto de prueba")
    r_rf_svr = float(np.corrcoef(g.RF, g.SVR)[0, 1])
    fig.suptitle("Random Forest y SVR frente a la misma referencia en el conjunto de prueba", x=0.01, ha="left", color=INK)
    fig.text(0.01, -0.01, f"{NOTA_TEST} Correlación entre las predicciones de RF y SVR: r = {r_rf_svr:.3f}.",
             fontsize=8.5, color=INK2, ha="left")
    fig.tight_layout()
    (ROOT / "material_adicional").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "material_adicional" / f"figura_4_rf_vs_svr_test.{ext}")
    plt.close(fig)
    return {"r_RF_SVR": round(r_rf_svr, 4)}


def main():
    FIG.mkdir(exist_ok=True)
    D = load()
    M, T = tables(D)
    M.to_csv(RES / "metricas_por_semilla.csv", index=False, float_format="%.10g")
    T.to_csv(RES / "metricas_finales.csv", index=False, float_format="%.10g")
    figura1(M)
    sel = figura2(D, M)
    print(f"{len(D)} ventanas de prueba, {D.seed.nunique()} semillas")
    for _, r in T.iterrows():
        pr = r.Pearson_nota or f"{r.Pearson_media:.3f} ± {r.Pearson_de:.3f}"
        print(f"  {r.metodo:4s} MAE {r.MAE_media:.3f} ± {r.MAE_de:.3f} | RMSE {r.RMSE_media:.3f} ± {r.RMSE_de:.3f} | r {pr}")
    print("Figura 2:", json.dumps(sel))
    print("Figura 3:", json.dumps(figura3(D, M)))
    print("Figura 4 (material adicional):", json.dumps(figura4(D, M)))


if __name__ == "__main__":
    main()
