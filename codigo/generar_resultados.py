"""
generar_resultados.py -- metricas y figuras del experimento final (escenario personalizado), desde los datos
congelados. NO entrena modelos y NO modifica predicciones, splits ni datos.

Lee (solo lectura):
  resultados/predicciones_rf.csv.gz, resultados/predicciones_svr.csv.gz   (33 semillas x ventanas de prueba)
  resultados/configuraciones_seleccionadas.csv                           (suavizado w elegido por semilla)
  datos_procesados/sincronizacion_ejemplo.json                           (generado por extraer_sincronizacion.py)
  datos_procesados/dataset_completo_v3_synced.csv                        (control de las etiquetas y_i)
Escribe:
  resultados/metricas_por_semilla.csv   MAE, RMSE y Pearson r por semilla y metodo (RF, SVR, B1, B0)
  resultados/metricas_finales.csv       resumen entre las 33 semillas (media, DE, IC95 del MAE)
  FIGURAS DE LA TESIS (figuras/):
    figura_sincronizacion_csi_smartwatch   Metodologia: lecturas del reloj -> interpolacion -> ventanas -> y_i
    figura_mae_33_corridas                 Resultados: distribucion del MAE de RF, SVR, B1 y B0 en 33 corridas
    figura_comparacion_rf_svr_test         Resultados: referencia vs RF y vs SVR en el test de la semilla 27
    figura_mejora_respecto_b1              Resultados (secundaria) o anexo: reduccion del MAE respecto de B1
  MATERIAL ADICIONAL (material_adicional/):
    figura_ejemplo_prediccion_1, figura_ejemplo_prediccion_2   ejemplos por grabacion (sustentacion)
  Cada figura se guarda en PNG (versionado) y PDF (solo local, .gitignore).

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
from csi_hr import config as C, splits  # noqa: E402

RES, FIG, ADIC = ROOT / "resultados", ROOT / "figuras", ROOT / "material_adicional"
SINCRO_JSON = ROOT / "datos_procesados" / "sincronizacion_ejemplo.json"
SEEDS = list(range(C.N_RUNS))                    # 0..32
METHODS = ("RF", "SVR", "B1", "B0")
KEYS = ["seed", "participante", "pos_id", "seg_id", "win_start"]
SEMILLA_REPRESENTATIVA = 27                      # desempeno mediano (criterio objetivo); se verifica contra los datos

COL = {"RF": "#2a78d6", "SVR": "#eb6834", "B1": "#7a7974", "B0": "#a9a8a2", "ref": "#0b0b0b"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASE = {"axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"}
# Estilos de las figuras aprobadas (cada figura se dibuja con el suyo, partiendo de la configuracion de arranque)
ESTILO_RESULTADOS = {**BASE, "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10, "axes.labelcolor": INK, "axes.grid": True, "axes.grid.axis": "y",
                     "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white"}
ESTILO_SINCRONIZACION = {**BASE, "font.size": 11}
ESTILO_COMPARACION = {**BASE, "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                      "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.labelcolor": INK,
                      "figure.facecolor": "white", "axes.facecolor": "white"}


class estilo:
    """Contexto: restaura la configuracion de arranque de matplotlib y aplica el estilo de la figura."""
    def __init__(self, rc):
        self.rc, self.ctx = rc, plt.rc_context()

    def __enter__(self):
        self.ctx.__enter__(); matplotlib.rc_file_defaults(); plt.rcParams.update(self.rc)

    def __exit__(self, *exc):
        return self.ctx.__exit__(*exc)


def guardar(fig, carpeta, nombre):
    carpeta.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(carpeta / f"{nombre}.{ext}")
    plt.close(fig)


# ====================================================================== datos y metricas
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


def semilla_mediana(M):
    """Semilla cuyo MAE medio (RF+SVR)/2 ocupa la posicion mediana de las 33 corridas."""
    S = (M[M.metodo.isin(["RF", "SVR"])].groupby("seed").MAE.mean().reset_index()
         .sort_values(["MAE", "seed"], kind="mergesort").reset_index(drop=True))
    return int(S.seed.iloc[len(S) // 2])


# ====================================================================== FIGURAS DE LA TESIS (figuras/)
def figura_sincronizacion():
    """Metodologia. Grabacion elegida por criterio objetivo (tabla incluida en el JSON, se recomprueba aqui)."""
    J = json.loads(SINCRO_JSON.read_text(encoding="utf-8"))
    R = pd.DataFrame(J["tabla_criterio"])
    E = R[(R.nw == 5) & (R.min_lect >= 2)].copy(); med = E.unicas.median(); E["dist"] = (E.unicas - med).abs()
    s = E.sort_values(["dist", "participante", "seg_id"], kind="mergesort").iloc[0]
    sel = J["seleccion"]
    if (s.participante, int(s.pos_id), int(s.seg_id)) != (sel["participante"], sel["pos_id"], sel["seg_id"]):
        raise RuntimeError("el criterio de seleccion no reproduce la grabacion guardada")
    d = splits.load_df_C()
    y_ds = d[(d.participante == sel["participante"]) & (d.pos_id == sel["pos_id"])].sort_values("win_start").bpm_watch.values
    win = J["ventanas"]
    if not np.allclose([w["y"] for w in win], y_ds, rtol=0, atol=1e-9):
        raise RuntimeError("las etiquetas y_i del JSON no coinciden con el dataset")
    a, b = J["xlim"]
    BLUE = COL["RF"]
    with estilo(ESTILO_SINCRONIZACION):
        fig, (ax, axw) = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.35], "hspace": 0.08})
        ax.plot(np.array(J["paquetes_t_s"]), np.array(J["paquetes_hr_interp"]), color="#9a9993", linewidth=1.6,
                label="Interpolación lineal", zorder=1)
        ax.scatter(np.array(J["lecturas_t_s"]), np.array(J["lecturas_hr"]), s=46, color=INK, zorder=3,
                   label="Lecturas del smartwatch")
        for k, w in enumerate(win):
            ax.scatter(w["centro"], w["y"], s=52, marker="D", color=BLUE, edgecolor="white", linewidth=0.8, zorder=4,
                       label="Etiqueta $y_i$ (centro de la ventana)" if k == 0 else None)
        ax.set_ylabel("Frecuencia cardíaca (BPM)"); ax.grid(True, axis="y", color=GRID)
        ax.legend(loc="upper right", ncol=1, fontsize=9.5); ax.set_xlim(a, b)
        for k, w in enumerate(win):
            lane = len(win) - k
            axw.hlines(lane, w["ini"], w["fin"], color=BLUE, linewidth=6, alpha=0.35)
            axw.plot([w["centro"]] * 2, [lane - 0.32, lane + 0.32], color=BLUE, linewidth=1.6)
        axw.set_yticks(range(1, len(win) + 1))
        axw.set_yticklabels([f"V{len(win) - j + 1}" for j in range(1, len(win) + 1)], fontsize=9)
        axw.set_ylim(0.4, len(win) + 0.6); axw.set_ylabel("Ventana CSI", fontsize=10)
        axw.set_xlabel("Tiempo desde el inicio de la grabación (s)")
        axw.spines["left"].set_visible(False); axw.tick_params(axis="y", length=0)
        guardar(fig, FIG, "figura_sincronizacion_csi_smartwatch")
    return {**sel, "ventanas": len(win), "mediana_lecturas_unicas": float(med)}


def figura_mae_33_corridas(M):
    """Resultados. Boxplot + 33 puntos (desplazamiento horizontal fijo) + media, para RF, SVR, B1 y B0."""
    with estilo(ESTILO_RESULTADOS):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        data = [M[M.metodo == m].sort_values("seed").MAE.values for m in METHODS]
        ax.boxplot(data, positions=range(4), widths=0.5, showfliers=False, patch_artist=True,
                   medianprops={"color": INK, "linewidth": 1.5}, boxprops={"facecolor": "white", "edgecolor": INK2},
                   whiskerprops={"color": INK2}, capprops={"color": INK2})
        rng = np.random.default_rng(0)
        for i, (m, v) in enumerate(zip(METHODS, data)):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=18, color=COL[m], alpha=0.85, edgecolor="white",
                       linewidth=0.6, zorder=3)
            ax.scatter(i, v.mean(), s=70, marker="D", facecolor="white", edgecolor=INK, linewidth=1.4, zorder=4,
                       label="Media" if i == 0 else None)
        ax.set_xticks(range(4)); ax.set_xticklabels(["RF", "SVR", "B1", "B0"])
        ax.set_ylabel("MAE (BPM)")
        ax.legend(loc="upper left")
        guardar(fig, FIG, "figura_mae_33_corridas")
    return {m: round(float(v.mean()), 4) for m, v in zip(METHODS, data)}


def figura_comparacion_rf_svr(D, M):
    """Resultados (figura principal). Referencia vs RF (arriba) y vs SVR (abajo) sobre las mismas 1241 ventanas de
    prueba de la semilla 27, ordenadas por participante, grabacion y win_start; curvas interrumpidas entre
    grabaciones; eje Y 40-140 BPM; MAE y r calculados desde las predicciones."""
    seed = SEMILLA_REPRESENTATIVA
    if semilla_mediana(M) != seed:
        raise RuntimeError("la semilla de desempeno mediano ya no es 27")
    g = D[D.seed == seed].sort_values(["participante", "seg_id", "win_start"]).reset_index(drop=True)
    rec = (g.participante + "_" + g.seg_id.astype(str)).values
    corte = np.r_[False, rec[1:] != rec[:-1]]
    n_rec = int(corte.sum() + 1)
    if not (len(g) == 1241 and n_rec == 249 == len(set(rec))):
        raise RuntimeError("el conjunto de prueba de la semilla 27 no tiene 1241 ventanas en 249 grabaciones contiguas")

    def tramos(v):                                   # NaN en cada cambio de grabacion
        x, y = [], []
        for i, (c, val) in enumerate(zip(corte, v)):
            if c:
                x.append(np.nan); y.append(np.nan)
            x.append(i); y.append(val)
        return np.array(x, float), np.array(y, float)

    v = g[["y", "RF", "SVR"]].values
    ylim = (np.floor(v.min() / 10) * 10, np.ceil(v.max() / 10) * 10)
    if ylim != (40.0, 140.0):
        raise RuntimeError(f"eje Y inesperado: {ylim}")
    res = {"semilla": seed, "ventanas": len(g), "grabaciones": n_rec, "ylim": list(ylim)}
    with estilo(ESTILO_COMPARACION):
        fig, axs = plt.subplots(2, 1, figsize=(13, 8.8), sharex=True, sharey=True, gridspec_kw={"hspace": 0.3})
        h = {}
        for ax, fam, nombre in zip(axs, ("RF", "SVR"), ("Random Forest", "SVR")):
            m = metrics(g.y.values, g[fam].values)
            xr, yr = tramos(g.y.values); xp, yp = tramos(g[fam].values)
            h["ref"], = ax.plot(xr, yr, color=INK, linewidth=1.1, label="Frecuencia cardíaca de referencia")
            h[fam], = ax.plot(xp, yp, color=COL[fam], linewidth=1.7, label=f"Frecuencia cardíaca estimada ({nombre})")
            ax.set_title(f"{nombre}: frecuencia cardíaca de referencia vs. estimada", loc="left", color=INK)
            ax.text(1.0, 1.015, f"MAE = {m['MAE']:.2f} BPM   r = {m['pearson_r']:.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=10.5, color=INK2)
            ax.set_ylabel("Frecuencia cardíaca (BPM)")
            ax.set_ylim(*ylim); ax.set_xlim(-5, len(g) + 4)
            ax.grid(True, axis="y", color=GRID, linewidth=0.8)
            res[fam] = {"MAE": round(m["MAE"], 4), "r": round(m["pearson_r"], 4)}
        axs[1].set_xlabel("Ventanas del conjunto de prueba")
        fig.suptitle("Frecuencia cardíaca de referencia y estimada en el conjunto de prueba", x=0.01, ha="left",
                     fontsize=13, color=INK, y=0.945)
        leg = fig.legend([h["ref"], h["RF"], h["SVR"]],
                         ["Frecuencia cardíaca de referencia", "Estimada por Random Forest", "Estimada por SVR"],
                         loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=10.5)
        fig.canvas.draw()
        if any(leg.get_window_extent().overlaps(ax.get_window_extent()) for ax in axs):
            raise RuntimeError("la leyenda se superpone con un panel")
        guardar(fig, FIG, "figura_comparacion_rf_svr_test")
    res["r_RF_SVR"] = round(float(np.corrcoef(g.RF, g.SVR)[0, 1]), 4)
    return res


def figura_mejora_respecto_b1(M):
    """Resultados (secundaria) o anexo. Reduccion del MAE respecto de B1 por semilla (> 0: menor MAE que B1)."""
    out = {}
    with estilo(ESTILO_RESULTADOS):
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        b1 = M[M.metodo == "B1"].sort_values("seed").MAE.values
        seeds = np.arange(33)
        ax.axhline(0, color=INK2, linewidth=1)
        for fam, mk, dx in (("RF", "o", -0.15), ("SVR", "s", 0.15)):
            d = b1 - M[M.metodo == fam].sort_values("seed").MAE.values
            ax.scatter(seeds + dx, d, s=30, marker=mk, color=COL[fam], edgecolor="white", linewidth=0.6, zorder=3, label=fam)
            out[fam] = {"media": round(float(d.mean()), 3), "min": round(float(d.min()), 3), "semillas_>0": int((d > 0).sum())}
        ax.set_xlabel("Semilla"); ax.set_ylabel("Reducción del MAE respecto de B1 (BPM)")
        ax.set_xticks(range(0, 33, 4)); ax.set_xlim(-1, 33)
        lo, hi = ax.get_ylim(); ax.set_ylim(min(lo, -0.1), hi)
        ax.legend(loc="upper right", ncol=2)
        guardar(fig, FIG, "figura_mejora_respecto_b1")
    return out


# ====================================================================== MATERIAL ADICIONAL (material_adicional/)
def ejemplos_por_grabacion(D, M):
    """Dos ejemplos (una figura cada uno, mismo eje Y): en la semilla mediana, la grabacion de prueba con MAE conjunto
    (RF+SVR)/2 mas cercano a la mediana por grabacion (desempate: participante, seg_id) y la otra grabacion de prueba
    del mismo participante. Se muestran referencia, SVR y B1."""
    seed = semilla_mediana(M)
    G = D[D.seed == seed]
    R = (G.groupby(["participante", "pos_id", "seg_id"])
         .apply(lambda g: pd.Series({"mae_conj": (np.mean(np.abs(g.RF - g.y)) + np.mean(np.abs(g.SVR - g.y))) / 2,
                                     "mae_svr": np.mean(np.abs(g.SVR - g.y)), "n": len(g)}), include_groups=False)
         .reset_index())
    R["dist"] = (R.mae_conj - R.mae_conj.median()).abs()
    sel = R.sort_values(["dist", "participante", "seg_id"], kind="mergesort").iloc[0]
    otra = R[(R.participante == sel.participante) & (R.seg_id != sel.seg_id)].iloc[0]
    ej = [G[(G.participante == r.participante) & (G.seg_id == r.seg_id)].sort_values("win_start") for r in (sel, otra)]
    allv = np.concatenate([g[["y", "SVR", "B1"]].values.ravel() for g in ej]); pad = 0.1 * np.ptp(allv)
    ylim = (allv.min() - pad, allv.max() + pad)
    info = {"semilla": seed}
    with estilo(ESTILO_RESULTADOS):
        for k, g in enumerate(ej, start=1):
            t = (g.win_start.values + C.WINDOW_SAMPLES / 2) / C.FS
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            ax.axhline(g.B1.iloc[0], color=COL["B1"], linestyle=(0, (2, 2)), linewidth=1.4, label="B1")
            ax.plot(t, g.y, color=COL["ref"], marker="D", markersize=6, linewidth=2, label="Referencia")
            ax.plot(t, g.SVR, color=COL["SVR"], marker="s", markersize=6, linewidth=2, markeredgecolor="white", label="SVR")
            ax.set_xlabel("Tiempo nominal del centro de la ventana (s)"); ax.set_ylabel("Frecuencia cardíaca (BPM)")
            ax.set_ylim(*ylim); ax.set_xticks(t); ax.set_xticklabels([f"{v:.1f}" for v in t])
            ax.legend(loc="lower right", ncol=3)
            guardar(fig, ADIC, f"figura_ejemplo_prediccion_{k}")
            info[f"ejemplo_{k}"] = {"participante": g.participante.iloc[0], "posicion": int(g.pos_id.iloc[0]),
                                    "seg_id": int(g.seg_id.iloc[0]), "MAE_SVR": round(float(np.mean(np.abs(g.SVR - g.y))), 3)}
    return info


def main():
    D = load()
    M, T = tables(D)
    M.to_csv(RES / "metricas_por_semilla.csv", index=False, float_format="%.10g")
    T.to_csv(RES / "metricas_finales.csv", index=False, float_format="%.10g")
    print(f"{len(D)} ventanas de prueba, {D.seed.nunique()} semillas")
    for _, r in T.iterrows():
        pr = r.Pearson_nota or f"{r.Pearson_media:.3f} ± {r.Pearson_de:.3f}"
        print(f"  {r.metodo:4s} MAE {r.MAE_media:.3f} ± {r.MAE_de:.3f} | RMSE {r.RMSE_media:.3f} ± {r.RMSE_de:.3f} | r {pr}")
    print("figuras/ sincronizacion:", json.dumps(figura_sincronizacion()))
    print("figuras/ MAE 33 corridas (medias):", json.dumps(figura_mae_33_corridas(M)))
    print("figuras/ comparacion RF/SVR:", json.dumps(figura_comparacion_rf_svr(D, M)))
    print("figuras/ mejora respecto de B1:", json.dumps(figura_mejora_respecto_b1(M)))
    print("material_adicional/ ejemplos:", json.dumps(ejemplos_por_grabacion(D, M)))


if __name__ == "__main__":
    main()
