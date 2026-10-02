"""
generar_resultados.py -- metricas y figuras del experimento final (escenario personalizado), desde los datos
congelados. NO entrena modelos y NO modifica predicciones, splits ni datos.

Lee (solo lectura):
  resultados/predicciones_rf.csv.gz, resultados/predicciones_svr.csv.gz   (33 semillas x ventanas de prueba)
  resultados/configuraciones_seleccionadas.csv                           (suavizado w elegido por semilla)
Escribe:
  resultados/metricas_por_semilla.csv   MAE, RMSE y Pearson r por semilla y metodo (RF, SVR, B1, B0)
  resultados/metricas_finales.csv       resumen entre las 33 semillas (media, DE, IC95 del MAE)
  FIGURAS DE LA TESIS (figuras/):
    figura_mae_33_corridas                 Resultados: distribucion del MAE de RF, SVR, B1 y B0 en 33 corridas
    figura_comparacion_rf_svr_test         Resultados: referencia vs RF y vs SVR en el test de la semilla 27
    figura_mejora_respecto_b1              Resultados (secundaria) o anexo: reduccion del MAE respecto de B1
  FIGURAS METODOLOGICAS (codigo/figuras_metodologia.py; lee datos_procesados/*_ejemplo.json y, para controlar las
  etiquetas y_i, datos_procesados/dataset_completo_v3_synced.csv):
    figuras/diagrama_proceso_general, figuras/preprocesamiento_antes_despues,
    figuras/sincronizacion_csi_smartwatch, figuras/diagrama_calibracion,
    material_adicional/preprocesamiento_etapas_anexo
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
sys.path.insert(0, str(ROOT / "codigo"))
from csi_hr import config as C  # noqa: E402
import figuras_metodologia  # noqa: E402

RES, FIG, ADIC = ROOT / "resultados", ROOT / "figuras", ROOT / "material_adicional"
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


# ====================================================================== PIEZAS CON EL ESTILO DE LAS FIGURAS ANTIGUAS
# Tamano de insercion: 16 cm de ancho (6.3 in), fuentes de 8-9 pt, 300 dpi.
ESTILO_CLASICO = {"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                  "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.facecolor": "white",
                  "axes.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03}
AZUL_PRED, GRIS_REF, AZUL_CAB, AZUL_FILA = "#2196f3", "#8a8a8a", "#1565c0", "#e3f2fd"


def generar_predicho_vs_referencia_test(D, M):
    """Una sola figura: referencia (gris) y prediccion SVR (azul) en las 1241 ventanas de prueba de la semilla 27,
    ordenadas por participante -> grabacion -> win_start. El eje X es el indice ordenado ("Muestra"), no tiempo."""
    seed = SEMILLA_REPRESENTATIVA
    if semilla_mediana(M) != seed:
        raise RuntimeError("la semilla de desempeno mediano ya no es 27")
    g = D[D.seed == seed].sort_values(["participante", "seg_id", "win_start"]).reset_index(drop=True)
    if len(g) != 1241:
        raise RuntimeError(f"se esperaban 1241 ventanas de prueba y hay {len(g)}")
    m = metrics(g.y.values, g.SVR.values)
    x = np.arange(len(g))
    with estilo(ESTILO_CLASICO):
        fig, ax = plt.subplots(figsize=(6.3, 2.6))
        ax.plot(x, g.y.values, color=GRIS_REF, linewidth=0.8, label="BPM Referencia")
        ax.plot(x, g.SVR.values, color=AZUL_PRED, linewidth=0.9, label="BPM Predicho (SVR)")
        ax.set_title(f"BPM Predicho vs Referencia - Conjunto de Test\nMAE = {m['MAE']:.2f} BPM | r = {m['pearson_r']:.3f}",
                     fontweight="bold", fontsize=9)
        ax.set_xlabel("Muestra"); ax.set_ylabel("Frecuencia cardíaca (BPM)")
        ax.grid(True, color="#e0e0e0", linewidth=0.6); ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        guardar(fig, FIG, "figura_predicho_vs_referencia_test")
    return {"semilla": seed, "modelo": "SVR", "ventanas": len(g), "MAE": round(m["MAE"], 4), "r": round(m["pearson_r"], 4)}


# Valores publicados (fuentes primarias verificadas):
#   Alzaabi, Saied y Arslan (2025), IEEE JTEHM, doi:10.1109/JTEHM.2025.3624469 (PMC12599888): MAE 9.26 BPM (DWT/CWT, sin ML).
#   Kocheta, Bhatia y Obraczka (2025), arXiv:2510.24744v1, Tabla I (eHealth): MAE 0.27 +- 0.03 (10 s), 0.17 +- 0.01 (30 s).
LITERATURA = [("Alzaabi et al. (2025)", "DWT/CWT (sin ML)", "9.26", "—", "—"),
              ("Kocheta et al. (2025)\n— PulseFi", "LSTM", "0.27 ± 0.03 (10 s)\n0.17 ± 0.01 (30 s)", "—", "—")]


def generar_tabla_comparacion_literatura(M):
    """Tabla visual: literatura (valores publicados) y este trabajo (media +- DE de las 33 semillas, calculada desde
    metricas_por_semilla)."""
    def fmt(v):
        return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
    propias = []
    for fam, metodo in (("RF", "RF residual\n+ suavizado"), ("SVR", "SVR residual\n+ suavizado")):
        g = M[M.metodo == fam]
        if len(g) != 33:
            raise RuntimeError(f"{fam}: se esperaban 33 semillas")
        propias.append((f"Este trabajo ({fam})", metodo, fmt(g.MAE), fmt(g.RMSE), fmt(g.pearson_r)))
    filas = LITERATURA + propias
    with estilo(ESTILO_CLASICO):
        lineas = [1] + [max(c.count("\n") + 1 for c in f) for f in filas]          # lineas por fila (cabecera = 1)
        alto_linea, alto_min = 0.155, 0.27                                          # pulgadas
        altos = [max(alto_min, alto_linea * n + 0.1) for n in lineas]
        fig = plt.figure(figsize=(6.3, sum(altos) + 0.05))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")                            # la tabla ocupa los 16 cm de ancho
        tab = ax.table(cellText=[list(f) for f in filas], colLabels=["Estudio", "Método", "MAE", "RMSE", "CC"],
                       colWidths=[0.235, 0.19, 0.215, 0.18, 0.18], cellLoc="center", loc="upper center")
        tab.auto_set_font_size(False); tab.set_fontsize(8)
        alto_fig = sum(altos) + 0.05
        for (fila, col), celda in tab.get_celld().items():
            celda.set_edgecolor("#000000"); celda.set_linewidth(0.6)
            celda.set_height(altos[fila] / alto_fig)
            if fila == 0:
                celda.set_facecolor(AZUL_CAB); celda.get_text().set_color("white"); celda.get_text().set_fontweight("bold")
            elif filas[fila - 1][0].startswith("Este trabajo"):
                celda.set_facecolor(AZUL_FILA); celda.get_text().set_fontweight("bold")
            else:
                celda.set_facecolor("white")
        ax.set_title("Comparación con la literatura", fontweight="bold", fontsize=9, pad=4)
        guardar(fig, FIG, "tabla_comparacion_literatura")
    return {f[0]: {"MAE": f[2], "RMSE": f[3], "CC": f[4]} for f in propias}


def _tabla_visual(filas, cabecera, anchos, titulo, nombre, resaltar=()):
    """Tabla visual de 16 cm (encabezado azul, texto blanco, bordes finos, texto centrado, 8 pt). Las filas cuyo primer
    campo esta en 'resaltar' se muestran en azul claro y negrita."""
    with estilo(ESTILO_CLASICO):
        lineas = [1] + [max(c.count("\n") + 1 for c in f) for f in filas]
        altos = [max(0.27, 0.155 * n + 0.1) for n in lineas]                       # pulgadas por fila
        alto_fig = sum(altos) + 0.05
        fig = plt.figure(figsize=(6.3, alto_fig))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        tab = ax.table(cellText=[list(f) for f in filas], colLabels=list(cabecera), colWidths=list(anchos),
                       cellLoc="center", loc="upper center")
        tab.auto_set_font_size(False); tab.set_fontsize(8)
        for (fila, col), celda in tab.get_celld().items():
            celda.set_edgecolor("#000000"); celda.set_linewidth(0.6)
            celda.set_height(altos[fila] / alto_fig)
            if fila == 0:
                celda.set_facecolor(AZUL_CAB); celda.get_text().set_color("white"); celda.get_text().set_fontweight("bold")
            elif filas[fila - 1][0] in resaltar:
                celda.set_facecolor(AZUL_FILA); celda.get_text().set_fontweight("bold")
            else:
                celda.set_facecolor("white")
        ax.set_title(titulo, fontweight="bold", fontsize=9, pad=4)
        guardar(fig, FIG, nombre)


def generar_tabla_metricas_finales(M):
    """Tabla visual de resultados: media +- DE de las 33 semillas para RF, SVR, B1 y B0, calculada desde
    metricas_por_semilla (M). Pearson de B0 'No definido' (prediccion constante en todas las semillas)."""
    def fmt(v):
        return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
    filas = []
    for met in ("RF", "SVR", "B1", "B0"):
        g = M[M.metodo == met]
        if len(g) != 33:
            raise RuntimeError(f"{met}: se esperaban 33 semillas")
        if g.pearson_r.isna().all():
            r = "No definido"
        elif g.pearson_r.notna().all():
            r = fmt(g.pearson_r)
        else:
            raise RuntimeError(f"{met}: Pearson indefinido solo en algunas semillas")
        filas.append((met, fmt(g.MAE), fmt(g.RMSE), r))
    _tabla_visual(filas, ("Método", "MAE (BPM)", "RMSE (BPM)", "Pearson r"), (0.22, 0.26, 0.26, 0.26),
                  "Resultados de los modelos y líneas base en las 33 corridas", "tabla_metricas_finales")
    return {f[0]: {"MAE": f[1], "RMSE": f[2], "r": f[3]} for f in filas}


def main():
    D = load()
    M, T = tables(D)
    M.to_csv(RES / "metricas_por_semilla.csv", index=False, float_format="%.10g")
    T.to_csv(RES / "metricas_finales.csv", index=False, float_format="%.10g")
    print(f"{len(D)} ventanas de prueba, {D.seed.nunique()} semillas")
    for _, r in T.iterrows():
        pr = r.Pearson_nota or f"{r.Pearson_media:.3f} ± {r.Pearson_de:.3f}"
        print(f"  {r.metodo:4s} MAE {r.MAE_media:.3f} ± {r.MAE_de:.3f} | RMSE {r.RMSE_media:.3f} ± {r.RMSE_de:.3f} | r {pr}")
    print("figuras/ MAE 33 corridas (medias):", json.dumps(figura_mae_33_corridas(M)))
    print("figuras/ comparacion RF/SVR:", json.dumps(figura_comparacion_rf_svr(D, M)))
    print("figuras/ mejora respecto de B1:", json.dumps(figura_mejora_respecto_b1(M)))
    print("material_adicional/ ejemplos:", json.dumps(ejemplos_por_grabacion(D, M)))
    print("figuras metodologicas (fuentes en pt):", json.dumps(figuras_metodologia.generar(ROOT)))
    print("figuras/ predicho vs referencia (test):", json.dumps(generar_predicho_vs_referencia_test(D, M)))
    print("figuras/ tabla comparacion literatura:", json.dumps(generar_tabla_comparacion_literatura(M), ensure_ascii=False))
    print("figuras/ tabla metricas finales:", json.dumps(generar_tabla_metricas_finales(M), ensure_ascii=False))


if __name__ == "__main__":
    main()
