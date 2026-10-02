"""
figuras_metodologia.py -- figuras metodologicas de la tesis (las llama generar_resultados.py).

Todas se dibujan a su tamano de insercion (16 cm de ancho = 6.3 in), con fuentes de 8-9 pt, 300 dpi.
Lee solo artefactos de datos_procesados/ (no entrena ni modifica nada):
  sincronizacion_ejemplo.json    (codigo/extraer_sincronizacion.py)
  preprocesamiento_ejemplo.json  (codigo/extraer_preprocesamiento.py)

  figuras/diagrama_proceso_general            Metodologia: flujo completo
  figuras/preprocesamiento_antes_despues      Metodologia: una subportadora real antes y despues
  figuras/sincronizacion_csi_smartwatch       Metodologia: lecturas del reloj -> interpolacion -> ventanas -> y_i
  figuras/diagrama_calibracion                Experimentacion: calibracion secuencial (Etapa 1 -> Etapa 2)
  material_adicional/preprocesamiento_etapas_anexo   Anexo: las cinco etapas del preprocesamiento
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy import signal

from csi_hr import config as C, preprocessing as PP

ANCHO = 6.3                                              # 16 cm
INK, INK2, GRID, FILL, ACC, ACC_L = "#0b0b0b", "#52514e", "#e4e3df", "#f4f3ef", "#2a78d6", "#dce9fa"
ESTILO_METODOLOGIA = {"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                      "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                      "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                      "ytick.color": INK2, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
                      "savefig.pad_inches": 0.03, "figure.facecolor": "white", "axes.facecolor": "white"}
R = 0.06                                                 # radio de las esquinas de las cajas (in)


class _estilo:
    """Restaura la configuracion de arranque de matplotlib y aplica el estilo de las figuras metodologicas."""
    def __init__(self):
        self.ctx = plt.rc_context()

    def __enter__(self):
        self.ctx.__enter__(); matplotlib.rc_file_defaults(); plt.rcParams.update(ESTILO_METODOLOGIA)

    def __exit__(self, *exc):
        return self.ctx.__exit__(*exc)


def _guardar(fig, carpeta, nombre):
    """PNG (versionado) y PDF (local). Devuelve el rango de tamanos de fuente visibles, en pt al tamano de insercion."""
    carpeta.mkdir(exist_ok=True)
    fig.canvas.draw()
    tam = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    if min(tam) < 8:
        raise RuntimeError(f"{nombre}: texto menor de 8 pt")
    for ext in ("png", "pdf"):
        fig.savefig(carpeta / f"{nombre}.{ext}")
    plt.close(fig)
    return {"min_pt": min(tam), "max_pt": max(tam)}


def _lienzo(alto):
    """Ejes en pulgadas reales (1 unidad = 1 in) para colocar cajas y flechas con precision."""
    fig = plt.figure(figsize=(ANCHO, alto)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, ANCHO); ax.set_ylim(0, alto); ax.axis("off")
    return fig, ax


def _caja(ax, cx, cy, w, h, txt, fill=FILL, edge=INK2, fs=8.5, peso="normal", dashed=False, lw=0.8):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={R}",
                                facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=(0, (3, 2)) if dashed else "-"))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs, color=INK, fontweight=peso, linespacing=1.2)


def _flecha(ax, p, q, color=INK2):
    ax.annotate("", xy=q, xytext=p, arrowprops={"arrowstyle": "-|>,head_length=0.35,head_width=0.18", "color": color,
                                                "linewidth": 0.9, "shrinkA": 0, "shrinkB": 0})


# ====================================================================== diagrama general
def diagrama_proceso_general(fig_dir):
    with _estilo():
        fig, ax = _lienzo(2.75)
        fases = [("Datos", ["eHealth CSI\n+ smartwatch", "Selección de\nregistros y posiciones", "Sincronización\ntemporal"]),
                 ("Señal CSI", ["Preprocesamiento", "Selección de\nsubportadoras", "Segmentación\nen ventanas",
                                "Extracción de\ncaracterísticas"]),
                 ("Modelado", ["Partición por\ngrabación", "Regresión\nresidual", "Random Forest\n/ SVR", "Suavizado"]),
                 ("Resultado", ["Estimación de la\nfrecuencia cardíaca", "MAE, RMSE\ny Pearson r"])]
        BW, BH, GAPX = 1.42, 0.42, 0.18
        x0 = (ANCHO - (4 * BW + 3 * GAPX)) / 2
        xs = [x0 + BW / 2 + k * (BW + GAPX) for k in range(4)]
        y_tit, y_top, paso = 2.58, 2.12, 0.56
        ultimo = None
        for (fase, pasos), cx in zip(fases, xs):
            ax.text(cx, y_tit, fase, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
            ax.plot([cx - BW / 2, cx + BW / 2], [y_tit - 0.14, y_tit - 0.14], color=ACC, linewidth=1.6, solid_capstyle="butt")
            ys = [y_top - k * paso for k in range(len(pasos))]
            for k, (txt, cy) in enumerate(zip(pasos, ys)):
                _caja(ax, cx, cy, BW, BH, txt, fill=ACC_L if fase == "Resultado" else FILL)
                if k:
                    _flecha(ax, (cx, ys[k - 1] - BH / 2), (cx, cy + BH / 2))
            if ultimo is not None:                   # sale por la derecha del ultimo bloque y entra por la izquierda del primero
                px, py = ultimo; xm = px + BW / 2 + GAPX / 2
                ax.plot([px + BW / 2, xm, xm], [py, py, y_top], color=INK2, linewidth=0.9, solid_joinstyle="miter")
                _flecha(ax, (xm, y_top), (cx - BW / 2, y_top))
            ultimo = (cx, ys[-1])
        return _guardar(fig, fig_dir, "diagrama_proceso_general")


# ====================================================================== preprocesamiento
def _etapas_preprocesamiento(P):
    """Recalcula las etapas con el codigo del pipeline y comprueba que coinciden con pipeline_amplitud."""
    e_sel = np.array(P["energias_banda_54"])
    if int(np.argsort(e_sel)[::-1][0]) != P["subportadora_indice_activo"]:
        raise RuntimeError("la subportadora guardada no es la de mayor energia en banda")
    sos = PP.disenar_butterworth(C.FS)
    x_raw = np.array(P["amplitud_original"], dtype=float)
    e1 = signal.detrend(x_raw, type="linear"); e2 = PP.hampel_filter(e1); e3 = signal.sosfiltfilt(sos, e2)
    e4 = 2.0 * (e3 - e3.min()) / (e3.max() - e3.min()) - 1.0
    if not (np.allclose(e4.astype(np.float32), PP.pipeline_amplitud(x_raw, sos))
            and np.allclose(e4.astype(np.float32), np.array(P["control_pipeline_amplitud"], dtype=np.float32))):
        raise RuntimeError("las etapas no reproducen pipeline_amplitud")
    return np.array(P["tiempo_s"]), x_raw, e1, e2, e3, e4


def preprocesamiento(P, fig_dir, adic_dir):
    t, x_raw, e1, e2, e3, e4 = _etapas_preprocesamiento(P)
    res = {}
    with _estilo():
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(ANCHO, 3.5), sharex=True, gridspec_kw={"hspace": 0.42})
        a1.plot(t, x_raw, color=INK2, linewidth=0.7)
        a1.set_title("Señal CSI antes del preprocesamiento", loc="left", color=INK)
        a1.set_ylabel("Amplitud |H|")
        a2.plot(t, e4, color=ACC, linewidth=0.8)
        a2.set_title("Señal CSI después del preprocesamiento", loc="left", color=INK)
        a2.set_ylabel("Amplitud\nnormalizada"); a2.set_ylim(-1.15, 1.15); a2.set_yticks([-1, 0, 1])
        a2.set_xlabel("Tiempo (s)"); a2.set_xlim(t[0] - 0.5, t[-1] + 0.5)
        for a in (a1, a2):
            a.grid(True, axis="y", color=GRID, linewidth=0.6)
        fig.align_ylabels([a1, a2])
        res["preprocesamiento_antes_despues"] = _guardar(fig, fig_dir, "preprocesamiento_antes_despues")

        fig, axs = plt.subplots(5, 1, figsize=(ANCHO, 7.2), sharex=True, gridspec_kw={"hspace": 0.62})
        etapas = [(x_raw, "1. Amplitud |H| original"), (e1, "2. Detrend lineal"), (e2, "3. Filtro de Hampel"),
                  (e3, "4. Filtro Butterworth pasabanda"), (e4, "5. Normalización min–max")]
        for a, (v, tit) in zip(axs, etapas):
            a.plot(t, v, color=ACC if tit.startswith("5") else INK2, linewidth=0.7)
            if tit.startswith("3"):
                m = e2 != e1
                a.scatter(t[m], e1[m], s=9, facecolor="none", edgecolor="#eb6834", linewidth=0.7, zorder=3,
                          label="Valor reemplazado")
                a.legend(loc="upper right", handletextpad=0.2, borderaxespad=0.1)
            a.set_title(tit, loc="left", color=INK); a.grid(True, axis="y", color=GRID, linewidth=0.6)
        axs[-1].set_xlabel("Tiempo (s)"); axs[-1].set_xlim(t[0] - 0.5, t[-1] + 0.5)
        res["preprocesamiento_etapas_anexo"] = _guardar(fig, adic_dir, "preprocesamiento_etapas_anexo")
    res["hampel_reemplazados"] = int(np.sum(e2 != e1))
    return res


# ====================================================================== sincronizacion
def _validar_sincronizacion(J):
    """El criterio objetivo vuelve a elegir la grabacion guardada y las y_i coinciden con el dataset congelado."""
    import pandas as pd
    from csi_hr import splits
    R_ = pd.DataFrame(J["tabla_criterio"])
    E = R_[(R_.nw == 5) & (R_.min_lect >= 2)].copy(); E["dist"] = (E.unicas - E.unicas.median()).abs()
    s = E.sort_values(["dist", "participante", "seg_id"], kind="mergesort").iloc[0]
    sel = J["seleccion"]
    if (s.participante, int(s.pos_id), int(s.seg_id)) != (sel["participante"], sel["pos_id"], sel["seg_id"]):
        raise RuntimeError("el criterio de seleccion no reproduce la grabacion guardada")
    d = splits.load_df_C()
    y_ds = d[(d.participante == sel["participante"]) & (d.pos_id == sel["pos_id"])].sort_values("win_start").bpm_watch.values
    if not np.allclose([w["y"] for w in J["ventanas"]], y_ds, rtol=0, atol=1e-9):
        raise RuntimeError("las etiquetas y_i del JSON no coinciden con el dataset")


def sincronizacion(J, fig_dir):
    _validar_sincronizacion(J)
    win = J["ventanas"]; a_x, b_x = J["xlim"]
    with _estilo():
        fig, (ax, axw) = plt.subplots(2, 1, figsize=(ANCHO, 3.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.28})
        ax.plot(J["paquetes_t_s"], J["paquetes_hr_interp"], color="#9a9993", linewidth=1.2, label="Interpolación lineal", zorder=1)
        ax.scatter(J["lecturas_t_s"], J["lecturas_hr"], s=34, color=INK, zorder=3, label="Lecturas del smartwatch")
        ax.set_ylabel("FC (BPM)"); ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.legend(loc="upper right", ncol=2, handletextpad=0.3, columnspacing=1.2)
        ylo, yhi = min(J["lecturas_hr"]) - 1.2, max(J["lecturas_hr"]) + 1.6; ax.set_ylim(ylo, yhi)
        n = len(win)
        for k, w in enumerate(win):
            lane = n - k
            axw.barh(lane, w["fin"] - w["ini"], left=w["ini"], height=0.62, color=ACC_L, edgecolor=ACC, linewidth=0.6)
            axw.plot([w["centro"]] * 2, [lane - 0.31, lane + 0.31], color=ACC, linewidth=1.4)
            axw.text(w["centro"] + 0.7, lane, f"$y_{k + 1}$ = {w['y']:.1f}", ha="left", va="center", fontsize=8, color=INK)
        w3 = win[2]; ly = n - 2 + 0.47              # criterio de +-15 s ilustrado solo en V3
        axw.annotate("", xy=(w3["centro"] - 15, ly), xytext=(w3["centro"] + 15, ly),
                     arrowprops={"arrowstyle": "|-|,widthA=0.25,widthB=0.25", "color": INK2, "linewidth": 0.7})
        axw.text(w3["centro"] + 15.6, ly, "±15 s del centro:\nal menos una lectura", ha="left", va="center", fontsize=8,
                 color=INK2, linespacing=1.1)
        axw.set_title("Ventanas CSI y referencia $y_i$ de cada ventana (BPM)", loc="left", fontsize=8.5, color=INK, pad=3)
        axw.set_yticks(range(1, n + 1)); axw.set_yticklabels([f"V{n - i + 1}" for i in range(1, n + 1)])
        axw.set_ylim(0.45, n + 0.85); axw.tick_params(axis="y", length=0)
        axw.spines["left"].set_visible(False)
        axw.set_xlabel("Tiempo (s)"); axw.set_xlim(a_x, b_x)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        fig.align_ylabels([ax, axw])
        return _guardar(fig, fig_dir, "sincronizacion_csi_smartwatch")


# ====================================================================== calibracion
def diagrama_calibracion(fig_dir):
    with _estilo():
        fig, ax = _lienzo(2.45)
        bw, bh, yc = 1.22, 0.86, 1.52
        cx = [0.70, 2.30, 3.82, 5.52]
        band_x0, band_x1 = cx[1] - bw / 2 - 0.17, cx[2] + bw / 2 + 0.17      # banda de la validacion interna
        ax.add_patch(FancyBboxPatch((band_x0, yc - bh / 2 - 0.12), band_x1 - band_x0, bh + 0.48,
                                    boxstyle=f"round,pad=0,rounding_size={R}", facecolor="white", edgecolor=ACC, linewidth=0.9))
        ax.text((band_x0 + band_x1) / 2, yc + bh / 2 + 0.2, "Validación interna: 5 folds agrupados por grabación",
                ha="center", va="center", fontsize=8, color=ACC)
        _caja(ax, cx[0], yc, bw, bh, "Datos de\nentrenamiento\nde la semilla")
        _caja(ax, cx[1], yc, bw, bh, "Etapa 1\nSelección de\ncaracterísticas\n(MI o PI, N, w)", fill=ACC_L)
        _caja(ax, cx[2], yc, bw, bh, "Etapa 2\nAjuste de\nhiperparámetros\n(RF o SVR, w)", fill=ACC_L)
        _caja(ax, cx[3], yc, 1.30, bh, "Reentrenamiento\ncon todo el\nentrenamiento")
        _flecha(ax, (cx[0] + bw / 2, yc), (band_x0, yc))
        _flecha(ax, (cx[1] + bw / 2, yc), (cx[2] - bw / 2, yc))
        _flecha(ax, (band_x1, yc), (cx[3] - 0.65, yc))
        ax.text((band_x0 + band_x1) / 2, yc - bh / 2 - 0.27,
                "Criterio: menor MAE de validación interna  ·  PI calculada con un RF base",
                ha="center", va="center", fontsize=8, color=INK2)
        _caja(ax, cx[3], 0.30, 1.30, 0.50, "Predicción única\ndel conjunto\nde prueba", fill="white", dashed=True)
        _flecha(ax, (cx[3], yc - bh / 2), (cx[3], 0.30 + 0.25))
        ax.text(cx[3] - 0.80, 0.30, "El conjunto de prueba no\ninterviene en la calibración", ha="right", va="center",
                fontsize=8, color=INK2)
        return _guardar(fig, fig_dir, "diagrama_calibracion")


def generar(root):
    """Genera las cinco figuras metodologicas. root = raiz de la entrega."""
    fig_dir, adic_dir = root / "figuras", root / "material_adicional"
    J = json.loads((root / "datos_procesados" / "sincronizacion_ejemplo.json").read_text(encoding="utf-8"))
    P = json.loads((root / "datos_procesados" / "preprocesamiento_ejemplo.json").read_text(encoding="utf-8"))
    if (P["participante"], P["pos_id"]) != (J["seleccion"]["participante"], J["seleccion"]["pos_id"]):
        raise RuntimeError("la grabacion del ejemplo de preprocesamiento no coincide con la de sincronizacion")
    return {"diagrama_proceso_general": diagrama_proceso_general(fig_dir), **preprocesamiento(P, fig_dir, adic_dir),
            "sincronizacion_csi_smartwatch": sincronizacion(J, fig_dir), "diagrama_calibracion": diagrama_calibracion(fig_dir)}
