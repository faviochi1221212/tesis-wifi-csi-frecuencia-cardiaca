"""
extraer_sincronizacion.py -- genera datos_procesados/sincronizacion_ejemplo.json, el insumo de la figura de
sincronizacion CSI-smartwatch (Metodologia). Es el UNICO script de la entrega que necesita los datos crudos de
eHealth CSI (PCAP y JSON del reloj); generar_resultados.py solo lee el JSON resultante.

Rutas de los datos crudos: variables de entorno CSI_HR_PCAP_ROOT y CSI_HR_WATCH_ROOT (ver codigo/src/csi_hr/config.py).

1) Para cada grabacion final (las 7397 ventanas de datos_procesados/dataset_completo_v3_synced.csv con posiciones
   estaticas y sync_ok) cuenta las lecturas del reloj dentro de cada ventana y las lecturas unicas dentro del
   intervalo de sus ventanas; comprueba que y_i recalculada (media de la interpolacion en los 231 paquetes)
   coincide con bpm_watch del dataset.
2) Criterio objetivo: entre las grabaciones con 5 ventanas y >= 2 lecturas en cada ventana, la de numero de lecturas
   unicas igual a la mediana (desempate: menor participante y seg_id).
3) Guarda la tabla del criterio y los valores exactos que se dibujan para la grabacion elegida.

Uso (desde la raiz de la entrega; ~15 min):  python codigo/extraer_sincronizacion.py
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codigo" / "src"))
from csi_hr import config as C, io_pcap, io_watch, splits  # noqa: E402

OUT = ROOT / "datos_procesados" / "sincronizacion_ejemplo.json"
WIN = C.WINDOW_SAMPLES


def main():
    d = splits.load_df_C()
    filas, cache = [], {}
    for part, g in d.groupby("participante"):
        tsw, hrw = io_watch.watch_all(part)
        for pos, gg in g.groupby("pos_id"):
            ts, _ = io_pcap.read_position(part, int(pos))
            interp = np.interp(ts, tsw, hrw, left=hrw[0], right=hrw[-1])
            for _, w in gg.sort_values("win_start").iterrows():
                i = int(w.win_start)
                if abs(float(np.mean(interp[i:i + WIN])) - float(w.bpm_watch)) > 1e-9:
                    raise RuntimeError(f"y_i recalculada distinta del dataset: {part} pos {pos} win {i}")
                filas.append({"participante": part, "pos_id": int(pos), "seg_id": int(w.seg_id),
                              "t_ini": ts[i], "t_fin": ts[i + WIN - 1],
                              "n_lect": int(((tsw >= ts[i]) & (tsw <= ts[i + WIN - 1])).sum())})
        cache[part] = tsw
    T = pd.DataFrame(filas)
    R = T.groupby(["participante", "pos_id", "seg_id"]).agg(nw=("n_lect", "size"), min_lect=("n_lect", "min"),
                                                            t0=("t_ini", "min"), t1=("t_fin", "max")).reset_index()
    R["unicas"] = [int(((cache[p] >= a) & (cache[p] <= b)).sum()) for p, a, b in zip(R.participante, R.t0, R.t1)]
    E = R[(R.nw == 5) & (R.min_lect >= 2)].copy(); med = float(E.unicas.median()); E["dist"] = (E.unicas - med).abs()
    s = E.sort_values(["dist", "participante", "seg_id"], kind="mergesort").iloc[0]
    part, pos = s.participante, int(s.pos_id)

    ts, _ = io_pcap.read_position(part, pos)
    tsw, hrw = io_watch.watch_all(part)
    interp = np.interp(ts, tsw, hrw, left=hrw[0], right=hrw[-1])
    g = d[(d.participante == part) & (d.pos_id == pos)].sort_values("win_start")
    t_ref = ts[0]; ventanas = []
    for _, w in g.iterrows():
        i = int(w.win_start); c = ts[min(i + WIN // 2, len(ts) - 1)]
        ventanas.append({"win_start": i, "ini": float(ts[i] - t_ref), "fin": float(ts[i + WIN - 1] - t_ref),
                         "centro": float(c - t_ref), "y": float(np.mean(interp[i:i + WIN])),
                         "dist_lectura_centro_s": float(np.min(np.abs(tsw - c)))})
    a, b = ventanas[0]["ini"] - 3, ventanas[-1]["fin"] + 3
    tt = ts - t_ref; vis = (tt >= a) & (tt <= b)
    mk = (tsw - t_ref >= a - 12) & (tsw - t_ref <= b + 12)
    out = {"descripcion": "Insumo de la figura de sincronizacion CSI-smartwatch (generado por codigo/extraer_sincronizacion.py)",
           "criterio": "grabaciones con 5 ventanas y >=2 lecturas del reloj por ventana; lecturas unicas = mediana; "
                       "desempate: menor participante y seg_id",
           "tabla_criterio": R[["participante", "pos_id", "seg_id", "nw", "min_lect", "unicas"]].to_dict(orient="list"),
           "mediana_lecturas_unicas": med,
           "seleccion": {"participante": part, "pos_id": pos, "seg_id": int(s.seg_id), "lecturas_unicas": int(s.unicas)},
           "xlim": [float(a), float(b)],
           "paquetes_t_s": tt[vis].tolist(), "paquetes_hr_interp": interp[vis].tolist(),
           "lecturas_t_s": (tsw[mk] - t_ref).tolist(), "lecturas_hr": hrw[mk].tolist(),
           "ventanas": ventanas}
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("mediana_lecturas_unicas", "seleccion")}), f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
