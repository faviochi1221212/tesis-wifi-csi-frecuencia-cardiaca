"""
extraer_preprocesamiento.py -- genera datos_procesados/preprocesamiento_ejemplo.json, el insumo de las figuras
preprocesamiento_antes_despues y preprocesamiento_etapas_anexo. Necesita los datos crudos de eHealth CSI (PCAP);
generar_resultados.py solo lee el JSON resultante.

Rutas de los datos crudos: variable de entorno CSI_HR_PCAP_ROOT (ver codigo/src/csi_hr/config.py).

- Grabacion: la misma de la figura de sincronizacion (datos_procesados/sincronizacion_ejemplo.json), elegida sin mirar
  la senal CSI.
- Subportadora: la primera de las 20 que el pipeline selecciona automaticamente por energia en la banda de analisis
  (csi_hr.subcarriers.seleccionar_subportadoras sobre las 54 subportadoras activas preprocesadas).
- Se guarda solo lo minimo para reproducir las figuras: tiempos de los paquetes, amplitud |H| original de esa
  subportadora y las energias en banda de las 54 subportadoras (para auditar la seleccion). No se guarda el PCAP.

Uso (desde la raiz de la entrega):  python codigo/extraer_preprocesamiento.py
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codigo" / "src"))
from csi_hr import config as C, io_pcap, preprocessing as PP, subcarriers as SC  # noqa: E402

SINCRO = ROOT / "datos_procesados" / "sincronizacion_ejemplo.json"
OUT = ROOT / "datos_procesados" / "preprocesamiento_ejemplo.json"


def main():
    sel = json.loads(SINCRO.read_text(encoding="utf-8"))["seleccion"]
    part, pos = sel["participante"], sel["pos_id"]
    ts, H = io_pcap.read_position(part, pos)
    A = SC.arm_matrix(H, "E1a")                                         # |H| de las 54 subportadoras activas
    sos = PP.disenar_butterworth(C.FS)
    proc = np.column_stack([PP.pipeline_amplitud(A[:, j], sos) for j in range(A.shape[1])])
    idx, energias = SC.seleccionar_subportadoras(proc, C.FS)            # top-20 por energia en banda
    j = int(idx[0])
    out = {"descripcion": "Insumo de las figuras de preprocesamiento (generado por codigo/extraer_preprocesamiento.py)",
           "participante": part, "pos_id": pos,
           "subportadora_indice_activo": j, "subportadora_columna_234": int(SC.selected_columns_234("E1a", np.array([j]))[0]),
           "criterio": "primera de las 20 subportadoras seleccionadas automaticamente por energia en la banda de analisis",
           "top20_indices_activos": [int(i) for i in idx], "energias_banda_54": [float(e) for e in energias],
           "fs_nominal_hz": C.FS, "tiempo_s": (ts - ts[0]).tolist(), "amplitud_original": A[:, j].astype(float).tolist(),
           "control_pipeline_amplitud": proc[:, j].astype(float).tolist()}
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("participante", "pos_id", "subportadora_indice_activo", "subportadora_columna_234")}),
          f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
