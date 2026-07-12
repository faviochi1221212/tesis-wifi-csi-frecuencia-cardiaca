"""
03_extraer_csi.py
Tesis: Monitoreo no invasivo de la frecuencia cardiaca mediante senales Wi-Fi CSI
Universidad de Lima - Ingenieria de Sistemas
Autores: Yadhira Sarmiento Escobar (20214190) y Favio Chavarry Minaya (20214680)

Extrae CSI complejo desde archivos .pcap usando CSIKit.
Regla de agrupacion:
  - Solo archivos con prefijo NUMERICO (ej: "1_2022...") se procesan.
  - Varios PCAP con el mismo numero de posicion -> se concatenan en orden temporal.
  - Archivos con prefijo no numerico (VAZIO, julio3, S_F_...) se omiten.

Salida: csi_csv/{participante}/pos_{N}.csv
  Columnas: timestamp (epoch s), sub_000 ... sub_233 (234 valores complejos, sin guard ni pilotos)

Requisito: pip install CSIKit
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

try:
    from CSIKit.reader import get_reader
except ImportError:
    print("ERROR: CSIKit no esta instalado.")
    print("  Ejecuta: pip install CSIKit")
    raise SystemExit(1)

# ─── RUTAS ────────────────────────────────────────────────────────────────────
PCAP_DIR  = r"C:\Users\LENOVO\Desktop\tesis-wifi-csi-frecuencia-cardiaca-main\Data"
OUT_DIR   = r"C:\Users\LENOVO\Desktop\csi_csv"

# ─── PARAMETROS ───────────────────────────────────────────────────────────────
MIN_FRAMES   = 100   # descartar archivos con muy pocas tramas (< ~13 segundos)
REESCRIBIR   = False # True = regenerar aunque el CSV ya exista

# ─── SUBPORTADORAS A ELIMINAR — 802.11ac 80 MHz (256 FFT, BCM43455c0) ─────────
# CSIKit devuelve 256 subportadoras sin filtrar. Se eliminan:
#
#  1) Guard bands (IEEE 802.11ac 80 MHz — siempre cero en hardware):
#     Guard inferior : indices   0-5   (-128 a -123)  6 subportadoras
#     Zona DC        : indices 127-129 ( -1, 0, +1)   3 subportadoras
#     Guard superior : indices 251-255 (+123 a +127)  5 subportadoras
#
#  2) Pilotos (referencia de canal, no varían con el cuerpo):
#     Calculo IEEE: piloto en freq f -> indice = f + 128
#     Pilotos 802.11ac 80MHz en ±103,±75,±39,±11 → indices 25,53,89,117,139,167,203,231
#     (CSIKit constants.py tiene un error de ±1 en estos valores)
#
# Total eliminadas: 6+3+5+8 = 22  ->  256-22 = 234 subportadoras de datos
# (coincide con el pipeline original csi_filtrado_v2)
_GUARD  = list(range(0, 6)) + [127, 128, 129] + list(range(251, 256))
_PILOTS = [25, 53, 89, 117, 139, 167, 203, 231]
_ELIMINAR = set(_GUARD + _PILOTS)

MASK_VALIDO = np.ones(256, dtype=bool)
for _i in _ELIMINAR:
    MASK_VALIDO[_i] = False
# Resultado: 234 subportadoras de datos puros

print("=" * 60)
print("  03 - EXTRACCION CSI DESDE ARCHIVOS .PCAP")
print("=" * 60)
print(f"  Entrada : {PCAP_DIR}")
print(f"  Salida  : {OUT_DIR}")

os.makedirs(OUT_DIR, exist_ok=True)

participantes = sorted(os.listdir(PCAP_DIR))
total_ok   = 0
total_skip = 0
total_err  = 0

for part in participantes:
    dir_part = os.path.join(PCAP_DIR, part)
    if not os.path.isdir(dir_part):
        continue

    # ─ Agrupar PCAP por numero de posicion ──────────────────────────────────
    # Las posturas numeradas (1-17) siguen el protocolo estandar del dataset
    # eHealth CSI. Archivos con prefijo no numerico (VAZIO=ambiente vacio,
    # julio3, S_F_*, etc.) son grabaciones de prueba/control fuera del
    # protocolo de posturas -> no tienen una posicion comparable entre
    # participantes, se omiten para no mezclar condiciones distintas.
    grupos = defaultdict(list)   # {pos_num: [ruta1, ruta2, ...]}
    for archivo in os.listdir(dir_part):
        if not archivo.endswith(".pcap"):
            continue
        # Solo prefijos numericos: "1_2022...", "10_2022..." etc.
        m = re.match(r'^(\d+)_', archivo)
        if not m:
            total_skip += 1
            continue
        pos_num = int(m.group(1))
        grupos[pos_num].append(os.path.join(dir_part, archivo))

    if not grupos:
        continue

    dir_out = os.path.join(OUT_DIR, part)
    os.makedirs(dir_out, exist_ok=True)

    for pos_num in sorted(grupos.keys()):
        out_csv = os.path.join(dir_out, f"pos_{pos_num}.csv")
        if os.path.exists(out_csv) and not REESCRIBIR:
            total_ok += 1
            continue

        # Ordenar los PCAP por timestamp del nombre (orden cronologico)
        rutas_ordenadas = sorted(grupos[pos_num])

        all_timestamps = []
        all_csi        = []
        n_sub_ref      = None

        for ruta in rutas_ordenadas:
            try:
                reader = get_reader(ruta)
                result = reader.read_file(ruta, scaled=True)
            except Exception as e:
                print(f"  [ERR lectura] {part}/{os.path.basename(ruta)}: {e}")
                continue

            if not result.frames:
                continue

            for frame in result.frames:
                try:
                    mat = np.array(frame.csi_matrix)  # (n_sub, 1) — NEXBeamformReader
                    if mat.ndim < 2 or mat.shape[0] < 1:
                        continue
                    csi_raw = mat[:, 0].flatten()   # 256 subcarriers del unico stream

                    # Eliminar guard bands y pilotos (802.11ac 80 MHz)
                    if len(csi_raw) == 256:
                        csi_vec = csi_raw[MASK_VALIDO]   # -> 234 subportadoras de datos
                    else:
                        csi_vec = csi_raw                # otro ancho de banda: pasar tal cual

                    n_sub   = len(csi_vec)

                    # Si una trama trae un numero de subportadoras distinto al de
                    # las anteriores (frame corrupto/anomalo del propio hardware),
                    # se descarta en vez de forzarla y romper la forma del array.
                    if n_sub_ref is None:
                        n_sub_ref = n_sub
                    elif n_sub != n_sub_ref:
                        continue   # subportadoras inconsistentes -> omitir trama

                    # Eliminar tramas con valores nulos o todo ceros
                    if np.any(np.isnan(csi_vec)) or np.any(np.isinf(csi_vec)):
                        continue
                    if np.all(csi_vec == 0):
                        continue

                    all_timestamps.append(float(frame.timestamp))
                    all_csi.append(csi_vec)
                except Exception:
                    continue

        # MIN_FRAMES=100 (~13s a 7.7Hz): grabaciones mas cortas no alcanzan ni
        # para una sola ventana de feature extraction (231 muestras = 30s), asi
        # que no vale la pena conservarlas.
        if len(all_csi) < MIN_FRAMES:
            print(f"  [SKIP] {part}/pos_{pos_num}: solo {len(all_csi)} tramas (min={MIN_FRAMES})")
            total_skip += 1
            continue

        # Convertir a DataFrame y guardar
        csi_arr  = np.array(all_csi)           # (N, n_sub) complejo
        if csi_arr.ndim == 3:
            csi_arr = csi_arr[:, :, 0]         # quitar dimension extra si existe
        n_sub    = csi_arr.shape[1]
        sub_cols = [f"sub_{i:03d}" for i in range(n_sub)]

        # CSV no tiene tipo nativo para numeros complejos: se guardan como texto
        # "(a+bj)" y se reconstruyen en 04_preprocesamiento.py con complex(str).
        # Convertir toda la matriz de una sola vez (evita error de dimensiones)
        csi_str  = csi_arr.astype(str)         # (N, n_sub) string
        df_csi   = pd.DataFrame(csi_str, columns=sub_cols)
        df_csi.insert(0, "timestamp", all_timestamps)

        # Cuando una posicion tiene varios PCAP (ej. grabacion se corto y se
        # reinicio), puede haber tramas duplicadas en el limite entre archivos.
        # Se eliminan duplicados exactos por timestamp y se reordena
        # cronologicamente (la concatenacion de varios PCAP no garantiza orden).
        df_csi = df_csi.drop_duplicates(subset=["timestamp"])
        df_csi = df_csi.sort_values("timestamp").reset_index(drop=True)

        df_csi.to_csv(out_csv, index=False)

        total_ok += 1
        print(f"  OK  {part}/pos_{pos_num}.csv  ({len(all_timestamps)} tramas, {n_sub} subportadoras)")

# ─── RESUMEN ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Archivos generados : {total_ok}")
print(f"  Omitidos           : {total_skip}")
print(f"  Errores            : {total_err}")
print(f"\n  CSV complejos en   : {OUT_DIR}")
print("\nScript 03 completado.")
print("  Siguiente paso: python 04_preprocesamiento.py\n")
