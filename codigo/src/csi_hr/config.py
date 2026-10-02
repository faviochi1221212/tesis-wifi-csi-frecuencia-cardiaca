"""
Configuracion centralizada. Todos los valores se extrajeron del codigo existente (se indica el origen);
ninguno es nuevo.

Rutas: sin rutas absolutas en el codigo. PROJECT_ROOT se deduce de la ubicacion de este archivo.
Los datos crudos viven fuera del repo y se configuran con (en este orden de prioridad):
  1. variables de entorno CSI_HR_PCAP_ROOT y CSI_HR_WATCH_ROOT;
  2. el archivo local NO versionado <PROJECT_ROOT>/csi_hr_local.json, p. ej.
       {"PCAP_ROOT": "D:/datos/Data_DS1_raspberry-main/Data",
        "WATCH_ROOT": "D:/datos/Data_DS1_smartwatch-main/Data"}
"""
import json
import os
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ rutas
PROJECT_ROOT = Path(os.environ.get("CSI_HR_PROJECT_ROOT", Path(__file__).resolve().parents[3]))   # raiz de la entrega
RESULTS_ROOT = PROJECT_ROOT / "reproduccion"    # entrega: salida de nuevas ejecuciones (nunca resultados/ entregados)
REPORTS_ROOT = PROJECT_ROOT / "reports"
LOCAL_CONFIG = PROJECT_ROOT / "csi_hr_local.json"

# archivos historicos de referencia (solo lectura)
DATASET_PUBLISHED = PROJECT_ROOT / "datos_procesados" / "dataset_completo_v3_synced.csv"   # entrega: artefacto congelado
PERMUTATION_IMPORTANCE_CSV = RESULTS_ROOT / "permutation_importance_features.csv"  # generado por 23
E1_DIR = PROJECT_ROOT / "datos_procesados"      # entrega: features_rf_3brazos.csv.gz y splits_rf.json
E5_DIR = RESULTS_ROOT / "fase_C" / "E5"


def _local_value(key):
    if LOCAL_CONFIG.exists():
        return json.loads(LOCAL_CONFIG.read_text(encoding="utf-8")).get(key)
    return None


def _data_root(key, env):
    v = os.environ.get(env) or _local_value(key)
    if not v:
        raise RuntimeError(f"Ruta de datos '{key}' no configurada: define la variable de entorno {env} "
                           f"o la clave '{key}' en {LOCAL_CONFIG} (archivo local no versionado).")
    p = Path(v)
    if not p.is_dir():
        raise RuntimeError(f"{key} = {p} no existe")
    return p


def get_pcap_root():
    """Carpeta Data/ de Data_DS1_raspberry-main (un subdirectorio por participante con los .pcap)."""
    return _data_root("PCAP_ROOT", "CSI_HR_PCAP_ROOT")


def get_watch_root():
    """Carpeta Data/ de Data_DS1_smartwatch-main (un subdirectorio por participante con *HeartRateData.json)."""
    return _data_root("WATCH_ROOT", "CSI_HR_WATCH_ROOT")


# ------------------------------------------------------------------ posiciones (tesis)
# Origen: 25_corridas_repetidas.py L75 (POSICIONES_SEDENTARIAS), E1_rf.py L45 (POS_SED), A3 L38 (STAT).
STATIC_POSITIONS = frozenset({1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})
NORMAL_BREATHING_POSITIONS = frozenset({1, 4, 6, 8, 10, 12})
ALTERNATE_BREATHING_POSITIONS = frozenset({2, 5, 7, 9, 11, 13})
# macro-posturas usadas como oraculo diagnostico en E5 (E5_generalizacion.py L31)
MACRO_POSTURE = {1: "G1", 2: "G1", 4: "G1", 5: "G1", 6: "G2", 7: "G2", 8: "G2", 9: "G2", 10: "G3", 11: "G3", 12: "G3", 13: "G3"}

# ------------------------------------------------------------------ muestreo y ventanas
# Origen: E1_rf.py L42-44 y 12_features_modelos_v3_synced.py L72-80.
FS = 7.7                                        # fs con la que se genero el dataset publicado (validado en A.1b)
WINDOW_SEC = 30.0
WINDOW_OVERLAP = 0.75
WINDOW_SAMPLES = int(WINDOW_SEC * FS)           # 231
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - WINDOW_OVERLAP))   # 57  (E1_rf: int(WIN * 0.25))
GAP_SEC = 1.0                                   # huecos > 1 s parten la captura en segmentos (seg_id)
MAX_DIST_WATCH_S = 15.0                         # sync_ok: lectura del reloj a <= 15 s del centro de la ventana
HR_MIN_BPM, HR_MAX_BPM = 40, 180                # filtro del reloj (A3.watch_all) y de la etiqueta de ventana (E1_rf L106)
MIN_CAPTURE_PACKETS = 100                       # E1_rf.build_features L126

# ------------------------------------------------------------------ protocolo personalizado (25 / E1)
TEST_FRAC = 0.20
N_RUNS = 33                                     # semillas de split = 0..32
# protocolo sujeto-independiente (E5)
N_FOLDS_SUBJECT = 5
SEED_BOOTSTRAP = 20260927                       # E1_comparar.SEED_BOOT, E5_comparar.SEED

# ------------------------------------------------------------------ subportadoras
# Origen: A1_auditoria_top20_subportadoras.py L31-34 (mascara de 03_extraer_csi.py L55-62) y E1_rf.py L48.
_ELIMINATED_BUFFER = set(list(range(0, 6)) + [127, 128, 129] + list(range(251, 256)) + [25, 53, 89, 117, 139, 167, 203, 231])
COL2BUF = np.array([b for b in range(256) if b not in _ELIMINATED_BUFFER])        # columna 0..233 -> indice de buffer
ACTIVE_BUFFER = frozenset(set(range(132, 189)) - {160})                            # bloque activo (FASE_AUDITORIA_ISPD.md)
ACTIVE_SUBCARRIERS = np.where(np.isin(COL2BUF, sorted(ACTIVE_BUFFER)))[0]          # 54 columnas (0..233) usadas por E1a
N_TOP_SUBCARRIERS = 20                          # 12 L74 N_SUBCAR

# ------------------------------------------------------------------ filtrado (04_preprocesamiento.py L45-50)
BUTTER_LOW_HZ = 0.8
BUTTER_HIGH_HZ = 2.17
BUTTER_ORDER = 5
HAMPEL_K = 10
HAMPEL_THRESHOLD = 3.0

# ------------------------------------------------------------------ features (HISTORICAS)
# Top-10 de permutation_importance_features.csv (generado por 23_seleccion_mi.py). ATENCION: el numero N=10 se
# eligio en 23 comparando MAE sobre un TEST (23 L156-179). Es la configuracion historica de E1a, no una seleccion
# metodologicamente limpia.
HISTORICAL_FEATURES_E1A = ("energia", "rms", "varianza", "std", "entropia", "e_banda", "pico_pico",
                           "snr_cardiac", "kurtosis_s", "wav_e1")

# ------------------------------------------------------------------ modelos (HISTORICOS, sin tuning nuevo)
SMOOTH_WINDOW = 9                               # suavizado centrado (25 L63, E1_rf L47)
# Origen: E1_rf.py L46 (= 25 L66-69). Procedencia del tuning NO trazable en el repo actual (ver auditoria).
RF_PARAMS_HISTORICAL = {"max_depth": 6, "max_features": "sqrt", "min_samples_leaf": 29, "n_estimators": 387,
                        "random_state": 42, "n_jobs": -1}
# Dos configuraciones historicas de SVR, sin decidir cual sera la final:
#   script25_33corridas: 25_corridas_repetidas.py L72 (la usada en el protocolo de 33 corridas; segun su comentario
#                        salio de GridSearchCV en 19_combinado_residual.py L145-151, sobre todas las posiciones).
#   script12_dataset   : 12_features_modelos_v3_synced.py L99 (modelos internos de 12).
SVR_PARAMS_HISTORICAL = {
    "script25_33corridas": {"kernel": "rbf", "C": 1, "epsilon": 0.5, "gamma": "scale"},
    "script12_dataset": {"kernel": "rbf", "C": 10, "epsilon": 0.5, "gamma": "scale"},
}

# ------------------------------------------------------------------ valores historicos de referencia
MAE_E1A_HISTORICAL = 8.400096407833395          # metricas_globales.csv (E1), MAE sobre las 41037 filas
