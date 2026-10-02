"""
Lectura del smartwatch (Samsung Galaxy Watch 4, JSON *HeartRateData.json).

Implementacion unica: copia de auditoria_fase_A/A3_muestreo_ventanas.py L40, L59-76 (watch_all / to_epoch), que a su
vez es copia de 12_features_modelos_v3_synced.leer_watch_completo (L297-321). Es la que usa E1_rf (etiquetas de E1a).
La asociacion CSI <-> reloj (interpolacion y sync_ok) vive en windows.py y no se modifica.
"""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from . import config

TZ = timezone(timedelta(hours=-3))      # hora literal de Brasil (UTC-3), correccion de 12


def to_epoch(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=TZ).timestamp()


def watch_all(subject, watch_root=None):
    """Union de JSON con clave heart_rate en la raiz, filtro 40-180."""
    root = str(watch_root if watch_root is not None else config.get_watch_root())
    ts_all, hr_all = [], []
    for f in os.listdir(os.path.join(root, subject)):
        if not f.endswith("HeartRateData.json"):
            continue
        try:
            d = json.load(open(os.path.join(root, subject, f)))
            hr = np.array(d.get("heart_rate", []), float); ts = np.array([to_epoch(t) for t in d.get("start_time", [])])
            mk = (hr >= config.HR_MIN_BPM) & (hr <= config.HR_MAX_BPM); ts_all += ts[mk].tolist(); hr_all += hr[mk].tolist()
        except Exception:
            continue
    o = np.argsort(ts_all)
    return np.array(ts_all)[o], np.array(hr_all)[o]
