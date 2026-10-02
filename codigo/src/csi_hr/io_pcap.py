"""
Lectura de CSI desde los PCAP crudos de nexmon (eHealth CSI).

LECTOR OFICIAL NUEVO: read_position (copia de experimentos_fase_C/E1_rf.py L53-74). Reproduce 03_extraer_csi.py
(CSIKit) sin depender de CSIKit; su equivalencia con CSIKit se valido en A.2 (residuo 2e-8 tras escalar) y en E1
(reconstruccion de las features publicadas). 03_extraer_csi.py queda como referencia historica.
pcap_t0 (A3) no se copia: E1a/E5 no lo usan.
"""
import glob
import os
import struct

import numpy as np

from . import config


def read_position(part, pos, pcap_root=None):
    """Replica 03_extraer_csi.py: concatena los PCAP de la posicion (orden de nombre), descarta tramas
    nulas/todo-cero, elimina timestamps duplicados y ordena. Devuelve ts y H (N x 256) complejo."""
    root = str(pcap_root if pcap_root is not None else config.get_pcap_root())
    ts, H = [], []
    # 03 agrupa por int(prefijo): "1_..." y "01_..." (participantes >=122) son la posicion 1
    files = sorted(f for f in glob.glob(os.path.join(root, part, "*.pcap"))
                   if os.path.basename(f).split("_")[0].isdigit() and int(os.path.basename(f).split("_")[0]) == pos)
    for f in files:
        d = open(f, "rb").read(); off = 24
        while off + 16 <= len(d):
            s, us, incl, _ = struct.unpack("<IIII", d[off:off + 16]); pk = d[off + 16:off + 16 + incl]; off += 16 + incl
            pl = pk[14 + (pk[14] & 0x0F) * 4 + 8:]
            if len(pl) != 18 + 1024:        # CSIKit descarta tramas con payload != 1042 bytes (verificado en 005/pos_2)
                continue
            iq = np.frombuffer(pl[18:18 + 1024], dtype="<i2").astype(float)
            h = iq[0::2] + 1j * iq[1::2]
            if np.all(h == 0):
                continue
            ts.append(s + us * 1e-6); H.append(h)
    ts = np.array(ts); H = np.array(H)
    _, first = np.unique(ts, return_index=True)          # drop_duplicates(timestamp) + sort
    return ts[first], H[first]
