"""
Entrypoints reproducibles de la tesis (usan exclusivamente src/csi_hr).

  reproduce_840.py              reproduccion HISTORICA de E1a (MAE 8.400096407833395)
  final_personalized.py         escenario personalizado (calibracion individual)
  final_subject_independent.py  escenario de persona nueva (GroupKFold por participante)

Uso desde la raiz del repositorio:  python -m pipeline.reproduce_840 --from-saved-predictions
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
