# Estimación de frecuencia cardíaca mediante Wi-Fi CSI — experimento final

## 1. Objetivo

Estimar la frecuencia cardíaca (BPM) a partir de señales Wi-Fi CSI del conjunto público **eHealth CSI**, usando
como referencia un smartwatch (Samsung Galaxy Watch 4). Se comparan dos modelos de regresión, **Random Forest
(RF)** y **Support Vector Regression (SVR, kernel RBF)**, con dos líneas base.

**Escenario evaluado:** nuevas grabaciones de participantes conocidos (escenario personalizado). Un mismo
participante aporta grabaciones al entrenamiento y a la prueba, pero ninguna grabación aparece en ambos conjuntos.
No es un escenario de participantes nuevos.

## 2. Estructura

```
codigo/src/csi_hr/        librería del experimento
codigo/pipeline/          final_personalized.py: ejecuta el experimento final
codigo/generar_resultados.py   métricas y figuras a partir de las predicciones entregadas
datos_procesados/         datos de entrada congelados (ventanas, características, particiones)
resultados/               predicciones, métricas, configuraciones elegidas y auditoría de fuga
figuras/                  figuras 1, 2 y 3 (cuerpo de la tesis)
material_adicional/       figura 4 (anexo / sustentación; no es figura principal)
```

Módulos de `codigo/src/csi_hr/`:

- **Usados por el experimento final:** `config`, `splits`, `nested` (validación anidada), `models`, `metrics`,
  `features`.
- **De apoyo** (`preprocessing`, `subcarriers`, `windows`, `io_pcap`, `io_watch`): documentan la implementación
  actual del preprocesamiento (detrend, Hampel, Butterworth, min-max, top-20 subportadoras, ventaneo,
  sincronización con el reloj). **El experimento final no los ejecuta**: se reproduce desde los artefactos
  procesados congelados de `datos_procesados/`. Usarlos requiere los datos crudos de eHealth CSI, que no se
  incluyen.

## 3. Instalación

Python 3.10:

```
pip install -r requirements.txt
```

## 4. Datos procesados

| Archivo | Contenido |
|---|---|
| `dataset_completo_v3_synced.csv` | Ventanas, referencia de frecuencia cardíaca (`bpm_watch`) y criterio de sincronización (`sync_ok`) |
| `features_rf_3brazos.csv.gz` | 24 características por ventana. El experimento usa las filas `arm == "E1a"` (54 subportadoras activas). |
| `splits_rf.json` | Particiones entrenamiento/prueba de las 33 semillas |

Muestra final: **125 participantes, 1481 grabaciones, 7397 ventanas**, 12 posiciones estáticas (códigos 1, 2 y
4–13). Cada ventana tiene 231 paquetes con un paso de 57 (≈30 s a la frecuencia nominal de 7,7 Hz; el muestreo
real es irregular, ≈7,6 Hz). Los archivos se entregan tal como se usaron en el experimento; la versión original
del código que generó el dataset no se conserva.

## 5. Ejecutar el experimento final

```
python codigo/pipeline/final_personalized.py --clean --seeds 0          # una semilla (~15 min)
python codigo/pipeline/final_personalized.py --clean --resume           # las 33 semillas (~15 h)
```

- Los resultados se escriben en `reproduccion/final_personalized/` y nunca sobrescriben `resultados/`.
- Con las versiones de `requirements.txt`, las predicciones coinciden con las entregadas.
- `--n-jobs` controla el paralelismo (con 6 procesos se usan hasta ~24 GB de RAM).
- Solo está soportado el modo `--clean`.

## 6. Regenerar tablas y figuras

```
python codigo/generar_resultados.py
```

El script lee las predicciones entregadas (no entrena) y escribe `resultados/metricas_por_semilla.csv`,
`resultados/metricas_finales.csv` y las figuras:

- **Figura 1:** (a) MAE de las 33 corridas para RF, SVR, B1 y B0; (b) diferencias B1−RF y B1−SVR por semilla.
- **Figura 2:** referencia, RF, SVR y B1 en las dos grabaciones de prueba del participante 048, semilla 27.
  La selección no se hizo por el error. Primero se toma la semilla cuyo MAE medio (RF+SVR) es la mediana de las
  33 corridas. Después, dentro de ella, la grabación de prueba con el MAE más cercano a la mediana. Se muestran
  las dos grabaciones de prueba de ese participante.
- **Figura 3:** frecuencia cardíaca de referencia y la estimada por SVR en todo el conjunto de prueba de la
  semilla 27 (1241 ventanas de 249 grabaciones; MAE = 8.42 BPM, r = 0.712). Las ventanas están ordenadas por
  participante, grabación y tiempo. Las curvas se interrumpen entre grabaciones: no es una señal continua.
  Se muestra SVR porque tuvo el menor MAE medio en las 33 corridas (8.384 frente a 8.406 de RF); la diferencia
  es pequeña y no indica superioridad.
- **Figura 4** (`material_adicional/`): RF y SVR frente a la misma referencia y las mismas ventanas. No es una
  figura principal porque las predicciones de ambos modelos son casi idénticas (r = 0.995 entre sí).

## 7. Modelos y líneas base

- **RF y SVR** aprenden la desviación de cada ventana respecto de la media del participante en el entrenamiento
  (regresión residual); la predicción final es esa media más la desviación estimada.
- **B0:** media global de frecuencia cardíaca del entrenamiento.
- **B1:** media de frecuencia cardíaca de cada participante en el entrenamiento.

Las dos líneas base no usan CSI: miden cuánto aporta la señal Wi-Fi.

## 8. Resultados (33 corridas, semillas 0–32)

Las métricas se calculan en cada corrida sobre sus ventanas de prueba y se resumen como media ± desviación
estándar.

| Método | MAE (BPM) | RMSE (BPM) | Pearson r |
|---|---|---|---|
| RF | 8.406 ± 0.284 | 10.578 ± 0.394 | 0.719 ± 0.031 |
| SVR | 8.384 ± 0.298 | 10.621 ± 0.412 | 0.717 ± 0.032 |
| B1 | 8.887 ± 0.291 | 11.090 ± 0.390 | 0.686 ± 0.032 |
| B0 | 12.196 ± 0.321 | 15.159 ± 0.450 | No definido (predicción constante) |

RF y SVR obtienen menor MAE que B1 en las 33 corridas. La diferencia entre RF y SVR no es concluyente. Como las
corridas comparten participantes, la dispersión refleja la variabilidad de la partición.

## 9. Separación entre entrenamiento y prueba

- En cada corrida, la selección de características (información mutua / importancia por permutación, N ∈ {5, 8,
  10, 12, 15, 20, todas}), los hiperparámetros (RF: 41 candidatos; SVR: 52) y el suavizado (w ∈ {1, 3, 5, 7, 9})
  se eligen **solo con validación cruzada interna sobre el entrenamiento** (5 particiones agrupadas por
  grabación).
- **Los datos de prueba no se usan en ninguna de esas decisiones.** La escala (StandardScaler) y las medias por
  participante se calculan solo con el entrenamiento.
- `resultados/leakage_audit.json` confirma, para las 33 semillas, que no hay grabaciones ni ventanas compartidas.
- `resultados/configuraciones_seleccionadas.csv` lista la configuración elegida en cada corrida.
- El suavizado es una media móvil centrada dentro de cada grabación. Como las grabaciones tienen 5 ventanas,
  w = 9 equivale a promediar la grabación completa.
