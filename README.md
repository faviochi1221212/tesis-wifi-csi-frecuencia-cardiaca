# Monitoreo No Invasivo de la Frecuencia Cardíaca mediante Señales Wi-Fi CSI

**Universidad de Lima — Ingeniería de Sistemas**
**Autores:** Yadhira Sarmiento Escobar (20214190) y Favio Chavarry Minaya (20214680)

---

## Descripción General

Este proyecto implementa un pipeline para estimar la frecuencia cardíaca (FC) de personas usando señales Wi-Fi CSI (*Channel State Information*), sin sensores de contacto. La señal Wi-Fi modifica su comportamiento ante micromovimientos del cuerpo humano, incluyendo la expansión torácica producida por el latido cardíaco. El objetivo es capturar esa modulación y convertirla en una estimación de BPM (latidos por minuto).

El repositorio contiene el pipeline completo que produjo el **resultado principal del proyecto: MAE = 9.51 ± 0.13 BPM** (media ± IC95% sobre 33 corridas con splits aleatorios, Random Forest), más una segunda ronda de experimentación (script 24) con Optuna y LSTM, y un escenario restringido a posiciones sedentarias que sí mejora el MAE de forma validada (ver más abajo). Las decisiones de diseño fueron exploradas en más de 25 escenarios alternativos documentados en las conclusiones.

> **Corrección metodológica (2026-07-08):** al regenerar el dataset desde cero se detectaron y corrigieron dos errores en los scripts 23 y 19 que habían quedado invisibles con la versión anterior de los datos (ver [Corrección metodológica](#corrección-metodológica-2026-07-08) más abajo).
>
> **Validación estadística con 33 corridas (2026-07-09):** a pedido del asesor, se dejó de reportar el MAE de un único split y se repitió el entrenamiento/evaluación 33 veces con splits aleatorios distintos (`25_corridas_repetidas.py`). Resultado: **el split original (semilla fija) daba MAE=8.7841, pero era un caso favorable — la media real sobre 33 corridas es 9.51 ± 0.13**. El 8.7841 queda documentado como el mejor caso individual dentro del rango observado (8.77–10.55), no como el resultado representativo. Ver [Validación estadística (33 corridas)](#paso-8--validación-estadística-33-corridas-25_corridas_repetidaspy).
>
> **Confirmación del asesor en asesoría (2026-07-10):** se validó con el asesor (Julio César Huarachi Soto) cómo presentar los 3 niveles de MAE del proyecto. Instrucción explícita: **no reportar solo el mejor número — narrar las 3 fases como una progresión** en la sección de Experimentación: (1) todas las posiciones, split único (~11.9–13, versión inicial), (2) todas las posiciones validado con 33 corridas (9.51 ± 0.13), (3) restringido a posiciones sedentarias por la interferencia entre movimiento corporal y señal cardíaca, también validado con 33 corridas (7.85–7.89 según variante). Cada fase se reporta como **media ± IC95% sobre 33 corridas** ("se ha realizado 33 veces el experimento con diferente distribución aleatoria... con un intervalo de confianza del 95%" — cita textual del asesor). La razón de restringir posiciones (fase 3) se usa también para plantear el **trabajo futuro**: no es optimizar más el modelo, es volver a incorporar posiciones con movimiento resolviendo la interferencia con la señal cardíaca (ver [Trabajo Futuro](#trabajo-futuro)). El asesor también confirmó como pendiente aclarar R vs R² frente al paper de referencia (ya cubierto en este README) y pidió generar **una figura única comparando las 3 fases** (pendiente de generar — ver TODO en Trabajo Futuro).

---

## Dataset Utilizado

**eHealth CSI** — Galdino et al. (2023)

| Característica | Valor |
|---|---|
| Participantes en el dataset | 156 personas |
| Participantes con CSI + smartwatch válidos | 125 |
| Participantes con solapamiento temporal real (TZ_OFFSET=−7200 s) | 123/125 |
| Archivos PCAP | 2806 grabaciones |
| Posiciones por participante | ~17–18 posiciones distintas |
| Duración por grabación | ~65 segundos |
| Hardware | Raspberry Pi 4 (BCM43455c0) |
| Estándar WiFi | 802.11ac, Canal 36, 80 MHz |
| Subportadoras totales | 256 (234 válidas tras eliminar pilotos y guardas) |
| Frecuencia de muestreo CSI | 7.7 Hz |
| Referencia de FC | Smartwatch (Samsung Galaxy Watch) |
| Biblioteca de extracción | CSIKit (NEXBeamformReader — nexmon_csi) |

**Obtención de los datos:** este repositorio no incluye los datos crudos ni los intermedios pesados (PCAP, CSI extraído/filtrado). Para reproducir el pipeline completo desde cero se debe descargar el dataset público eHealth CSI (Galdino et al., 2023) desde su fuente original <!-- TODO: agregar aquí el enlace de descarga del dataset --> y organizarlo según las rutas esperadas por `03_extraer_csi.py` (ver sección [Requisitos de Software](#requisitos-de-software) y [Orden de Ejecución](#orden-de-ejecución)).

Para evitar que quien evalúe este proyecto tenga que repetir la extracción y el filtrado (pasos 1 y 2, ~4-5 horas en total), la carpeta `csi_filtrado_v3/` (CSI ya filtrada, salida de `04_preprocesamiento.py`) se comparte directamente de forma externa (fuera de este repositorio, por su peso). Con esa carpeta ya es posible ejecutar el pipeline desde el **Paso 3** (`12_features_modelos_v3_synced.py`) en adelante.

---

## Estructura de Archivos

```
tesis-wifi-csi-frecuencia-cardiaca-main/
├── 03_extraer_csi.py                  # Paso 1: PCAP -> CSI complejo (234 subportadoras)
├── 04_preprocesamiento.py             # Paso 2: filtrado (Hampel + Butterworth + ISPD)
├── 12_features_modelos_v3_synced.py   # Paso 3: features + sincronización + diagnóstico
├── 23_seleccion_mi.py                 # Paso 4: selección de features por permutation importance
├── 19_combinado_residual.py           # Paso 5: modelo final RF residual (single split, MAE=8.7841)
├── 24_optuna_sedentario_lstm.py       # Paso 6: escenarios Optuna + LSTM (segunda ronda)
├── 08_figuras_tesis.py                # Paso 7: figuras finales
├── 25_corridas_repetidas.py           # Paso 8: 33 corridas Escenarios A y C -> media +/- IC95% (MAE real = 9.51)
├── 26_corridas_repetidas_bde.py       # Paso 9: 33 corridas Escenarios B, D y E -> confirma que ninguno supera a C
├── resultados/                        # Datasets intermedios, CSVs de resultados y figuras (generado por los scripts)
└── README.md
```

> **Nota (2026-07-08):** antes de esta fecha, `OUTPUT_DIR` apuntaba a `C:\Users\LENOVO\Desktop\Data_DS1_raspberry-main\` (el clon del dataset original), que también aloja un proyecto distinto (una réplica del paper Pulse-Fi con LSTM). Para evitar confusión entre ambos, todos los scripts de este repo (12, 23, 19, 24, 08) ahora escriben en `resultados/`, dentro del propio repo.

**Datos externos requeridos (no incluidos en el repositorio):**
```
C:\Users\LENOVO\Desktop\
├── Data_DS1_smartwatch-main\Data\     # JSONs de frecuencia cardíaca del smartwatch
├── csi_csv\                           # Generado por 03_extraer_csi.py
└── csi_filtrado_v3\                   # Generado por 04_preprocesamiento.py
```

---

## Pipeline Completo

### Paso 1 — Extracción de CSI desde PCAP (`03_extraer_csi.py`)

Lee cada archivo `.pcap` con CSIKit y extrae la matriz CSI compleja de 234 subportadoras válidas por fotograma.

- Se eliminan 14 subportadoras de guarda y 8 pilotos IEEE 802.11ac, quedando **234 subportadoras de datos válidas**.
- Los índices de pilotos se corrigen manualmente porque CSIKit usa índices incorrectos para 802.11ac 80 MHz.
- Solo archivos con prefijo numérico (`1_2022...`) se procesan; archivos de control (`VAZIO`, `julio3`, etc.) se omiten.

**Salida:** `csi_csv/{participante}/pos_N.csv` — número complejo por subportadora por fotograma.

---

### Paso 2 — Preprocesamiento (`04_preprocesamiento.py`)

Filtra la señal CSI cruda para aislar la componente en la banda cardíaca (0.8–2.17 Hz).

1. **Detrend lineal** — elimina deriva lenta por temperatura o movimiento ambiental.
2. **Filtro Hampel** (k=10, 3σ) — reemplaza spikes por la mediana local. Se usa mediana/MAD en lugar de media/std porque son robustas al propio outlier que se quiere detectar.
3. **Filtro Butterworth bandpass** (orden 5, 0.8–2.17 Hz, `sosfiltfilt`) — fase cero (sosfiltfilt aplica el filtro dos veces: forward y backward, duplicando el rolloff efectivo a 60 dB/octava). Se usa `sos` en vez de coeficientes `(b,a)` directos para evitar inestabilidad numérica en orden 5.
4. **ISPD** (Inter-Subcarrier Phase Difference) — cancela ruido de fase común entre subportadoras (CFO/SFO), mejora la SNR.

**Salida:** `csi_filtrado_v3/{participante}/pos_N.csv` — amplitud filtrada (`sub_XXX`) + ISPD (`ipd_XXX`) + timestamp.

---

### Paso 3 — Features y sincronización (`12_features_modelos_v3_synced.py`)

Extrae 29 características por ventana temporal, sincroniza con el smartwatch y genera el dataset base.

**Sincronización CSI ↔ Smartwatch (corrección crítica):**
`TZ_OFFSET = −7200 segundos`. Una versión anterior usaba `+10800`, lo que dejaba al 100% de los 125 participantes sin solapamiento temporal real (desfase constante de ~5 horas). Con el valor corregido, 123/125 participantes quedan correctamente sincronizados. La sincronización fue verificada mediante BPM fisiológicamente plausibles (media=86.6 BPM, 76.6% entre 60–100 BPM en reposo).

**Procesamiento por posición:** cada `pos_N.csv` se procesa individualmente para conservar `pos_id` en el dataset, necesario para filtrar escenarios sedentarios. Se añaden columnas `sync_ok` (lectura del smartwatch a menos de 15 s del centro de la ventana) y `min_dist_watch_s` por ventana.

**Ventaneo:** 30 segundos (≈231 muestras a 7.7 Hz), solapamiento 75% (paso de ~57 muestras). Confirmado como óptimo experimentalmente.

**29 features por ventana:**

| Grupo | Features (cantidad) |
|---|---|
| Temporales | media, std, varianza, energía, RMS, pico-pico, kurtosis, skewness (8) |
| Espectrales | energía en banda, ratio de energía, BPM por FFT parabólico, BPM armónico, centroide, ancho de banda, SNR, BPM AC parabólico (8) |
| Autocorrelación | pico AC, BPM por AC (2) |
| Wavelet `db4` | 4 ratios de energía + entropía (5) |
| ISPD | BPM FFT, BPM armónico, BPM AC, ratio de energía, SNR (5) |
| **Total** | **28 base + 1 extra = 29** |

**Salidas:**
- `dataset_completo_v3_synced.csv` — dataset con pos_id, sync_ok, 29 features
- `diagnostico_sincronizacion.csv` — solapamiento, delta temporal y conteo de ventanas por participante
- `feature_importance_v3_synced.csv` — importancias RF
- `fig_sync_alineamiento.png` — figura de alineamiento temporal CSI vs smartwatch

---

### Paso 4 — Selección de features (`23_seleccion_mi.py`)

Selecciona las **top-10 features por permutation importance** (caída real de desempeño al barajar cada feature en validación) en vez de `feature_importances_` nativo de RF (sesgado hacia variables continuas de alta cardinalidad). El pool de candidatas excluye explícitamente `pos_id`, `sync_ok` y `min_dist_watch_s` (metadatos de la etapa 3, no señal CSI — ver corrección metodológica).

Top-10 seleccionadas: `energia, rms, varianza, std, entropia, e_banda, pico_pico, snr_cardiac, kurtosis_s, wav_e1`.

**Salida:** `permutation_importance_features.csv`

---

### Paso 5 — Modelo final (`19_combinado_residual.py`)

**Mejor MAE del proyecto: 8.7841 BPM.**

1. **Target = residual:** el modelo predice `bpm_watch − media_persona_en_calibración`; al inferir se suma la media de la persona. Funciona mejor que predecir HR absoluto porque la varianza intra-sujeto en grabaciones de 65 s es muy baja.
2. **Split por grabación completa 80/20** (`seg_id`), sin fuga de datos: cada grabación entera va a calibración o a test.
3. **Top-10 features por permutation importance** (del paso 4).
4. **Random Forest tuneado:** `max_depth=6, max_features=sqrt, min_samples_leaf=29, n_estimators=387`.
5. **Suavizado media móvil w=9** sobre predicciones, agrupado por (participante, grabación). Cada grupo (participante, seg_id) tiene exactamente 5 ventanas, así que w≥9 promedia efectivamente todo el grupo.

**Salidas:** `resultados_combinado_residual.csv`, `predicciones_finales_v19.csv`

#### Corrección metodológica (2026-07-08)

Al regenerar el pipeline completo desde cero se encontraron y corrigieron dos errores que la versión anterior de los datos no dejaba ver:

1. **Fuga de metadatos en la selección de features (`23_seleccion_mi.py`):** `ID_COLS` excluía las columnas de identificación pero no `pos_id`, `sync_ok` ni `min_dist_watch_s` — estas tres quedaban disponibles como candidatas para el ranking de permutation importance. `sync_ok` y `min_dist_watch_s` describen qué tan cerca estaba la lectura del smartwatch de referencia respecto al centro de la ventana; en un despliegue real sin smartwatch esa información no existiría, así que su uso como feature predictiva es inválido (mismo tipo de problema que el escenario 24 descartado en la ronda anterior). El propio script 12 ya las trataba correctamente como metadatos (`META_COLS`) al entrenar sus modelos de referencia — el script 23 simplemente no se había actualizado para igualar ese criterio. Con las tres columnas excluidas, el MAE volvió a un rango consistente con el histórico (8.78–8.85 en vez de 5.9–7.0).
2. **Desalineación de índices en el suavizado (`19_combinado_residual.py`):** las predicciones (`pred`, `pred_svr`) se calculaban en el orden original del DataFrame, pero se reasignaban con `pd.Series(pred, index=df_te.index)` **después** de reordenar `df_te` por `(participante, seg_id, win_start)` — esto empareja cada predicción con la fila de otra ventana antes de aplicar la media móvil. No se notaba con el dataset anterior porque el orden natural de las filas coincidía por casualidad con el orden ordenado; con el nuevo esquema de `seg_id = pos_id*100+gap_id` (paso 3) dejó de coincidir, y el suavizado empezó a **empeorar** el MAE en vez de mejorarlo. Se corrigió capturando el índice original antes de ordenar.

Tras corregir ambos y volver a barrer N (features) × w (suavizado), el óptimo sigue siendo top-10 + w=9, con **MAE=8.7841**, mejor que el 9.1049 histórico.

---

### Paso 6 — Escenarios Optuna + LSTM (`24_optuna_sedentario_lstm.py`)

Segunda ronda de experimentación. Requiere haber ejecutado los pasos 3, 4 y 5 primero.

**Escenario A:** Todas las posiciones (referencia, reproduce el resultado del script 19).

**Escenario B:** Solo posiciones sedentarias `{1,2,3,4,5,6,11,12,13,14}` (sentado y acostado). Requiere `pos_id` en el dataset (disponible solo con la versión actual del script 12).

**Escenario C:** Sedentario + sincronización estricta (`sync_ok=True`).

**Escenario D:** Sedentario + optimización de hiperparámetros con **Optuna** (50 trials, GroupKFold por participante para evitar fuga). Modelos: RF, XGBoost, LightGBM.

**Escenario E:** Sedentario + **LSTM** de 2 capas sobre secuencias de 5 ventanas consecutivas de features, con early stopping y suavizado w=9.

**Salidas:**
- `dataset_sedentario_synced.csv`
- `resultados_optuna.csv`
- `resultados_lstm.csv`
- `comparacion_escenarios.csv`
- `predicciones_finales_mejoradas.csv`
- `fig_comparacion_escenarios.png`
- `fig_lstm_predicho_vs_real.png`

---

### Paso 7 — Figuras (`08_figuras_tesis.py`)

Genera 6 figuras a partir de `predicciones_finales_v19.csv` y `comparacion_escenarios.csv`:

- `fig1_predicho_vs_real.png` — serie temporal predicho vs real
- `fig2_dispersion.png` — dispersión predicho vs real
- `fig3_distribucion_error.png` — distribución del error absoluto
- `fig4_comparacion_literatura.png` — tabla comparativa con literatura (incluye scripts 19 y 24)
- `fig5_bland_altman.png` — análisis de Bland-Altman
- `fig_sync_alineamiento.png` — alineamiento temporal CSI vs smartwatch

---

### Paso 8 — Validación estadística (33 corridas) (`25_corridas_repetidas.py`)

El script 19 evalúa un **único** split aleatorio (semilla fija = 42). A pedido del asesor de tesis, este paso repite el entrenamiento/evaluación **33 veces**, cada vez con un split distinto (por grabación dentro de cada persona — se preserva la calibración por `media_persona`; no se excluyen personas completas, ver discusión más abajo), y reporta la **media ± intervalo de confianza al 95%** de MAE, RMSE y MAPE para RF y SVR, en vez de un número puntual — igual en espíritu a como reportan sus métricas los papers de referencia (media ± IC sobre múltiples muestreos aleatorios).

Los hiperparámetros de RF y SVR se mantienen **fijos** (los ya tuneados en el script 19); solo se repite el split y el entrenamiento, para no mezclar la varianza de remuestreo con la varianza de la búsqueda de hiperparámetros.

Se corre sobre dos escenarios: **A** (todas las posiciones, el modelo final) y **C** (sedentario + sincronización estricta, el mejor escenario de la segunda ronda), para verificar si la mejora de C es real o también es un split con suerte.

**Salidas:**
- `resultados_33_corridas.csv` — una fila por corrida × escenario × modelo
- `resumen_33_corridas.csv` — media, std e IC95% por escenario × modelo × métrica
- `fig_33_corridas_barras.png` — barras con error bars, Escenario A vs C

---

### Paso 9 — Validación estadística de B, D y E (`26_corridas_repetidas_bde.py`)

Extiende el Paso 8 a los escenarios de `24_optuna_sedentario_lstm.py` que solo se habían evaluado con un único split: **B** (sedentario, sin filtro de sync), **D** (RF/XGBoost/LightGBM con hiperparámetros de Optuna) y **E** (LSTM). Mismo criterio que el Paso 8: los hiperparámetros de Optuna y la arquitectura del LSTM se mantienen **fijos** (los ya encontrados en el script 24); solo se repite el split y el entrenamiento 33 veces.

**Salidas:**
- `resultados_33_corridas_bde.csv` — una fila por corrida × escenario
- `resumen_33_corridas_bde.csv` — media, std e IC95% por escenario
- `fig_33_corridas_bde_barras.png` — barras con error bars, B/D/E

---

## Resultados

### Resultado principal: media ± IC95% sobre 33 corridas

| Escenario | Modelo | MAE | RMSE | MAPE | r |
|---|---|---|---|---|---|
| A: Todas las posiciones | Baseline (solo promedio persona) | 9.892 ± 0.129 | 12.894 ± 0.166 | 11.61% ± 0.14 | 0.616 ± 0.008 |
| **A: Todas las posiciones (modelo final)** | **RF** | **9.507 ± 0.128** | 12.276 ± 0.150 | 11.19% ± 0.15 | 0.660 ± 0.007 |
| A: Todas las posiciones | SVR | 9.512 ± 0.127 | 12.336 ± 0.152 | 11.16% ± 0.14 | 0.656 ± 0.007 |
| C: Sedentario + sync estricta | Baseline (solo promedio persona) | 7.906 ± 0.123 | 10.013 ± 0.155 | 9.80% ± 0.15 | 0.711 ± 0.010 |
| **C: Sedentario + sync estricta** | **RF** | **7.891 ± 0.122** | 9.978 ± 0.153 | 9.78% ± 0.15 | 0.713 ± 0.010 |
| C: Sedentario + sync estricta | SVR | 7.883 ± 0.123 | 9.995 ± 0.156 | 9.81% ± 0.16 | 0.712 ± 0.010 |

Dos hallazgos importantes de esta tabla:

1. **RF y SVR son estadísticamente indistinguibles entre sí** en ambos escenarios (intervalos superpuestos) — se mantiene RF como modelo final por ser más simple e igual de bueno.
2. **El CSI sí aporta señal real en el Escenario A, pero casi nada en el Escenario C.** En "todas las posiciones", el modelo (9.51) mejora sobre el baseline (9.89) de forma estadísticamente clara (intervalos no se superponen). En "sedentario + sync estricta", el modelo (7.89) es **prácticamente idéntico** al baseline (7.91) — el CSI casi no aporta nada ahí. Esto significa que el buen MAE del Escenario C se explica principalmente porque el HR en reposo es más estable/predecible desde el simple promedio de la persona, **no** porque el modelo esté aprovechando mejor la señal CSI en ese subconjunto. La comparación entre A y C sigue siendo válida como "mejor MAE alcanzable restringiendo el alcance", pero el crédito de esa mejora es del **filtro de posiciones**, no del modelo en sí.

### Evolución del MAE (mejor resultado de cada etapa, split único semilla=42)

| Etapa | MAE (BPM) |
|---|---|
| Baseline (promedio por persona, sin CSI) | 9.328 |
| RF tuneado sobre residual (top-10 PermImp, sin suavizado) | 8.854 |
| RF + suavizado w=9 + features por permutation importance, sin fuga | 8.784 |

> Esta tabla usa el split de semilla fija (42) del script 19 — es la evolución tal como se fue mejorando el pipeline, pero **8.784 es un caso favorable dentro del rango normal**, no el valor esperado (ver tabla de arriba: la media real es 9.51 ± 0.13). Se mantiene esta tabla porque documenta correctamente el efecto relativo de cada mejora (selección de features, suavizado, corrección de bugs) bajo el mismo split, aun si el punto de partida absoluto no es representativo.

### Segunda ronda — Escenarios adicionales (scripts 24, 25 y 26)

> **Actualizado 2026-07-09:** todos los escenarios (A, B, C, D, E) ya están validados con media ± IC95% sobre 33 corridas (Pasos 8 y 9), no solo con un split único. El split único (columna aparte abajo, referencial) ya no es el número que se reporta.

| Escenario | MAE (33 corridas) | RMSE | MAPE | r |
|---|---|---|---|---|
| A: Todas las posiciones (referencia) | 9.507 ± 0.128 | 12.276 ± 0.150 | 11.19% ± 0.15 | 0.660 ± 0.007 |
| B: Solo posiciones sedentarias | 7.854 ± 0.107 | 9.906 ± 0.140 | 9.77% ± 0.14 | 0.711 ± 0.008 |
| **C: Sedentario + sincronización estricta** | **7.891 ± 0.122** | 9.978 ± 0.153 | 9.78% ± 0.15 | 0.713 ± 0.010 |
| D: RF + Optuna (50 trials) | 7.847 ± 0.105 | 9.903 ± 0.139 | 9.75% ± 0.14 | 0.711 ± 0.008 |
| D: XGBoost + Optuna | 7.849 ± 0.107 | 9.908 ± 0.140 | 9.76% ± 0.14 | 0.711 ± 0.008 |
| D: LightGBM + Optuna | 7.844 ± 0.106 | 9.902 ± 0.140 | 9.75% ± 0.14 | 0.711 ± 0.008 |
| E: LSTM 2 capas + suavizado w=9 | 7.895 ± 0.121 | 10.110 ± 0.160 | 9.93% ± 0.15 | 0.708 ± 0.009 |

**Hallazgo principal, ahora con validación estadística completa: B, C, D (RF/XGBoost/LightGBM) y E (LSTM) son estadísticamente indistinguibles entre sí** — los cinco caen en el rango 7.84–7.90 con intervalos de confianza que se superponen totalmente. Esto confirma con rigor lo que el split único ya insinuaba:

- **Optuna no aporta nada real** sobre los hiperparámetros por defecto de RF, ni cambiando de RF a XGBoost o LightGBM — las 3 variantes de Optuna dan prácticamente el mismo MAE que el RF sin tunear (B).
- **El LSTM tampoco supera a los modelos de árboles** — de hecho tiene el intervalo de confianza más ancho (más inestable entre corridas) y una media nominal ligeramente peor, aunque no estadísticamente distinta.
- **La sincronización estricta (C vs B) tampoco aporta una mejora estadísticamente separable** — 7.891±0.122 vs 7.854±0.107, intervalos casi idénticos. Toda la mejora observada al pasar de "todas las posiciones" (9.51) a "sedentario" (~7.85) se explica por el **filtro de posición en sí**, no por ningún ajuste posterior (modelo, hiperparámetros, o sincronización más estricta).

En otras palabras: de más de 25 escenarios probados en total, **la única palanca que mueve el MAE de forma estadísticamente sostenida es restringir las posiciones a sedentarias** — todo lo demás (qué modelo, qué hiperparámetros, filtrar por sync) queda dentro del margen de ruido estadístico (±0.10–0.15 BPM).

### Comparación con la literatura

| Trabajo | MAE (BPM) | Por qué es más bajo |
|---|---|---|
| Este trabajo (modelo final, media ± IC95% sobre 33 corridas) | **9.51 ± 0.13** | Antena única, dataset público, grabaciones cortas |
| Este trabajo (solo posiciones sedentarias, validado con 33 corridas) | **7.89 ± 0.12** | Subconjunto restringido, más fácil de predecir |
| Gouveia et al. (2024) | 2.72 | Evaluación por sesión, entorno controlado |
| Gu et al. (2021) | 3.53 | Múltiples antenas, grabaciones largas |
| Liu et al. (2022) | 0.6 | CNN profunda + hardware especializado |

### Aclaración: R vs R² (pedido explícito del asesor)

En asesoría se pidió aclarar esto porque el R² bajo del modelo generó confusión al comparar con la literatura:

- **r (coeficiente de correlación de Pearson):** mide qué tan bien se mueven juntas dos variables (aquí, BPM real y BPM predicho), en escala de −1 a 1. Es la métrica que reportan casi todos los papers de la literatura citados en este README (Liu et al., Sun et al., Gu et al.). Nuestro modelo: **r=0.714** (Escenario A) — correlación fuerte.
- **R² (coeficiente de determinación):** mide qué proporción de la varianza de la variable real explica el modelo, en escala de 0 a 1 (puede ser negativo si el modelo es peor que predecir la media). No es lo mismo que r² (el cuadrado de Pearson) cuando el modelo tiene sesgo o cuando se calcula sobre un conjunto distinto al de entrenamiento — por eso nuestro R²=0.508 es más bajo que r²≈0.51 solo por coincidencia numérica en este caso, pero conceptualmente son cálculos distintos (R² penaliza más los errores grandes y el sesgo del modelo).
- **Por qué el R² de este proyecto es más bajo que en papers con antena única similares:** los papers de referencia casi nunca reportan R² (reportan r, MAE o RMSE), precisamente porque R² es más sensible a la varianza intra-sujeto pequeña de este tipo de datasets (grabaciones cortas, HR casi constante por persona) — un R² moderado (0.4–0.5) es normal y esperable en este tipo de problema, no indica que el modelo esté mal.

---

## Análisis de Sincronización CSI–Smartwatch

La sincronización fue verificada en profundidad con los siguientes resultados:

| Indicador | Valor | Interpretación |
|---|---|---|
| BPM medio (smartwatch) | 86.6 BPM | Normal para reposo |
| BPM rango | 44–149 BPM | Fisiológicamente plausible |
| Ventanas con BPM en 60–100 BPM | 76.6% | Coherente con protocolo sedentario |
| r(bpm_CSI, bpm_watch) | −0.03 | Esencialmente cero |
| MAE directo bpm_est vs bpm_watch | 21.45 BPM | CSI no estima BPM directamente con antena única |

**La correlación nula entre el BPM estimado desde el CSI y el BPM del reloj es una limitante de hardware, no de sincronización.** El CSI de antena única captura principalmente la respiración (~0.25 Hz) y movimiento ambiental; la señal cardíaca (~1–2 Hz) es demasiado débil para distinguirla en crudo. El modelo RF no estima frecuencia directamente — aprende correlaciones estadísticas indirectas entre features y BPM del reloj.

Adicionalmente, filtrar segmentos con BPM estático (std < 0.01, 6.8% de los segmentos) **empeora** el MAE a 9.28, confirmando que esos segmentos son los más fáciles de predecir (persona en reposo absoluto) y no son artefactos de mala sync.

---

## Por Qué No Es Viable Bajar Más el MAE con Este Dataset (para todas las posiciones)

Se probaron sistemáticamente más de 25 escenarios alternativos en dos rondas de experimentación, más una validación estadística final:

**Primera ronda:** ventaneo, modelos (SVR, XGBoost, stacking), suavizado, outliers, fracción de calibración, estimación directa sin ML (MAE≈17–23 BPM), VMD (MAE=9.15), ICA sobre subportadoras (MAE=9.28).

**Segunda ronda (sugerida por el jurado):**
- **Optuna RF/XGB/LGB** (50 trials, GroupKFold): MAE=9.13–9.15 — el RF ya estaba en su óptimo.
- **LSTM** sobre secuencias de features: MAE=9.69 — grabaciones muy cortas, pocas secuencias por segmento.
- **Calibración adaptativa (EMA)**: MAE=4.07 — **descartada por ser metodológicamente inválida** (requiere conocer el HR real en tiempo real).
- **Selección por permutation importance** (en vez de `feature_importances_`): MAE=9.105 — mejora real, incorporada al modelo final.
- **Corrección de fuga de metadatos + bug de desalineación en el suavizado** (2026-07-08): MAE=8.784 (split único) / **9.51 ± 0.13 (media real, 33 corridas)** — eliminación de dos errores de implementación, no un cambio de metodología de fondo.
- **Restringir a posiciones sedentarias + sincronización estricta** (Escenario C): **única mejora real y validada estadísticamente** — MAE=7.89 ± 0.12, ~1.6 BPM mejor que el modelo general, al precio de no cubrir posiciones activas/de pie.

**¿Entonces sí hay una forma de bajar el MAE?** Sí, una sola, y ya está validada con rigor estadístico completo (Pasos 8 y 9): **restringir el sistema a posiciones sedentarias**. Da MAE≈7.85-7.89 (BD/C/E, todos estadísticamente iguales entre sí) en vez de 9.51. El costo es de alcance, no de metodología: el sistema deja de cubrir posiciones activas/de pie. Dentro de ese subconjunto sedentario, **ninguna otra variante probada (Optuna, XGBoost, LightGBM, LSTM, sincronización más estricta) mueve el MAE más allá del margen de ruido estadístico** — no es que falte explorar mejor esa via, es que ya se exploró exhaustivamente y no hay más margen ahí. Para bajar el MAE en la población completa (todas las posiciones) sin restringir el alcance, no encontramos ninguna variante de features, modelo, suavizado o post-procesamiento que lo logre — el techo parece ser estructural (ver abajo), no un óptimo de hiperparámetros sin explorar.

**Causas estructurales del techo de ~9.5 BPM (todas las posiciones):**

1. **Antena única (sin diversidad espacial).** BCM43455c0 tiene una sola antena Tx/Rx; los sistemas MIMO mejoran la SNR porque el ruido es parcialmente independiente entre antenas.
2. **Grabaciones cortas (65 s) con HR casi constante.** Insuficiente variación real de HR para que el modelo aprenda más allá del promedio por persona.
3. **Baja frecuencia de muestreo (7.7 Hz).** Resolución FFT en ventana de 30 s: Δf = 1/30 ≈ 2 BPM/bin.
4. **Entorno no controlado.** Respiración, postura y movimiento generan variaciones en el CSI muchas veces mayores que la microvibración cardíaca — esto explica también por qué restringir a posiciones sedentarias (Escenario C) da un MAE más bajo: hay menos artefactos de movimiento contaminando la señal, y el HR en reposo es más estable/predecible.

La mejora del modelo final sobre el baseline trivial (media ± IC95% sobre 33 corridas) es de **0.39 BPM en el Escenario A** (9.89 → 9.51, estadísticamente real, intervalos de confianza no se superponen) — confirma que el CSI aporta señal cardíaca real pero débil. En el Escenario C (sedentario + sync estricta) esa mejora prácticamente **desaparece** (7.91 → 7.89, intervalos superpuestos): ahí el buen MAE viene sobre todo del subconjunto más fácil, no del modelo.

---

## Requisitos de Software

```
Python 3.10+
numpy
pandas
scipy
scikit-learn
matplotlib
pywavelets
xgboost
csikit
```

Dependencias opcionales para el script 24:
```
optuna          # optimización de hiperparámetros (Escenario D)
lightgbm        # modelo alternativo con Optuna (Escenario D)
torch           # LSTM (Escenario E)
```

Instalación:
```bash
pip install numpy pandas scipy scikit-learn matplotlib PyWavelets xgboost csikit
pip install optuna lightgbm torch   # opcional, para script 24
```

---

## Orden de Ejecución

> **Nota:** los scripts tienen rutas absolutas hardcodeadas al entorno original (p. ej. `C:\Users\LENOVO\Desktop\csi_csv`). Antes de ejecutar, ajusta las variables `CSI_CRUDO`, `CSI_FILTRADO` y `OUTPUT_DIR` al inicio de cada script según la ubicación de tus propios datos y carpeta de salida.

```bash
# 1. Extraer CSI desde PCAP (~3-4 horas)
python 03_extraer_csi.py

# 2. Preprocesar CSI (~1 hora)
python 04_preprocesamiento.py

# 3. Extraer features, sincronizar y generar diagnóstico (~10 min)
#    → dataset_completo_v3_synced.csv (con pos_id, sync_ok)
#    → diagnostico_sincronizacion.csv
#    → fig_sync_alineamiento.png
python 12_features_modelos_v3_synced.py

# 4. Seleccionar features por permutation importance (~5 min)
python 23_seleccion_mi.py

# 5. Modelo final: RF residual + suavizado (~1 min)
#    → MAE = 8.7841 BPM
python 19_combinado_residual.py

# 6. Escenarios Optuna + LSTM (~10-60 min según hardware)
#    → comparacion_escenarios.csv
#    → fig_comparacion_escenarios.png
python 24_optuna_sedentario_lstm.py

# 7. Generar figuras finales (~1 min)
python 08_figuras_tesis.py

# 8. Validacion estadistica: 33 corridas x 2 escenarios (~10-15 min)
#    -> MAE real (media +/- IC95%) = 9.51 +/- 0.13 (todas posiciones)
#                                  = 7.89 +/- 0.12 (sedentario + sync estricta)
python 25_corridas_repetidas.py

# 9. Validacion estadistica de B, D (Optuna) y E (LSTM): 33 corridas (~20-30 min, LSTM incluido)
#    -> confirma que B/C/D/E son estadisticamente indistinguibles entre si (7.84-7.90)
python 26_corridas_repetidas_bde.py
```

---

## Trabajo Futuro

Confirmado en asesoría con el asesor de tesis (Julio César Huarachi Soto, 2026-07-10):

- **La vía de mejora futura es reincorporar posiciones con movimiento, no seguir optimizando el modelo sobre el subconjunto sedentario.** El asesor lo planteó explícitamente: como este proyecto restringió el alcance a posiciones sedentarias porque el movimiento corporal interfiere con la señal cardíaca en el CSI (ver [Por Qué No Es Viable Bajar Más el MAE](#por-qué-no-es-viable-bajar-más-el-mae-con-este-dataset-para-todas-las-posiciones)), el trabajo futuro natural es resolver esa interferencia (p. ej. separar movimiento y microvibración cardíaca en el dominio de frecuencia o con un modelo que module por tipo de movimiento) para poder ampliar el sistema de vuelta a todas las posiciones sin perder precisión.
- Hardware con diversidad espacial (MIMO, múltiples antenas) y mayor frecuencia de muestreo (≥30 Hz) siguen siendo vías estructurales no exploradas por falta de hardware propio (ver limitaciones).
- Deep learning sobre la señal CSI cruda (CNN-1D/LSTM sobre la serie temporal completa, no sobre features estadísticos por ventana) sigue sin explorarse — es la única palanca de modelo con potencial de mejora real no descartada aún por experimentación directa.

**TODO pendiente (pedido explícito del asesor, no generado todavía):** figura única que compare las 3 fases de MAE en un solo gráfico (split único inicial ~11.9–13 → 33 corridas todas las posiciones 9.51±0.13 → 33 corridas sedentario 7.85–7.89), a diferencia de `fig_33_corridas_barras.png` (solo A vs C) y `fig_33_corridas_bde_barras.png` (solo B/D/E), que comparan subconjuntos por separado. El asesor pidió esta figura para la sección de Experimentación.

---

## Conclusiones

El sistema desarrollado alcanza un **MAE = 9.51 ± 0.13 BPM** (media ± IC95% sobre 33 corridas con splits aleatorios independientes, Random Forest, calibración por sujeto sin fuga de datos), tras una corrección crítica de sincronización temporal CSI↔smartwatch (TZ_OFFSET=−7200 s), dos rondas exhaustivas de experimentación sobre ventaneo, selección de features, modelos, descomposición de señal y post-procesamiento, una corrección metodológica adicional (2026-07-08) que eliminó una fuga de metadatos y un bug de desalineación de índices en el suavizado, y finalmente una validación estadística rigurosa (33 corridas) que reemplazó el reporte de un único split por una media con intervalo de confianza — el split único usado originalmente (MAE=8.7841) resultó ser un caso favorable dentro del rango observado (8.77–10.55), no el valor representativo.

La mejora sobre el baseline trivial (promedio histórico por persona, también evaluado en las 33 corridas) es de **0.39 BPM** (9.89 → 9.51, estadísticamente significativa), lo que indica que la señal cardíaca presente en el CSI de este dataset es real pero débil. El BPM estimado directamente desde el CSI tiene correlación r=−0.03 con el BPM del reloj, confirmando que el modelo no detecta la frecuencia cardíaca de forma directa sino mediante correlaciones estadísticas indirectas entre features espectrales y BPM.

**La única vía encontrada, de más de 25 escenarios probados, que reduce el MAE de forma real y estadísticamente sostenida es restringir el sistema a posiciones sedentarias** (MAE≈7.85-7.89 según variante, ~1.6 BPM mejor que el modelo general de 9.51). Sin embargo, en ese subconjunto el modelo apenas mejora sobre el baseline trivial (7.91 → 7.89, diferencia no significativa) — la ganancia viene principalmente de que el HR en reposo es más estable y predecible por sí solo, no de que el CSI aporte más señal ahí. Es una mejora real pero de alcance limitado (excluye posiciones activas/de pie), no una mejora del modelo en sí.

La segunda ronda de experimentación (Optuna con 50 trials para RF, XGBoost y LightGBM; LSTM de 2 capas; sincronización estricta) **no mejoró el MAE sobre el subconjunto sedentario de forma estadísticamente significativa** — los 5 escenarios (B, C, D×3, E) validados con 33 corridas cada uno caen todos en el rango 7.84–7.90 con intervalos de confianza superpuestos. Esto confirma el **techo estructural del dataset**, tanto para la población completa como, de forma más granular, dentro del propio subconjunto sedentario: antena única, baja frecuencia de muestreo, grabaciones cortas y entorno no controlado. Para superar este límite se requeriría hardware MIMO, frecuencia de muestreo ≥30 Hz, grabaciones de varios minutos con variación real de HR, y/o arquitecturas de deep learning sobre la señal CSI cruda (no sobre features estadísticos de ventanas de 30 s) — no más ajuste de hiperparámetros o cambio de modelo sobre los datos ya disponibles.

---

*Tesis de pregrado — Ingeniería de Sistemas — Universidad de Lima*
