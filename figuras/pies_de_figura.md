# Pies de figura (referencia para el documento de tesis)

## Metodología

**diagrama_proceso_general.png**
Esquema general del proceso seguido en la tesis, desde la preparación y sincronización de los registros hasta la
estimación y evaluación de la frecuencia cardíaca.

**preprocesamiento_antes_despues.png**
Ejemplo del efecto del preprocesamiento sobre una subportadora CSI. Se muestra la amplitud original y la señal obtenida
después de aplicar detrend lineal, filtrado de Hampel, filtro Butterworth pasabanda y normalización min-max.

**sincronizacion_csi_smartwatch.png**
Ejemplo de sincronización temporal entre las mediciones CSI y la referencia cardíaca. Las lecturas del smartwatch se
interpolan en las marcas temporales de los paquetes CSI. Para cada ventana, la referencia y_i corresponde a la media de
los valores interpolados de sus 231 paquetes. La ventana se conserva si existe al menos una lectura real del smartwatch a
15 s o menos del paquete central.

## Experimentación

**diagrama_calibracion.png**
Procedimiento de calibración utilizado en cada corrida experimental. La selección de características y el ajuste de
hiperparámetros se realizan secuencialmente mediante validación interna agrupada por grabación. El conjunto de prueba no
interviene en ninguna decisión de selección.

## Resultados

**figura_comparacion_rf_svr_test.png**
Comparación entre la frecuencia cardíaca de referencia y la estimada por Random Forest y SVR en el conjunto de prueba de
la semilla 27. Las ventanas se ordenan por participante, grabación y tiempo, y las curvas se interrumpen entre
grabaciones.

**figura_predicho_vs_referencia_test.png**
Comparación entre la frecuencia cardíaca de referencia y la estimada mediante SVR en el conjunto de prueba de la semilla
27. Las ventanas se presentan según su índice ordenado por participante, grabación y posición temporal dentro de cada
grabación; el eje horizontal no representa una única serie temporal continua.

**tabla_comparacion_literatura.png**
Comparación de las métricas de estimación de frecuencia cardíaca reportadas en trabajos relacionados y en esta tesis. Los
valores se presentan como referencia y no corresponden a condiciones experimentales equivalentes.
