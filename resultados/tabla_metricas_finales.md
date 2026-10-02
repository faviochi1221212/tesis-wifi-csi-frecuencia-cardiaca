**Tabla X. Resultados de los modelos y líneas base en las 33 corridas**

| Método | MAE (BPM) | RMSE (BPM) | Pearson r |
|---|---|---|---|
| Random Forest (RF) | 8.406 ± 0.284 | 10.578 ± 0.394 | 0.719 ± 0.031 |
| SVR | 8.384 ± 0.298 | 10.621 ± 0.412 | 0.717 ± 0.032 |
| B1 (media del participante) | 8.887 ± 0.291 | 11.090 ± 0.390 | 0.686 ± 0.032 |
| B0 (media global) | 12.196 ± 0.321 | 15.159 ± 0.450 | No definido |

*Nota.* Los valores corresponden a la media ± desviación estándar de las 33 corridas experimentales. El coeficiente de Pearson no se reporta para B0 debido a que genera una predicción constante.

*Fuente.* Calculado a partir de `resultados/metricas_por_semilla.csv` (métricas por semilla sobre las ventanas de prueba de cada corrida).
