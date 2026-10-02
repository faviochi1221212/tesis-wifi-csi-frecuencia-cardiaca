**Tabla X. Calibración de características e hiperparámetros**

| Etapa | Elemento | Valores evaluados | Procedimiento | Resultado general |
|---|---|---|---|---|
| Validación interna | Particiones | 5 folds agrupados por grabación | Solo con el entrenamiento de cada semilla | — |
| Etapa 1 | Ranking de características | Información mutua (MI) o importancia por permutación (PI) | PI calculada con un Random Forest base, incluso para la familia SVR | PI seleccionada en las 66 combinaciones familia/semilla |
| Etapa 1 | Número de características N | 5, 8, 10, 12, 15, 20 o las 24 | El subconjunto seleccionado se evaluó posteriormente con el modelo base correspondiente a cada familia | N seleccionado entre 10 y 20 |
| Etapa 1 | Suavizado w | 1, 3, 5, 7, 9 | Evaluado junto con cada subconjunto | — |
| Etapa 2 — RF | Hiperparámetros y suavizado | 41 candidatos: 40 configuraciones aleatorias + 1 control histórico; w ∈ {1, 3, 5, 7, 9} | Con el subconjunto elegido en la Etapa 1 | Ningún control histórico seleccionado |
| Etapa 2 — SVR | Hiperparámetros y suavizado | 52 candidatos: 50 configuraciones aleatorias + 2 controles históricos; w ∈ {1, 3, 5, 7, 9} | Con el subconjunto elegido en la Etapa 1 | Ningún control histórico seleccionado |
| Criterio | Selección | — | Menor MAE de validación interna, en ambas etapas | — |

*Nota.* El conjunto de prueba de cada semilla no interviene en ninguna decisión de selección. Tras la calibración, cada modelo se reentrena utilizando todo el subconjunto de entrenamiento y genera una única predicción sobre el conjunto de prueba.
