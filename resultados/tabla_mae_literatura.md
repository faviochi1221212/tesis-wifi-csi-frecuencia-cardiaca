**Tabla X. Comparación del error de estimación de frecuencia cardíaca con trabajos relacionados**

| Estudio | Dataset / hardware | Participantes | Método | Escenario de evaluación | MAE reportado (BPM) | Observación de comparabilidad |
|---|---|---|---|---|---|---|
| Alzaabi et al. (2025) | Datos propios en domicilios; dos ESP32-DevKitC-VE | 12 adultos ≥ 60 años; 8 mediciones emparejadas de 7 voluntarios | Procesamiento de señal con wavelets (DWT y CWT), sin aprendizaje supervisado | Validación por sesión frente a un sensor PPG; sin partición de entrenamiento y prueba | 9.26 | Otro hardware y otra población; error por sesión de medición y sin modelo entrenado |
| Kocheta et al. — PulseFi (2025) | eHealth CSI; Raspberry Pi 4B con Nexmon | 118 | Red LSTM | Partición 64/16/20 % en entrenamiento, validación y prueba; promedio de 3 repeticiones. | 0.27 ± 0.03 (ventana de 10 s); 0.17 ± 0.01 (ventana de 30 s) | Mismo dataset, pero distinto protocolo de partición, modelo, representación y número de participantes. |
| Esta tesis — RF | eHealth CSI; Raspberry Pi 4B (54 subportadoras activas) | 125 | Random Forest con regresión residual | Grabaciones nuevas de participantes conocidos; partición por grabación dentro de cada participante; 33 corridas | 8.406 ± 0.284 | Media ± DE de 33 corridas; ventanas de aproximadamente 30 s |
| Esta tesis — SVR | eHealth CSI; Raspberry Pi 4B (54 subportadoras activas) | 125 | SVR con kernel RBF y regresión residual | Mismo protocolo que RF; 33 corridas | 8.384 ± 0.298 | Media ± DE de 33 corridas; ventanas de aproximadamente 30 s |

*Nota.* Los valores se presentan como referencia y no corresponden a condiciones experimentales equivalentes. Los estudios difieren en hardware, número de participantes, estrategia de partición, duración de las ventanas, representación de la señal y método de estimación. Por ello, los valores no deben interpretarse como una comparación directa de desempeño.

*Fuentes verificadas.* Alzaabi, Saied y Arslan (2025), IEEE Journal of Translational Engineering in Health and Medicine, doi:10.1109/JTEHM.2025.3624469 (texto completo en PMC12599888). Kocheta, Bhatia y Obraczka (2025), arXiv:2510.24744v1, Tabla I (eHealth).

*No incluidos hasta disponer de fuente primaria verificable:* Sazid et al. (2024) y Gouveia et al. (2024).
