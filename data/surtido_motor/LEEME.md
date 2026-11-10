# data/surtido_motor/ — Impulso (Surtido Ideal)

**Qué colocar (obligatorio):** la carpeta `outputs/Surtido Ideal - Medición/` completa de tu carpeta de trabajo anterior
(`df_matches.csv`, `df_sin_pareja.csv`, `estado_matching.json` y los demás CSV). Ahí están los pares congelados de G01-G03 y el estado del G04.

**No la borres ni la reemplaces a mano.** El motor la actualiza solo cuando un cierre se **carga**; las pruebas (`dry_run`) la dejan como estaba.
Respáldala antes de cada `cargar`, sobre todo antes del cierre de diciembre (el G04 pasa a definitivo y cambia sus pares).

Las ventas, cosechas, territorio, segmentación y lista de bodega vienen de BigQuery.
