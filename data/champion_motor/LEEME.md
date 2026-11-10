# data/champion_motor/ — Champion Minimayorista

**Qué colocar (obligatorio), en esta carpeta:**

| Archivo | Para qué |
|---|---|
| `1. Base de clientes desagregada por mes.xlsx` | clientes y mes de entrada (ya cerrada: no se actualiza) |
| `Base Minimayorista vf.xlsx` | segmentos |
| `BD_Clientes_*Snack*_<Mes>.xlsx` | lista vigente de clientes y su `Código antiguo` (cambios de código). **Se usa el archivo cuyo nombre trae el mes del cierre** (octubre → `..._Octubre.xlsx`), aunque haya otros; funciona con la columna `cod_cliente_dex` o `Cliente DEX Cod`. Si no hay uno del mes, se usa el más reciente. El motor imprime `[LEGACY] archivo de clientes: ...` |
| `outputs/Champion Minimayorista - Medición/` (carpeta) | pares congelados: `df_matches.csv`, `meta_pares_congelados.json` |

**Opcionales (acumulativos: se agregan filas, no se borran):**

| Archivo | Para qué | Columnas |
|---|---|---|
| `equivalencias_codigos.csv` | cambios de código que NO vengan en `Código antiguo` de la lista | `cod_cliente_nuevo, cod_cliente_antiguo, vigente_desde, origen, observacion` |
| `segmentos_manual.csv` | segmento de clientes que `Base Minimayorista vf.xlsx` no trae (clientes nuevos). **Solo completa a quienes la base no trae**: si la base ya los tiene, manda la base. Se puede poner el código nuevo o el antiguo. | `customer_id, Segmento, observacion` |

Los segmentos válidos son los de la base: `Premium`, `Plus`, `Alto`, `Base`. (No son los de la lista de clientes: `Oro +`, `Oro`, `Plata` son otra clasificación.)

## Rutina de cada mes (Champion)

1. Copia a esta carpeta la lista de clientes del mes (`BD_Clientes_..._<Mes>.xlsx`). El motor usa la que trae el mes del cierre en el nombre.
2. Si hay una `Base Minimayorista vf.xlsx` más nueva, reemplázala.
3. Corre en `dry_run` y lee, en el log, **`[CONCILIACIÓN MENSUAL DE CLIENTES]`**: altas, bajas, códigos antiguos nuevos y clientes sin segmento. El detalle queda en
   `conciliacion/clientes_<AAAAMM>.csv` (una fila por cliente, con el motivo).
4. Completa `segmentos_manual.csv` con las **altas del mes** que salen en `sin_segmento` (las demás pueden seguir como `Sin Asignar`; completarlas cambia su etiqueta
   en los cortes por segmento solo desde ese mes). Si falta algún cambio de código, agrégalo a `equivalencias_codigos.csv`.
5. Vuelve a correr en `dry_run` hasta que la conciliación sea la esperada y después `cargar`.

**Altas, bajas y cambios de código:** un cliente que está en la lista y no en las hojas mensuales es una **alta** y entra en el mes de cierre (si ya tiene entrada publicada o
congelada, **la conserva**). Una **baja** (estaba y ya no está) se mantiene en sus pares y el motor solo avisa. Un cambio de código no cuenta como alta ni baja: el historial se une
al código antiguo.
