# Mediciones de brechas: CPB, CPM, Champion Minimayorista e Impulso (surtido)

Motor único para los 4 programas. Calcula pares acción-control, brechas (DiD con IC), toneladas atribuibles y UB atribuible, y publica en
BigQuery (`acpe-dev-uc-sandbox-aa.mediciones`).

**Regla central:** lo publicado **no cambia**. Enero-setiembre 2026 es exactamente lo reportado. Cada cierre publica **solo su mes**, salvo el
G04 de Impulso, que se republica completo hasta que su retroactivo queda final (febrero 2027).

---

## Qué colocar (una sola vez)

Descomprime este zip en una carpeta nueva y copia desde tu carpeta de trabajo anterior **solo esto** (respetando las rutas):

| Programa | Qué copiar a `data/` | Para qué |
|---|---|---|
| **Impulso (surtido)** | la carpeta completa `data/surtido_motor/outputs/Surtido Ideal - Medición/` (`df_matches.csv`, `df_sin_pareja.csv`, `estado_matching.json` y los demás CSV) | pares congelados de G01-G03 y estado del G04 |
| **Champion** | `data/champion_motor/1. Base de clientes desagregada por mes.xlsx`, `Base Minimayorista vf.xlsx`, `BD_Clientes_Perfectos_Snack_-_<Mes>.xlsx` y la carpeta `outputs/Champion Minimayorista - Medición/` (`df_matches.csv`, `meta_pares_congelados.json`) | clientes, segmentos, códigos antiguos y pares congelados |
| **CPB, CPM** | nada | sus pares se restauran solos desde BigQuery |

Opcionales (Champion): `data/champion_motor/equivalencias_codigos.csv` (cambios de código) y `segmentos_manual.csv` (segmento de clientes nuevos). Vienen vacíos.

No hace falta copiar `runs/` ni nada más. Credenciales: las mismas de BigQuery que ya usas.

Comprobación: abre `notebooks/cierre_mensual.ipynb`, corre hasta la sección 1 ("Archivos que tienen que estar"). Todo debe decir `OK`.

---

## Cómo se usa cada mes: `notebooks/cierre_mensual.ipynb`

Solo se edita `MES_CIERRE` y `MODO`.

**Probar** (cualquier día, cuantas veces quieras): `MODO = 'dry_run'` y correr todo. No escribe en BigQuery y **no deja estado**: los pares
y el estado congelado quedan como estaban, así que la prueba nunca ensucia el cierre real. Revisas la sección 3 y el Excel de la vista previa
(sección 4), que muestra lo que quedaría publicado. Cada prueba del mismo mes reemplaza ese Excel.

**Cerrar el mes** (cuando cierren las ventas: ~día 4 del mes siguiente):
1. `MODO = 'dry_run'` → correr → revisar (frontera, meses a publicar, celdas fuera de tolerancia, vista previa).
2. `MODO = 'cargar'` → correr. Publica el mes y actualiza el estado local (solo si la carga se hizo).
3. La sección 5 exporta los Excel finales desde lo publicado.

| Cierre | `MES_CIERRE` | Cuándo |
|---|---|---|
| octubre | `2026-10-01` | ~4 de noviembre |
| noviembre | `2026-11-01` | ~4 de diciembre |
| diciembre | `2026-12-01` | ~4 de enero |
| enero, febrero 2027 | `2027-01-01`, `2027-02-01` | completan el retroactivo del G04 |

Siempre en orden: un cierre anterior al último publicado se bloquea.

**Qué revisar en cada cierre** (sección 3):
- `frontera` = el mes anterior al cierre;
- `se publican` = el mes del cierre (en Impulso, además, los meses del G04);
- celdas fuera de tolerancia. Esperadas y explicadas en octubre: **abril-Galletas G01** (base negativa) y **julio G03** (su retroactivo completa
  la referencia con octubre; lo publicado se mantiene). Si solo aparecen esas, se carga con `ACEPTAR_DERIVA = True` y un `MOTIVO_DERIVA`.
  Cualquier otra: no cargar y revisar.

---

## Qué hace cada programa en cada cierre

**CPB y CPM:** calculan **solo el mes nuevo** (cada mes se empareja por separado; es rápido). `VALIDAR_CERRADOS = True` recalcula también los
meses cerrados y mide su deriva (más lento; sugerido una vez por trimestre). Limitación conocida 2026: Lavavajillas JW (CPM) sin UB.

**Champion:** agrega el mes nuevo para sus pares congelados. Ya no entran clientes por hojas mensuales; las altas, bajas y cambios de código vienen en la **lista de clientes
del mes** (`BD_Clientes_..._<Mes>.xlsx`), que se copia cada mes a `data/champion_motor/`. El motor la compara con la del mes anterior y deja la conciliación en el log y en
`data/champion_motor/conciliacion/clientes_<AAAAMM>.csv`. Los clientes nuevos que la base de minimayoristas no trae se completan en `segmentos_manual.csv`. Rutina completa
en `data/champion_motor/LEEME.md`.

**Impulso (surtido)** | la carpeta completa `data/surtido_motor/outputs/Surtido Ideal - Medición/` (`df_matches.csv`, `df_sin_pareja.csv`, `estado_matching.json` y los demás CSV) | pares congelados de G01-G03 y estado del G04 |
| **Champion** | `data/champion_motor/1. Base de clientes desagregada por mes.xlsx`, `Base Minimayorista vf.xlsx`, `BD_Clientes_Perfectos_Snack_-_<Mes>.xlsx` y la carpeta `outputs/Champion Minimayorista - Medición/` (`df_matches.csv`, `meta_pares_congelados.json`) | clientes, segmentos, códigos antiguos y pares congelados |
| **CPB, CPM** | nada | sus pares se restauran solos desde BigQuery |

Opcionales (Champion): `data/champion_motor/equivalencias_codigos.csv` (cambios de código) y `segmentos_manual.csv` (segmento de clientes nuevos). Vienen vacíos.

No hace falta copiar `runs/` ni nada más. Credenciales: las mismas de BigQuery que ya usas.

Comprobación: abre `notebooks/cierre_mensual.ipynb`, corre hasta la sección 1 ("Archivos que tienen que estar"). Todo debe decir `OK`.

---

## Cómo se usa cada mes: `notebooks/cierre_mensual.ipynb`

Solo se edita `MES_CIERRE` y `MODO`.

**Probar** (cualquier día, cuantas veces quieras): `MODO = 'dry_run'` y correr todo. No escribe en BigQuery y **no deja estado**: los pares
y el estado congelado quedan como estaban, así que la prueba nunca ensucia el cierre real. Revisas la sección 3 y el Excel de la vista previa
(sección 4), que muestra lo que quedaría publicado. Cada prueba del mismo mes reemplaza ese Excel.

**Cerrar el mes** (cuando cierren las ventas: ~día 4 del mes siguiente):
1. `MODO = 'dry_run'` → correr → revisar (frontera, meses a publicar, celdas fuera de tolerancia, vista previa).
2. `MODO = 'cargar'` → correr. Publica el mes y actualiza el estado local (solo si la carga se hizo).
3. La sección 5 exporta los Excel finales desde lo publicado.

| Cierre | `MES_CIERRE` | Cuándo |
|---|---|---|
| octubre | `2026-10-01` | ~4 de noviembre |
| noviembre | `2026-11-01` | ~4 de diciembre |
| diciembre | `2026-12-01` | ~4 de enero |
| enero, febrero 2027 | `2027-01-01`, `2027-02-01` | completan el retroactivo del G04 |

Siempre en orden: un cierre anterior al último publicado se bloquea.

**Qué revisar en cada cierre** (sección 3):
- `frontera` = el mes anterior al cierre;
- `se publican` = el mes del cierre (en Impulso, además, los meses del G04);
- celdas fuera de tolerancia. Esperadas y explicadas en octubre: **abril-Galletas G01** (base negativa) y **julio G03** (su retroactivo completa
  la referencia con octubre; lo publicado se mantiene). Si solo aparecen esas, se carga con `ACEPTAR_DERIVA = True` y un `MOTIVO_DERIVA`.
  Cualquier otra: no cargar y revisar.

---

## Qué hace cada programa en cada cierre

**CPB y CPM:** calculan **solo el mes nuevo** (cada mes se empareja por separado; es rápido). `VALIDAR_CERRADOS = True` recalcula también los
meses cerrados y mide su deriva (más lento; sugerido una vez por trimestre). Limitación conocida 2026: Lavavajillas JW (CPM) sin UB.

**Champion:** agrega el mes nuevo para sus pares congelados. Ya no entran clientes; los **cambios de código** se registran como filas en
`data/champion_motor/equivalencias_codigos.csv` (una fila por cambio, nunca se borran; el historial se une al código más antiguo):

| cod_cliente_nuevo | cod_cliente_antiguo | vigente_desde | origen | observacion |
|---|---|---|---|---|
| 0004246795 | 0001523339 | 2026-10-01 | manual | cambio de distribuidora |

Si el archivo de clientes legacy llega con otro mes en el nombre (`BD_Clientes_Perfectos_Snack_-_Octubre.xlsx`), se usa el más reciente.

**Impulso (surtido)** — grupos en `config/programas.yaml` (`calendario_cosechas`):

| Grupo | Altas | Cosecha | Desde octubre |
|---|---|---|---|
| G01 | feb-abr | abr | cerrado: agrega el mes nuevo |
| G02 | may-jun | jun | cerrado: agrega el mes nuevo |
| G03 | jul-set | ago | cerrado (ya no recibe altas); su julio retroactivo queda como se reportó |
| **G04** | **oct-dic** | **dic** | preliminar en oct-nov, definitivo desde dic |

| Cierre | G04 | `estado_publicacion` |
|---|---|---|
| octubre | empareja las altas de octubre (anclado en octubre) y mide octubre directo | `preliminar` |
| noviembre | conserva esos pares, suma las altas de noviembre; republica oct-nov | `preliminar` |
| diciembre | **pasa a definitivo**: descarta los pares preliminares y empareja completo con cosecha diciembre; oct-nov pasan a retroactivos | `provisional` |
| enero, febrero 2027 | completa la referencia del retroactivo; republica | `provisional` |
| desde marzo 2027 | fijo | `definitivo` |

Los números del G04 **cambian en diciembre** (otro método y otros pares): es lo previsto. Para ver cuánto: `exportar(..., modo='historico')`.
En el log de surtido: `[CALENDARIO] ... grupo abierto ... | rehacer ...` (en diciembre `rehacer: ['2026-12']`) y `entrantes de la cosecha abierta`.

---

## Qué outputs se obtienen

**En BigQuery** (`mediciones`), lo que se publica en cada `cargar`: `base_ton_atribuibles`, `base_brechas`, `base_espejos` y sus vistas
`vw_*_vigente` (lo que vale hoy), más `vw_espejos_vigencia` (cada espejo con su tramo desde-hasta) y el log de cada corrida.

**En `exportes/`** (Excel, un archivo por programa, 4 hojas + diccionario):

| Hoja | Nivel | Contenido |
|---|---|---|
| `Toneladas` | cliente × categoría × mes | toneladas, toneladas atribuibles, UB atribuible |
| `Brechas_par_mes` | par × mes | base de brechas: pre/post de acción y control, DiD, brecha de la celda |
| `Brechas_celda` | mes × categoría (× cosecha) | pares, brecha, IC80, reportable, DiD, toneladas y UB atribuibles |
| `Espejos_vigencia` | cliente de acción × categoría | cada espejo con desde-hasta |

Todas con `periodo` (mes medido), `periodo_reporte` (**fecha de reporte**: el cierre que publicó la fila) y `estado_publicacion`.

- `exportes/vista_previa_AAAAMM/`: lo que quedaría publicado, desde una prueba (`dry_run`). Se reemplaza en cada prueba del mes.
- `exportes/<programa>_vigente_<desde>_<hasta>.xlsx`: lo publicado, después de `cargar` (o desde `notebooks/exportar.ipynb`: rango, modo
  vigente/histórico, destino Excel o BigQuery `mediciones_reporte`, que reemplaza solo el rango pedido).

**En `runs/<corrida>/`**: resultados crudos, manifiesto e insumos de cada corrida (para reproducirla exactamente).

---

## Control de deriva (qué bloquea la carga)

En los meses cerrados que se recalculan (Champion y surtido siempre; CPB y CPM con `VALIDAR_CERRADOS`) se compara contra lo publicado **por celda
mes × categoría** (× cosecha en brechas). Bloquea si una brecha cambia ≥ **0.25 pp** o unas toneladas ≥ **1 %** (y ≥ 1 kg). Las altas tardías no
cuentan (no se publican). En `dry_run` solo avisa; en `cargar` se continúa con `ACEPTAR_DERIVA = True` + motivo. Lo publicado nunca se modifica.

## Si hay que corregir algo

| Situación | Qué hacer |
|---|---|
| un cierre publicado salió mal | `REPROCESO = True`, `MOTIVO = '...'` (versión nueva del mismo cierre) |
| cambiar meses ya cerrados (decisión de negocio) | `REABRIR_DESDE = 'AAAA-MM-01'`, `MOTIVO = '...'` |
| repetir una corrida exactamente | `correr_mes(..., insumos='reproducir', reproducir_de='<id de corrida>')` |
| congelar el G04 antes/después de febrero | `provisional_hasta` en `config/programas.yaml` |

## Pendiente para 2027

Definir cómo se mide 2027 (nuevos espejos o insumos, año base) y el modo `redefinir_grupos`. Nada de eso se necesita para octubre-diciembre.

## Estructura

```
config/programas.yaml        los 4 programas, calendario de cosechas, UB, lo provisional
notebooks/cierre_mensual     el cierre de cada mes (prueba + carga + exportación)
notebooks/exportar           exportar lo publicado (rango, vigente/histórico, Excel/BigQuery)
src/mediciones/              el motor (programas/, familias/, metodologia/, nucleo/, runner, frontera, salidas, ...)
sql/ddl_mediciones.sql       tablas y vistas de BigQuery
data/<programa>/             entradas y estado congelado de cada programa
tests/                       pruebas del motor (python -m pytest)
docs/decisiones.md           las reglas y decisiones técnicas acordadas
```
