# Decisiones y reglas técnicas acordadas (octubre 2026)

## Publicación: lo reportado no cambia
- **Línea base:** lo reportado de setiembre 2026 se sembró en `mediciones` como el cierre `2026-09-01` (versión 1, `config_hash = 'reportado'`).
  CPB 774.0 t, CPM 1,316.3 t, surtido 307.8 t, Champion 328.8 t.
- **Frontera** (`src/mediciones/frontera.py`): cada cierre publica solo lo posterior al último mes publicado por cierres anteriores. Lo cerrado
  se recalcula únicamente para medir la deriva y nunca se publica. Un cierre anterior al último publicado se bloquea; repetir un cierre ya
  publicado no carga nada (salvo `reproceso`).
- **Lo vigente** (vistas `vw_*_vigente` = `frontera.vigente_local`): última versión de cada (programa, cierre) y, por (programa, mes, cosecha),
  el cierre más reciente que lo publicó.
- **Provisional por cosecha:** solo el G04 de Impulso (cosecha `2026-12-01`) se republica completo en cada cierre entre `provisional_desde`
  (`2026-10-01`) y `provisional_hasta` (`2027-02-01`). Sus pares no se verifican contra lo publicado mientras es provisional.
- **Estado local** (`estado_local.py`): pares fijos de CPB/CPM y la carpeta `outputs/` de surtido y Champion se respaldan antes de correr y
  se restauran al terminar, salvo carga real. Un `dry_run`, una carga bloqueada o un error no dejan rastro.

## Deriva (validación en cada cierre)
- Por celda mes × categoría (× cosecha en brechas): bloquea con ≥ 0.25 pp en brechas o ≥ 1 % y ≥ 1 kg en toneladas
  (`incremental.TOLERANCIA_CELDA`). Las altas tardías (llaves que no estaban publicadas) no cuentan. Los conteos por fila son informativos.
- Validación de setiembre (motor contra reportado, por mes × categoría): CPB y Champion exactos; CPM 99.2 % (máx. 0.08 pp); surtido dentro
  de 0.22 pp y +0.10 % en el total, con residuo por ventas corregidas de unos 22 controles.

## Impulso (surtido)
- **Calendario** (`calendario.py`, `programas.yaml`): G01 feb-abr (cosecha abr), G02 may-jun (jun), G03 jul-set (ago), G04 oct-dic (dic).
  Retroactivo: los meses previos a la cosecha de quien entró antes toman la brecha de los meses 0-2 de su cosecha.
- **G04 en dos etapas:** preliminar en oct-nov (anclado en octubre, incremental; en las salidas se rotula con la cosecha diciembre) y
  definitivo desde diciembre (se descartan los pares preliminares y se empareja completo con cosecha diciembre; se congela).
- **G03:** cerró altas en setiembre. Su julio retroactivo completa la referencia con octubre, pero queda como se reportó (frontera).
- **Controles congelados** conservan su serie aunque hoy los saque un filtro de la base: territorio de impulso (`terr_impulso`, 19 controles
  salieron después del reporte), compañía excluida o lista de bodega (`surtido_ideal_bodeg`, la lista creció hacia atrás).
- **Consultas ancladas al cierre:** segmentación vigente al cierre (no la última que exista) y marca ROMA con tope al cierre.

## Champion
- Cosecha ancla con altas acumulativas; sus funciones idénticas a surtido (tamaño de muestra, hojas Excel) se comparten.
- **Lista de clientes mensual:** el motor usa el archivo cuyo nombre trae el mes del cierre (aunque haya otros y aunque el nombre tenga un error de tipeo) y acepta la columna
  `cod_cliente_dex` o `Cliente DEX Cod`. Las altas que solo vienen en la lista entran en el mes de cierre, **salvo que ya tengan entrada publicada** (el primer mes publicado del
  cliente, que el runner le pasa como `ENTRADAS_PUBLICADAS`) o congelada: reasignarlas al mes de cierre les quitaría sus meses anteriores (falló en octubre con 11 clientes).
- **Conciliación mensual** (informativa): altas, bajas, cambios de código y clientes sin segmento contra la lista del mes anterior; un cambio de código no cuenta como alta ni baja.
- **Segmento:** sale de `Base Minimayorista vf.xlsx` (`Premium/Plus/Alto/Base`). `segmentos_manual.csv` completa solo a quienes la base no trae. El segmento de la lista de clientes
  (`Oro +/Oro/Plata`) es otra clasificación, sin equivalencia 1 a 1, y no se usa.
- Cambios de código: `equivalencias_codigos.csv` (acumulativo; cadenas al código más antiguo). El archivo legacy toma la versión más reciente.

## CPB y CPM
- Cada mes se empareja por separado: un cierre calcula solo el mes nuevo (`RECALCULAR_DESDE`); `validar_cerrados=True` recalcula los cerrados.
- Pares de meses cerrados: se restauran desde `base_espejos` publicada.

## UB atribuible
- Los 4 programas usan el método de CPB/CPM (`metodo_ub: notebook`): con año anterior, atribuible × (UB base / toneladas año anterior); sin año
  anterior, **atribuible** × UB/TN de referencia de la categoría y mes. CPB/CPM filtran el negocio CP1 (su perímetro de ventas); surtido y
  Champion no filtran (su sell-out tampoco).
- Corrección del 2026-10-25: en cosechas, la UB de los clientes sin año anterior se calculaba sobre las toneladas actuales, y los
  `reactivado` (0 t atribuibles) recibían UB. Corregido. La UB de setiembre de surtido y Champion se republicó como versión 3: surtido
  S/ 1,265,712, Champion S/ 2,054,716. Guarda: si hay UB atribuible con 0 toneladas atribuibles, no se carga.
- Lavavajillas JW (CPM): UB 0 en 2026, limitación conocida.

## BigQuery
- `sql/ddl_mediciones.sql` crea tablas y vistas. `bigquery.actualizar_esquema(cli, ejecutar=True)` las deja al día.
- **Tablas de una versión anterior:** si quedaran columnas obligatorias que el motor no carga, `bigquery.recrear_tablas_vacias(cli, confirmar=True)`
  (solo si están vacías). La siembra y el runner revisan las 3 tablas antes de cargar la primera: nunca queda una carga a medias.
