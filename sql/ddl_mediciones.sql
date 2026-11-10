-- =============================================================================
-- AMBIENTE DE MEDICIONES — acpe-dev-uc-sandbox-aa.mediciones
-- Brechas / toneladas atribuibles de: surtido (piloto CMP), champion, cpb, cpm
--
-- Diseño:
--   * 3 bases de negocio (append-only, con programa como columna), con la
--     estructura de base_calculo_piloto_cmp_202609.xlsx:
--       base_ton_atribuibles  -> cliente-mes (solo accion) + UB
--       base_brechas          -> par-mes (DiD por par); IC y distribuciones se
--                                calculan por bootstrap sobre esta base
--       base_espejos          -> cliente de acción x categoría x mes con su espejo
--   * 1 log de metricas de ejecucion (embudo, cascada, QA, tiempos)
--   * 1 tabla de apoyo: dim_cosechas (y base_espejos, con el espejo de cada cliente de acción)
--   * 2 vistas "vigente": ultima version de cada (programa, periodo_reporte)
--
-- Regla de version: una carga nueva de un cierre ya publicado NO pisa nada; crea
-- version = max(version) + 1 para ese (programa, periodo_reporte). La vista
-- vigente toma solo la ultima version de cada rebanada.
--
-- Ejecutar en una sola corrida (script de BigQuery). Es re-ejecutable: todo usa
-- IF NOT EXISTS y las semillas se insertan solo si la tabla esta vacia.
-- =============================================================================

-- ── 0) Dataset ───────────────────────────────────────────────────────────────
-- AJUSTAR: la ubicacion debe coincidir con la del dataset `dev` y la de las
-- tablas de origen, o los JOIN entre regiones fallan. Se consulta con:
--   bq show --format=prettyjson acpe-dev-uc-sandbox-aa:dev | grep location
DECLARE ubicacion STRING DEFAULT 'US';

EXECUTE IMMEDIATE FORMAT("""
  CREATE SCHEMA IF NOT EXISTS `acpe-dev-uc-sandbox-aa.mediciones`
  OPTIONS (
    location = '%s',
    description = 'Ambiente de mediciones: brechas y toneladas atribuibles (surtido, champion, cpb, cpm)'
  )
""", ubicacion);


-- ── 1) base_ton_atribuibles (nivel cliente-categoria-mes) ───────────────────
-- Estructura de "Base toneladas (cliente-mes)" de base_calculo_piloto_cmp_202609.xlsx
-- + columnas de UB + trazabilidad.
-- Contiene SOLO clientes de accion. El control espejo vive en base_espejos y
-- agregado en base_brechas (pre_control / post_control).
-- Llave logica: (programa, cod_cliente, des_categoria, periodo, periodo_reporte, version)
-- CPB y CPM: cosecha_grupo y mes_rel van NULL (evaluacion mes calendario).
CREATE TABLE IF NOT EXISTS `acpe-dev-uc-sandbox-aa.mediciones.base_ton_atribuibles` (
  programa                    STRING  NOT NULL OPTIONS (description = 'surtido | champion | cpb | cpm'),
  cod_cliente                 STRING  NOT NULL OPTIONS (description = 'STRING para conservar ceros a la izquierda'),
  cod_cliente_dex             STRING,
  des_categoria               STRING  NOT NULL OPTIONS (description = 'Galletas, Cereales; Papas Fritas (sin control, 100% atribuible) en el piloto'),
  des_oficina                 STRING,
  des_segmento_transaccional  STRING,
  nom_compania                STRING,
  flg_cliente_perfecto        BOOL    OPTIONS (description = 'NULL si no hay dato del flag en ese mes'),
  cosecha_grupo               DATE    OPTIONS (description = 'Ancla de la cosecha; NULL en cpb/cpm'),
  periodo                     DATE    NOT NULL OPTIONS (description = 'Mes calendario evaluado (dia 1)'),
  mes_rel                     INT64   OPTIONS (description = 'Mes relativo a la cosecha (negativos = retroactivo); NULL en cpb/cpm'),

  -- Volumen y crecimiento
  toneladas_ant               FLOAT64 OPTIONS (description = 'Mismo mes del anio anterior; NULL si la categoria no tiene base (ej. Papas Fritas)'),
  toneladas_actual            FLOAT64,
  yoy_pct_raw                 FLOAT64,
  ic_pct_inf_80               FLOAT64 OPTIONS (description = 'Limite inferior IC80 del crecimiento esperado (%)'),
  ic_pct_sup_80               FLOAT64 OPTIONS (description = 'Limite superior IC80 del crecimiento esperado (%)'),
  flg_estado_yoy              STRING  OPTIONS (description = 'CPB/CPM: antiguo = tuvo compra el mismo mes del anio anterior (toneladas_ant > 0); nuevo = toneladas_ant = 0 (atribuible = todo); NULL = sin base ni compra actual. Surtido/Champion: normal | no_existia_periodo_pasado (atribuible = todo) | reactivado (atribuible = 0) | sin_compra_ambos | cien_pct_atribuible_sin_control (Inka Chips)'),
  crec_atrib_80_pct           FLOAT64,
  ton_atrib_80                FLOAT64 OPTIONS (description = 'toneladas_ant x crec_atrib_80_pct; sin tope ni ajuste por compra'),
  flg_estimado_retroactivo    BOOL    NOT NULL OPTIONS (description = 'TRUE = mes sin medicion propia, estimado con la brecha de referencia'),
  kg_atribuibles              FLOAT64 OPTIONS (description = 'valor_atribuible x 1000'),
  valor_atribuible            FLOAT64 OPTIONS (description = 'TONELADAS atribuibles finales (no soles): antiguos = ton_atrib_80 si hubo compra en el mes, 0 si no; nuevos = toneladas_actual'),

  -- UB (ratio_ub sobre la venta). metodo_ub = actual (surtido, champion): ub_actual x valor/actual. metodo_ub = notebook (CPB, CPM): UB base por tonelada x valor; nuevos: toneladas x ub_por_ton_ref
  venta_actual_soles          FLOAT64 OPTIONS (description = 'Venta bruta del mes (mnt_bruto), cliente x categoria'),
  venta_sin_ratio_soles       FLOAT64 OPTIONS (description = 'Parte de la venta sin ratio_ub disponible (no se imputa)'),
  ub_actual_soles             FLOAT64 OPTIONS (description = 'Suma venta x ratio_ub a nivel cliente SAP x familia'),
  ub_base_soles               FLOAT64 OPTIONS (description = 'UB del MISMO MES DEL AÑO ANTERIOR del cliente x categoria (venta base x ratio_ub). Solo con metodo_ub = notebook (CPB, CPM); en surtido y champion es nulo'),
  ub_por_ton_ref              FLOAT64 OPTIONS (description = 'UB por tonelada de referencia de la celda mes x categoria: sum(ub_base) / sum(toneladas_ant) de los antiguos de accion. Solo con metodo_ub = notebook; se usa para los clientes nuevos'),
  ub_atribuible_soles         FLOAT64 OPTIONS (description = 'ub_actual x (valor_atribuible / toneladas_actual). SIN tope: puede superar ub_actual cuando valor_atribuible > toneladas_actual'),

  -- Trazabilidad
  periodo_reporte             DATE      NOT NULL OPTIONS (description = 'Mes de cierre en que se publico la fila'),
  corrida_id                  STRING    NOT NULL,
  version                     INT64     NOT NULL OPTIONS (description = '1 = primera carga; reprocesos explicitos suben la version'),
  config_hash                 STRING    OPTIONS (description = 'Hash de la config con la que corrio (parametros, cosechas, overrides)'),
  fecha_carga                 TIMESTAMP NOT NULL
)
PARTITION BY DATE_TRUNC(periodo, MONTH)
CLUSTER BY programa, cod_cliente, des_categoria
OPTIONS (description = 'Base general de toneladas y UB atribuibles (cliente-mes, solo accion). Append-only; usar vw_base_ton_atribuibles_vigente');

-- Tablas creadas con una versión anterior de este script: agrega las columnas de UB base (re-ejecutar el script también recrea las vistas vigentes de abajo).
ALTER TABLE `acpe-dev-uc-sandbox-aa.mediciones.base_ton_atribuibles` ADD COLUMN IF NOT EXISTS ub_base_soles FLOAT64 OPTIONS (description = 'UB del mismo mes del año anterior (metodo_ub = notebook)');
ALTER TABLE `acpe-dev-uc-sandbox-aa.mediciones.base_ton_atribuibles` ADD COLUMN IF NOT EXISTS ub_por_ton_ref FLOAT64 OPTIONS (description = 'UB por tonelada de referencia de la celda mes x categoria (metodo_ub = notebook)');


-- ── 2) base_brechas (nivel par-mes) ──────────────────────────────────────────
-- Estructura de "Base brecha (par-mes)" de base_calculo_piloto_cmp_202609.xlsx.
-- Un par = un cliente de accion con su control espejo; el control va agregado
-- (pre_control / post_control). Los IC 80/90/95 y las distribuciones NO se
-- guardan aqui: salen del bootstrap (semilla fija) sobre esta base. Lo publicado
-- (brecha + IC) queda registrado en log_metricas_ejecucion, etapa = 'brecha_publicada'.
-- Filas retroactivas (flg_estimado_retroactivo = TRUE): post_* y control NULL,
-- did_kg es la estimacion con la brecha de referencia.
CREATE TABLE IF NOT EXISTS `acpe-dev-uc-sandbox-aa.mediciones.base_brechas` (
  programa                    STRING  NOT NULL,
  par_id                      STRING  NOT NULL OPTIONS (description = 'cosecha|categoria|cod_cliente_accion; une con base_espejos.par_id'),
  cod_cliente_accion_dex      STRING  NOT NULL,
  nom_compania_accion         STRING,
  flg_cliente_perfecto        BOOL,
  cosecha_grupo               DATE    OPTIONS (description = 'NULL en cpb/cpm'),
  cosecha_label               STRING  OPTIONS (description = 'Nombre del grupo; tambien derivable de dim_cosechas'),
  des_categoria               STRING  NOT NULL,
  des_oficina                 STRING,
  des_segmento_transaccional  STRING,
  periodo                     DATE    NOT NULL,
  mes_rel                     INT64   OPTIONS (description = 'NULL en cpb/cpm'),

  -- Panel del par (kg)
  pre_accion                  FLOAT64,
  post_accion                 FLOAT64,
  pre_control                 FLOAT64,
  post_control                FLOAT64,
  did_kg                      FLOAT64 OPTIONS (description = 'Diferencias-en-diferencias del par en kg'),
  flg_estimado_retroactivo    BOOL    NOT NULL,

  -- Atributos de la celda (repetidos por fila, como en el entregable)
  brecha_pct_celda            FLOAT64 OPTIONS (description = 'Brecha % de la celda cosecha x categoria x mes_rel'),
  reportable                  BOOL    OPTIONS (description = 'valida AND paralelismo_ok de la celda cosecha x categoria'),

  -- Opcionales: los exporta la familia mensual (CPB/CPM); en cosechas quedan nulos
  cod_cliente_control         STRING  OPTIONS (description = 'Control espejo del par (STRING, 10 caracteres)'),
  n_pares_celda               INT64   OPTIONS (description = 'Pares de la celda mes x categoria'),
  ic_pct_inf_80_celda         FLOAT64 OPTIONS (description = 'IC80 inferior de la brecha % de la celda (bootstrap del notebook)'),
  ic_pct_sup_80_celda         FLOAT64 OPTIONS (description = 'IC80 superior de la brecha % de la celda (bootstrap del notebook)'),

  -- Trazabilidad
  periodo_reporte             DATE      NOT NULL,
  corrida_id                  STRING    NOT NULL,
  version                     INT64     NOT NULL,
  config_hash                 STRING,
  fecha_carga                 TIMESTAMP NOT NULL
)
PARTITION BY DATE_TRUNC(periodo, MONTH)
CLUSTER BY programa, des_categoria, par_id
OPTIONS (description = 'Base general de brechas (par-mes). Append-only; usar vw_base_brechas_vigente');


-- ── 3) log_metricas_ejecucion (formato largo) ────────────────────────────────
-- Embudo de viabilidad, cascada de matching, filas cargadas, chequeos de QA,
-- tiempos. Se acumula por corrida; no lleva version.
CREATE TABLE IF NOT EXISTS `acpe-dev-uc-sandbox-aa.mediciones.log_metricas_ejecucion` (
  corrida_id                  STRING    NOT NULL,
  programa                    STRING    NOT NULL,
  periodo_reporte             DATE      NOT NULL,
  etapa                       STRING    NOT NULL OPTIONS (description = 'embudo | viabilidad | cascada | carga | qa | tiempo | ...'),
  cosecha_grupo               DATE,
  des_categoria               STRING,
  des_oficina                 STRING,
  metrica                     STRING    NOT NULL,
  valor                       FLOAT64,
  valor_txt                   STRING    OPTIONS (description = 'Para metricas no numericas (ej. OK / FALLA)'),
  fecha_registro              TIMESTAMP NOT NULL
)
PARTITION BY DATE_TRUNC(periodo_reporte, MONTH)
CLUSTER BY programa, etapa
OPTIONS (description = 'Metricas de cada ejecucion del pipeline de mediciones');


-- ── 4) dim_cosechas (apoyo) ──────────────────────────────────────────────────
-- Las cosechas son datos, no codigo. Agregar un grupo (ej. G04 en octubre) =
-- insertar una fila y poner fecha_fin_ingreso al grupo anterior.
CREATE TABLE IF NOT EXISTS `acpe-dev-uc-sandbox-aa.mediciones.dim_cosechas` (
  programa                    STRING  NOT NULL,
  cosecha_grupo               DATE    NOT NULL OPTIONS (description = 'Ancla (cosecha_agrupada)'),
  nombre_grupo                STRING  NOT NULL,
  fecha_inicio_ingreso        DATE    NOT NULL,
  fecha_fin_ingreso           DATE    OPTIONS (description = 'NULL = grupo abierto (recibe ingresos nuevos)'),
  vigente_desde_cierre        DATE    NOT NULL OPTIONS (description = 'Primer cierre en que aplica esta regla; nunca reasigna el pasado'),
  parametros                  JSON    OPTIONS (description = 'Overrides por grupo: reuso de control, pesos, semillas, MIN_MESES'),
  fecha_carga                 TIMESTAMP NOT NULL
)
CLUSTER BY programa
OPTIONS (description = 'Definicion versionada de cosechas por programa');

-- Semillas (solo si la tabla esta vacia). Valores tomados de los notebooks v8 / v3.
-- OJO: max_reuso_control = 1 porque en el v8 del piloto el reuso 1:N esta
-- DESACTIVADO ("corrida de comparacion"). Confirmar con que valor se calculo lo
-- ya entregado antes del backfill. Fechas de vigente_desde_cierre: ajustar.
INSERT INTO `acpe-dev-uc-sandbox-aa.mediciones.dim_cosechas`
SELECT * FROM UNNEST([
  STRUCT('surtido' AS programa, DATE '2026-04-01' AS cosecha_grupo, 'GRUPO 01 (Feb-Abr)' AS nombre_grupo,
         DATE '2026-02-01' AS fecha_inicio_ingreso, DATE '2026-04-30' AS fecha_fin_ingreso,
         DATE '2026-02-01' AS vigente_desde_cierre,
         PARSE_JSON('{"max_reuso_control": 1, "overrides": {"Cereales": {"W_NIVEL": 0.3, "W_FORMA": 0.7, "N_SEMILLAS": 150}}}') AS parametros,
         CURRENT_TIMESTAMP() AS fecha_carga),
  STRUCT('surtido', DATE '2026-06-01', 'GRUPO 02 (May-Jun)',
         DATE '2026-05-01', DATE '2026-06-30', DATE '2026-02-01',
         PARSE_JSON('{"max_reuso_control": 1, "overrides": {"Galletas": {"W_NIVEL": 0.3, "W_FORMA": 0.7, "N_SEMILLAS": 150}}}'),
         CURRENT_TIMESTAMP()),
  STRUCT('surtido', DATE '2026-08-01', 'GRUPO 03 (Jul-Ago, abierto)',
         DATE '2026-07-01', CAST(NULL AS DATE), DATE '2026-02-01',
         PARSE_JSON('{"max_reuso_control": 1}'),
         CURRENT_TIMESTAMP()),
  STRUCT('champion', DATE '2026-03-01', 'COSECHA UNICA (Mar)',
         DATE '2026-03-01', CAST(NULL AS DATE), DATE '2026-03-01',
         PARSE_JSON('{"max_reuso_control": 1}'),
         CURRENT_TIMESTAMP())
])
WHERE NOT EXISTS (SELECT 1 FROM `acpe-dev-uc-sandbox-aa.mediciones.dim_cosechas`);


-- ── 5) base_espejos (cliente de acción x categoría x mes, con su espejo) ─────────
-- Mismo grano que base_ton_atribuibles (acción), con el control espejo de ese mes y categoría; o sin espejo y el motivo.
-- Reemplaza al borrador `pares_congelados`: contiene los pares por mes y el espejo de cada cliente de acción, para detectar desfases.
-- En surtido/Champion el par es fijo por cosecha y categoría y se expande a cada mes (incluidos los retroactivos).
-- Inka Chips no tiene espejo (diferencial YoY sin matching ni control) y no aparece aquí.
CREATE TABLE IF NOT EXISTS `acpe-dev-uc-sandbox-aa.mediciones.base_espejos` (
  programa                    STRING  NOT NULL,
  cod_cliente_accion          STRING  NOT NULL OPTIONS (description = 'STRING de 10 caracteres'),
  des_categoria               STRING  NOT NULL,
  periodo                     DATE    NOT NULL,
  cosecha_grupo               DATE    OPTIONS (description = 'NULL en cpb/cpm'),
  mes_rel                     INT64   OPTIONS (description = 'NULL en cpb/cpm'),

  cod_cliente_espejo          STRING  OPTIONS (description = 'Control espejo; NULL si el cliente de acción no tiene espejo'),
  estado_espejo               STRING  NOT NULL OPTIONS (description = 'con_espejo | sin_espejo'),
  motivo_sin_espejo           STRING  OPTIONS (description = 'Por qué no hay espejo: cliente_nuevo, celda_sin_matching, no_en_pares_fijos, sin_control_en_cascada, fuera_de_muestra, sin_pareja...'),
  par_id                      STRING  OPTIONS (description = 'Une con base_brechas.par_id'),

  des_oficina                 STRING  OPTIONS (description = 'Del cliente de acción'),
  nom_compania                STRING,
  des_segmento_transaccional  STRING,
  des_oficina_espejo          STRING  OPTIONS (description = 'Del espejo'),
  nom_compania_espejo         STRING,
  des_segmento_espejo         STRING,
  nivel_match                 INT64   OPTIONS (description = 'Nivel de la cascada con que se emparejó (1 = más estricto); NULL si no se registra'),
  flg_estimado_retroactivo    BOOL    NOT NULL,

  periodo_reporte             DATE      NOT NULL,
  corrida_id                  STRING    NOT NULL,
  version                     INT64     NOT NULL,
  config_hash                 STRING,
  fecha_carga                 TIMESTAMP NOT NULL
)
PARTITION BY DATE_TRUNC(periodo, MONTH)
CLUSTER BY programa, des_categoria, cod_cliente_accion
OPTIONS (description = 'Espejo de cada cliente de acción por categoría y mes. Append-only; usar vw_base_espejos_vigente');


-- ── 6) Vistas vigentes ───────────────────────────────────────────────────────
-- Regla única (la misma de src/mediciones/frontera.py, probada en tests/test_frontera.py):
--   1) en cada (programa, periodo_reporte) solo cuenta la última version cargada;
--   2) cada mes y cosecha (programa, periodo, cosecha_grupo) lo define la rebanada MÁS RECIENTE que lo publicó.
-- Con la frontera de meses cerrados, un mes lo publica un solo cierre y queda fijo; un mes provisional
-- o reabierto se reemplaza COMPLETO por el cierre más reciente que lo publicó.
CREATE OR REPLACE VIEW `acpe-dev-uc-sandbox-aa.mediciones.vw_base_ton_atribuibles_vigente` AS
SELECT * EXCEPT (_rep_max)
FROM (
  SELECT *, MAX(periodo_reporte) OVER (PARTITION BY programa, DATE_TRUNC(periodo, MONTH), IFNULL(CAST(cosecha_grupo AS STRING), '')) AS _rep_max
  FROM (
    SELECT *
    FROM `acpe-dev-uc-sandbox-aa.mediciones.base_ton_atribuibles`
    QUALIFY version = MAX(version) OVER (PARTITION BY programa, periodo_reporte)
  )
)
WHERE periodo_reporte = _rep_max;

CREATE OR REPLACE VIEW `acpe-dev-uc-sandbox-aa.mediciones.vw_base_espejos_vigente` AS
SELECT * EXCEPT (_rep_max)
FROM (
  SELECT *, MAX(periodo_reporte) OVER (PARTITION BY programa, DATE_TRUNC(periodo, MONTH), IFNULL(CAST(cosecha_grupo AS STRING), '')) AS _rep_max
  FROM (
    SELECT *
    FROM `acpe-dev-uc-sandbox-aa.mediciones.base_espejos`
    QUALIFY version = MAX(version) OVER (PARTITION BY programa, periodo_reporte)
  )
)
WHERE periodo_reporte = _rep_max;

CREATE OR REPLACE VIEW `acpe-dev-uc-sandbox-aa.mediciones.vw_base_brechas_vigente` AS
SELECT * EXCEPT (_rep_max)
FROM (
  SELECT *, MAX(periodo_reporte) OVER (PARTITION BY programa, DATE_TRUNC(periodo, MONTH), IFNULL(CAST(cosecha_grupo AS STRING), '')) AS _rep_max
  FROM (
    SELECT *
    FROM `acpe-dev-uc-sandbox-aa.mediciones.base_brechas`
    QUALIFY version = MAX(version) OVER (PARTITION BY programa, periodo_reporte)
  )
)
WHERE periodo_reporte = _rep_max;


-- ── 7) Espejos con vigencia: cada espejo de un cliente de acción con su tramo desde-hasta (meses consecutivos) ──
-- Misma lógica que salidas.vigencia_espejos (exportable 'Espejos_vigencia').
CREATE OR REPLACE VIEW `acpe-dev-uc-sandbox-aa.mediciones.vw_espejos_vigencia` AS
WITH e AS (
  SELECT programa, cod_cliente_accion, des_categoria, IFNULL(CAST(cosecha_grupo AS STRING), '—') AS cosecha_grupo,
         cod_cliente_espejo, nivel_match, DATE_TRUNC(periodo, MONTH) AS periodo, periodo_reporte,
         IFNULL(flg_estimado_retroactivo, FALSE) AS retro
  FROM `acpe-dev-uc-sandbox-aa.mediciones.vw_base_espejos_vigente`
  WHERE cod_cliente_espejo IS NOT NULL
),
islas AS (
  SELECT *, DATE_DIFF(periodo, DATE '2000-01-01', MONTH)
            - ROW_NUMBER() OVER (PARTITION BY programa, cod_cliente_accion, des_categoria, cosecha_grupo, cod_cliente_espejo ORDER BY periodo) AS isla
  FROM e
)
SELECT programa, cod_cliente_accion, des_categoria, cosecha_grupo, cod_cliente_espejo,
       MIN(periodo) AS desde, MAX(periodo) AS hasta, COUNT(*) AS meses, COUNTIF(retro) AS meses_retroactivos,
       ANY_VALUE(nivel_match) AS nivel_match, MIN(periodo_reporte) AS primer_cierre, MAX(periodo_reporte) AS ultimo_cierre
FROM islas
GROUP BY programa, cod_cliente_accion, des_categoria, cosecha_grupo, cod_cliente_espejo, isla;

-- =============================================================================
-- QA / CONTROL DE CARGA (ejecutar desde el orquestador, no forman parte del DDL)
-- =============================================================================
-- A) Bloqueo de cierre ya publicado (debe devolver 0 filas para cargar version 1):
--   SELECT programa, periodo_reporte, MAX(version) AS ultima_version
--   FROM `acpe-dev-uc-sandbox-aa.mediciones.base_ton_atribuibles`
--   WHERE programa = @programa AND periodo_reporte = @periodo_reporte
--   GROUP BY 1, 2;
--
-- B) Duplicados en la base vigente (debe devolver 0 filas):
--   SELECT programa, cod_cliente, des_categoria, periodo, COUNT(*) AS n
--   FROM `acpe-dev-uc-sandbox-aa.mediciones.vw_base_ton_atribuibles_vigente`
--   GROUP BY 1, 2, 3, 4 HAVING COUNT(*) > 1;
--
-- C) Atribuible por encima de lo comprado (informativo, NO bloquea; piloto sep-2026: 3,259 filas):
--   SELECT programa, periodo, COUNT(*) AS n_filas
--   FROM `acpe-dev-uc-sandbox-aa.mediciones.vw_base_ton_atribuibles_vigente`
--   WHERE toneladas_actual > 0 AND valor_atribuible > toneladas_actual
--   GROUP BY 1, 2;
