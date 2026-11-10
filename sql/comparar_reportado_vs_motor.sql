-- =============================================================================
-- REPORTADO vs MOTOR — compara la línea base reportada (dataset {rep}) con lo que cargó el motor (dataset {mot}), para un programa (@p).
-- Las dos fuentes son las vistas "vigente" (última versión de cada cierre). Se corre desde Python (reportado.comparar_en_bigquery) o a mano:
-- reemplaza {rep} y {mot} por `proyecto.dataset` y define @p (p. ej. 'surtido_motor').
-- Cada consulta empieza con una marca `-- @nombre`. Tolerancia 1e-9 en toneladas y kg; 1e-6 en soles (la UB suma muchos términos y el orden de la suma puede diferir).
--   estado de una fila: comunes (en ambos) · solo_reportado (el motor no la tiene) · solo_motor (el motor agrega filas que no estaban reportadas)
--   retro: la fila es una estimación retroactiva (Impulso): se espera que pueda cambiar
-- =============================================================================

-- @ton_resumen
-- Toneladas, por categoría × mes × retroactivo: cuántas filas difieren y en qué columna, y los totales de cada lado.
WITH r AS (
  SELECT cod_cliente, des_categoria, periodo, toneladas_ant, toneladas_actual, ic_pct_inf_80, ic_pct_sup_80, valor_atribuible, ub_base_soles, ub_atribuible_soles, flg_estimado_retroactivo
  FROM `{rep}.vw_base_ton_atribuibles_vigente` WHERE programa = @p),
m AS (
  SELECT cod_cliente, des_categoria, periodo, toneladas_ant, toneladas_actual, ic_pct_inf_80, ic_pct_sup_80, valor_atribuible, ub_base_soles, ub_atribuible_soles, flg_estimado_retroactivo
  FROM `{mot}.vw_base_ton_atribuibles_vigente` WHERE programa = @p),
j AS (
  SELECT
    COALESCE(r.des_categoria, m.des_categoria) AS des_categoria,
    COALESCE(r.periodo, m.periodo) AS periodo,
    COALESCE(r.flg_estimado_retroactivo, m.flg_estimado_retroactivo, FALSE) AS retro,
    r.cod_cliente IS NULL AS solo_motor, m.cod_cliente IS NULL AS solo_reportado,
    r.toneladas_ant AS ant_rep, m.toneladas_ant AS ant_mot, r.toneladas_actual AS act_rep, m.toneladas_actual AS act_mot,
    r.ic_pct_inf_80 AS inf_rep, m.ic_pct_inf_80 AS inf_mot, r.ic_pct_sup_80 AS sup_rep, m.ic_pct_sup_80 AS sup_mot,
    r.valor_atribuible AS val_rep, m.valor_atribuible AS val_mot,
    r.ub_base_soles AS ubb_rep, m.ub_base_soles AS ubb_mot, r.ub_atribuible_soles AS ub_rep, m.ub_atribuible_soles AS ub_mot
  FROM r FULL OUTER JOIN m
    ON r.cod_cliente = m.cod_cliente AND r.des_categoria = m.des_categoria AND r.periodo = m.periodo)
SELECT
  des_categoria, FORMAT_DATE('%Y-%m', periodo) AS mes, retro,
  COUNTIF(NOT solo_motor AND NOT solo_reportado) AS filas_comunes,
  COUNTIF(solo_reportado) AS solo_reportado, COUNTIF(solo_motor) AS solo_motor,
  COUNTIF(NOT solo_motor AND NOT solo_reportado AND (ABS(IFNULL(ant_rep, 0) - IFNULL(ant_mot, 0)) > 1e-9 OR ABS(IFNULL(act_rep, 0) - IFNULL(act_mot, 0)) > 1e-9)) AS dif_toneladas_entrada,
  COUNTIF(NOT solo_motor AND NOT solo_reportado AND (ABS(IFNULL(inf_rep, -1e9) - IFNULL(inf_mot, -1e9)) > 1e-9 OR ABS(IFNULL(sup_rep, -1e9) - IFNULL(sup_mot, -1e9)) > 1e-9)) AS dif_banda_ic,
  COUNTIF(NOT solo_motor AND NOT solo_reportado AND ABS(IFNULL(val_rep, 0) - IFNULL(val_mot, 0)) > 1e-9) AS dif_atribuible,
  ROUND(SUM(IFNULL(act_rep, 0)), 3) AS ton_actual_rep, ROUND(SUM(IFNULL(act_mot, 0)), 3) AS ton_actual_mot,
  ROUND(SUM(IFNULL(val_rep, 0)), 3) AS atribuible_rep, ROUND(SUM(IFNULL(val_mot, 0)), 3) AS atribuible_mot,
  ROUND(SUM(IFNULL(val_mot, 0)) - SUM(IFNULL(val_rep, 0)), 3) AS dif_atribuible_t,
  COUNTIF(NOT solo_motor AND NOT solo_reportado AND ABS(IFNULL(ubb_rep, 0) - IFNULL(ubb_mot, 0)) > 1e-6) AS dif_ub_base,
  COUNTIF(NOT solo_motor AND NOT solo_reportado AND ABS(IFNULL(ub_rep, 0) - IFNULL(ub_mot, 0)) > 1e-6) AS dif_ub_atribuible,
  ROUND(SUM(IFNULL(ub_rep, 0)), 2) AS ub_atribuible_rep, ROUND(SUM(IFNULL(ub_mot, 0)), 2) AS ub_atribuible_mot,
  ROUND(SUM(IFNULL(ub_mot, 0)) - SUM(IFNULL(ub_rep, 0)), 2) AS dif_ub_soles
FROM j
GROUP BY des_categoria, mes, retro
ORDER BY des_categoria, mes, retro;

-- @ton_top
-- Las 50 filas (cliente × categoría × mes) con mayor diferencia de atribuible, con sus entradas y bandas de cada lado.
SELECT
  r.cod_cliente, r.des_categoria, FORMAT_DATE('%Y-%m', r.periodo) AS mes, r.flg_estado_yoy AS estado_rep, m.flg_estado_yoy AS estado_mot,
  r.toneladas_ant AS ant_rep, m.toneladas_ant AS ant_mot, r.toneladas_actual AS act_rep, m.toneladas_actual AS act_mot,
  r.ic_pct_inf_80 AS inf_rep, m.ic_pct_inf_80 AS inf_mot, r.ic_pct_sup_80 AS sup_rep, m.ic_pct_sup_80 AS sup_mot,
  r.valor_atribuible AS atribuible_rep, m.valor_atribuible AS atribuible_mot,
  ROUND(IFNULL(m.valor_atribuible, 0) - IFNULL(r.valor_atribuible, 0), 6) AS dif
FROM `{rep}.vw_base_ton_atribuibles_vigente` AS r
JOIN `{mot}.vw_base_ton_atribuibles_vigente` AS m
  ON r.programa = m.programa AND r.cod_cliente = m.cod_cliente AND r.des_categoria = m.des_categoria AND r.periodo = m.periodo
WHERE r.programa = @p AND ABS(IFNULL(m.valor_atribuible, 0) - IFNULL(r.valor_atribuible, 0)) > 1e-9
ORDER BY ABS(IFNULL(m.valor_atribuible, 0) - IFNULL(r.valor_atribuible, 0)) DESC
LIMIT 50;

-- @ub_top
-- Las 50 filas con mayor diferencia de UB atribuible, con lo que la explica: atribuible en toneladas, UB base y UB por tonelada de referencia de cada lado.
SELECT
  r.cod_cliente, r.des_categoria, FORMAT_DATE('%Y-%m', r.periodo) AS mes, r.flg_estado_yoy AS estado_rep, m.flg_estado_yoy AS estado_mot,
  r.valor_atribuible AS atribuible_t_rep, m.valor_atribuible AS atribuible_t_mot,
  r.ub_base_soles AS ub_base_rep, m.ub_base_soles AS ub_base_mot, r.ub_por_ton_ref AS ub_ton_ref_rep, m.ub_por_ton_ref AS ub_ton_ref_mot,
  r.ub_atribuible_soles AS ub_atribuible_rep, m.ub_atribuible_soles AS ub_atribuible_mot,
  ROUND(IFNULL(m.ub_atribuible_soles, 0) - IFNULL(r.ub_atribuible_soles, 0), 6) AS dif_soles
FROM `{rep}.vw_base_ton_atribuibles_vigente` AS r
JOIN `{mot}.vw_base_ton_atribuibles_vigente` AS m
  ON r.programa = m.programa AND r.cod_cliente = m.cod_cliente AND r.des_categoria = m.des_categoria AND r.periodo = m.periodo
WHERE r.programa = @p AND ABS(IFNULL(m.ub_atribuible_soles, 0) - IFNULL(r.ub_atribuible_soles, 0)) > 1e-6
ORDER BY ABS(IFNULL(m.ub_atribuible_soles, 0) - IFNULL(r.ub_atribuible_soles, 0)) DESC
LIMIT 50;

-- @brecha_resumen
-- Brechas por cosecha × categoría × mes relativo: pares que solo están de un lado, pares con DiD distinto, y la brecha % y los kg de cada lado.
WITH r AS (
  SELECT par_id, periodo, cosecha_grupo, des_categoria, mes_rel, did_kg, brecha_pct_celda
  FROM `{rep}.vw_base_brechas_vigente` WHERE programa = @p),
m AS (
  SELECT par_id, periodo, cosecha_grupo, des_categoria, mes_rel, did_kg, brecha_pct_celda
  FROM `{mot}.vw_base_brechas_vigente` WHERE programa = @p)
SELECT
  COALESCE(r.cosecha_grupo, m.cosecha_grupo) AS cosecha_grupo, COALESCE(r.des_categoria, m.des_categoria) AS des_categoria, COALESCE(r.mes_rel, m.mes_rel) AS mes_rel,
  COUNTIF(m.par_id IS NULL) AS solo_reportado, COUNTIF(r.par_id IS NULL) AS solo_motor,
  COUNTIF(r.par_id IS NOT NULL AND m.par_id IS NOT NULL AND ABS(IFNULL(r.did_kg, 0) - IFNULL(m.did_kg, 0)) > 1e-9) AS dif_did,
  ROUND(AVG(r.brecha_pct_celda), 4) AS brecha_pct_rep, ROUND(AVG(m.brecha_pct_celda), 4) AS brecha_pct_mot,
  ROUND(SUM(IFNULL(r.did_kg, 0)) / 1000, 3) AS did_t_rep, ROUND(SUM(IFNULL(m.did_kg, 0)) / 1000, 3) AS did_t_mot
FROM r FULL OUTER JOIN m ON r.par_id = m.par_id AND r.periodo = m.periodo
GROUP BY cosecha_grupo, des_categoria, mes_rel
ORDER BY cosecha_grupo, des_categoria, mes_rel;

-- @espejos_resumen
-- Espejos (el control de cada cliente de acción, mes a mes): clientes con el mismo espejo, con espejo distinto, o presentes de un solo lado.
WITH r AS (
  SELECT cod_cliente_accion, des_categoria, periodo, cod_cliente_espejo FROM `{rep}.vw_base_espejos_vigente` WHERE programa = @p),
m AS (
  SELECT cod_cliente_accion, des_categoria, periodo, cod_cliente_espejo FROM `{mot}.vw_base_espejos_vigente` WHERE programa = @p)
SELECT
  COALESCE(r.des_categoria, m.des_categoria) AS des_categoria,
  COUNTIF(m.cod_cliente_accion IS NULL) AS solo_reportado, COUNTIF(r.cod_cliente_accion IS NULL) AS solo_motor,
  COUNTIF(r.cod_cliente_accion IS NOT NULL AND m.cod_cliente_accion IS NOT NULL AND IFNULL(r.cod_cliente_espejo, '~') = IFNULL(m.cod_cliente_espejo, '~')) AS espejo_igual,
  COUNTIF(r.cod_cliente_accion IS NOT NULL AND m.cod_cliente_accion IS NOT NULL AND IFNULL(r.cod_cliente_espejo, '~') != IFNULL(m.cod_cliente_espejo, '~')) AS espejo_distinto
FROM r FULL OUTER JOIN m
  ON r.cod_cliente_accion = m.cod_cliente_accion AND r.des_categoria = m.des_categoria AND r.periodo = m.periodo
GROUP BY des_categoria
ORDER BY des_categoria;
