"""Esquema único de las 2 bases de negocio y sus llaves."""

PROYECTO = 'acpe-dev-uc-sandbox-aa'
DATASET = 'mediciones'
T_TON, T_BRE, T_LOG = 'base_ton_atribuibles', 'base_brechas', 'log_metricas_ejecucion'
# Columnas de la tabla de log (las de bigquery.fila_log; el DDL tiene las mismas)
COLS_LOG = ['corrida_id', 'programa', 'periodo_reporte', 'etapa', 'cosecha_grupo', 'des_categoria', 'des_oficina', 'metrica', 'valor', 'valor_txt', 'fecha_registro']
COLS_UB = ['venta_actual_soles', 'venta_sin_ratio_soles', 'ub_actual_soles', 'ub_base_soles', 'ub_por_ton_ref', 'ub_atribuible_soles']
COLS_TEC = ['periodo_reporte', 'corrida_id', 'version', 'config_hash', 'fecha_carga']
COLS_TON = (['programa', 'cod_cliente', 'cod_cliente_dex', 'des_categoria', 'des_oficina',
             'des_segmento_transaccional', 'nom_compania', 'flg_cliente_perfecto', 'cosecha_grupo',
             'periodo', 'mes_rel', 'toneladas_ant', 'toneladas_actual', 'yoy_pct_raw', 'ic_pct_inf_80',
             'ic_pct_sup_80', 'flg_estado_yoy', 'crec_atrib_80_pct', 'ton_atrib_80',
             'flg_estimado_retroactivo', 'kg_atribuibles', 'valor_atribuible'] + COLS_UB + COLS_TEC)
COLS_BRE = (['programa', 'par_id', 'cod_cliente_accion_dex', 'nom_compania_accion', 'flg_cliente_perfecto',
             'cosecha_grupo', 'cosecha_label', 'des_categoria', 'des_oficina', 'des_segmento_transaccional',
             'periodo', 'mes_rel', 'pre_accion', 'post_accion', 'pre_control', 'post_control', 'did_kg',
             'flg_estimado_retroactivo', 'brecha_pct_celda', 'reportable',
             # opcionales: las exporta la familia mensual (CPB/CPM); en cosechas quedan nulas
             'cod_cliente_control', 'n_pares_celda', 'ic_pct_inf_80_celda', 'ic_pct_sup_80_celda'] + COLS_TEC)
T_ESP = 'base_espejos'
COLS_ESP = (['programa', 'cod_cliente_accion', 'des_categoria', 'periodo', 'cosecha_grupo', 'mes_rel',
             'cod_cliente_espejo', 'estado_espejo', 'motivo_sin_espejo', 'par_id',
             'des_oficina', 'nom_compania', 'des_segmento_transaccional',                  # del cliente de acción
             'des_oficina_espejo', 'nom_compania_espejo', 'des_segmento_espejo',           # del espejo
             'nivel_match', 'flg_estimado_retroactivo'] + COLS_TEC)
CLAVE = {T_TON: ['cod_cliente', 'des_categoria', 'periodo'], T_BRE: ['par_id', 'periodo'],
         T_ESP: ['cod_cliente_accion', 'des_categoria', 'periodo']}
VALOR = {T_TON: 'valor_atribuible', T_BRE: 'did_kg', T_ESP: 'cod_cliente_espejo'}   # valor que se vigila para detectar deriva
