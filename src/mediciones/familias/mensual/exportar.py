# flake8: noqa
# ruff: noqa
# Script de FAMILIA: se ejecuta con exec() en un espacio de nombres compartido con el script del programa
# (mediciones.ejecucion.ejecutar_familia). No es un módulo importable: usa variables globales definidas por el programa.
# =============================================================================
# EXPORT ESTÁNDAR — arma las 2 bases (cliente-mes y par-mes) con el mismo esquema que surtido y Champion
# Deja `base_calculo_toneladas` (cliente-mes, solo clientes de acción) y `base_calculo_brecha` (par-mes) en el espacio de nombres;
# el runner los recoge. Los controles viven en la base de brechas (cod_cliente_control + pre/post_control).
# =============================================================================
import numpy as np
import pandas as pd

_K = ['cod_cliente', 'des_categoria', 'periodo']

# ── Base TONELADAS (cliente-mes): solo clientes de acción ─────────────────────
_acc = df_base_completa[df_base_completa['rol'].astype(str).str.startswith('accion')].copy()
_acc['cod_cliente'] = _acc['cod_cliente'].astype(str).str.zfill(10)
_acc['periodo'] = pd.to_datetime(_acc['periodo'])
_acc['cod_cliente_dex'] = _acc['cod_cliente']
_acc['flg_estimado_retroactivo'] = False
for _c in ('cosecha_grupo', 'mes_rel', 'flg_cliente_perfecto'):      # CPB y CPM no tienen cosechas
    if _c not in _acc.columns:
        _acc[_c] = np.nan
_cols_ton = ['cod_cliente', 'cod_cliente_dex', 'des_categoria', 'des_oficina', 'des_segmento_transaccional',
             'nom_compania', 'flg_cliente_perfecto', 'cosecha_grupo', 'periodo', 'mes_rel', 'toneladas_ant',
             'toneladas_actual', 'yoy_pct_raw', 'ic_pct_inf_80', 'ic_pct_sup_80', 'flg_estado_yoy',
             'crec_atrib_80_pct', 'ton_atrib_80', 'flg_estimado_retroactivo', 'valor_atribuible']
base_calculo_toneladas = _acc[_cols_ton].sort_values(_K).reset_index(drop=True)

# ── Base BRECHA (par-mes) ─────────────────────────────────────────────────────
_pan = df_panel.copy()
_pan['cod_cliente_accion_dex'] = _pan['cod_accion'].astype(str).str.zfill(10)
_pan['periodo'] = pd.to_datetime(_pan['periodo'])
_kp = ['cod_cliente_accion_dex', 'des_categoria', 'periodo']
_ref_attr = (_acc.drop_duplicates(_K)[_K + ['nom_compania', 'des_oficina', 'des_segmento_transaccional']]
             .rename(columns={'cod_cliente': 'cod_cliente_accion_dex', 'nom_compania': 'nom_compania_accion'}))
_ref_attr = _ref_attr.drop(columns=[c for c in _ref_attr.columns if c in _pan.columns and c not in _kp])
_pan = _pan.merge(_ref_attr, on=_kp, how='left')
_pan['cod_cliente_control'] = _pan['cod_control'].astype(str).str.zfill(10) if 'cod_control' in _pan.columns else np.nan
# Datos de la celda (mes x categoría): mismo cálculo que la base de cálculo de los notebooks CPB/CPM — la brecha % y el IC80 del
# notebook cuando la celda tiene >= N_MIN_PARES; si no, la brecha implícita del panel (100 x Σ did_kg / Σ pre_accion).
_cel = (_pan.groupby(['periodo_str', 'des_categoria'])
        .apply(lambda g: pd.Series({'n_pares_celda': len(g), 'brecha_pct_celda': 100 * g['did_kg'].sum() / g['pre_accion'].sum()}))
        .reset_index())
_bn = (df_brechas[['periodo_str', 'des_categoria', 'brecha_pct', 'ic_pct_inf_80', 'ic_pct_sup_80']]
       .rename(columns={'brecha_pct': '_bp_nb', 'ic_pct_inf_80': 'ic_pct_inf_80_celda', 'ic_pct_sup_80': 'ic_pct_sup_80_celda'}))
_cel = _cel.merge(_bn, on=['periodo_str', 'des_categoria'], how='left')
_cel['brecha_pct_celda'] = _cel['_bp_nb'].fillna(_cel['brecha_pct_celda'].round(2))
_cel['n_pares_celda'] = _cel['n_pares_celda'].astype('Int64')
_pan = _pan.merge(_cel.drop(columns='_bp_nb'), on=['periodo_str', 'des_categoria'], how='left')
for _c, _v in (('flg_cliente_perfecto', np.nan), ('cosecha_grupo', np.nan), ('cosecha_label', None),
               ('mes_rel', np.nan), ('flg_estimado_retroactivo', False)):
    _pan[_c] = _v
_cols_bre = ['par_id', 'cod_cliente_accion_dex', 'nom_compania_accion', 'flg_cliente_perfecto', 'cosecha_grupo',
             'cosecha_label', 'des_categoria', 'des_oficina', 'des_segmento_transaccional', 'periodo', 'mes_rel',
             'pre_accion', 'post_accion', 'pre_control', 'post_control', 'did_kg', 'flg_estimado_retroactivo',
             'brecha_pct_celda', 'reportable', 'cod_cliente_control', 'n_pares_celda', 'ic_pct_inf_80_celda', 'ic_pct_sup_80_celda']
base_calculo_brecha = _pan[_cols_bre].sort_values(['des_categoria', 'periodo', 'par_id']).reset_index(drop=True)

# ── Base ESPEJOS (cliente de acción x categoría x mes, con su espejo) ─────────────────────────────
# Mismo grano que la base de toneladas: cada fila de acción (matcheada, sin pareja o nueva) con el control espejo de ese mes y categoría,
# o sin espejo y el motivo. Los atributos del espejo salen de los pares (ofi_ct, comp_ct, seg_ct).
_m = df_matches.copy()
_m['cod_cliente_accion'] = _m['cod_accion'].astype(str).str.zfill(10)
_m['cod_cliente_espejo'] = _m['cod_control'].astype(str).str.zfill(10)
_m['periodo'] = pd.to_datetime(_m['periodo'])
_cm = ['cod_cliente_accion', 'des_categoria', 'periodo', 'cod_cliente_espejo'] + [c for c in ('nivel_match', 'ofi_ct', 'comp_ct', 'seg_ct') if c in _m.columns]
_m = _m[_cm].drop_duplicates(['cod_cliente_accion', 'des_categoria', 'periodo'])
_ce = ['cod_cliente', 'des_categoria', 'periodo', 'des_oficina', 'nom_compania', 'des_segmento_transaccional', 'rol'] + \
      [c for c in ('motivo_sin_pareja', 'par_id') if c in _acc.columns]
_e = _acc[_ce].rename(columns={'cod_cliente': 'cod_cliente_accion'}).merge(_m, on=['cod_cliente_accion', 'des_categoria', 'periodo'], how='left')
_e = _e.rename(columns={'ofi_ct': 'des_oficina_espejo', 'comp_ct': 'nom_compania_espejo', 'seg_ct': 'des_segmento_espejo'})
_motivo = _e['motivo_sin_pareja'] if 'motivo_sin_pareja' in _e.columns else pd.Series('sin_pareja', index=_e.index)
_e['motivo_sin_espejo'] = np.where(_e['cod_cliente_espejo'].notna(), None, np.where(_e['rol'] == 'accion_nuevo', 'cliente_nuevo', _motivo))
for _c, _v in (('cosecha_grupo', np.nan), ('mes_rel', np.nan), ('nivel_match', np.nan), ('flg_estimado_retroactivo', False)):
    if _c not in _e.columns:
        _e[_c] = _v
_cols_esp = ['cod_cliente_accion', 'des_categoria', 'periodo', 'cosecha_grupo', 'mes_rel', 'cod_cliente_espejo', 'motivo_sin_espejo', 'par_id',
             'des_oficina', 'nom_compania', 'des_segmento_transaccional', 'des_oficina_espejo', 'nom_compania_espejo', 'des_segmento_espejo',
             'nivel_match', 'flg_estimado_retroactivo']
for _c in _cols_esp:
    if _c not in _e.columns:
        _e[_c] = np.nan
base_calculo_espejos = _e[_cols_esp].sort_values(['des_categoria', 'periodo', 'cod_cliente_accion']).reset_index(drop=True)
_falta = (_e['rol'] == 'accion_matcheado') & _e['cod_cliente_espejo'].isna()
assert not _falta.any(), f'{int(_falta.sum())} clientes matcheados sin espejo en los pares: revisar df_matches'
assert not base_calculo_espejos.duplicated(['cod_cliente_accion', 'des_categoria', 'periodo']).any(), 'Base espejos: llaves duplicadas'

# ── Chequeos de cruce (si fallan, los ids no coinciden entre el panel y la base) ──
_sin_ofi = float(base_calculo_brecha['des_oficina'].isna().mean())
print(f"[motor] pares sin oficina tras el cruce: {100 * _sin_ofi:.1f}%")
assert _sin_ofi < 0.05, 'El cruce panel-base falló (ids distintos): revisar cod_accion vs cod_cliente'
assert not base_calculo_toneladas.duplicated(_K).any(), 'Base toneladas: llaves duplicadas'
