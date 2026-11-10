# flake8: noqa
# ruff: noqa
# Script de FAMILIA: se ejecuta con exec() en un espacio de nombres compartido con el script del programa
# (mediciones.ejecucion.ejecutar_familia). No es un módulo importable: usa variables globales definidas por el programa.
# =============================================================================
# EXPORT ESTÁNDAR del motor de cosechas — deja `base_calculo_toneladas` (cliente-mes, acción + Inka Chips) y `base_calculo_brecha` (par-mes)
# con las columnas estándar; el runner los recoge. Las columnas opcionales de la base de brechas (control y datos de la celda) quedan nulas.
# =============================================================================
import numpy as np
import pandas as pd

_COLS_TON = ['cod_cliente', 'cod_cliente_dex', 'des_categoria', 'des_oficina', 'des_segmento_transaccional',
             'nom_compania', 'flg_cliente_perfecto', 'cosecha_grupo', 'periodo', 'mes_rel', 'toneladas_ant',
             'toneladas_actual', 'yoy_pct_raw', 'ic_pct_inf_80', 'ic_pct_sup_80', 'flg_estado_yoy',
             'crec_atrib_80_pct', 'ton_atrib_80', 'flg_estimado_retroactivo', 'valor_atribuible']
_COLS_BRE = ['par_id', 'cod_cliente_accion_dex', 'nom_compania_accion', 'flg_cliente_perfecto', 'cosecha_grupo',
             'cosecha_label', 'des_categoria', 'des_oficina', 'des_segmento_transaccional', 'periodo', 'mes_rel',
             'pre_accion', 'post_accion', 'pre_control', 'post_control', 'did_kg', 'flg_estimado_retroactivo',
             'brecha_pct_celda', 'reportable']
_OPC_BRE = ['cod_cliente_control', 'n_pares_celda', 'ic_pct_inf_80_celda', 'ic_pct_sup_80_celda']      # opcionales (los trae la familia mensual)

_ton = base_calculo_toneladas.copy()
_bre = base_calculo_brecha.copy()
if 'cosecha_label' not in _bre.columns:
    _bre['cosecha_label'] = _bre['cosecha_grupo'].map(GRUPO_COSECHA)
_falt_ton = [c for c in _COLS_TON if c not in _ton.columns]
_falt_bre = [c for c in _COLS_BRE if c not in _bre.columns]
assert not _falt_ton, f'Base toneladas: faltan columnas estándar {_falt_ton}'
assert not _falt_bre, f'Base brecha: faltan columnas estándar {_falt_bre}'
# Inka Chips agrupa 4 categorías reales de venta (Snacks SSCC, Papas Fritas, Plátano Frito, Vegetales) bajo una sola 'Inkachips' SIN agregar:
# queda más de una fila por cliente-mes. La base estándar es cliente×categoría×mes, así que se suman (toneladas y diferencial YoY son aditivos;
# valor_atribuible = toneladas_actual - toneladas_ant, se conserva al sumar). El % de crecimiento no es aditivo: se recalcula.
# Solo aplica a categorías sin matching; en Galletas/Cereales una llave duplicada sigue siendo un error (assert de abajo).
_K_TON = ['cod_cliente', 'des_categoria', 'periodo']
_dup_agr = ~_ton['des_categoria'].isin(CATEGORIAS) & _ton.duplicated(_K_TON, keep=False)
if _dup_agr.any():
    _ADIT = ['toneladas_ant', 'toneladas_actual', 'valor_atribuible']
    _g = _ton[_dup_agr]
    _agr = _g.drop_duplicates(_K_TON).drop(columns=_ADIT).merge(_g.groupby(_K_TON, as_index=False)[_ADIT].sum(), on=_K_TON)
    _agr['crec_atrib_80_pct'] = np.where(_agr['toneladas_ant'] > 0,
                                         (_agr['toneladas_actual'] - _agr['toneladas_ant']) / _agr['toneladas_ant'].where(_agr['toneladas_ant'] > 0) * 100, np.nan)
    print(f"[motor] toneladas: {int(_dup_agr.sum()):,} filas de {sorted(_g['des_categoria'].unique())} agregadas a {len(_agr):,} (cliente x categoría x mes)")
    _ton = pd.concat([_ton[~_dup_agr], _agr[_ton.columns]], ignore_index=True)
assert not _ton.duplicated(['cod_cliente', 'des_categoria', 'periodo']).any(), 'Base toneladas: llaves duplicadas'

for _c in _OPC_BRE:
    if _c not in _bre.columns:
        _bre[_c] = pd.NA
base_calculo_toneladas = _ton[_COLS_TON].reset_index(drop=True)
base_calculo_brecha = _bre[_COLS_BRE + _OPC_BRE].reset_index(drop=True)
# ── Base ESPEJOS (cliente de acción x categoría x mes, con su espejo) ─────────────────────────────
# Mismo grano que la base de toneladas (sin Inka Chips: es un diferencial YoY sin matching ni control). El par es fijo por cosecha y categoría
# (df_matches); se expande a cada mes de la base de toneladas, incluidos los meses retroactivos, y además a los par-mes de la base de brechas que
# no tienen fila de toneladas: la brecha retroactiva es una estimación por celda (cosecha x categoría x mes) que se replica a TODOS los pares de
# la cosecha, incluidos clientes que aún no aparecían en el sell-out esos meses. Esas filas se reportan, así que su par debe tener espejo registrado.
_tn = base_calculo_toneladas[base_calculo_toneladas['des_categoria'].isin(CATEGORIAS)]
_e = (_tn[['cod_cliente', 'des_categoria', 'periodo', 'cosecha_grupo', 'mes_rel', 'des_oficina', 'nom_compania',
           'des_segmento_transaccional', 'flg_estimado_retroactivo']]
      .rename(columns={'cod_cliente': 'cod_cliente_accion'}).copy())
_e['cod_cliente_accion'] = _e['cod_cliente_accion'].astype(str).str.zfill(10)
_ids = ['cod_cliente_accion', 'des_categoria', 'periodo']
_b = (base_calculo_brecha[base_calculo_brecha['des_categoria'].isin(CATEGORIAS)]
      .rename(columns={'cod_cliente_accion_dex': 'cod_cliente_accion', 'nom_compania_accion': 'nom_compania'})
      [_ids + ['cosecha_grupo', 'mes_rel', 'des_oficina', 'nom_compania', 'des_segmento_transaccional', 'flg_estimado_retroactivo']].copy())
_b['cod_cliente_accion'] = _b['cod_cliente_accion'].astype(str).str.zfill(10)
_b = _b.merge(_e[_ids].drop_duplicates(), on=_ids, how='left', indicator=True)
_b = _b[_b['_merge'] == 'left_only'].drop(columns='_merge').drop_duplicates(_ids)
if len(_b):
    print(f"[motor] espejos: {len(_b):,} par-mes de la brecha sin fila de toneladas (estimación retroactiva por celda) se agregan a la base de espejos")
    _e = pd.concat([_e, _b[_e.columns]], ignore_index=True)
_m = df_matches.copy()
_m['par_id'] = [f"{r.cosecha.date()}|{r.des_categoria}|{r.cod_accion}" for r in _m.itertuples()]
_m['cod_cliente_accion'] = _m['cod_accion'].astype(str).str.zfill(10)
_m['cod_cliente_espejo'] = _m['cod_control'].astype(str).str.zfill(10)
_m = _m.rename(columns={'cosecha': 'cosecha_grupo', 'ofi_ct': 'des_oficina_espejo', 'comp_ct': 'nom_compania_espejo', 'seg_ct': 'des_segmento_espejo'})
_cm = ['cosecha_grupo', 'des_categoria', 'cod_cliente_accion', 'cod_cliente_espejo', 'par_id', 'des_oficina_espejo',
       'nom_compania_espejo', 'des_segmento_espejo'] + [c for c in ('nivel_match',) if c in _m.columns]
_m = _m[_cm].drop_duplicates(['cosecha_grupo', 'des_categoria', 'cod_cliente_accion'])
_e = _e.merge(_m, on=['cosecha_grupo', 'des_categoria', 'cod_cliente_accion'], how='left')
_sp = set(zip(df_sin_pareja['cosecha'], df_sin_pareja['des_categoria'], df_sin_pareja['cod_accion'].astype(str).str.zfill(10)))
_e['motivo_sin_espejo'] = [None if pd.notna(ce) else ('sin_pareja' if (cg, ct, ca) in _sp else 'sin_match_registrado')
                           for ce, cg, ct, ca in zip(_e['cod_cliente_espejo'], _e['cosecha_grupo'], _e['des_categoria'], _e['cod_cliente_accion'])]
if 'nivel_match' not in _e.columns:
    _e['nivel_match'] = np.nan
_cols_esp = ['cod_cliente_accion', 'des_categoria', 'periodo', 'cosecha_grupo', 'mes_rel', 'cod_cliente_espejo', 'motivo_sin_espejo', 'par_id',
             'des_oficina', 'nom_compania', 'des_segmento_transaccional', 'des_oficina_espejo', 'nom_compania_espejo', 'des_segmento_espejo',
             'nivel_match', 'flg_estimado_retroactivo']
base_calculo_espejos = _e[_cols_esp].sort_values(['des_categoria', 'cosecha_grupo', 'periodo', 'cod_cliente_accion']).reset_index(drop=True)
assert not base_calculo_espejos.duplicated(['cod_cliente_accion', 'des_categoria', 'periodo']).any(), 'Base espejos: llaves duplicadas'
assert len(df_matches) == 0 or base_calculo_espejos['cod_cliente_espejo'].notna().any(), \
    'Ningún cliente de acción quedó con espejo: revisar el cruce de df_matches (cosecha / ids) con la base de toneladas'

print(f"[motor] bases estándar: {len(base_calculo_toneladas):,} filas de toneladas | {len(base_calculo_brecha):,} filas de brecha")

# ── Calendario (surtido): en etapa preliminar el G04 se ancla en su primer mes de altas, pero en las salidas se rotula con su cosecha definitiva,
#    así conserva la misma identidad al pasar a definitivo (lo provisional se reemplaza por cosecha: ver mediciones/frontera.py). `cosecha_label` dice "preliminar".
_ALIAS = {pd.Timestamp(_k): pd.Timestamp(_v) for _k, _v in (globals().get('ALIAS_COSECHA') or {}).items()}
if _ALIAS:
    for _df in (base_calculo_toneladas, base_calculo_brecha, base_calculo_espejos):
        _df['cosecha_grupo'] = pd.to_datetime(_df['cosecha_grupo'], errors='coerce').replace(_ALIAS)
    print(f"[motor] calendario: cosechas preliminares rotuladas con su cosecha definitiva { {k.strftime('%Y-%m'): v.strftime('%Y-%m') for k, v in _ALIAS.items()} }")
