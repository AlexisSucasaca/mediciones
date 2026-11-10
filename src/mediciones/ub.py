"""
UB atribuible, con dos métodos (`metodo_ub` de cada programa en config/programas.yaml). Sin tope en ninguno.

  'actual'    (surtido, Champion)  UB actual x (valor_atribuible / toneladas_actual): la UB por tonelada del mes ACTUAL.
  'notebook'  (CPB, CPM)           como el notebook: la UB por tonelada del MISMO MES DEL AÑO ANTERIOR (UB base).
                                     antiguo con compra : valor_atribuible x (ub_base / toneladas_ant)
                                     antiguo sin compra : 0
                                     nuevo              : toneladas_actual x ub_por_ton_ref, con ub_por_ton_ref = sum(ub_base) / sum(toneladas_ant)
                                                          de los antiguos de acción de la celda mes x categoría
                                   El notebook calcula ub_base con la misma venta y los mismos ratios que `sql_ub`, pero en los meses de hace un año:
                                   `consulta_ub(..., meses_atras=12)` los trae y aquí se desplazan 1 año.
Surtido y Champion no tienen UB en sus notebooks: no hay contra qué comparar y se mantiene el método actual del motor.
"""
import numpy as np
import pandas as pd

from .esquema import COLS_TON, COLS_UB

TABLA_DETALLE = 'acpe-prod-dlk-shrd-gld.gld_cliente.dex_cliente_detalle'
TABLA_RATIOS = 'acpe-dev-uc-sandbox-aa.dev.ratios_uv_bain_2025'


def sql_ub(con_negocios):
    """Venta bruta del mes por cliente x categoría y su UB (venta x ratio_ub a nivel cliente SAP x familia)."""
    filtro = 'AND v.cod_negocio IN UNNEST(@negocios)' if con_negocios else ''
    return f"""
WITH ratios_ub AS (
  SELECT cod_cliente AS cod_cliente_sap, cod_familia,
         mnt_utilidad_bruta / NULLIF(mnt_venta_bruta, 0) AS ratio_ub
  FROM `{TABLA_RATIOS}`
),
ventas AS (
  SELECT v.periodo,
         LPAD(TRIM(v.id_interlocutor_origen), 10, '0') AS cod_cliente,
         v.des_categoria, v.cod_cliente_sap, v.cod_familia,
         SUM(v.mnt_bruto) AS venta_soles,
         SUM(v.num_peso_neto) / 1000 AS ton_detalle
  FROM `{TABLA_DETALLE}` v
  WHERE v.periodo BETWEEN @desde AND @hasta
    AND v.des_categoria IN UNNEST(@categorias)
    AND LPAD(TRIM(v.id_interlocutor_origen), 10, '0') IN UNNEST(@clientes)
    {filtro}
  GROUP BY 1, 2, 3, 4, 5
)
SELECT v.periodo, v.cod_cliente, v.des_categoria,
       SUM(v.venta_soles)                                          AS venta_actual_soles,
       SUM(IF(r.ratio_ub IS NULL, v.venta_soles, 0))               AS venta_sin_ratio_soles,
       SUM(v.venta_soles * r.ratio_ub)                             AS ub_actual_soles,
       SUM(v.ton_detalle)                                          AS ton_detalle
FROM ventas v
LEFT JOIN ratios_ub r
  ON v.cod_cliente_sap = r.cod_cliente_sap AND v.cod_familia = r.cod_familia
GROUP BY 1, 2, 3
"""


def consulta_ub(cli, ton, negocios=None, meses_atras=0):
    """Trae la venta y la UB de los clientes/categorías/periodos de `ton`; con `meses_atras` también los meses anteriores (la UB base del método notebook: 12)."""
    from google.cloud import bigquery
    ids = ton['cod_cliente'].astype(str).str.zfill(10).unique().tolist()   # texto: hay ids con letras (ej. 000C000993)
    params = [
        bigquery.ScalarQueryParameter('desde', 'DATE', (pd.Timestamp(min(ton['periodo'])) - pd.DateOffset(months=meses_atras)).date()),
        bigquery.ScalarQueryParameter('hasta', 'DATE', max(ton['periodo'])),
        bigquery.ArrayQueryParameter('categorias', 'STRING', sorted(ton['des_categoria'].dropna().unique().tolist())),
        bigquery.ArrayQueryParameter('clientes', 'STRING', ids),
    ]
    if negocios:
        params.append(bigquery.ArrayQueryParameter('negocios', 'STRING', list(negocios)))
    df = cli.query(sql_ub(bool(negocios)), job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
    df['periodo'] = pd.to_datetime(df['periodo']).dt.date
    return df


def _ub_notebook(d, u, con_compra):
    """Método del notebook de CPB/CPM (misma fórmula que su celda de UB; probado contra una copia literal en tests/legado/ub_notebook.py)."""
    base = u[['_k', 'des_categoria', 'periodo', 'ub_actual_soles']].rename(columns={'ub_actual_soles': 'ub_base_soles'}).copy()
    base['_pk'] = (pd.to_datetime(base['periodo']) + pd.DateOffset(years=1)).dt.date            # la UB de hace un año se atribuye al mes que corresponde
    d['_pk'] = pd.to_datetime(d['periodo']).dt.date
    d = d.merge(base[['_k', 'des_categoria', '_pk', 'ub_base_soles']], on=['_k', 'des_categoria', '_pk'], how='left')
    d['ub_base_soles'] = d['ub_base_soles'].fillna(0.0)
    estados = set(d['flg_estado_yoy'].dropna().unique())
    if estados & {'antiguo', 'nuevo'}:                                   # familia mensual (CPB/CPM): sus estados tal cual
        ant_m, nue_m = d['flg_estado_yoy'] == 'antiguo', d['flg_estado_yoy'] == 'nuevo'
        ton_nuevo = d['toneladas_actual']                                 # a un 'nuevo' se le atribuye todo lo que compra: valor_atribuible == toneladas_actual
    else:
        # cosechas (surtido/Champion) usan otros estados (normal, reactivado, no_existia_periodo_pasado, sin_compra_ambos, ...):
        # mismo criterio de fondo que 'antiguo'/'nuevo' en CPB/CPM: si el cliente tenía toneladas el año anterior.
        ton_ant = d['toneladas_ant'].fillna(0.0)
        ant_m = ton_ant > 0
        nue_m = ton_ant <= 0
        # Sin año anterior, la UB se atribuye sobre las toneladas ATRIBUIBLES, no sobre las actuales: en cosechas (regla 'cliente_creado') solo
        # 'no_existia_periodo_pasado' recibe todo lo que compra; un 'reactivado' recibe ton_atrib (0 en setiembre). Con toneladas_actual, la UB de
        # los reactivados quedaba inflada (más de la mitad de la UB atribuible de surtido y Champion) con 0 toneladas atribuidas.
        ton_nuevo = d['valor_atribuible'].fillna(0.0)
    ref = (d[ant_m & (d['toneladas_ant'] > 0)].groupby(['_pk', 'des_categoria']).agg(_u=('ub_base_soles', 'sum'), _t=('toneladas_ant', 'sum')))
    ref['ub_por_ton_ref'] = ref['_u'] / ref['_t']
    d = d.merge(ref[['ub_por_ton_ref']].reset_index(), on=['_pk', 'des_categoria'], how='left')
    ub_por_ton = np.where(d['toneladas_ant'] > 0, d['ub_base_soles'] / d['toneladas_ant'], np.nan)       # S/ por TN base del cliente
    d['ub_atribuible_soles'] = np.select([ant_m & con_compra, ant_m, nue_m],
                                         [d['valor_atribuible'] * ub_por_ton, 0.0, ton_nuevo * d['ub_por_ton_ref'].fillna(0.0)], default=np.nan)
    return d.drop(columns=['_pk'])


def aplicar_ub(ton, ub, metodo='actual'):
    """
    Llena las columnas de UB de la base de toneladas (sin tope). `metodo`: 'actual' o 'notebook' (ver el docstring del módulo). Método 'actual':
      ub_atribuible = ub_actual x (valor_atribuible / toneladas_actual)   si hubo compra en el mes
                    = 0                                                   si no hubo compra y valor_atribuible == 0
      nuevos: valor_atribuible == toneladas_actual -> ub_atribuible = ub_actual completa.
    Devuelve (base con UB, cobertura). La cobertura compara las toneladas del detalle DEX contra
    las del notebook: si difieren mucho, la UB prorrateada deja de ser comparable.
    """
    if metodo not in ('actual', 'notebook'):
        raise ValueError(f"metodo_ub debe ser 'actual' o 'notebook' (recibí {metodo!r})")
    d = ton.drop(columns=[c for c in COLS_UB if c in ton.columns]).copy()
    d['_k'] = d['cod_cliente'].astype(str).str.zfill(10)
    u = ub.rename(columns={'cod_cliente': '_k'}).copy()
    u['_k'] = u['_k'].astype(str).str.zfill(10)
    # BigQuery devuelve NUMERIC/SUM como decimal.Decimal (dtype object): rompe el producto con float.
    for c in ['venta_actual_soles', 'venta_sin_ratio_soles', 'ub_actual_soles', 'ton_detalle']:
        u[c] = pd.to_numeric(u[c], errors='coerce').astype(float)
    d = d.merge(u[['_k', 'des_categoria', 'periodo', 'venta_actual_soles', 'venta_sin_ratio_soles',
                   'ub_actual_soles', 'ton_detalle']], on=['_k', 'des_categoria', 'periodo'], how='left')

    act, val = d['toneladas_actual'], d['valor_atribuible']
    con_compra = act > 0
    if metodo == 'actual':
        ratio = (val / act).where(con_compra)
        d['ub_atribuible_soles'] = d['ub_actual_soles'] * ratio
        d.loc[~con_compra & (val == 0), 'ub_atribuible_soles'] = 0.0
        d['ub_base_soles'] = np.nan
        d['ub_por_ton_ref'] = np.nan
    else:
        d = _ub_notebook(d, u, con_compra)

    m = con_compra & d['ton_detalle'].notna()
    cobertura = {
        'filas_con_compra': int(con_compra.sum()),
        'filas_con_ub': int((con_compra & d['ub_actual_soles'].notna()).sum()),
        'pct_cobertura_ub': round(100 * float((con_compra & d['ub_actual_soles'].notna()).sum()) / max(int(con_compra.sum()), 1), 1),
        'pct_venta_sin_ratio': round(100 * float(d['venta_sin_ratio_soles'].sum()) / max(float(d['venta_actual_soles'].sum()), 1e-9), 2),
        'ton_detalle_vs_base': round(float(d.loc[m, 'ton_detalle'].sum()) / max(float(act[m].sum()), 1e-9), 3),
        'ub_atribuible_soles_total': round(float(d['ub_atribuible_soles'].sum()), 0),
        'filas_ub_atrib_mayor_a_ub_actual (informativo)': int((d['ub_atribuible_soles'] > d['ub_actual_soles'] + 1e-9).sum()),
        # coherencia: sin toneladas atribuibles no puede haber UB atribuible (debe ser 0)
        'filas_ub_con_atribuible_cero': int(((d['valor_atribuible'].abs() < 1e-12) & (d['ub_atribuible_soles'].abs() > 1e-9)).sum()),
    }
    d = d.drop(columns=['_k', 'ton_detalle'])
    return d[COLS_TON], cobertura
