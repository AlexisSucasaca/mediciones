"""Paridad: compara la base de toneladas del motor contra un resultado de referencia."""
import pandas as pd

from .normalizacion import asignar_estado_yoy


def paridad_con_legacy(ton, legacy, tol=1e-6, regla_estado='estandar'):
    """
    Compara la base de toneladas del motor contra la 'Base completa' (hoja) que dejó el notebook anterior,
    solo filas de acción. Sirve para validar que unificar la metodología no cambió los números.
    `legacy` = DataFrame de esa hoja (cpb_v2_base_completa.xlsx / cpm_v2_base_completa.xlsx).
    `regla_estado`: la misma que usó el motor (`estado_yoy` de la familia). Con 'programa' la referencia se compara TAL CUAL la reportó el notebook
    (sin reaplicar ninguna regla); con 'estandar' se lleva primero a la regla estándar (nuevo = toneladas_actual).
    """
    k = ['cod_cliente', 'des_categoria', 'periodo']
    if 'valor_atribuible' not in legacy.columns and 'ton_atribuible' in legacy.columns:       # nombre en la base de cálculo de CPB/CPM
        legacy = legacy.rename(columns={'ton_atribuible': 'valor_atribuible'})
    lg = (legacy[legacy['rol'].astype(str).str.startswith('accion')].copy() if 'rol' in legacy.columns
          else legacy.copy())   # base estándar (sin rol) o 'Base completa' del notebook anterior
    lg['cod_cliente'] = lg['cod_cliente'].astype(str).str.zfill(10)
    lg['periodo'] = pd.to_datetime(lg['periodo']).dt.date
    # Llave repetida en la referencia (p. ej. Inka Chips: 4 categorías reales bajo una sola etiqueta, sin agregar en el notebook). El motor las
    # suma al grano cliente x categoría x mes (familias/cosechas/exportar.py); se hace lo mismo aquí ANTES de aplicar la regla de negocio.
    n_agregadas = int(lg.duplicated(k).sum())
    if n_agregadas:
        adit = [c for c in ('toneladas_ant', 'toneladas_actual', 'valor_atribuible') if c in lg.columns]
        lg = lg.groupby(k, as_index=False)[adit].sum(min_count=1)
    # Con la regla estándar la referencia se lleva a la MISMA regla (nuevo = toneladas_actual) para medir solo diferencias del motor.
    # Con la regla del programa NO se toca: debe coincidir con lo ya reportado (p. ej. reactivados = 0).
    if regla_estado == 'estandar' and 'toneladas_ant' in lg.columns:
        _nuevo = asignar_estado_yoy(lg['toneladas_ant'], lg['toneladas_actual']).eq('nuevo')
        lg.loc[_nuevo, 'valor_atribuible'] = lg.loc[_nuevo, 'toneladas_actual']
    mo = ton[k + ['toneladas_actual', 'valor_atribuible']].copy()
    mo['cod_cliente'] = mo['cod_cliente'].astype(str).str.zfill(10)
    m = mo.merge(lg[k + ['toneladas_actual', 'valor_atribuible']], on=k, how='outer',
                 suffixes=('_motor', '_legacy'), indicator=True)
    both = m[m['_merge'] == 'both']
    d_ton = (both['toneladas_actual_motor'] - both['toneladas_actual_legacy']).abs()
    d_val = (both['valor_atribuible_motor'].astype(float) - both['valor_atribuible_legacy'].astype(float)).abs()
    d_val = d_val.where(~(both['valor_atribuible_motor'].isna() & both['valor_atribuible_legacy'].isna()), 0.0)
    res = {'filas_motor': len(mo), 'filas_legacy': len(lg), 'filas_legacy_agregadas': n_agregadas,
           'solo_en_motor': int((m['_merge'] == 'left_only').sum()),
           'solo_en_legacy': int((m['_merge'] == 'right_only').sum()),
           'max_dif_toneladas_actual': float(d_ton.max()) if len(both) else None,
           'max_dif_valor_atribuible': float(d_val.max(skipna=True)) if len(both) else None,
           'filas_con_dif_valor_atribuible': int((d_val > tol).sum()),
           'suma_valor_atribuible_motor': round(float(mo['valor_atribuible'].sum()), 3),
           'suma_valor_atribuible_legacy': round(float(lg['valor_atribuible'].sum()), 3)}
    res['coincide'] = bool(res['solo_en_motor'] == 0 and res['solo_en_legacy'] == 0 and res['filas_con_dif_valor_atribuible'] == 0
                           and (res['max_dif_toneladas_actual'] or 0) <= tol)
    return res
