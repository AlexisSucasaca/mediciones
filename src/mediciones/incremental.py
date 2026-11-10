"""Incremental: solo llaves nuevas; lo ya publicado no se pisa (deriva = aviso); reproceso explícito por bloque."""
import numpy as np
import pandas as pd

from .esquema import CLAVE, VALOR


def comparar_con_publicado(df, publicado, tabla, tol=1e-9):
    """
    Devuelve (filas_nuevas, resumen, deriva).
      filas_nuevas : llaves que no existen en lo publicado (esto es lo único que se carga).
      deriva       : llaves ya publicadas cuyo valor recalculado cambió (se REPORTA, no se aplica).
    """
    clave, vcol = CLAVE[tabla], VALOR[tabla]
    if publicado is None or len(publicado) == 0:
        return df, {'filas_nuevas': len(df), 'filas_ya_publicadas': 0, 'filas_con_deriva': 0,
                    'filas_publicadas_ausentes_hoy': 0, 'suma_abs_deriva': 0.0}, pd.DataFrame()
    pub = publicado[clave + [vcol]].rename(columns={vcol: '_publicado'})
    m = df.merge(pub, on=clave, how='left', indicator=True)
    nuevas = m[m['_merge'] == 'left_only'].drop(columns=['_merge', '_publicado'])
    comunes = m[m['_merge'] == 'both']
    if pd.api.types.is_numeric_dtype(df[vcol]):
        a, b = comunes[vcol].astype(float).to_numpy(), comunes['_publicado'].astype(float).to_numpy()
        cambio = ~np.isclose(a, b, rtol=0, atol=tol, equal_nan=True)
        suma_deriva = float(np.nansum(np.abs(a[cambio] - b[cambio])))
    else:                                                   # valores de texto (p. ej. el id del espejo): cambia si es distinto
        a, b = comunes[vcol].reset_index(drop=True), comunes['_publicado'].reset_index(drop=True)
        cambio = ~((a.isna() & b.isna()) | (a.notna() & b.notna() & (a == b))).to_numpy()
        suma_deriva = float(cambio.sum())
    deriva = comunes.loc[cambio, clave + [vcol, '_publicado']].rename(columns={vcol: 'recalculado', '_publicado': 'publicado'})
    ausentes = pub.merge(df[clave], on=clave, how='left', indicator=True)
    resumen = {'filas_nuevas': len(nuevas), 'filas_ya_publicadas': len(comunes),
               'filas_con_deriva': int(cambio.sum()),
               'filas_publicadas_ausentes_hoy': int((ausentes['_merge'] == 'left_only').sum()),
               'suma_abs_deriva': suma_deriva}
    return nuevas, resumen, deriva


def seleccionar_para_reproceso(df, publicado, tabla, periodo_reporte):
    """
    Reproceso explícito de un cierre: se republica el bloque COMPLETO de ese periodo_reporte
    (llaves ya publicadas en él, recalculadas) más las llaves nuevas. Los bloques de otros
    cierres no se tocan. Se carga con version + 1 y la vista vigente toma esa versión.
    """
    clave = CLAVE[tabla]
    if publicado is None or len(publicado) == 0:
        return df
    k_bloque = publicado.loc[publicado['periodo_reporte'] == periodo_reporte, clave]
    nuevas, _, _ = comparar_con_publicado(df, publicado, tabla)
    en_bloque = df.merge(k_bloque, on=clave, how='inner')
    return pd.concat([en_bloque, nuevas], ignore_index=True)


TOLERANCIA_CELDA = {'brecha_pp': 0.25, 'ton_pct': 1.0, 'ton_abs': 0.001}   # 0.25 pp en brechas; 1 % (y al menos 1 kg) en toneladas


def deriva_por_celda(tabla, recalculado, publicado, tol=None):
    """
    Deriva de los meses cerrados al nivel en que se reporta: mes x categoría (x cosecha en brechas). Es la vara de la validación de setiembre.
      toneladas: suma de valor_atribuible de la celda, publicada vs recalculada (sin las altas tardías: llaves que no estaban publicadas);
                 fuera de tolerancia si cambia >= ton_pct % Y >= ton_abs t.
      brechas  : brecha_pct_celda publicada vs recalculada; fuera si cambia >= brecha_pp puntos.
    Devuelve una fila por celda con publicado, recalculado, diferencia y `fuera` (bool). Vacío si no aplica.
    """
    tol = {**TOLERANCIA_CELDA, **(tol or {})}
    vacio = pd.DataFrame(columns=['periodo', 'des_categoria', 'cosecha_grupo', 'publicado', 'recalculado', 'dif', 'dif_pct', 'fuera'])
    if recalculado is None or publicado is None or len(recalculado) == 0 or len(publicado) == 0:
        return vacio

    def _n(d):
        d = d.copy()
        d['periodo'] = pd.to_datetime(d['periodo']).dt.to_period('M').dt.to_timestamp()
        d['cosecha_grupo'] = (pd.to_datetime(d['cosecha_grupo'], errors='coerce').dt.strftime('%Y-%m').fillna('—')
                              if 'cosecha_grupo' in d.columns else '—')
        return d

    a, b = _n(recalculado), _n(publicado)
    if tabla == 'base_ton_atribuibles':
        k = CLAVE[tabla]
        comunes = a.merge(b[k].drop_duplicates(), on=k)                         # las altas tardías no cuentan (no se publican)
        g = ['periodo', 'des_categoria']
        cel = pd.concat({'publicado': b.groupby(g)['valor_atribuible'].sum(),
                         'recalculado': comunes.groupby(g)['valor_atribuible'].sum()}, axis=1).fillna(0.0).reset_index()
        cel['cosecha_grupo'] = '—'
        cel['dif'] = cel['recalculado'] - cel['publicado']
        cel['dif_pct'] = np.where(cel['publicado'].abs() > 1e-12, 100 * cel['dif'] / cel['publicado'].abs(), np.nan)
        cel['fuera'] = (cel['dif'].abs() >= tol['ton_abs']) & ~(cel['dif_pct'].abs() < tol['ton_pct'])
    elif tabla == 'base_brechas':
        if 'brecha_pct_celda' not in a or 'brecha_pct_celda' not in b:
            return vacio
        g = ['periodo', 'des_categoria', 'cosecha_grupo']
        cel = pd.concat({'publicado': b.groupby(g)['brecha_pct_celda'].first(),
                         'recalculado': a.groupby(g)['brecha_pct_celda'].first()}, axis=1).dropna(how='all').reset_index()
        cel['dif'] = cel['recalculado'] - cel['publicado']
        cel['dif_pct'] = np.nan
        cel['fuera'] = (cel['dif'].abs() >= tol['brecha_pp']) | (cel['publicado'].notna() & cel['recalculado'].isna())   # o la celda desapareció
    else:
        return vacio
    return cel[vacio.columns].sort_values(['periodo', 'des_categoria', 'cosecha_grupo']).reset_index(drop=True)
