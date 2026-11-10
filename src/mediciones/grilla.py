"""
Lectura de una consulta SQL de BigQuery con caché CSV local (surtido y Champion).

Una sola implementación con parámetros EXPLÍCITOS para lo que hoy difiere entre los dos programas (no se unifica el comportamiento: eso es una decisión aparte):
                                   surtido                      champion
  sufijo_cache                     AAAAMM del cierre            None (sin sufijo)
  ids_como_texto_en_cache          False                        True  (al leer el CSV, evita perder los ceros a la izquierda)
  fechas_en_consulta               True                         False (en la ruta de BigQuery NO convierte las fechas; en la del caché sí)
  decimales_en_consulta            True                         False (surtido convierte Decimal -> número; Champion no lo hace)
`pd.read_gbq` se invoca como atributo de pandas en cada llamada: así la capa de insumos (mediciones.insumos) puede interceptarlo.
CPB y CPM tienen otra función (sin caché CSV, propaga los errores) y no pasan por aquí.
"""
import decimal
import os

import pandas as pd

from .metodologia.ids import es_columna_id_cliente, normalizar_ids

PATRONES_FECHA = ['periodo', 'fecha', 'fec_', 'cosecha']


def _fechas(df, patrones):
    for c in df.columns:
        if any(p in c.lower() for p in patrones):
            df[c] = pd.to_datetime(df[c], errors='coerce')


def ejecutar_grilla(query, nombre_cache, *, cache_dir, sufijo_cache=None, project_id='acpe-dev-uc-sandbox-aa', verbose=True, forzar_descarga=True,
                    ids_como_texto_en_cache=False, fechas_en_consulta=True, decimales_en_consulta=True, patrones_fecha=PATRONES_FECHA):
    """
    Ejecuta la consulta en BigQuery y devuelve un DataFrame, con caché local en CSV (`<cache_dir>/<nombre_cache>[_<sufijo>].csv`).
    `forzar_descarga=True` ignora el caché existente y vuelve a consultar (el motor lo usa por defecto: los snapshots con hash los maneja mediciones.insumos).
    Ante un error de BigQuery imprime el mensaje y devuelve un DataFrame vacío (comportamiento de los notebooks).
    """
    ruta = os.path.join(cache_dir, f'{nombre_cache}_{sufijo_cache}.csv' if sufijo_cache else f'{nombre_cache}.csv')
    if os.path.exists(ruta) and not forzar_descarga:
        if verbose:
            print(f"[CACHE] '{nombre_cache}' ya existe en {ruta} -> se lee del disco, NO se vuelve a descargar de BigQuery.")
        if ids_como_texto_en_cache:
            # Los códigos de cliente se leen como string: si no, read_csv los vuelve int64 y pierde los ceros a la izquierda
            # (los códigos DEX son de 10 dígitos con ceros).
            cols_csv = pd.read_csv(ruta, nrows=0).columns
            df = pd.read_csv(ruta, low_memory=False, dtype={c: str for c in cols_csv if es_columna_id_cliente(c)})
        else:
            df = pd.read_csv(ruta, low_memory=False)
        _fechas(df, patrones_fecha)
        df = normalizar_ids(df)
        if verbose:
            print(f'  Filas cargadas desde cache: {len(df):,}')
        return df
    if verbose:
        print(f"[BIGQUERY] '{nombre_cache}' no esta en cache -> ejecutando consulta...")
    try:
        df = pd.read_gbq(query, project_id=project_id, dialect='standard', progress_bar_type='tqdm' if verbose else None)
        df = normalizar_ids(df)
        if fechas_en_consulta:
            _fechas(df, patrones_fecha)
        if decimales_en_consulta:
            for c in df.columns:
                if df[c].dtype == object and df[c].map(lambda v: isinstance(v, decimal.Decimal)).any():
                    df[c] = pd.to_numeric(df[c], errors='coerce')
        if verbose:
            print(f'Consulta completada. Filas obtenidas: {len(df):,}')
        df.to_csv(ruta, index=False)
        if verbose:
            print(f'[CACHE] Guardado para proximas corridas en: {ruta}')
        return df
    except Exception as e:
        print('Error al ejecutar la grilla:', e)
        return pd.DataFrame()
