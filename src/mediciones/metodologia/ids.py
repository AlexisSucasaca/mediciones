"""Normalización de los códigos de cliente (compartida por los programas que leen de BigQuery o de un CSV)."""
import pandas as pd


def es_columna_id_cliente(nombre_col):
    n = nombre_col.lower()
    return n.startswith('cod_') and 'cliente' in n


def normalizar_ids(df):
    """Los códigos de cliente quedan como texto sin espacios y con nulos reales (<NA>), nunca como float ni 'nan'."""
    for c in df.columns:
        if not es_columna_id_cliente(c):
            continue
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            # típico cuando el id trae nulos -> BigQuery/CSV lo sube a float64
            # (ej. 123.0) -> Int64 nullable lo deja como entero, sin decimales
            s = s.astype('Int64')
        df[c] = s.astype(str).str.strip()
        df[c] = df[c].replace({'<NA>': pd.NA, 'nan': pd.NA, 'None': pd.NA})
    return df
