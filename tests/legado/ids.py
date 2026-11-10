# COPIA LITERAL del código anterior (ids, surtido y champion (idénticas)), líneas 135-151 de programas/surtido.py, antes de extraerlo a un módulo compartido.
# No se edita: es la referencia de las pruebas diferenciales.
def _es_columna_id_cliente(nombre_col):
    n = nombre_col.lower()
    return n.startswith('cod_') and 'cliente' in n

def _normalizar_ids(df):
    for c in df.columns:
        if not _es_columna_id_cliente(c):
            continue
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            # típico cuando el id trae nulos -> BigQuery/CSV lo sube a float64
            # (ej. 123.0) -> Int64 nullable lo deja como entero, sin decimales
            s = s.astype('Int64')
        df[c] = s.astype(str).str.strip()
        df[c] = df[c].replace({'<NA>': pd.NA, 'nan': pd.NA, 'None': pd.NA})
    return df
