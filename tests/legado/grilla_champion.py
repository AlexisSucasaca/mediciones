# COPIA LITERAL del código anterior (ejecutar_grilla, champion), líneas 101-160 de programas/champion.py, antes de extraerlo a un módulo compartido.
# No se edita: es la referencia de las pruebas diferenciales.
def ejecutar_grilla(query: str, nombre_cache: str, project_id: str = "acpe-dev-uc-sandbox-aa",
                     verbose: bool = True, forzar_descarga: bool = True):   # el motor usa mediciones.insumos (snapshots con hash de SQL); caché CSV del notebook desactivado
    """
    Ejecuta una consulta SQL en BigQuery y devuelve el resultado como DataFrame,
    con CACHE LOCAL EN CSV para evitar redescargas innecesarias entre corridas.

    Parametros:
    ----------
    query : str
        Consulta SQL completa.
    nombre_cache : str
        Nombre unico para el archivo de cache (se guarda en 'data_cache/<nombre_cache>.csv').
        Debe ser distinto para cada consulta del notebook.
    project_id : str
        ID del proyecto en GCP.
    verbose : bool
        Si True, muestra el progreso.
    forzar_descarga : bool
        Si True, ignora el cache existente y vuelve a consultar BigQuery
        (util cuando sabes que la fuente cambio y quieres refrescar).
    """
    ruta = os.path.join(CACHE_DIR, f"{nombre_cache}.csv")

    if os.path.exists(ruta) and not forzar_descarga:
        if verbose:
            print(f"[CACHE] '{nombre_cache}' ya existe en {ruta} -> se lee del disco, "
                  f"NO se vuelve a descargar de BigQuery.")
        # Los códigos de cliente se leen como string: si no, read_csv los vuelve int64 y
        # pierde los ceros a la izquierda (los códigos DEX son de 10 dígitos con ceros).
        _cols_csv = pd.read_csv(ruta, nrows=0).columns
        _dtype_ids = {c: str for c in _cols_csv if _es_columna_id_cliente(c)}
        df = pd.read_csv(ruta, low_memory=False, dtype=_dtype_ids)
        for c in df.columns:
            if any(p in c.lower() for p in _PATRONES_FECHA):
                df[c] = pd.to_datetime(df[c], errors='coerce')
        df = _normalizar_ids(df)
        if verbose:
            print(f"  Filas cargadas desde cache: {len(df):,}")
        return df

    if verbose:
        print(f"[BIGQUERY] '{nombre_cache}' no esta en cache -> ejecutando consulta...")

    try:
        df = pd.read_gbq(
            query,
            project_id=project_id,
            dialect="standard",
            progress_bar_type="tqdm" if verbose else None
        )
        df = _normalizar_ids(df)
        if verbose:
            print(f"Consulta completada. Filas obtenidas: {len(df):,}")
        df.to_csv(ruta, index=False)
        if verbose:
            print(f"[CACHE] Guardado para proximas corridas en: {ruta}")
        return df
    except Exception as e:
        print("Error al ejecutar la grilla:", e)
        return pd.DataFrame()
