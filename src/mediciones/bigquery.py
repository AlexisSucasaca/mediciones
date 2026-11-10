"""Acceso a BigQuery: lectura de lo publicado, carga append-only, log de ejecución y exportes desde la vista vigente."""
import os

import pandas as pd

from .esquema import CLAVE, DATASET, PROYECTO, T_BRE, T_ESP, T_TON, T_LOG, VALOR


def cliente_bq():
    from google.cloud import bigquery
    return bigquery.Client(project=PROYECTO)


def leer_publicado(cli, tabla, programa):
    """Llaves y valor ya publicados (vista vigente) del programa."""
    from google.cloud import bigquery
    extra = {'base_brechas': ['des_categoria', 'cosecha_grupo', 'brecha_pct_celda']}.get(tabla, ['cosecha_grupo'])   # para la deriva por celda
    cols = ', '.join(dict.fromkeys(CLAVE[tabla] + [VALOR[tabla], 'periodo_reporte', 'version', 'config_hash'] + extra))
    q = f'SELECT {cols} FROM `{PROYECTO}.{DATASET}.vw_{tabla}_vigente` WHERE programa = @p'
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('p', 'STRING', programa)])
    df = cli.query(q, job_config=cfg).to_dataframe()
    df['periodo'] = pd.to_datetime(df['periodo']).dt.date
    df['periodo_reporte'] = pd.to_datetime(df['periodo_reporte']).dt.date
    return df


def leer_rebanada(cli, tabla, programa, periodo_reporte, dataset=None):
    """Todas las columnas de lo que publicó un cierre (su última versión): la base para completar o corregir esa publicación."""
    from google.cloud import bigquery
    q = (f'SELECT * FROM `{PROYECTO}.{dataset or DATASET}.{tabla}` WHERE programa = @p AND periodo_reporte = @r '
         f'QUALIFY version = MAX(version) OVER ()')
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('p', 'STRING', programa),
                                                    bigquery.ScalarQueryParameter('r', 'DATE', pd.Timestamp(periodo_reporte).date())])
    return cli.query(q, job_config=cfg).to_dataframe()


def leer_pares(cli, programa):
    """Pares acción-espejo vigentes del programa (solo los que tienen espejo)."""
    from google.cloud import bigquery
    q = (f'SELECT periodo, cosecha_grupo, des_categoria, cod_cliente_accion, cod_cliente_espejo, nivel_match '
         f'FROM `{PROYECTO}.{DATASET}.vw_{T_ESP}_vigente` WHERE programa = @p AND cod_cliente_espejo IS NOT NULL')
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('p', 'STRING', programa)])
    df = cli.query(q, job_config=cfg).to_dataframe()
    df['periodo'] = pd.to_datetime(df['periodo'])
    return df


def ultima_version(cli, tabla, programa, periodo_reporte):
    from google.cloud import bigquery
    q = (f'SELECT MAX(version) AS v FROM `{PROYECTO}.{DATASET}.{tabla}` '
         'WHERE programa = @p AND periodo_reporte = @r')
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter('p', 'STRING', programa),
        bigquery.ScalarQueryParameter('r', 'DATE', periodo_reporte)])
    v = list(cli.query(q, job_config=cfg).result())[0]['v']
    return None if v is None else int(v)


def verificar_carga(cli, df, tabla, dataset=None):
    """
    Revisa, SIN cargar, que `df` se pueda cargar en la tabla real: no trae columnas que la tabla no tenga (se perderían en silencio) y no deja vacía
    ninguna columna obligatoria (REQUIRED) de la tabla. Devuelve el esquema real. Se usa antes de cargar varias tablas, para no dejar una carga a medias.
    """
    ref = f'{PROYECTO}.{dataset or DATASET}.{tabla}'
    esquema = cli.get_table(ref).schema
    nombres = [f.name for f in esquema]
    sobran = [c for c in df.columns if c not in nombres]
    if sobran:                                   # antes se descartaban en silencio (el dato se cargaba como NULL): ahora se falla
        raise ValueError(f'{ref} no tiene las columnas {sobran}: la tabla se creó con otra versión del DDL y cargar ahora perdería esos datos en silencio. '
                         f"Revisa el esquema con bigquery.diagnosticar_esquema(cli, '{dataset or DATASET}') y corrígelo (docs/decisiones.md, 'Esquema desactualizado').")
    vacias = [f.name for f in esquema if getattr(f, 'mode', 'NULLABLE') == 'REQUIRED' and (f.name not in df.columns or df[f.name].isna().any())]
    if vacias:
        raise ValueError(f'{ref} tiene columnas obligatorias que esta carga deja vacías: {vacias}. Si la tabla es de una versión anterior del DDL '
                         f'(diagnosticar_esquema las muestra como "sobra"), recréala con el DDL actual (docs/decisiones.md, "Tablas de una versión anterior").')
    return esquema


def cargar_df(cli, df, tabla, dataset=None):
    """Append alineado al esquema real de la tabla (columnas faltantes -> nulo). `dataset`: por defecto el de producción del motor (esquema.DATASET)."""
    from google.cloud import bigquery
    ref = f'{PROYECTO}.{dataset or DATASET}.{tabla}'
    esquema = verificar_carga(cli, df, tabla, dataset)
    nombres = [f.name for f in esquema]
    df = df.reindex(columns=nombres)
    cfg = bigquery.LoadJobConfig(schema=esquema, write_disposition='WRITE_APPEND')
    cli.load_table_from_dataframe(df, ref, job_config=cfg).result()
    return len(df)


TABLAS_ESQUEMA = {'base_ton_atribuibles': 'COLS_TON', 'base_brechas': 'COLS_BRE', 'base_espejos': 'COLS_ESP', 'log_metricas_ejecucion': 'COLS_LOG'}   # el motor escribe en las 4


def _tipos_ddl():
    """Tipo de cada columna de cada tabla, leído de sql/ddl_mediciones.sql: {tabla: {columna: tipo}}."""
    import re
    from pathlib import Path
    ddl = (Path(__file__).resolve().parents[2] / 'sql' / 'ddl_mediciones.sql').read_text(encoding='utf-8')
    out = {}
    for tabla in TABLAS_ESQUEMA:
        m = re.search(r'CREATE TABLE IF NOT EXISTS `[^`]*\.' + tabla + r'`\s*\((.*?)\n\)', ddl, re.S)
        out[tabla] = {} if not m else {c.group(1): c.group(2) for c in re.finditer(r'^\s{2}([a-z_0-9]+)\s+(STRING|FLOAT64|INT64|DATE|BOOL|TIMESTAMP|NUMERIC)\b', m.group(1), re.M)}
    return out


def diagnosticar_esquema(cli, dataset=None):
    """
    Compara las tablas reales de un dataset con las que el motor espera (esquema.py). Devuelve un DataFrame (vacío = todo bien) con una fila por diferencia y la
    sentencia sugerida: RENAME COLUMN cuando la tabla trae el nombre antiguo (p. ej. ic_pct_inf en vez de ic_pct_inf_80) o ADD COLUMN para una columna que falta.
    Un RENAME conserva las filas ya cargadas, pero con el valor que tenían (si el motor cargó con el nombre antiguo, esas columnas están en NULL: conviene recargar).
    """
    import pandas as pd
    from . import esquema as esq
    from google.api_core.exceptions import NotFound
    ds = dataset or DATASET
    tipos = _tipos_ddl()
    filas = []
    for tabla, nombre_cols in TABLAS_ESQUEMA.items():
        ref = f'{PROYECTO}.{ds}.{tabla}'
        esperadas = list(getattr(esq, nombre_cols))
        try:
            reales = [f.name for f in cli.get_table(ref).schema]
        except NotFound:
            filas.append({'tabla': tabla, 'columna': '(toda la tabla)', 'problema': 'la tabla no existe', 'sugerencia': f'ejecuta sql/ddl_mediciones.sql en {ds}'})
            continue
        faltan, sobran = [c for c in esperadas if c not in reales], [c for c in reales if c not in esperadas]
        for c in faltan:
            antiguo = next((x for x in sobran if c.startswith(x) or c.replace('_80', '') == x), None)
            sug = (f'ALTER TABLE `{ref}` RENAME COLUMN {antiguo} TO {c};' if antiguo
                   else f'ALTER TABLE `{ref}` ADD COLUMN IF NOT EXISTS {c} {tipos.get(tabla, {}).get(c, "FLOAT64")};')
            filas.append({'tabla': tabla, 'columna': c, 'problema': 'falta en la tabla' + (f' (trae {antiguo!r})' if antiguo else ''), 'sugerencia': sug,
                          'antiguo': antiguo, 'tipo': tipos.get(tabla, {}).get(c, 'FLOAT64')})
        for c in sobran:
            if not any(f['columna'] == c or f"trae {c!r}" in f['problema'] for f in filas if f['tabla'] == tabla):
                filas.append({'tabla': tabla, 'columna': c, 'problema': 'sobra en la tabla (el motor no la carga)', 'sugerencia': ''})
    return pd.DataFrame(filas, columns=['tabla', 'columna', 'problema', 'sugerencia', 'antiguo', 'tipo'])


def crear_dataset(cli, dataset, ubicacion='US'):
    """
    Crea un dataset GEMELO del de mediciones (mismas tablas y vistas vigentes) ejecutando sql/ddl_mediciones.sql con el nombre del dataset cambiado.
    Es re-ejecutable (todo es IF NOT EXISTS / CREATE OR REPLACE VIEW). Pensado para la línea base REPORTADA (p. ej. 'mediciones_reportado').
    """
    if dataset == DATASET:
        raise ValueError(f'{dataset!r} es el dataset de producción del motor: para actualizarlo usa actualizar_esquema(cli) o ejecuta sql/ddl_mediciones.sql directamente')
    _ejecutar_ddl(cli, dataset, ubicacion)
    return f'{PROYECTO}.{dataset}'


def _ejecutar_ddl(cli, dataset, ubicacion='US'):
    """Ejecuta sql/ddl_mediciones.sql sobre `dataset`. Es idempotente: no toca tablas existentes (IF NOT EXISTS), agrega las columnas de UB y recrea las vistas vigentes."""
    from pathlib import Path
    ddl = (Path(__file__).resolve().parents[2] / 'sql' / 'ddl_mediciones.sql').read_text(encoding='utf-8')
    ddl = ddl.replace(f'{PROYECTO}.{DATASET}', f'{PROYECTO}.{dataset}').replace("DEFAULT 'US'", f"DEFAULT '{ubicacion}'")
    cli.query(ddl).result()


def actualizar_esquema(cli, dataset=None, ejecutar=False, modo='agregar'):
    """
    Pone las tablas de un dataset al día SIN borrar datos: compara con lo que espera el motor y construye las sentencias mínimas.
      modo='agregar'    (por defecto) agrega las columnas que faltan y deja las antiguas como están (no supone nada sobre su contenido).
      modo='renombrar'  si la tabla trae el nombre antiguo (ic_pct_inf), lo renombra (conserva sus valores; solo si sabes que significan lo mismo).
    Con `ejecutar=False` (por defecto) solo devuelve el plan. Con `ejecutar=True` aplica las sentencias, re-ejecuta el DDL (recrea las vistas vigentes, que son `SELECT *`
    y no ven las columnas nuevas hasta recrearlas) y devuelve el diagnóstico final sin las columnas sobrantes (vacío = al día).
    Ojo: las filas cargadas ANTES tienen NULL en las columnas nuevas (la carga descartaba lo que la tabla no tenía): recarga esa corrida (ver borrar_corrida).
    """
    import pandas as pd
    if modo not in ('agregar', 'renombrar'):
        raise ValueError("modo debe ser 'agregar' o 'renombrar'")
    ds = dataset or DATASET
    d = diagnosticar_esquema(cli, ds)
    plan = []
    for r in d.itertuples():
        ref = f'{PROYECTO}.{ds}.{r.tabla}'
        if r.problema == 'la tabla no existe':
            plan.append({'tabla': r.tabla, 'accion': 'se crea al re-ejecutar el DDL', 'sentencia': ''})
        elif r.problema.startswith('falta'):
            antiguo = r.antiguo if isinstance(r.antiguo, str) else None                 # pandas guarda el 'sin antecesor' como NaN, que es verdadero: no usarlo como condición
            if modo == 'renombrar' and antiguo:
                plan.append({'tabla': r.tabla, 'accion': f'renombrar {antiguo} -> {r.columna}', 'sentencia': f'ALTER TABLE `{ref}` RENAME COLUMN {antiguo} TO {r.columna};'})
            else:
                plan.append({'tabla': r.tabla, 'accion': f'agregar {r.columna}', 'sentencia': f'ALTER TABLE `{ref}` ADD COLUMN IF NOT EXISTS {r.columna} {r.tipo};'})
    plan = pd.DataFrame(plan, columns=['tabla', 'accion', 'sentencia'])
    if not ejecutar:
        return plan
    for sentencia in plan['sentencia']:
        if sentencia:
            cli.query(sentencia).result()
    _ejecutar_ddl(cli, ds)
    d2 = diagnosticar_esquema(cli, ds)
    return d2[~d2['problema'].str.startswith('sobra')].reset_index(drop=True)        # las columnas antiguas que quedan sin usar no son un problema


def recrear_tablas_vacias(cli, dataset=None, confirmar=False):
    """
    Recrea con el DDL actual las 3 tablas base de un dataset cuyas tablas quedaron de una versión anterior del diseño (columnas sobrantes u obligatorias que
    el motor ya no carga), SOLO si están vacías (ningún programa, ningún cierre). Con alguna fila se niega. Sin `confirmar=True` solo dice qué haría.
    Devuelve el diagnóstico del esquema al terminar (vacío = al día). El log de ejecución y dim_cosechas no se tocan.
    """
    ds = dataset or DATASET
    tablas = ('base_ton_atribuibles', 'base_brechas', 'base_espejos')
    filas = {t: int(list(cli.query(f'SELECT COUNT(*) AS n FROM `{PROYECTO}.{ds}.{t}`').result())[0]['n']) for t in tablas}
    if any(filas.values()):
        raise RuntimeError(f'{ds}: las tablas tienen filas {filas}; no se recrean. Corrige el esquema con actualizar_esquema o borra antes esas corridas (borrar_corrida).')
    if not confirmar:
        print(f'[{ds}] tablas vacías {filas}: con confirmar=True se borran y se recrean con el DDL actual (también sus vistas vigentes).')
        return None
    for t in tablas:
        cli.delete_table(f'{PROYECTO}.{ds}.{t}', not_found_ok=True)
    return actualizar_esquema(cli, dataset=ds, ejecutar=True)


def listar_corridas(cli, dataset=None):
    """Qué hay cargado en un dataset: una fila por programa, corrida y cierre, con su versión y las filas de la tabla de toneladas."""
    q = (f'SELECT programa, corrida_id, periodo_reporte, MAX(version) AS version, COUNT(*) AS filas FROM `{PROYECTO}.{dataset or DATASET}.base_ton_atribuibles` '
         'GROUP BY programa, corrida_id, periodo_reporte ORDER BY periodo_reporte, programa, corrida_id')
    return cli.query(q).to_dataframe()


def borrar_corrida(cli, corrida_id, dataset=None, programas=None, confirmar=False):
    """
    Borra de las 3 tablas SOLO las filas de una corrida (no las tablas). Sin `confirmar=True` solo cuenta lo que borraría. `programas` limita a esos programas.
    Úsalo para quitar una carga de prueba y recargarla con el esquema corregido, sin tocar el resto del dataset.
    """
    from google.cloud import bigquery as bq
    ds = dataset or DATASET
    filtro = ' AND programa IN UNNEST(@programas)' if programas else ''
    params = [bq.ScalarQueryParameter('c', 'STRING', corrida_id)] + ([bq.ArrayQueryParameter('programas', 'STRING', list(programas))] if programas else [])
    cfg = bq.QueryJobConfig(query_parameters=params)
    cuenta = {}
    for tabla in TABLAS_ESQUEMA:
        ref = f'`{PROYECTO}.{ds}.{tabla}`'
        cuenta[tabla] = int(list(cli.query(f'SELECT COUNT(*) AS n FROM {ref} WHERE corrida_id = @c{filtro}', job_config=cfg).result())[0]['n'])
    if confirmar:
        for tabla in TABLAS_ESQUEMA:
            cli.query(f'DELETE FROM `{PROYECTO}.{ds}.{tabla}` WHERE corrida_id = @c{filtro}', job_config=cfg).result()
    return cuenta


def contar_programa(cli, tabla, programa, dataset=None):
    """Filas (de cualquier cierre y versión) que un programa ya tiene publicadas en una tabla (la siembra de la línea base exige 0)."""
    from google.cloud import bigquery
    q = f'SELECT COUNT(*) AS n FROM `{PROYECTO}.{dataset or DATASET}.{tabla}` WHERE programa = @p'
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('p', 'STRING', programa)])
    return int(list(cli.query(q, job_config=cfg).result())[0]['n'])


def contar_corrida(cli, tabla, programa, corrida_id, dataset=None):
    """Filas de una corrida en una tabla (para no cargar dos veces la misma línea base)."""
    from google.cloud import bigquery
    q = f'SELECT COUNT(*) AS n FROM `{PROYECTO}.{dataset or DATASET}.{tabla}` WHERE programa = @p AND corrida_id = @c'
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('p', 'STRING', programa),
                                                    bigquery.ScalarQueryParameter('c', 'STRING', corrida_id)])
    return int(list(cli.query(q, job_config=cfg).result())[0]['n'])


def fila_log(corrida_id, programa, periodo_reporte, etapa, metrica, valor=None, valor_txt=None):
    return {'corrida_id': corrida_id, 'programa': programa, 'periodo_reporte': periodo_reporte,
            'etapa': etapa, 'cosecha_grupo': None, 'des_categoria': None, 'des_oficina': None,
            'metrica': metrica, 'valor': None if valor is None else float(valor), 'valor_txt': valor_txt,
            'fecha_registro': pd.Timestamp.now(tz='UTC')}


def registrar_log(cli, filas):
    if filas and cli is not None:
        cargar_df(cli, pd.DataFrame(filas), T_LOG)


def sql_vigente(tabla, hasta=None, dataset=None):
    """
    Lo vigente de `tabla` con la misma regla que las vistas vw_*_vigente (ver frontera.py), filtrado por el parámetro @p (programa).
    `hasta` ('AAAA-MM-DD'): lo vigente A ESE CIERRE, ignorando cierres posteriores (sirve para reconstruir el Excel de un mes pasado tal como se publicó).
    """
    filtro = f"AND periodo_reporte <= DATE '{pd.Timestamp(hasta):%Y-%m-%d}'" if hasta is not None else ''
    return (f"SELECT * EXCEPT (_rep_max) FROM (\n"
            f"  SELECT *, MAX(periodo_reporte) OVER (PARTITION BY programa, DATE_TRUNC(periodo, MONTH), IFNULL(CAST(cosecha_grupo AS STRING), '')) AS _rep_max FROM (\n"
            f"    SELECT * FROM `{PROYECTO}.{dataset or DATASET}.{tabla}` WHERE programa = @p {filtro}\n"
            f"    QUALIFY version = MAX(version) OVER (PARTITION BY programa, periodo_reporte)))\n"
            f"WHERE periodo_reporte = _rep_max")


def exportar_vigente(cli, programa, mes, carpeta='exportes'):
    os.makedirs(carpeta, exist_ok=True)
    try:
        import xlsxwriter  # noqa: F401
        motor = 'xlsxwriter'
    except ImportError:
        motor = 'openpyxl'
    rutas = []
    for tabla, hoja in ((T_TON, 'Base toneladas (cliente-mes)'), (T_BRE, 'Base brecha (par-mes)'), (T_ESP, 'Base espejos (accion-mes)')):
        from google.cloud import bigquery as _bq
        cfg = _bq.QueryJobConfig(query_parameters=[_bq.ScalarQueryParameter('p', 'STRING', programa)])
        df = cli.query(sql_vigente(tabla, hasta=mes), job_config=cfg).to_dataframe()   # lo vigente a ese cierre (ver frontera.py)
        for c in df.select_dtypes(include=['datetimetz']).columns:
            df[c] = df[c].dt.tz_localize(None)
        ruta = os.path.join(carpeta, f'{tabla}_{programa}_{mes:%Y%m}.xlsx')
        with pd.ExcelWriter(ruta, engine=motor) as w:
            df.to_excel(w, sheet_name=hoja[:31], index=False)
        rutas.append(ruta)
    return rutas
