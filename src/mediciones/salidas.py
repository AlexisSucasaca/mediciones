"""
Exportables desde lo PUBLICADO (no recalcula nada): la última foto vigente, o todo lo que se publicó cierre a cierre.

Por programa entrega 4 bases:
  * Toneladas      : cliente x categoría x mes, con toneladas atribuibles y UB atribuible.
  * Brechas_par_mes: la base de brechas (par x mes): pre/post de acción y control, DiD, brecha de la celda.
  * Brechas_celda  : resumen por mes x categoría (x cosecha): pares, brecha, IC80, reportable, toneladas y UB atribuibles.
  * Espejos_vigencia: por cliente de acción x categoría, cada espejo con su tramo desde-hasta.
Todas llevan `periodo` (mes medido), `periodo_reporte` (fecha de reporte: el cierre que publicó la fila) y `estado_publicacion`
(definitivo / provisional). En modo 'historico', además `vigente` (si esa fila es la que vale hoy).

Rango: `desde` y `hasta` filtran por mes medido (por defecto, el año en curso). Destino 'excel' (un archivo por programa, en exportes/),
'bigquery' (tablas de reporte en `dataset_destino`, reemplazando SOLO el rango pedido: nunca duplica ni toca otros meses) o 'ambos'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .esquema import DATASET, PROYECTO, T_BRE, T_ESP, T_TON
from .frontera import vigente_local

LIMITE_FILAS_EXCEL = 1_048_000
TABLAS_REPORTE = {'Toneladas': 'rep_ton_atribuibles', 'Brechas_par_mes': 'rep_brechas', 'Brechas_celda': 'rep_brechas_celda',
                  'Espejos_vigencia': 'rep_espejos_vigencia'}
DICCIONARIO = [
    ('periodo', 'mes medido'),
    ('periodo_reporte', 'fecha de reporte: el cierre que publicó la fila (la línea base sembrada es 2026-09-01)'),
    ('estado_publicacion', "'definitivo' (no cambia más), 'provisional' (se republica en cada cierre: el G04 hasta feb-2027) o 'preliminar' (el G04 antes de su cosecha de diciembre)"),
    ('vigente', "solo en modo histórico: True si esa fila es la que vale hoy"),
    ('valor_atribuible', 'toneladas atribuibles'),
    ('ub_atribuible_soles', 'UB atribuible (soles)'),
    ('brecha_pct_celda', 'brecha de la celda (%), igual para todos los pares de la celda'),
    ('did_kg', 'diferencias en diferencias del par (kg): (post - pre) acción menos (post - pre) control'),
    ('desde / hasta', 'primer y último mes de un tramo continuo con el mismo espejo'),
]


# ── lectura ───────────────────────────────────────────────────────────────────
def _mes(x):
    return None if x is None else pd.Timestamp(x).to_period('M').to_timestamp()


def rango_por_defecto(desde=None, hasta=None, hoy=None):
    """Por defecto: desde enero del año en curso, sin tope."""
    hoy = pd.Timestamp(hoy or pd.Timestamp.today())
    return (_mes(desde) if desde is not None else pd.Timestamp(year=hoy.year, month=1, day=1)), _mes(hasta)


def leer_bigquery(cli, tabla, programa, desde, hasta, modo, dataset=None):
    """Filas publicadas de `tabla` para `programa` en el rango. 'vigente' = la vista vigente; 'historico' = todo lo cargado (todas las versiones)."""
    from google.cloud import bigquery
    ds = dataset or DATASET
    rango = "periodo >= @desde" + (" AND periodo <= @hasta" if hasta is not None else '')
    if modo == 'vigente':
        q = f'SELECT * FROM `{PROYECTO}.{ds}.vw_{tabla}_vigente` WHERE programa = @p AND {rango}'
    else:
        q = f'SELECT * FROM `{PROYECTO}.{ds}.{tabla}` WHERE programa = @p AND {rango}'
    params = [bigquery.ScalarQueryParameter('p', 'STRING', programa), bigquery.ScalarQueryParameter('desde', 'DATE', desde.date())]
    if hasta is not None:
        params.append(bigquery.ScalarQueryParameter('hasta', 'DATE', hasta.date()))
    return cli.query(q, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()


# ── armado de las bases ───────────────────────────────────────────────────────
def estado_publicacion(d, provisional_desde=None, provisional_cosechas=None, provisional_hasta=None):
    """
    'preliminar'  : fila de una cosecha provisional publicada ANTES de su cosecha (el G04 en los cierres de oct-nov);
    'provisional' : se republica en cada cierre hasta `provisional_hasta` (el G04 hasta febrero 2027, cuando su retroactivo queda final);
    'definitivo'  : no cambia más.
    """
    prov = _mes(provisional_desde)
    if prov is None or d is None or len(d) == 0:
        return pd.Series('definitivo', index=d.index if d is not None else None, dtype='object')
    es = d['periodo'] >= prov
    cg = pd.to_datetime(d['cosecha_grupo'], errors='coerce').dt.to_period('M').dt.to_timestamp() if 'cosecha_grupo' in d else None
    if provisional_cosechas:
        coh = {_mes(c) for c in provisional_cosechas}
        es = es & (cg.isin(coh) if cg is not None else False)
    ultimo = d['periodo_reporte'].max() if 'periodo_reporte' in d else None
    activo = provisional_hasta is None or ultimo is None or ultimo <= _mes(provisional_hasta)
    if not activo:
        return pd.Series('definitivo', index=d.index, dtype='object')
    prelim = es & (cg is not None) & (d['periodo_reporte'] < cg) if cg is not None and 'periodo_reporte' in d else pd.Series(False, index=d.index)
    return pd.Series(np.where(prelim, 'preliminar', np.where(es, 'provisional', 'definitivo')), index=d.index)


def _preparar(df, prog, modo):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = df.copy()
    for c in ('periodo', 'periodo_reporte'):
        if c in d:
            d[c] = pd.to_datetime(d[c]).dt.to_period('M').dt.to_timestamp()
    d['estado_publicacion'] = estado_publicacion(d, getattr(prog, 'provisional_desde', None), getattr(prog, 'provisional_cosechas', None),
                                                 getattr(prog, 'provisional_hasta', None))
    if modo == 'historico':
        vig = vigente_local(d)
        d['vigente'] = d.index.isin(vig.index)
    return d


def _ordenar(d, primero):
    if d is None or len(d) == 0:
        return d
    tec = [c for c in ('periodo_reporte', 'estado_publicacion', 'vigente', 'corrida_id', 'version', 'config_hash', 'fecha_carga') if c in d]
    cols = [c for c in primero if c in d] + [c for c in d.columns if c not in primero and c not in tec] + tec
    return d[cols].sort_values([c for c in ('periodo', 'des_categoria', 'cosecha_grupo') if c in d]).reset_index(drop=True)


def brechas_por_celda(ton, bre):
    """Resumen por mes x categoría (x cosecha): pares, brecha, IC80, reportable, DiD, toneladas y UB atribuibles, clientes."""
    if bre is None or len(bre) == 0:
        return pd.DataFrame()
    llave = ['programa', 'periodo', 'des_categoria', 'cosecha_grupo']
    b, t = bre.copy(), (ton.copy() if ton is not None and len(ton) else pd.DataFrame(columns=llave))
    for d in (b, t):
        d['cosecha_grupo'] = d['cosecha_grupo'].astype('object').where(d['cosecha_grupo'].notna(), '—') if 'cosecha_grupo' in d else '—'
    extra = [c for c in ('vigente',) if c in b]
    gb = b.groupby(llave + extra, dropna=False)
    cel = gb.agg(pares=('par_id', 'nunique'), brecha_pct=('brecha_pct_celda', 'first'), did_t=('did_kg', lambda s: s.sum() / 1000),
                 reportable=('reportable', 'first'), periodo_reporte=('periodo_reporte', 'max'), estado_publicacion=('estado_publicacion', 'max'))
    for c_bre, c_out in (('ic_pct_inf_80_celda', 'ic80_inf'), ('ic_pct_sup_80_celda', 'ic80_sup')):
        cel[c_out] = gb[c_bre].first() if c_bre in b else np.nan
    if len(t):
        tg = t.groupby(llave + [c for c in extra if c in t], dropna=False)
        agg = {'ton_atribuible': ('valor_atribuible', 'sum'), 'clientes': ('cod_cliente', 'nunique')}
        if 'ub_atribuible_soles' in t:
            agg['ub_atribuible_soles'] = ('ub_atribuible_soles', lambda s: s.sum(min_count=1))
        if 'ic_pct_inf_80' in t:
            agg.update({'_ic_inf_ton': ('ic_pct_inf_80', 'first'), '_ic_sup_ton': ('ic_pct_sup_80', 'first')})
        tt = tg.agg(**agg)
        cel = cel.join(tt, how='left')
        if '_ic_inf_ton' in cel:                                     # en cosechas la banda viene en la base de toneladas
            cel['ic80_inf'] = cel['ic80_inf'].fillna(cel.pop('_ic_inf_ton'))
            cel['ic80_sup'] = cel['ic80_sup'].fillna(cel.pop('_ic_sup_ton'))
    return cel.reset_index()


def vigencia_espejos(esp):
    """Tramos continuos (meses consecutivos) con el mismo espejo, por programa x cliente de acción x categoría (x cosecha)."""
    if esp is None or len(esp) == 0:
        return pd.DataFrame()
    e = esp[esp['cod_cliente_espejo'].notna()].copy()
    if 'vigente' in e:
        e = e[e['vigente']]
    if e.empty:
        return pd.DataFrame()
    e['cosecha_grupo'] = e['cosecha_grupo'].astype('object').where(e['cosecha_grupo'].notna(), '—') if 'cosecha_grupo' in e else '—'
    grupo = ['programa', 'cod_cliente_accion', 'des_categoria', 'cosecha_grupo', 'cod_cliente_espejo']
    e = e.sort_values(grupo + ['periodo'])
    n_mes = e['periodo'].dt.year * 12 + e['periodo'].dt.month
    e['_isla'] = n_mes - e.groupby(grupo, dropna=False).cumcount()
    retro = e['flg_estimado_retroactivo'].fillna(False).astype(bool) if 'flg_estimado_retroactivo' in e else False
    e['_retro'] = retro
    agg = {'desde': ('periodo', 'min'), 'hasta': ('periodo', 'max'), 'meses': ('periodo', 'size'), 'meses_retroactivos': ('_retro', 'sum'),
           'primer_cierre': ('periodo_reporte', 'min'), 'ultimo_cierre': ('periodo_reporte', 'max')}
    if 'nivel_match' in e:
        agg['nivel_match'] = ('nivel_match', 'first')
    v = e.groupby(grupo + ['_isla'], dropna=False).agg(**agg).reset_index().drop(columns='_isla')
    return v.sort_values(['programa', 'cod_cliente_accion', 'des_categoria', 'desde']).reset_index(drop=True)


def armar_bases(programa, ton, bre, esp, modo='vigente', prog=None, provisional_desde=None):
    """Las 4 bases de un programa a partir de las filas publicadas de cada tabla. `prog`: su configuración (lo provisional)."""
    if prog is None and provisional_desde is not None:
        prog = type('P', (), {'provisional_desde': provisional_desde, 'provisional_cosechas': None, 'provisional_hasta': None})()
    t = _preparar(ton, prog, modo)
    b = _preparar(bre, prog, modo)
    e = _preparar(esp, prog, modo)
    return {
        'Toneladas': _ordenar(t, ['programa', 'periodo', 'des_categoria', 'cosecha_grupo', 'cod_cliente', 'cod_cliente_dex', 'des_oficina',
                                  'flg_estado_yoy', 'toneladas_ant', 'toneladas_actual', 'valor_atribuible', 'ub_atribuible_soles']),
        'Brechas_par_mes': _ordenar(b, ['programa', 'periodo', 'des_categoria', 'cosecha_grupo', 'par_id', 'cod_cliente_accion_dex',
                                        'cod_cliente_control', 'pre_accion', 'post_accion', 'pre_control', 'post_control', 'did_kg',
                                        'brecha_pct_celda', 'reportable', 'flg_estimado_retroactivo']),
        'Brechas_celda': _ordenar(brechas_por_celda(t, b), ['programa', 'periodo', 'des_categoria', 'cosecha_grupo', 'pares', 'brecha_pct',
                                                            'ic80_inf', 'ic80_sup', 'reportable', 'did_t', 'ton_atribuible',
                                                            'ub_atribuible_soles', 'clientes']),
        'Espejos_vigencia': vigencia_espejos(e),
    }


# ── destinos ─────────────────────────────────────────────────────────────────
def _a_excel(bases, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        import xlsxwriter  # noqa: F401
        motor = 'xlsxwriter'
    except ImportError:
        motor = 'openpyxl'
    with pd.ExcelWriter(ruta, engine=motor) as xw:
        for hoja, df in bases.items():
            if df is None or len(df) == 0:
                pd.DataFrame({'sin filas en el rango': []}).to_excel(xw, sheet_name=hoja, index=False)
                continue
            d = df.copy()
            for c in d.columns:                                      # Excel: fechas sin hora y sin zona
                if pd.api.types.is_datetime64_any_dtype(d[c]):
                    d[c] = d[c].dt.tz_localize(None) if getattr(d[c].dt, 'tz', None) is not None else d[c]
            partes = max(1, int(np.ceil(len(d) / LIMITE_FILAS_EXCEL)))
            for i in range(partes):
                nombre = hoja if partes == 1 else f'{hoja}_{i + 1}'
                d.iloc[i * LIMITE_FILAS_EXCEL:(i + 1) * LIMITE_FILAS_EXCEL].to_excel(xw, sheet_name=nombre[:31], index=False)
        pd.DataFrame(DICCIONARIO, columns=['columna', 'qué es']).to_excel(xw, sheet_name='Diccionario', index=False)
    return ruta


def _a_bigquery(cli, bases, programa, desde, hasta, dataset_destino, sufijo=''):
    """Reemplaza SOLO el rango pedido de este programa en cada tabla de reporte (borra el rango e inserta)."""
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery
    filas = {}
    for hoja, tabla in TABLAS_REPORTE.items():
        df = bases.get(hoja)
        if df is None or len(df) == 0:
            continue
        ref = f'{PROYECTO}.{dataset_destino}.{tabla}{sufijo}'
        col_mes = 'desde' if hoja == 'Espejos_vigencia' else 'periodo'
        cond = f'programa = @p AND {col_mes} >= @desde' + (f' AND {col_mes} <= @hasta' if hasta is not None else '')
        params = [bigquery.ScalarQueryParameter('p', 'STRING', programa), bigquery.ScalarQueryParameter('desde', 'DATE', desde.date())]
        if hasta is not None:
            params.append(bigquery.ScalarQueryParameter('hasta', 'DATE', hasta.date()))
        try:
            cli.query(f'DELETE FROM `{ref}` WHERE {cond}', job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        except NotFound:
            pass                                                     # primera exportación: la carga crea la tabla
        d = df.copy()
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]) and c in ('periodo', 'periodo_reporte', 'desde', 'hasta', 'primer_cierre', 'ultimo_cierre'):
                d[c] = d[c].dt.date
        cfg = bigquery.LoadJobConfig(write_disposition='WRITE_APPEND', schema_update_options=['ALLOW_FIELD_ADDITION'])
        cli.load_table_from_dataframe(d, ref, job_config=cfg).result()
        filas[tabla + sufijo] = len(d)
    return filas


def exportar(programas, desde=None, hasta=None, modo='vigente', destino='excel', raiz='.', carpeta='exportes', cli=None,
             dataset_origen=None, dataset_destino='mediciones_reporte', leer=None, registro=None, verbose=True):
    """
    Exporta desde lo publicado. `modo`: 'vigente' (lo que vale hoy, una fila por llave) o 'historico' (todo lo publicado, con `vigente`).
    `desde`/`hasta`: meses medidos (por defecto, el año en curso). `destino`: 'excel', 'bigquery' o 'ambos'.
    `leer(tabla, programa, desde, hasta, modo)` permite inyectar la lectura (pruebas); por defecto lee BigQuery.
    """
    if modo not in ('vigente', 'historico'):
        raise ValueError("modo debe ser 'vigente' o 'historico'")
    if destino not in ('excel', 'bigquery', 'ambos'):
        raise ValueError("destino debe ser 'excel', 'bigquery' o 'ambos'")
    desde, hasta = rango_por_defecto(desde, hasta)
    if hasta is not None and hasta < desde:
        raise ValueError(f'hasta ({hasta.date()}) es anterior a desde ({desde.date()})')
    if registro is None:
        from .registro import cargar_registro
        _, registro = cargar_registro(Path(raiz) / 'config' / 'programas.yaml')
    if leer is None or destino in ('bigquery', 'ambos'):
        if cli is None:
            from . import bigquery as _bq
            cli = _bq.cliente_bq()
    if leer is None:
        leer = lambda tabla, p, d, h, m: leer_bigquery(cli, tabla, p, d, h, m, dataset=dataset_origen)
    rango_txt = f"{desde:%Y%m}_{hasta:%Y%m}" if hasta is not None else f"{desde:%Y%m}_en_adelante"
    out = {}
    for p in programas:
        bases = armar_bases(p, leer(T_TON, p, desde, hasta, modo), leer(T_BRE, p, desde, hasta, modo), leer(T_ESP, p, desde, hasta, modo),
                            modo=modo, prog=registro.get(p) if registro else None)
        r = {'filas': {h: (0 if d is None else len(d)) for h, d in bases.items()}}
        if destino in ('excel', 'ambos'):
            r['excel'] = str(_a_excel(bases, Path(raiz) / carpeta / f'{p}_{modo}_{rango_txt}.xlsx'))
        if destino in ('bigquery', 'ambos'):
            r['bigquery'] = _a_bigquery(cli, bases, p, desde, hasta, dataset_destino, sufijo='' if modo == 'vigente' else '_historico')
        out[p] = r
        if verbose:
            print(f'[{p}] {modo} {desde.date()} -> {hasta.date() if hasta is not None else "último mes"}: {r["filas"]}'
                  + (f" · {r['excel']}" if 'excel' in r else '') + (f" · BigQuery {r['bigquery']}" if 'bigquery' in r else ''))
    return out


def vista_previa(corrida, desde=None, hasta=None, raiz='.', carpeta='exportes', cli=None, leer=None, registro=None, verbose=True):
    """
    Lo que quedaría publicado si esta corrida se cargara, SIN cargar nada: lo publicado + lo que la corrida publicaría (`cargar_*`), con la
    misma regla de vigencia (el mes nuevo se agrega; el G04 provisional reemplaza al anterior). Un Excel por programa en
    exportes/vista_previa_<AAAAMM>/, con las mismas 4 hojas de `exportar`. Pensado para revisar un `dry_run` antes de cargar.
    """
    desde, hasta = rango_por_defecto(desde, hasta)
    if registro is None:
        from .registro import cargar_registro
        _, registro = cargar_registro(Path(raiz) / 'config' / 'programas.yaml')
    if leer is None:
        if cli is None:
            from . import bigquery as _bq
            cli = _bq.cliente_bq()
        leer = lambda tabla, p, d, h, m: leer_bigquery(cli, tabla, p, d, h, m)
    mes = pd.Timestamp(corrida.mes).to_period('M').to_timestamp()
    rango_txt = f"{desde:%Y%m}_{hasta:%Y%m}" if hasta is not None else f"{desde:%Y%m}_en_adelante"
    out = {}
    for p, r in corrida.res.items():
        if 'cargar_ton' not in r:
            if verbose:
                print(f'[{p}] sin vista previa: la corrida no llegó a calcular qué publicar ({r.get("error", "sin resultados")})')
            continue
        tablas = {}
        for tabla, k in ((T_TON, 'ton'), (T_BRE, 'bre'), (T_ESP, 'esp')):
            pub = leer(tabla, p, desde, hasta, 'historico')
            pub = pub if pub is not None else pd.DataFrame()
            nuevo = r.get(f'cargar_{k}')
            partes = [pub] if len(pub) else []
            if nuevo is not None and len(nuevo):
                n = nuevo.copy()
                per = pd.to_datetime(n['periodo']).dt.to_period('M').dt.to_timestamp()
                n = n[(per >= desde) & ((per <= hasta) if hasta is not None else True)]
                n['periodo_reporte'] = mes
                v_prev = (pd.to_numeric(pub.loc[pd.to_datetime(pub['periodo_reporte']).dt.to_period('M').dt.to_timestamp() == mes, 'version']).max()
                          if len(pub) else float('nan'))
                n['version'] = 1 if pd.isna(v_prev) else int(v_prev) + 1            # siempre gana la rebanada de esta corrida
                n['programa'] = p
                partes.append(n)
            comb = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
            tablas[k] = vigente_local(comb) if len(comb) else comb
        bases = armar_bases(p, tablas['ton'], tablas['bre'], tablas['esp'], modo='vigente', prog=registro.get(p) if registro else None)
        ruta = _a_excel(bases, Path(raiz) / carpeta / f'vista_previa_{mes:%Y%m}' / f'{p}_vista_previa_{rango_txt}.xlsx')
        t = bases['Toneladas']
        nuevo_mes = t[pd.to_datetime(t['periodo_reporte']) == mes] if len(t) else t
        resumen = {'excel': str(ruta), 'meses_que_publicaria': sorted(pd.to_datetime(nuevo_mes['periodo']).dt.strftime('%Y-%m').unique().tolist()) if len(nuevo_mes) else [],
                   'filas_que_publicaria': int(len(nuevo_mes)),
                   'ton_atribuible_que_publicaria': round(float(nuevo_mes['valor_atribuible'].sum()), 3) if len(nuevo_mes) else 0.0,
                   'ub_que_publicaria': (round(float(nuevo_mes['ub_atribuible_soles'].sum()), 0)
                                         if len(nuevo_mes) and 'ub_atribuible_soles' in nuevo_mes else None)}
        out[p] = resumen
        if verbose:
            print(f"[{p}] vista previa {mes:%Y-%m}: publicaría {resumen['meses_que_publicaria']} · {resumen['filas_que_publicaria']:,} filas · "
                  f"{resumen['ton_atribuible_que_publicaria']:,} t · UB {resumen['ub_que_publicaria']} · {ruta}")
    return out
