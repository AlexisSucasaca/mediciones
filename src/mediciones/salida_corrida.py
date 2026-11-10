"""
Lo que produjo una corrida del motor, para VERLO y compararlo: un Excel por programa y/o las mismas 3 tablas en un dataset gemelo (`mediciones_motor`).

DESDE LOS PARQUET (sin BigQuery, sin correr el motor): cada corrida deja en `runs/<corrida_id>/` un parquet por programa y base. `corrida_desde_runs` los lee y los normaliza
con los mismos pasos que el motor, y de ahí se exporta igual que si la corrida estuviera en memoria:

    from mediciones.salida_corrida import corrida_desde_runs, exportar_corrida_excel, cargar_corrida_prueba
    c = corrida_desde_runs(RAIZ)                                     # la última corrida del motor (o corrida='202609_...'); 'reportado_202609' es la línea base
    exportar_corrida_excel(c)                                        # 1 Excel por programa en exportes/: resúmenes agregados + las bases que caben (las grandes, en CSV al lado)
    cargar_corrida_prueba(corrida)                                   # prepara y valida (no toca BigQuery)
    cargar_corrida_prueba(corrida, cargar=True, reemplazar=True)     # sube las 3 bases a `mediciones_motor` (mismo esquema que `mediciones_reportado`)

`corrida` es lo que devuelve `correr_mes` (la variable `corrida` del orquestador) o `corrida_desde_runs`. Se usan las bases NORMALIZADAS, tal como el motor las cargaría:
las mismas columnas y el mismo esquema que la línea base reportada, para que se puedan comparar fila a fila.

El dataset gemelo NO es el de producción: no hay deriva ni "cierres congelados" y se puede reemplazar la corrida de un programa y cierre (`reemplazar=True`).
La UB solo viene calculada si el motor la calculó en esa corrida (`calcular_ub=True`) y, cuando ya hay datos publicados en `mediciones`, solo para las filas nuevas
(el motor aplica la UB únicamente a lo que va a cargar): en ese caso la hoja/tabla de toneladas lo avisa.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import bigquery
from . import normalizacion as nz
from . import ub as _ub
from .esquema import DATASET, T_BRE, T_ESP, T_TON
from .registro import cargar_registro
from .runner import Corrida
from .validacion import _carpeta_corrida

TABLAS = ((T_TON, 'ton', 'Base toneladas'), (T_BRE, 'bre', 'Base brechas'), (T_ESP, 'esp', 'Base espejos'))
LIMITE_FILAS_EXCEL = 1_048_575
MAX_FILAS_HOJA = 100_000           # ~35 s por hoja de este tamaño; las bases de 300 a 500 mil filas (CPB, CPM, surtido) van en CSV


def listar_corridas_locales(raiz):
    """Qué hay en runs/: una fila por carpeta, con el tipo (corrida del motor o línea base reportada), los programas y sus filas de toneladas. La más reciente primero."""
    import pyarrow.parquet as pq
    filas = []
    for d in Path(raiz, 'runs').glob('*'):
        archivos = sorted(d.glob('*_ton.parquet')) if d.is_dir() else []
        if not archivos:
            continue
        filas.append({'corrida': d.name, 'tipo': 'línea base reportada' if d.name.startswith('reportado_') else 'motor',
                      'programas': {a.name.rsplit('_ton.parquet', 1)[0]: pq.ParquetFile(a).metadata.num_rows for a in archivos},
                      'modificada': pd.Timestamp(d.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')})
    return pd.DataFrame(sorted(filas, key=lambda f: f['modificada'], reverse=True), columns=['corrida', 'tipo', 'programas', 'modificada'])


def corrida_desde_runs(raiz, corrida=None, programas=None, config=None, con_ub=False, cli=None, verbose=True):
    """
    Lee los parquet de `runs/<corrida>/` (sin BigQuery y sin volver a correr el motor) y devuelve una `Corrida` con las 3 bases normalizadas por programa, lista para
    `exportar_corrida_excel` y `cargar_corrida_prueba`. `corrida=None` toma la última corrida del MOTOR (nunca la línea base `reportado_AAAAMM`, que se pide por nombre).
    Los parquet son la salida del motor ANTES de la UB: la UB se calcula contra BigQuery (solo lectura). `con_ub=True` la calcula como lo hace el motor (según `metodo_ub` de cada programa);
    sin eso, las columnas de UB quedan vacías.
    """
    import re
    raiz = Path(raiz)
    carpeta = _carpeta_corrida(raiz, corrida)
    m = re.search(r'(\d{4})(\d{2})', carpeta.name)
    mes = pd.Timestamp(f'{m.group(1)}-{m.group(2)}-01')
    familias, registro = cargar_registro(Path(config) if config else raiz / 'config' / 'programas.yaml')
    disponibles = sorted(a.name.rsplit('_ton.parquet', 1)[0] for a in carpeta.glob('*_ton.parquet'))
    if cli is None and con_ub:
        cli = bigquery.cliente_bq()
    res = {}
    for p in disponibles:
        if programas and p not in programas:
            continue
        if not (carpeta / f'{p}_brecha.parquet').exists():
            continue
        try:
            regla = familias[registro[p].familia].get('estado_yoy', 'estandar') if p in registro else 'estandar'
            rep = mes.date()
            ton = nz.preparar_base_ton(pd.read_parquet(carpeta / f'{p}_ton.parquet'), p, rep, carpeta.name, regla_estado=regla)
            bre = nz.preparar_base_brechas(pd.read_parquet(carpeta / f'{p}_brecha.parquet'), p, rep, carpeta.name)
            ruta_esp = carpeta / f'{p}_espejos.parquet'
            esp = nz.preparar_base_espejos(pd.read_parquet(ruta_esp), p, rep, carpeta.name) if ruta_esp.exists() else None
            errores, resumen = nz.validar(ton, bre, rep, esp=esp)
            if con_ub:
                metodo = registro[p].metodo_ub if p in registro else 'actual'
                ub_df = _ub.consulta_ub(cli, ton, registro[p].negocios_ub if p in registro else None, meses_atras=12 if metodo == 'notebook' else 0)
                ton, _ = _ub.aplicar_ub(ton, ub_df, metodo=metodo)
            res[p] = {'ton': ton, 'bre': bre, 'esp': esp, 'errores': errores, 'resumen': resumen}
            if verbose:
                print(f"[{p}] {len(ton):,} filas de toneladas · {len(bre):,} de brechas · atribuible {resumen['valor_atribuible_tn_total']} t" + (f"  ⚠ {errores}" if errores else ''))
        except Exception as e:                                         # un programa que no se puede leer no impide los demás
            res[p] = {'error': f'{type(e).__name__}: {e}'}
            if verbose:
                print(f'✘ [{p}] {type(e).__name__}: {e}')
    return Corrida(corrida_id=carpeta.name, mes=mes, modo='local', res=res)


def bases_de_corrida(corrida, programas=None):
    """
    {programa: {'ton', 'bre', 'esp', 'avisos', 'errores'}} con las 3 bases normalizadas completas de cada programa que llegó a normalizarse sin error de ejecución.
    La de toneladas es `cargar_ton` (con UB) cuando cubre todas las filas; si no, es `ton` sin UB y se deja un aviso.
    """
    salida = {}
    for p, r in corrida.res.items():
        if programas and p not in programas:
            continue
        if 'ton' not in r or 'bre' not in r:
            continue                                              # falló antes de normalizar: no hay nada que mostrar
        ton, avisos = r['ton'], []
        cargar = r.get('cargar_ton')
        if cargar is not None and len(cargar) == len(ton):
            ton = cargar                                          # primera carga / nada publicado: es la corrida completa y trae la UB
        if ton['ub_atribuible_soles'].isna().all():
            avisos.append(f'UB vacía: ya hay datos publicados y el motor solo calculó la UB de las {len(cargar):,} filas nuevas de {len(ton):,}.' if cargar is not None and len(cargar) != len(ton)
                          else 'UB vacía: no se calculó en esta corrida (calcular_ub = False, o falló la consulta).')
        salida[p] = {'ton': ton, 'bre': r['bre'], 'esp': r.get('esp'), 'avisos': avisos, 'errores': list(r.get('errores') or [])}
    return salida


def _sin_zona_horaria(df):
    """Excel no admite fechas con zona horaria."""
    df = df.copy()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].dt.tz_localize(None)
    return df


def _agregados(ton, bre):
    """Resúmenes para VER la corrida sin abrir cientos de miles de filas."""
    t = ton.assign(mes=pd.to_datetime(ton['periodo']).dt.strftime('%Y-%m'))
    suma = dict(filas=('cod_cliente', 'size'), clientes=('cod_cliente', 'nunique'), ton_ant=('toneladas_ant', 'sum'), ton_actual=('toneladas_actual', 'sum'),
                atribuible_t=('valor_atribuible', 'sum'), ub_atribuible_soles=('ub_atribuible_soles', 'sum'))
    por_mes = t.groupby(['des_categoria', 'mes'], dropna=False).agg(**suma).round(3).reset_index()
    por_estado = t.groupby(['des_categoria', 'flg_estado_yoy'], dropna=False).agg(**suma).round(3).reset_index()
    claves = [c for c in ('cosecha_grupo', 'des_categoria', 'mes_rel') if c in bre.columns]
    brechas = (bre.groupby(claves, dropna=False).agg(pares=('par_id', 'nunique'), did_t=('did_kg', lambda x: x.sum() / 1000), brecha_pct=('brecha_pct_celda', 'mean'),
                                                       ic80_inf=('ic_pct_inf_80_celda', 'mean'), ic80_sup=('ic_pct_sup_80_celda', 'mean')).round(3).reset_index())
    return {'Por categoría y mes': por_mes, 'Por estado': por_estado, 'Brechas por celda': brechas}


def exportar_corrida_excel(corrida, carpeta='exportes', programas=None, formato='xlsx', max_filas=MAX_FILAS_HOJA, etiqueta='motor', verbose=True):
    """
    Un archivo por programa: `<carpeta>/<programa>_<etiqueta>_AAAAMM.xlsx` (`etiqueta='motor'`; usa 'reportado' para la línea base) con 'Resumen', 'Por categoría y mes', 'Por estado', 'Brechas por celda' y las bases
    'Base toneladas', 'Base brechas', 'Base espejos'. Una base con más de `max_filas` filas NO va en el Excel (tardaría minutos y nadie la revisa en una hoja):
    se escribe como CSV al lado (`<programa>_motor_AAAAMM_<base>.csv`, ~10 s) y la hoja 'Archivos' lo indica. `max_filas=None` las pone todas en el Excel.
    `formato='csv'` escribe solo CSV (las 3 bases). Devuelve las rutas escritas.
    """
    if formato not in ('xlsx', 'csv'):
        raise ValueError("formato debe ser 'xlsx' o 'csv'")
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    aaaamm = pd.Timestamp(corrida.mes).strftime('%Y%m')
    rutas = []
    for p, b in bases_de_corrida(corrida, programas).items():
        resumen = pd.DataFrame({'dato': ['programa', 'corrida_id', 'cierre', 'filas de toneladas', 'filas de brechas', 'filas de espejos', 'atribuible total (t)', 'ub atribuible total (S/)',
                                         'errores de validación', 'avisos'],
                                'valor': [p, corrida.corrida_id, pd.Timestamp(corrida.mes).strftime('%Y-%m-%d'), len(b['ton']), len(b['bre']), 0 if b['esp'] is None else len(b['esp']),
                                          round(float(b['ton']['valor_atribuible'].sum()), 3), round(float(b['ton']['ub_atribuible_soles'].sum()), 0),
                                          '; '.join(b['errores']) or 'ninguno', '; '.join(b['avisos']) or 'ninguno']})
        bases = [(h, b[k]) for _, k, h in TABLAS if b[k] is not None]
        nombre_csv = lambda h: carpeta / f'{p}_{etiqueta}_{aaaamm}_{h.replace("Base ", "").replace(" ", "_")}.csv'
        escritas = []
        if formato == 'csv':
            for h, df in bases:
                df.to_csv(nombre_csv(h), index=False)
                escritas.append(nombre_csv(h))
        else:
            en_excel = [(h, df) for h, df in bases if max_filas is None or len(df) <= max_filas]
            fuera = [(h, df) for h, df in bases if (h, df) not in en_excel]
            grandes = [h for h, df in en_excel if len(df) > LIMITE_FILAS_EXCEL]
            if grandes:
                raise ValueError(f'{p}: {grandes} superan el límite de filas de Excel ({LIMITE_FILAS_EXCEL:,}); usa formato="csv" o max_filas')
            ruta = carpeta / f'{p}_{etiqueta}_{aaaamm}.xlsx'
            with pd.ExcelWriter(ruta, engine='xlsxwriter', date_format='yyyy-mm-dd', datetime_format='yyyy-mm-dd hh:mm:ss') as w:
                resumen.to_excel(w, sheet_name='Resumen', index=False)
                for hoja, df in _agregados(b['ton'], b['bre']).items():
                    df.to_excel(w, sheet_name=hoja, index=False)
                for h, df in en_excel:
                    _sin_zona_horaria(df).to_excel(w, sheet_name=h, index=False)
                if fuera:
                    pd.DataFrame({'base': [h for h, _ in fuera], 'filas': [len(df) for _, df in fuera], 'archivo': [nombre_csv(h).name for h, _ in fuera],
                                  'motivo': f'más de {max_filas:,} filas: en CSV'}).to_excel(w, sheet_name='Archivos', index=False)
            escritas.append(ruta)
            for h, df in fuera:
                df.to_csv(nombre_csv(h), index=False)
                escritas.append(nombre_csv(h))
        rutas.extend(escritas)
        if verbose:
            print(f"[{p}] {len(b['ton']):,} filas de toneladas · {len(b['bre']):,} de brechas · atribuible {b['ton']['valor_atribuible'].sum():,.1f} t -> {', '.join(x.name for x in escritas)}"
                  + (f"  ⚠ {'; '.join(b['avisos'])}" if b['avisos'] else ''))
    return rutas


def cargar_corrida_prueba(corrida, dataset='mediciones_motor', programas=None, cargar=False, reemplazar=False, cli=None, verbose=True):
    """
    Prepara (y con `cargar=True` sube) las 3 bases de la corrida a un dataset GEMELO de producción (por defecto `mediciones_motor`), que se crea solo con el mismo DDL.
    Nunca escribe en `mediciones`. Los programas con errores de validación o que fallaron no se cargan.
    Si ya hay una corrida de ese programa y cierre en el dataset, falla; con `reemplazar=True` borra esas filas (solo esas) y carga la nueva.
    Devuelve un DataFrame con lo cargado (o lo que se cargaría) por programa y tabla.
    """
    if dataset == DATASET:
        raise ValueError(f'{dataset!r} es el dataset de producción: para publicar usa MODO = \'cargar\' en el orquestador. Esta función es para la copia de prueba (p. ej. mediciones_motor).')
    mes = pd.Timestamp(corrida.mes)
    filas, listos = [], {}
    for p, b in bases_de_corrida(corrida, programas).items():
        if b['errores']:
            if verbose:
                print(f"✘ [{p}] no se carga: tiene errores de validación: {b['errores']}")
            continue
        listos[p] = b
        for tabla, k, _ in TABLAS:
            if b[k] is not None:
                filas.append({'programa': p, 'tabla': tabla, 'filas': len(b[k]), 'corrida_id': corrida.corrida_id, 'cargado': False})
    plan = pd.DataFrame(filas, columns=['programa', 'tabla', 'filas', 'corrida_id', 'cargado'])
    if not cargar:
        if verbose:
            print(f'Se cargarían {len(listos)} programa(s) a {dataset} (corrida {corrida.corrida_id}). Usa cargar=True para subirlos.')
        return plan
    cli = cli or bigquery.cliente_bq()
    bigquery.crear_dataset(cli, dataset)
    previas = bigquery.listar_corridas(cli, dataset)
    for p in listos:
        ya = previas[(previas['programa'] == p) & (pd.to_datetime(previas['periodo_reporte']).dt.date == mes.date())]['corrida_id'].unique() if len(previas) else []
        if len(ya) and not reemplazar:
            raise RuntimeError(f'{dataset} ya tiene una corrida de {p} para {mes:%Y-%m}: {list(ya)}. Usa reemplazar=True para quitarla y cargar la nueva '
                               '(si no, las vistas vigentes mostrarían las dos).')
        for corrida_previa in ya:
            bigquery.borrar_corrida(cli, corrida_previa, dataset, programas=[p], confirmar=True)
            if verbose:
                print(f'[{p}] se quitó la corrida anterior {corrida_previa} de {dataset}')
    for p, b in listos.items():
        for tabla, k, _ in TABLAS:
            if b[k] is None:
                continue
            n = bigquery.cargar_df(cli, b[k], tabla, dataset=dataset)                       # tal como lo cargaría el motor
            plan.loc[(plan.programa == p) & (plan.tabla == tabla), 'cargado'] = True
            if verbose:
                print(f'[{p}] {tabla}: {n:,} filas -> {dataset}')
    return plan
