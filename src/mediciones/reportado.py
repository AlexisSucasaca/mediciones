"""
Línea base REPORTADA: lo que ya entregaste (Excel de cada notebook + sus pares) llevado al esquema estándar de las 3 bases, para compararlo con lo que produce el motor.

    from mediciones.reportado import cargar_reportado, comparar_en_bigquery
    cargar_reportado(RAIZ, 'surtido_motor', r'...\\Cosecha_Brechas_202609.xlsx', '2026-09-01')                  # prepara y valida; guarda copia local (no toca BigQuery)
    cargar_reportado(RAIZ, 'surtido_motor', r'...\\Cosecha_Brechas_202609.xlsx', '2026-09-01', cargar=True)     # además crea `mediciones_reportado` y la carga
    comparar_en_bigquery(cli, 'mediciones_reportado', 'mediciones', 'surtido_motor')                             # reportado vs corrida del motor (cargada con MODO='cargar')

Cómo se lleva lo reportado al esquema estándar (mismos pasos que el motor, para que la diferencia sea del cálculo y no del formato):
  1. Se leen las hojas de toneladas y brecha del Excel y los pares (df_matches.csv, df_sin_pareja.csv) de `data/<programa>/outputs/<carpeta>/`: el Excel reportado NO trae los pares.
  2. Se ejecutan los scripts de salida de la familia del programa (familias/<familia>/): suman las filas repetidas de Inka Chips y arman la base de ESPEJOS desde los pares.
  3. Se normaliza y valida con normalizacion.py. Se etiqueta `corrida_id = reportado_AAAAMM`, `version = 1`, `config_hash = 'reportado'`.
Alcance: surtido (`cosechas`), Champion y CPB/CPM (`mensual`). En CPB/CPM la hoja 'Base ton + UB atribuibles' es el `df_base_completa` del motor (con `rol`, `motivo_sin_pareja` y `par_id`;
el atribuible viene como `ton_atribuible`) y la hoja 'Base brechas (par-mes)' trae los pares (`cod_accion`, `cod_control`): de ahí salen la celda (brecha % e IC80) y los espejos.
Los espejos reportados son DERIVADOS (de los pares y de las toneladas), no un archivo que hayas entregado; las toneladas y brechas sí son tal cual el Excel.
La UB de CPB/CPM se lleva tal cual la reportó el notebook (`ub_base_soles`, `ub_por_ton_ref`, `ub_atribuible_soles`) y se compara con la que calcula el motor con el mismo método (metodo_ub: notebook, ver ub.py).
Surtido y Champion no tienen UB en sus notebooks.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pandas as pd

from . import bigquery
from . import normalizacion as nz
from .ejecucion import DIR_PAQUETE
from .esquema import DATASET, T_BRE, T_ESP, T_TON
from .registro import cargar_registro
from .runner import _guardar_crudas

CARPETA_OUTPUTS = {'surtido_motor': 'Surtido Ideal - Medición', 'champion_motor': 'Champion Minimayorista - Medición'}
HOJAS_TON = ('Base toneladas (cliente-mes)', 'Base toneladas (cli-mes)', 'Base toneladas')
HOJAS_TON_MENSUAL = ('Base ton + UB atribuibles', 'Base ton')
HOJAS_BRE = ('Base brecha (par-mes)', 'Base brecha')          # también 'Base brechas (par-mes)' (coincide por prefijo)
HOJAS_MATCHES = ('Matches',)                                   # Champion las escribe en su Excel; surtido no
HOJAS_SIN_PAREJA = ('Sin pareja (en muestra)', 'Sin pareja')
SCRIPTS_DE_SALIDA = ('salida', 'exportar')                  # de cada familia solo se ejecutan estos: no se recalcula nada
IDS = {'cod_cliente': str, 'cod_cliente_dex': str, 'cod_cliente_accion': str, 'cod_cliente_accion_dex': str, 'cod_cliente_control': str, 'cod_accion': str, 'cod_control': str}
SQL_COMPARACION = Path(__file__).resolve().parents[2] / 'sql' / 'comparar_reportado_vs_motor.sql'


def _elegir_hoja(hojas, candidatas, forzada=None):
    if forzada:
        if forzada not in hojas:
            raise ValueError(f'No existe la hoja {forzada!r}. Hojas del Excel: {hojas}')
        return forzada
    for c in candidatas:
        if c in hojas:
            return c
    for c in candidatas:                                                   # por prefijo ('Base brechas (par-mes)' empieza con 'Base brecha')
        h = next((x for x in hojas if x.startswith(c)), None)
        if h:
            return h
    raise ValueError(f'No encuentro una hoja entre {list(candidatas)}. Hojas del Excel: {hojas}')


def _hoja_opcional(hojas, candidatas):
    for c in candidatas:
        if c in hojas:
            return c
    return next((h for c in candidatas for h in hojas if h.startswith(c)), None)


def _exigir(df, cols, hoja, programa, ayuda=''):
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"[{programa}] La hoja {hoja!r} del Excel reportado no tiene las columnas {faltan}. Columnas que trae: {list(df.columns)}. "
                         f"{ayuda}¿Es el Excel del cierre correcto, generado por la versión actual del notebook?")


def _par(xl, explicito, hojas, defecto, dtype, programa):
    """Pares de lo reportado, por prioridad: ruta indicada > hoja del propio Excel (son los del cierre reportado) > CSV de data/<programa>/outputs."""
    if explicito:
        if not Path(explicito).exists():
            raise FileNotFoundError(f'No existe {explicito}')
        return pd.read_csv(explicito, dtype=dtype), f'CSV {explicito}'
    h = _hoja_opcional(xl.sheet_names, hojas)
    if h:
        return xl.parse(h, dtype=dtype), f'hoja {h!r} del Excel'
    if not Path(defecto).exists():
        raise FileNotFoundError(f'No existe {defecto} (los pares del reportado: df_matches.csv y df_sin_pareja.csv; pásalos con matches= / sin_pareja=, o usa un Excel que traiga las hojas {list(hojas)})')
    return pd.read_csv(defecto, dtype=dtype), f'CSV {defecto}'


def _ns_cosechas(raiz, programa, familia, xl, matches, sin_pareja, categorias, hoja_ton, hoja_brecha):
    """Variables que espera la salida de surtido/Champion: toneladas y brecha del Excel, y los pares (del Excel si los trae, o de data/<programa>/outputs/)."""
    carpeta = Path(raiz) / 'data' / programa / 'outputs' / CARPETA_OUTPUTS[programa]
    h_ton, h_bre = _elegir_hoja(xl.sheet_names, HOJAS_TON, hoja_ton), _elegir_hoja(xl.sheet_names, HOJAS_BRE, hoja_brecha)
    ton, bre = xl.parse(h_ton, dtype=IDS), xl.parse(h_bre, dtype=IDS)
    m, o_m = _par(xl, matches, HOJAS_MATCHES, carpeta / 'df_matches.csv', {'cod_accion': str, 'cod_control': str}, programa)
    sp, o_sp = _par(xl, sin_pareja, HOJAS_SIN_PAREJA, carpeta / 'df_sin_pareja.csv', {'cod_accion': str}, programa)
    _exigir(m, ['cosecha', 'des_categoria', 'cod_accion', 'cod_control'], 'Matches', programa)
    _exigir(sp, ['cosecha', 'des_categoria', 'cod_accion'], 'Sin pareja', programa)
    m['cosecha'], sp['cosecha'] = pd.to_datetime(m['cosecha']), pd.to_datetime(sp['cosecha'])
    unica = m['cosecha'].unique()
    for hoja, d in ((h_ton, ton), (h_bre, bre)):
        if 'cosecha_grupo' not in d.columns:
            alt = next((c for c in ('cosecha_agrupada', 'cosecha') if c in d.columns), None)
            if alt:
                d['cosecha_grupo'] = d[alt]
            elif len(unica) == 1:                                    # programa de una sola cosecha (Champion): es la de sus pares
                d['cosecha_grupo'] = unica[0]
            else:
                _exigir(d, ['cosecha_grupo'], hoja, programa, f'{programa} tiene varias cosechas ({len(unica)}) y no se puede deducir cuál es cuál. ')
        d['cosecha_grupo'] = pd.to_datetime(d['cosecha_grupo'])
    _exigir(ton, ['periodo', 'des_categoria', 'cod_cliente'], h_ton, programa)
    _exigir(bre, ['periodo', 'des_categoria'], h_bre, programa)
    for d in (ton, bre):
        d['periodo'] = pd.to_datetime(d['periodo'])
    etiquetas = dict(zip(bre['cosecha_grupo'], bre['cosecha_label'])) if 'cosecha_label' in bre.columns else {}
    ns = {'base_calculo_brecha': bre, 'df_matches': m, 'df_sin_pareja': sp, 'CATEGORIAS': list(categorias), 'GRUPO_COSECHA': etiquetas,
          'base_toneladas_export' if familia == 'champion' else 'base_calculo_toneladas': ton}
    if familia == 'champion':
        ns['base_calculo_toneladas'] = ton[ton['des_categoria'].isin(categorias)]
    ns['_origen_pares'] = f'matches: {o_m} · sin pareja: {o_sp}'
    return ns


def _ns_mensual(xl, hoja_ton, hoja_brecha):
    """
    Variables que espera la salida de CPB/CPM, reconstruidas de las dos hojas del Excel reportado:
      df_base_completa = 'Base ton + UB atribuibles' (con `rol`; `ton_atribuible` -> `valor_atribuible`)
      df_panel         = 'Base brechas (par-mes)' (con `periodo_str`)         df_matches = sus pares (cod_accion, cod_control)
      df_brechas       = la celda mes × categoría (brecha % e IC80), una fila por celda, de las columnas *_celda de la misma hoja
    """
    ton = xl.parse(_elegir_hoja(xl.sheet_names, HOJAS_TON_MENSUAL, hoja_ton), dtype=IDS)
    bre = xl.parse(_elegir_hoja(xl.sheet_names, HOJAS_BRE, hoja_brecha), dtype=IDS)
    for nombre, d, cols in (('toneladas', ton, ('rol', 'ton_atribuible' if 'valor_atribuible' not in ton.columns else 'valor_atribuible')),
                            ('brechas', bre, ('cod_accion', 'cod_control', 'did_kg', 'brecha_pct_celda', 'ic_pct_inf_80_celda', 'ic_pct_sup_80_celda'))):
        faltan = [c for c in cols if c not in d.columns]
        if faltan:
            raise ValueError(f'La hoja de {nombre} del Excel reportado no tiene las columnas {faltan}. Columnas: {list(d.columns)}')
    if 'valor_atribuible' not in ton.columns:
        ton = ton.rename(columns={'ton_atribuible': 'valor_atribuible'})
    ton['periodo'] = pd.to_datetime(ton['periodo'])
    dt = pd.to_datetime(bre['periodo'])
    bre = bre.assign(periodo=dt, periodo_str=dt.dt.strftime('%Y-%m'))
    celdas = (bre.groupby(['periodo_str', 'des_categoria'], as_index=False)
              .agg(brecha_pct=('brecha_pct_celda', 'first'), ic_pct_inf_80=('ic_pct_inf_80_celda', 'first'), ic_pct_sup_80=('ic_pct_sup_80_celda', 'first')))
    matches = bre[['periodo', 'des_categoria', 'cod_accion', 'cod_control']].drop_duplicates()
    panel = bre.drop(columns=['n_pares_celda', 'brecha_pct_celda', 'ic_pct_inf_80_celda', 'ic_pct_sup_80_celda'], errors='ignore')      # el exportador las vuelve a derivar de la celda
    cols_ub = [c for c in ('ub_base_soles', 'ub_por_ton_ref', 'ub_atribuible_soles') if c in ton.columns]
    ub = None
    if cols_ub:                                                    # el exportador no lleva la UB: se une después, tal cual la reportó el notebook
        ub = ton[ton['rol'].astype(str).str.startswith('accion')][['cod_cliente', 'des_categoria', 'periodo'] + cols_ub].copy()
        ub['cod_cliente'] = ub['cod_cliente'].astype(str).str.zfill(10)
        ub = ub.drop_duplicates(['cod_cliente', 'des_categoria', 'periodo'])
    return {'df_base_completa': ton, 'df_panel': panel, 'df_brechas': celdas, 'df_matches': matches, '_ub_reportado': ub}


def preparar_reportado(raiz, programa, excel, *, matches=None, sin_pareja=None, categorias=('Galletas', 'Cereales'), config=None,
                       hoja_ton=None, hoja_brecha=None, verbose=True):
    """Devuelve {'ton','bre','esp'}: las 3 bases ya como las deja el exportador del motor (aún sin normalizar), calculadas desde la salida REPORTADA."""
    raiz = Path(raiz)
    familias, registro = cargar_registro(Path(config) if config else raiz / 'config' / 'programas.yaml')
    if programa not in registro:
        raise ValueError(f'Programa {programa!r} no está en config/programas.yaml')
    prog = registro[programa]
    if prog.motor != 'familia' or prog.familia not in ('cosechas', 'champion', 'mensual'):
        raise NotImplementedError(f'{programa} (familia {prog.familia}): la línea base reportada está implementada para surtido, Champion, CPB y CPM.')
    if not Path(excel).exists():
        raise FileNotFoundError(f'No existe {excel}')
    xl = pd.ExcelFile(excel)
    if prog.familia == 'mensual':
        ns = _ns_mensual(xl, hoja_ton, hoja_brecha)
    else:
        ns = _ns_cosechas(raiz, programa, prog.familia, xl, matches, sin_pareja, categorias, hoja_ton, hoja_brecha)
    ns['__name__'] = '__main__'
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        for s in [x for x in familias[prog.familia]['scripts'] if x.rsplit('/', 1)[-1] in SCRIPTS_DE_SALIDA]:
            exec(compile((DIR_PAQUETE / 'familias' / prog.familia / f'{s}.py').read_text(encoding='utf-8'), s, 'exec'), ns)
    ub = ns.get('_ub_reportado')
    if ub is not None:
        ns['base_calculo_toneladas'] = ns['base_calculo_toneladas'].merge(ub, on=['cod_cliente', 'des_categoria', 'periodo'], how='left')
    if verbose and ns.get('_origen_pares'):
        print(f"[reportado:{programa}] pares tomados de -> {ns['_origen_pares']}")
    if verbose:
        print(f"[reportado:{programa}] toneladas {len(ns['base_calculo_toneladas']):,} · brechas {len(ns['base_calculo_brecha']):,} · espejos {len(ns['base_calculo_espejos']):,}")
    return {'ton': ns['base_calculo_toneladas'], 'bre': ns['base_calculo_brecha'], 'esp': ns['base_calculo_espejos'], 'regla_estado': familias[prog.familia].get('estado_yoy', 'estandar')}


def _preparar_linea_base(raiz, programa, excel, mes, *, matches=None, sin_pareja=None, config=None, verbose=True, dataset=None, **kw):
    """Prepara y valida las 3 bases de lo reportado (cierre `mes`, version 1, config_hash 'reportado') y deja la copia local en runs/reportado_AAAAMM/."""
    raiz = Path(raiz)
    rep, corrida_id = mes.date(), f'reportado_{mes:%Y%m}'
    raw = preparar_reportado(raiz, programa, excel, matches=matches, sin_pareja=sin_pareja, config=config, verbose=verbose, **kw)
    comunes = dict(version=1, config_hash='reportado')
    ton = nz.preparar_base_ton(raw['ton'], programa, rep, corrida_id, regla_estado=raw['regla_estado'], **comunes)
    bre = nz.preparar_base_brechas(raw['bre'], programa, rep, corrida_id, **comunes)
    esp = nz.preparar_base_espejos(raw['esp'], programa, rep, corrida_id, **comunes)
    errores, resumen = nz.validar(ton, bre, rep, esp=esp)
    ruta_local = raiz / 'runs' / corrida_id                          # reportado_AAAAMM: NO empieza con AAAAMM_, para que `EJECUTAR = False` nunca lo tome por una corrida del motor
    _guardar_crudas(raiz, ruta_local.name, programa, raw['ton'], raw['bre'], raw['esp'])
    if verbose:
        print(f"[reportado:{programa}] atribuible {resumen.get('valor_atribuible_tn_total')} t · errores de validación: {errores or 'ninguno'} · copia local: {ruta_local}")
    out = dict(programa=programa, corrida_id=corrida_id, errores=errores, resumen=resumen, ruta_local=str(ruta_local), cargado=False, dataset=dataset)
    return out, ((T_TON, ton), (T_BRE, bre), (T_ESP, esp))


def sembrar_linea_base(raiz, programa, excel, mes_cierre, *, matches=None, sin_pareja=None, cargar=False, cli=None, config=None, verbose=True, **kw):
    """
    Publica lo REPORTADO como línea base en el dataset del motor (`mediciones`), como el cierre `mes_cierre` (version 1, config_hash 'reportado').
    Desde ahí el motor solo agrega meses posteriores (frontera de meses cerrados, ver frontera.py y README.md):
    lo reportado queda fijo en brechas, toneladas atribuibles y UB.
    Una sola vez por programa: se niega si el programa ya tiene algo publicado en `mediciones`. Sin `cargar=True` solo prepara, valida y muestra qué cargaría.
    """
    mes = pd.Timestamp(mes_cierre)
    if mes.day != 1:
        raise ValueError(f'mes_cierre debe ser día 1 (recibí {mes.date()})')
    out, tablas = _preparar_linea_base(raiz, programa, excel, mes, matches=matches, sin_pareja=sin_pareja, config=config, verbose=verbose,
                                       dataset=DATASET, **kw)
    out['filas_a_sembrar'] = {t: len(df) for t, df in tablas}
    if not cargar:
        return out
    if out['errores']:
        raise ValueError(f'La línea base reportada tiene errores de validación; no se siembra: {out["errores"]}')
    cli = cli or bigquery.cliente_bq()
    bigquery.actualizar_esquema(cli, ejecutar=True)                  # tablas y vistas vigentes al día (la vista aplica la regla de frontera)
    ya = {t: bigquery.contar_programa(cli, t, programa) for t, _ in tablas}
    if any(ya.values()):
        raise RuntimeError(f'{programa} ya tiene filas publicadas en {DATASET} ({ya}): la línea base solo se siembra una vez. '
                           'Para corregir un cierre publicado usa reproceso (con motivo); para reabrir meses, reabrir_desde.')
    for t, df in tablas:                                             # las 3 tablas se revisan ANTES de cargar la primera: nada queda a medias
        bigquery.verificar_carga(cli, df, t)
    out['filas_cargadas'] = {t: bigquery.cargar_df(cli, df, t) for t, df in tablas}
    out['cargado'], out['dataset'] = True, DATASET
    if verbose:
        print(f'[siembra:{programa}] línea base {out["corrida_id"]} publicada en {DATASET}: {out["filas_cargadas"]}')
    return out


def completar_ub_linea_base(raiz, programa, cierre='2026-09-01', *, cargar=False, cli=None, verbose=True, reemplazar=False):
    """
    Completa la UB atribuible de una línea base que se publicó SIN UB (surtido y Champion: sus notebooks no la calculaban), con el mismo método
    de CPB/CPM (`metodo_ub` y `negocios_ub` de programas.yaml). Toma las filas publicadas de ese cierre, les calcula la UB y las vuelve a publicar
    como versión siguiente del MISMO cierre: toneladas, valor atribuible y llaves quedan idénticos (se verifica antes de cargar). Brechas y espejos
    no se tocan. Se niega si esas filas ya tienen UB, salvo `reemplazar=True` (corrige una UB ya publicada: recalcula y publica la versión siguiente).
    Sin `cargar=True` solo calcula y muestra.
    """
    import numpy as np
    from datetime import datetime, timezone
    from . import ub as _ub
    raiz = Path(raiz)
    prog = cargar_registro(raiz / 'config' / 'programas.yaml')[1][programa]
    metodo = getattr(prog, 'metodo_ub', None) or 'actual'
    cli = cli or bigquery.cliente_bq()
    ton = bigquery.leer_rebanada(cli, T_TON, programa, cierre)
    if ton.empty:
        raise ValueError(f'{programa} no tiene nada publicado en el cierre {cierre}')
    ub_previa = float(ton['ub_atribuible_soles'].sum()) if 'ub_atribuible_soles' in ton else 0.0
    if 'ub_atribuible_soles' in ton and ton['ub_atribuible_soles'].notna().any() and not reemplazar:
        raise RuntimeError(f'{programa} ({cierre}) ya tiene UB en {int(ton["ub_atribuible_soles"].notna().sum()):,} filas: no se completa '
                           f'(si es una corrección, usa reemplazar=True)')
    t = ton.copy()
    t['periodo'] = pd.to_datetime(t['periodo']).dt.date
    u = _ub.consulta_ub(cli, t, getattr(prog, 'negocios_ub', None), meses_atras=12 if metodo == 'notebook' else 0)
    con, cobertura = _ub.aplicar_ub(t, u, metodo=metodo)
    # guardas: mismas filas, mismas llaves y mismo atribuible (solo cambian las columnas de UB)
    k = ['cod_cliente', 'des_categoria', 'periodo']
    a, b = ton.assign(periodo=pd.to_datetime(ton['periodo'])).sort_values(k), con.assign(periodo=pd.to_datetime(con['periodo'])).sort_values(k)
    if len(a) != len(b) or not (a[k].astype(str).values == b[k].astype(str).values).all() \
            or not np.allclose(a['valor_atribuible'].astype(float).values, b['valor_atribuible'].astype(float).values, equal_nan=True):
        raise RuntimeError('el cálculo de UB alteró filas, llaves o el atribuible: no se carga')
    if cobertura.get('filas_ub_con_atribuible_cero', 0):
        raise RuntimeError(f"{cobertura['filas_ub_con_atribuible_cero']:,} filas con UB atribuible y 0 toneladas atribuibles: la UB no es coherente con el atribuible; no se carga")
    con['version'] = int(ton['version'].max()) + 1
    import re as _re
    con['corrida_id'] = _re.sub(r'_ub\d*$', '', str(ton['corrida_id'].iloc[0])) + ('_ub' if int(con['version'].iloc[0]) == 2 else f"_ub{int(con['version'].iloc[0])}")
    con['fecha_carga'] = datetime.now(timezone.utc)
    for c in ('periodo_reporte', 'config_hash', 'programa'):
        con[c] = ton[c].iloc[0]
    out = {'programa': programa, 'cierre': str(pd.Timestamp(cierre).date()), 'metodo_ub': metodo, 'filas': len(con),
           'ton_atribuible': round(float(con['valor_atribuible'].sum()), 3),
           'ub_atribuible_soles': round(float(con['ub_atribuible_soles'].sum()), 0), 'ub_publicada_antes': round(ub_previa, 0), 'cobertura': cobertura,
           'version_nueva': int(con['version'].iloc[0]), 'cargado': False}
    if cargar:
        bigquery.verificar_carga(cli, con, T_TON)
        bigquery.cargar_df(cli, con, T_TON)
        out['cargado'] = True
    if verbose:
        print(f"[UB:{programa}] {out['filas']:,} filas · {out['ton_atribuible']:,} t · UB S/ {out['ub_atribuible_soles']:,.0f}"
              + (f" (publicada antes: S/ {ub_previa:,.0f})" if ub_previa else '') + " · "
              f"cobertura {cobertura} · {'publicada como versión ' + str(out['version_nueva']) if cargar else 'vista previa (cargar=True para publicar)'}")
    return out


def cargar_reportado(raiz, programa, excel, mes_cierre, *, matches=None, sin_pareja=None, dataset='mediciones_reportado', cargar=False, cli=None, config=None, verbose=True, **kw):
    """
    Prepara y valida la línea base reportada y guarda una copia local en runs/reportado_AAAAMM/ (con la misma forma que las corridas del motor: sirve para
    `validar_corrida(corrida='reportado_AAAAMM')`). Con `cargar=True` crea el dataset gemelo `dataset` y carga las 3 bases (versión 1). No carga dos veces la misma línea base.
    """
    raiz = Path(raiz)
    mes = pd.Timestamp(mes_cierre)
    if mes.day != 1:
        raise ValueError(f'mes_cierre debe ser día 1 (recibí {mes.date()})')
    out, tablas = _preparar_linea_base(raiz, programa, excel, mes, matches=matches, sin_pareja=sin_pareja, config=config, verbose=verbose, dataset=dataset, **kw)
    if not cargar:
        return out
    errores, corrida_id = out['errores'], out['corrida_id']
    if errores:
        raise ValueError(f'La línea base reportada tiene errores de validación; no se carga: {errores}')
    cli = cli or bigquery.cliente_bq()
    bigquery.crear_dataset(cli, dataset)
    ya = {t: bigquery.contar_corrida(cli, t, programa, corrida_id, dataset) for t, _ in tablas}
    if any(ya.values()):
        raise RuntimeError(f'La línea base {corrida_id} de {programa} ya está cargada en {dataset} ({ya}); no se carga dos veces.')
    out['filas_cargadas'] = {t: bigquery.cargar_df(cli, df, t, dataset=dataset) for t, df in tablas}
    out['cargado'] = True
    if verbose:
        print(f'[reportado:{programa}] cargado en {dataset}: {out["filas_cargadas"]}')
    return out


def inspeccionar_excel(ruta):
    """
    Hojas, filas aproximadas y columnas de un Excel, sin cargarlo entero (lee solo el encabezado). Útil para ver qué trae un reportado antes de cargarlo
    y para diagnosticar `cargar_reportado` cuando dice que falta una hoja o una columna.
    """
    import openpyxl
    wb = openpyxl.load_workbook(ruta, read_only=True)
    try:
        filas = []
        for ws in wb.worksheets:
            primera = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
            filas.append({'hoja': ws.title, 'filas': ws.max_row, 'columnas': [c for c in primera if c is not None]})
    finally:
        wb.close()
    return pd.DataFrame(filas, columns=['hoja', 'filas', 'columnas'])


def _consultas():
    """Divide sql/comparar_reportado_vs_motor.sql en {nombre: sql} por las marcas `-- @nombre`."""
    partes, actual = {}, None
    for linea in SQL_COMPARACION.read_text(encoding='utf-8').split('\n'):
        if linea.startswith('-- @'):
            actual = linea[4:].strip()
            partes[actual] = []
        elif actual:
            partes[actual].append(linea)
    return {k: '\n'.join(v).strip().rstrip(';') for k, v in partes.items()}


def sql_comparacion(dataset_reportado, dataset_motor, proyecto=None):
    """Las consultas de comparación con los datasets ya sustituidos (por si prefieres correrlas a mano en la consola; el parámetro de programa es @p)."""
    from .esquema import PROYECTO
    pr = proyecto or PROYECTO
    return {k: q.replace('{rep}', f'{pr}.{dataset_reportado}').replace('{mot}', f'{pr}.{dataset_motor}') for k, q in _consultas().items()}


def verificar_comparacion(cli, dataset_reportado, dataset_motor, programa):
    """
    Revisa ANTES de comparar que todo lo necesario exista y devuelve la lista de problemas (vacía = listo), cada uno con el paso que lo resuelve:
    los dos datasets, la vista vigente de toneladas con las columnas de UB, y filas del programa en ambos lados.
    """
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery as bq
    from . import esquema as esq
    from .esquema import PROYECTO
    problemas = []
    paso_motor = ("corre el orquestador con MODO = 'cargar'" if dataset_motor == esq.DATASET
                  else f"carga la corrida del motor desde el orquestador con cargar_corrida_prueba(corrida, dataset='{dataset_motor}', cargar=True)")
    lados = (('reportado', dataset_reportado, 'corre el paso 2 de notebooks/comparar_reportado.ipynb (CARGAR = True)'),
             ('motor', dataset_motor, paso_motor))
    for etiqueta, ds, paso in lados:
        try:
            cli.get_dataset(f'{PROYECTO}.{ds}')
        except NotFound:
            problemas.append(f'El dataset {ds} ({etiqueta}) no existe: {paso}.')
            continue
        for vista, esperadas in (('vw_base_ton_atribuibles_vigente', esq.COLS_TON), ('vw_base_brechas_vigente', esq.COLS_BRE), ('vw_base_espejos_vigente', esq.COLS_ESP)):
            try:
                reales = {f.name for f in cli.get_table(f'{PROYECTO}.{ds}.{vista}').schema}
            except NotFound:
                problemas.append(f'{ds}.{vista} no existe: ejecuta sql/ddl_mediciones.sql en ese dataset.')
                continue
            faltan = [c for c in esperadas if c not in reales]
            if faltan:
                problemas.append(f'{ds}.{vista} no tiene las columnas {faltan}: la tabla se creó con una versión anterior del DDL (re-ejecutar sql/ddl_mediciones.sql solo agrega columnas, no renombra). '
                                 f"Mira exactamente qué difiere con bigquery.diagnosticar_esquema(cli, '{ds}').")
        if any(p.startswith(ds + '.') for p in problemas):
            continue
        cfg = bq.QueryJobConfig(query_parameters=[bq.ScalarQueryParameter('p', 'STRING', programa)])
        n = int(list(cli.query(f'SELECT COUNT(*) AS n FROM `{PROYECTO}.{ds}.vw_base_ton_atribuibles_vigente` WHERE programa = @p', job_config=cfg).result())[0]['n'])
        if n == 0:
            problemas.append(f'{ds} existe pero no tiene filas de {programa!r}: {paso}.')
    return problemas


def comparar_en_bigquery(cli, dataset_reportado, dataset_motor, programa, *, verbose=True):
    """Ejecuta las comparaciones reportado vs motor para un programa y devuelve {nombre: DataFrame}. Si falta algo, lo dice (RuntimeError) antes de consultar."""
    from google.cloud import bigquery as bq
    problemas = verificar_comparacion(cli, dataset_reportado, dataset_motor, programa)
    if problemas:
        raise RuntimeError(f'No se puede comparar {programa}:\n  - ' + '\n  - '.join(problemas))
    cfg = bq.QueryJobConfig(query_parameters=[bq.ScalarQueryParameter('p', 'STRING', programa)])
    out = {}
    for nombre, q in sql_comparacion(dataset_reportado, dataset_motor).items():
        out[nombre] = cli.query(q, job_config=cfg).to_dataframe()
        if verbose:
            print(f'[comparación:{programa}] {nombre}: {len(out[nombre]):,} filas')
    return out
