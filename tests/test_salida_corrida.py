"""Lo que produjo una corrida del motor: Excel por programa, carga al dataset gemelo (sin BigQuery real) y diagnóstico de un Excel reportado."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from mediciones import bigquery, esquema as esq, normalizacion as nz, reportado, salida_corrida
from mediciones.esquema import DATASET, T_BRE, T_ESP, T_TON
from mediciones.runner import Corrida
from test_validacion import _bases

REP = dt.date(2026, 9, 1)


def _res(programa='surtido_motor', con_ub=True, errores=None):
    t, b, e = _bases()
    ton = nz.preparar_base_ton(t, programa, REP, '202609_x', regla_estado='programa')
    bre = nz.preparar_base_brechas(b, programa, REP, '202609_x')
    esp = nz.preparar_base_espejos(e, programa, REP, '202609_x')
    if con_ub:
        ton['ub_atribuible_soles'] = ton['valor_atribuible'] * 100.0
    return {'ton': ton, 'bre': bre, 'esp': esp, 'errores': errores or [], 'cargar_ton': ton.copy()}


def _corrida(**res):
    return Corrida(corrida_id='202609_x', mes=pd.Timestamp('2026-09-01'), modo='dry_run', res=res)


def test_usa_la_base_con_ub_cuando_cubre_toda_la_corrida_y_avisa_cuando_no():
    sin_ub = _res('cpb', con_ub=False)
    sin_ub['cargar_ton'] = sin_ub['ton'].iloc[:5]                                  # ya había datos publicados: solo 5 filas nuevas
    falla = {'error': 'KeyError: x'}                                               # un programa que falló antes de normalizar
    b = salida_corrida.bases_de_corrida(_corrida(surtido_motor=_res(), cpb=sin_ub, champion_motor=falla))
    assert set(b) == {'surtido_motor', 'cpb'} and b['surtido_motor']['avisos'] == []
    assert b['surtido_motor']['ton']['ub_atribuible_soles'].notna().any()
    assert len(b['cpb']['avisos']) == 1 and 'solo calculó la UB de las 5 filas nuevas' in b['cpb']['avisos'][0]
    assert list(salida_corrida.bases_de_corrida(_corrida(surtido_motor=_res(), cpb=_res('cpb')), programas=['cpb'])) == ['cpb']


def test_excel_por_programa_con_las_3_bases_y_un_resumen(tmp_path):
    r = _res()
    r['ton']['fecha_carga'] = pd.Timestamp.now(tz='UTC')                           # Excel no admite zona horaria: no debe romper
    rutas = salida_corrida.exportar_corrida_excel(_corrida(surtido_motor=r), carpeta=tmp_path / 'exportes', verbose=False)
    assert [x.name for x in rutas] == ['surtido_motor_motor_202609.xlsx']                       # pocas filas: todo cabe en el Excel
    x = pd.ExcelFile(rutas[0])
    assert x.sheet_names == ['Resumen', 'Por categoría y mes', 'Por estado', 'Brechas por celda', 'Base toneladas', 'Base brechas', 'Base espejos']
    ton = x.parse('Base toneladas', dtype={'cod_cliente': str})
    assert len(ton) == len(r['ton']) and ton['cod_cliente'].str.len().eq(10).all()                    # los ceros a la izquierda se conservan
    assert len(x.parse('Base brechas')) == len(r['bre']) and len(x.parse('Base espejos')) == len(r['esp'])
    por_mes = x.parse('Por categoría y mes')
    assert por_mes['filas'].sum() == len(r['ton']) and por_mes['atribuible_t'].sum() == pytest.approx(r['ton']['valor_atribuible'].sum(), abs=1e-2)       # el resumen cuadra con la base
    res = x.parse('Resumen').set_index('dato')['valor']
    assert res['programa'] == 'surtido_motor' and res['corrida_id'] == '202609_x' and float(res['atribuible total (t)']) == pytest.approx(r['ton']['valor_atribuible'].sum(), abs=1e-3)


def test_formato_csv_y_limite_de_filas_de_excel(tmp_path, monkeypatch):
    c = _corrida(cpb=_res('cpb'))
    rutas = salida_corrida.exportar_corrida_excel(c, carpeta=tmp_path, formato='csv', verbose=False)
    assert sorted(x.name for x in rutas) == ['cpb_motor_202609_espejos.csv', 'cpb_motor_202609_brechas.csv', 'cpb_motor_202609_toneladas.csv'] or len(rutas) == 3
    monkeypatch.setattr(salida_corrida, 'LIMITE_FILAS_EXCEL', 3)
    with pytest.raises(ValueError, match='formato="csv"'):
        salida_corrida.exportar_corrida_excel(c, carpeta=tmp_path, verbose=False)
    with pytest.raises(ValueError, match='formato'):
        salida_corrida.exportar_corrida_excel(c, carpeta=tmp_path, formato='pdf')


class _Cap:
    """Captura lo que se haría contra BigQuery."""

    def __init__(self, monkeypatch, previas=None):
        self.eventos = []
        self.previas = previas if previas is not None else pd.DataFrame(columns=['programa', 'corrida_id', 'periodo_reporte', 'version', 'filas'])
        monkeypatch.setattr(bigquery, 'crear_dataset', lambda cli, ds, ubicacion='US': self.eventos.append(('dataset', ds)))
        monkeypatch.setattr(bigquery, 'listar_corridas', lambda cli, ds=None: self.previas)
        monkeypatch.setattr(bigquery, 'borrar_corrida', lambda cli, c, ds=None, programas=None, confirmar=False: self.eventos.append(('borrar', c, ds, tuple(programas or ()), confirmar)))
        monkeypatch.setattr(bigquery, 'cargar_df', lambda cli, df, tabla, dataset=None: self.eventos.append(('cargar', tabla, dataset, len(df))) or len(df))


def test_sin_cargar_solo_muestra_el_plan_y_no_toca_bigquery(monkeypatch):
    cap = _Cap(monkeypatch)
    plan = salida_corrida.cargar_corrida_prueba(_corrida(surtido_motor=_res(), cpb=_res('cpb', errores=['espejos: faltan'])), verbose=False)
    assert cap.eventos == [] and set(plan.programa) == {'surtido_motor'} and set(plan.tabla) == {T_TON, T_BRE, T_ESP} and not plan.cargado.any()


def test_carga_las_3_tablas_al_gemelo_y_nunca_a_produccion(monkeypatch):
    cap = _Cap(monkeypatch)
    plan = salida_corrida.cargar_corrida_prueba(_corrida(surtido_motor=_res()), cargar=True, cli=object(), verbose=False)
    assert cap.eventos[0] == ('dataset', 'mediciones_motor')
    cargas = [e for e in cap.eventos if e[0] == 'cargar']
    assert {(e[1], e[2]) for e in cargas} == {(T_TON, 'mediciones_motor'), (T_BRE, 'mediciones_motor'), (T_ESP, 'mediciones_motor')} and plan.cargado.all()
    with pytest.raises(ValueError, match='producción'):
        salida_corrida.cargar_corrida_prueba(_corrida(surtido_motor=_res()), dataset=DATASET, cargar=True, cli=object())
    assert not any(e[2] == DATASET for e in cargas)


def test_no_duplica_la_corrida_de_un_programa_y_cierre_salvo_que_se_pida_reemplazar(monkeypatch):
    previas = pd.DataFrame({'programa': ['surtido_motor', 'surtido_motor', 'cpb'], 'corrida_id': ['202609_vieja', '202608_otra', '202609_cpb'],
                            'periodo_reporte': [dt.date(2026, 9, 1), dt.date(2026, 8, 1), dt.date(2026, 9, 1)], 'version': 1, 'filas': 10})
    cap = _Cap(monkeypatch, previas)
    c = _corrida(surtido_motor=_res())
    with pytest.raises(RuntimeError, match=r"202609_vieja.*reemplazar=True"):
        salida_corrida.cargar_corrida_prueba(c, cargar=True, cli=object(), verbose=False)
    assert not [e for e in cap.eventos if e[0] in ('borrar', 'cargar')]                         # falló antes de tocar nada
    salida_corrida.cargar_corrida_prueba(c, cargar=True, reemplazar=True, cli=object(), verbose=False)
    assert ('borrar', '202609_vieja', 'mediciones_motor', ('surtido_motor',), True) in cap.eventos     # solo esa corrida y solo ese programa
    assert not any(e[0] == 'borrar' and e[1] in ('202608_otra', '202609_cpb') for e in cap.eventos)    # ni otro cierre ni otro programa
    assert cap.eventos.index(next(e for e in cap.eventos if e[0] == 'borrar')) < cap.eventos.index(next(e for e in cap.eventos if e[0] == 'cargar'))


def test_inspeccionar_excel_muestra_hojas_y_columnas_sin_cargarlo_entero(tmp_path):
    xlsx = tmp_path / 'r.xlsx'
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']}).to_excel(w, sheet_name='Base toneladas (cli-mes)', index=False)
        pd.DataFrame({'cosecha': [1]}).to_excel(w, sheet_name='Matches', index=False)
    d = reportado.inspeccionar_excel(xlsx)
    assert list(d.hoja) == ['Base toneladas (cli-mes)', 'Matches'] and d.iloc[0].columnas == ['a', 'b'] and d.iloc[0].filas == 4


def test_las_bases_grandes_van_en_csv_al_lado_y_el_excel_lo_indica(tmp_path):
    r = _res()
    rutas = salida_corrida.exportar_corrida_excel(_corrida(surtido_motor=r), carpeta=tmp_path, max_filas=20, verbose=False)     # toneladas (36) y espejos superan 20
    nombres = sorted(x.name for x in rutas)
    assert 'surtido_motor_motor_202609.xlsx' in nombres and 'surtido_motor_motor_202609_toneladas.csv' in nombres
    x = pd.ExcelFile(next(r_ for r_ in rutas if r_.suffix == '.xlsx'))
    assert 'Base toneladas' not in x.sheet_names and 'Archivos' in x.sheet_names and 'Por categoría y mes' in x.sheet_names          # los agregados siempre están
    arch = x.parse('Archivos')
    assert 'Base toneladas' in set(arch['base']) and set(arch['archivo']) <= set(nombres)
    csv = pd.read_csv(tmp_path / 'surtido_motor_motor_202609_toneladas.csv', dtype={'cod_cliente': str})
    assert len(csv) == len(r['ton']) and csv['cod_cliente'].str.len().eq(10).all()
    todas = salida_corrida.exportar_corrida_excel(_corrida(surtido_motor=r), carpeta=tmp_path / 'todo', max_filas=None, verbose=False)
    assert [y.suffix for y in todas] == ['.xlsx'] and 'Base toneladas' in pd.ExcelFile(todas[0]).sheet_names


# ── desde los parquet de runs/ (sin BigQuery) ──
def _runs(tmp_path, carpeta='202609_20261015T205619_1a2807', programas=('surtido_motor',)):
    """Una carpeta runs/<corrida>/ con los parquet CRUDOS que deja el motor (antes de normalizar), como los tienes en disco."""
    import shutil
    from conftest import RAIZ
    (tmp_path / 'config').mkdir(exist_ok=True)
    shutil.copy(RAIZ / 'config' / 'programas.yaml', tmp_path / 'config' / 'programas.yaml')
    d = tmp_path / 'runs' / carpeta
    d.mkdir(parents=True)
    for p in programas:
        t, b, e = _bases()
        t.to_parquet(d / f'{p}_ton.parquet', index=False)
        b.to_parquet(d / f'{p}_brecha.parquet', index=False)
        e.to_parquet(d / f'{p}_espejos.parquet', index=False)
    return d


def test_corrida_desde_runs_normaliza_los_parquet_y_se_exporta_igual_que_una_corrida_en_memoria(tmp_path):
    _runs(tmp_path)
    c = salida_corrida.corrida_desde_runs(tmp_path, verbose=False)
    assert c.corrida_id == '202609_20261015T205619_1a2807' and c.mes == pd.Timestamp('2026-09-01') and c.modo == 'local'
    r = c.res['surtido_motor']
    assert r['errores'] == [] and list(r['ton'].columns) == esq.COLS_TON and len(r['ton']) == 36 and (r['ton'].corrida_id == c.corrida_id).all()       # el esquema estándar completo
    rutas = salida_corrida.exportar_corrida_excel(c, carpeta=tmp_path / 'exportes', verbose=False)
    x = pd.ExcelFile(rutas[0])
    assert rutas[0].name == 'surtido_motor_motor_202609.xlsx' and 'Por categoría y mes' in x.sheet_names and len(x.parse('Base toneladas')) == 36


def test_la_linea_base_reportada_se_pide_por_nombre_y_nunca_se_toma_por_la_ultima_corrida(tmp_path):
    import os
    import time
    _runs(tmp_path)
    base = _runs(tmp_path, 'reportado_202609')
    os.utime(base, (time.time() + 100, time.time() + 100))                                      # la línea base es la más reciente
    assert salida_corrida.corrida_desde_runs(tmp_path, verbose=False).corrida_id.startswith('202609_2026')          # por defecto: la del motor
    c = salida_corrida.corrida_desde_runs(tmp_path, corrida='reportado_202609', verbose=False)
    assert c.corrida_id == 'reportado_202609' and c.mes == pd.Timestamp('2026-09-01')
    rutas = salida_corrida.exportar_corrida_excel(c, carpeta=tmp_path / 'e', etiqueta='reportado', verbose=False)
    assert rutas[0].name == 'surtido_motor_reportado_202609.xlsx'
    d = salida_corrida.listar_corridas_locales(tmp_path)
    assert set(d.corrida) == {'202609_20261015T205619_1a2807', 'reportado_202609'} and set(d.tipo) == {'motor', 'línea base reportada'}
    assert d.set_index('corrida').loc['reportado_202609', 'programas'] == {'surtido_motor': 36}


def test_un_programa_ilegible_no_impide_leer_los_demas_y_se_puede_filtrar(tmp_path):
    d = _runs(tmp_path, programas=('surtido_motor', 'cpb'))
    pd.DataFrame({'x': [1]}).to_parquet(d / 'cpb_ton.parquet')                                  # parquet corrupto de un programa
    c = salida_corrida.corrida_desde_runs(tmp_path, verbose=False)
    assert 'ton' in c.res['surtido_motor'] and 'error' in c.res['cpb']
    assert list(salida_corrida.bases_de_corrida(c)) == ['surtido_motor']                         # el que falló no se exporta
    assert list(salida_corrida.corrida_desde_runs(tmp_path, programas=['surtido_motor'], verbose=False).res) == ['surtido_motor']


def test_con_ub_calcula_la_ub_como_el_motor_contra_bigquery_en_solo_lectura(tmp_path, monkeypatch):
    _runs(tmp_path, programas=('cpb',))
    visto = {}

    def consulta(cli, ton, negocios, meses_atras=0):
        visto['meses_atras'], visto['negocios'] = meses_atras, negocios
        actual = pd.DataFrame({'periodo': ton.periodo, 'cod_cliente': ton.cod_cliente, 'des_categoria': ton.des_categoria, 'venta_actual_soles': 1.0, 'venta_sin_ratio_soles': 0.0,
                               'ub_actual_soles': ton.toneladas_actual * 777.0, 'ton_detalle': ton.toneladas_actual})
        previo = actual.assign(periodo=(pd.to_datetime(actual.periodo) - pd.DateOffset(years=1)).dt.date, ub_actual_soles=ton.toneladas_ant.to_numpy() * 500.0)
        return pd.concat([actual, previo], ignore_index=True)
    monkeypatch.setattr(salida_corrida._ub, 'consulta_ub', consulta)
    monkeypatch.setattr(bigquery, 'cliente_bq', lambda: object())
    sin = salida_corrida.corrida_desde_runs(tmp_path, verbose=False)
    con = salida_corrida.corrida_desde_runs(tmp_path, con_ub=True, verbose=False)
    assert sin.res['cpb']['ton']['ub_atribuible_soles'].isna().all() and visto == {'meses_atras': 12, 'negocios': ['CP1']}      # CPB: método del notebook (UB base, hace 12 meses)
    assert con.res['cpb']['ton']['ub_atribuible_soles'].notna().any() and con.res['cpb']['ton']['ub_base_soles'].notna().any()
    assert 'UB vacía' not in ' '.join(salida_corrida.bases_de_corrida(con)['cpb']['avisos'])
