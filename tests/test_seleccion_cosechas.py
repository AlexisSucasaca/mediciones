"""Bloque real de la familia de cosechas que decide qué se empareja en cada cierre (G04: preliminar incremental -> definitivo completo)."""
import pandas as pd

from conftest import RAIZ
from mediciones.calendario import Calendario

T = pd.Timestamp


def _bloque():
    src = (RAIZ / 'src' / 'mediciones' / 'familias' / 'cosechas' / 'metodologia.py').read_text(encoding='utf-8')
    return src[src.index('# [seleccion_cosechas:inicio]'):src.index('# [seleccion_cosechas:fin]')]


def _correr(cierre, congelado, ultimo_cierre, clientes):
    cal, mes = Calendario(), T(cierre)
    cli_min_per = pd.Series({c: T(f) for c, f in clientes.items()})
    cli_cosecha = pd.Series({c: T(cal.asignar(pd.Series([f]), mes).iloc[0]) for c, f in clientes.items()})
    ns = {'pd': pd, 'REHACER_MATCHING': False, 'MES_CIERRE': mes, 'COSECHAS': cal.cosechas(mes), 'COSECHA_ABIERTA': cal.abierta(mes),
          'COSECHAS_REHACER': cal.a_rehacer(mes), 'COSECHAS_DESCARTAR': cal.a_descartar(mes), 'cli_min_per': cli_min_per, 'cli_cosecha': cli_cosecha,
          'cargar_matching_congelado': lambda: {k: v.copy() for k, v in congelado.items()}, 'leer_estado_matching': lambda: T(ultimo_cierre)}
    exec(_bloque(), ns)
    return ns


def _cong(cosechas):
    m = pd.DataFrame({'cosecha': [T(c) for c in cosechas], 'des_categoria': 'Galletas', 'cod_accion': [f'a{i}' for i in range(len(cosechas))],
                      'cod_control': [f'c{i}' for i in range(len(cosechas))]})
    return {'matches': m, 'sin_pareja': m.iloc[0:0][['cosecha', 'des_categoria', 'cod_accion']], 'resumen': m[['cosecha']].assign(categoria='Galletas')}


ALTAS = {'g03_set': '2026-09-01', 'g04_oct': '2026-10-01', 'g04_nov': '2026-11-01', 'g04_dic': '2026-12-01'}


def test_octubre_g04_preliminar_incremental_y_g03_cerrado():
    ns = _correr('2026-10-01', _cong(['2026-04-01', '2026-08-01']), '2026-09-01', ALTAS)
    assert ns['COSECHAS_A_CORRER'] == [T('2026-10-01')]                         # el G04 anclado en octubre
    assert ns['ENTRANTES_G03'] == {'g04_oct'} and ns['COSECHAS_REHACER_AHORA'] == []   # solo las altas de octubre; el G03 ya no recibe


def test_noviembre_suma_las_altas_de_noviembre():
    ns = _correr('2026-11-01', _cong(['2026-04-01', '2026-08-01', '2026-10-01']), '2026-10-01', ALTAS)
    assert ns['COSECHAS_A_CORRER'] == [T('2026-10-01')] and ns['ENTRANTES_G03'] == {'g04_nov'}
    assert (ns['CONGELADO']['matches']['cosecha'] == T('2026-10-01')).sum() == 1      # los preliminares de octubre se conservan


def test_diciembre_descarta_preliminares_y_empareja_completo():
    ns = _correr('2026-12-01', _cong(['2026-04-01', '2026-08-01', '2026-10-01', '2026-10-01']), '2026-11-01', ALTAS)
    assert ns['COSECHAS_REHACER_AHORA'] == [T('2026-12-01')] and ns['COSECHAS_A_CORRER'] == [T('2026-12-01')]
    assert ns['ENTRANTES_G03'] == set()                                               # completo, no incremental
    assert set(ns['CONGELADO']['matches']['cosecha']) == {T('2026-04-01'), T('2026-08-01')}   # preliminares fuera


def test_diciembre_corrido_de_nuevo_no_rehace_lo_definitivo():
    ns = _correr('2026-12-01', _cong(['2026-04-01', '2026-08-01', '2026-12-01']), '2026-12-01', ALTAS)
    assert ns['COSECHAS_A_CORRER'] == [] and (ns['CONGELADO']['matches']['cosecha'] == T('2026-12-01')).sum() == 1


def test_enero_sin_grupo_abierto_no_empareja():
    ns = _correr('2027-01-01', _cong(['2026-04-01', '2026-08-01', '2026-12-01']), '2026-12-01', {**ALTAS, 'ene': '2027-01-01'})
    assert ns['COSECHAS_A_CORRER'] == [] and ns['ENTRANTES_G03'] == set()
