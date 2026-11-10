"""Calendario de cosechas de surtido (G01..G04) y su uso: G04 preliminar en oct-nov, definitivo desde dic; lo provisional por cosecha."""
import pandas as pd
import pytest

from mediciones import frontera
from mediciones.calendario import Calendario
from mediciones.registro import cargar_registro
from conftest import RAIZ

T = pd.Timestamp


def test_etapas_del_g04():
    c = Calendario()
    assert c.abierta('2026-09-01') == T('2026-08-01')                  # setiembre: el G03 todavía recibe altas
    for cierre in ('2026-10-01', '2026-11-01'):                          # preliminar: anclado en octubre, incremental
        assert c.abierta(cierre) == T('2026-10-01') and c.a_rehacer(cierre) == [] and c.a_descartar(cierre) == []
        assert c.alias(cierre) == {T('2026-10-01'): T('2026-12-01')}
        assert c.cosechas(cierre) == [T('2026-04-01'), T('2026-06-01'), T('2026-08-01'), T('2026-10-01')]
    assert c.abierta('2026-12-01') == T('2026-12-01')                   # definitivo: cosecha diciembre
    assert c.a_rehacer('2026-12-01') == [T('2026-12-01')] and c.a_descartar('2026-12-01') == [T('2026-10-01')]
    assert c.alias('2026-12-01') == {}
    assert c.abierta('2027-01-01') is None                              # sin grupo: nadie recibe altas
    assert c.cosechas('2026-09-01') == [T('2026-04-01'), T('2026-06-01'), T('2026-08-01')]   # el G04 aún no existe


def test_nombres_y_rangos():
    c = Calendario()
    n = c.nombres('2026-10-01')
    assert n[T('2026-10-01')] == 'GRUPO 04 (Oct-Dic) preliminar' and n[T('2026-12-01')] == 'GRUPO 04 (Oct-Dic)'
    assert c.rangos_12m('2026-12-01')['2026-12-01'] == (T('2025-12-01'), T('2026-11-01'))
    assert c.rangos_12m('2026-10-01')['2026-10-01'] == (T('2025-10-01'), T('2026-09-01'))


def test_sql_case_coincide_con_asignar():
    c = Calendario()
    sql = c.sql_case('min_periodo_cosecha', '2026-11-01')
    assert "BETWEEN DATE '2026-07-01' AND DATE '2026-09-01' THEN DATE '2026-08-01'" in sql
    assert "BETWEEN DATE '2026-10-01' AND DATE '2026-12-01' THEN DATE '2026-10-01'" in sql


def test_calendario_invalido():
    with pytest.raises(ValueError, match='superponen'):
        Calendario([{'grupo': 'A', 'altas_desde': '2026-01-01', 'altas_hasta': '2026-03-01', 'cosecha': '2026-03-01'},
                    {'grupo': 'B', 'altas_desde': '2026-03-01', 'altas_hasta': '2026-04-01', 'cosecha': '2026-04-01'}])


def test_el_calendario_de_la_configuracion_es_el_acordado():
    _, reg = cargar_registro(RAIZ / 'config' / 'programas.yaml')
    s = reg['surtido_motor']
    c = Calendario(s.calendario_cosechas)
    assert [(g.grupo, g.altas_desde, g.altas_hasta, g.cosecha, g.preliminar) for g in c.grupos] == [
        ('G01', T('2026-02-01'), T('2026-04-01'), T('2026-04-01'), False), ('G02', T('2026-05-01'), T('2026-06-01'), T('2026-06-01'), False),
        ('G03', T('2026-07-01'), T('2026-09-01'), T('2026-08-01'), False), ('G04', T('2026-10-01'), T('2026-12-01'), T('2026-12-01'), True)]
    assert s.provisional_desde == '2026-10-01' and s.provisional_cosechas == ['2026-12-01'] and s.provisional_hasta == '2027-02-01'


# ── lo provisional por cosecha ──
def _filas(periodos, cosechas, rep, v=1.0):
    return pd.DataFrame({'programa': 'surtido_motor', 'periodo': pd.to_datetime(periodos), 'cosecha_grupo': pd.to_datetime(cosechas),
                         'periodo_reporte': T(rep), 'version': 1, 'v': v})


def test_separar_republica_solo_la_cosecha_provisional():
    df = _filas(['2026-10-01', '2026-10-01', '2026-11-01'], ['2026-04-01', '2026-12-01', '2026-04-01'], '2026-11-01')
    pub, cer = frontera.separar(df, T('2026-10-01'), {'desde': '2026-10-01', 'cosechas': ['2026-12-01']})
    assert list(zip(pub.periodo.dt.month, pub.cosecha_grupo.dt.month)) == [(10, 12), (11, 4)]     # G04 de octubre se republica; G01 de octubre no
    assert list(zip(cer.periodo.dt.month, cer.cosecha_grupo.dt.month)) == [(10, 4)]


def test_vigente_reemplaza_la_cosecha_provisional_sin_tocar_las_demas():
    d = pd.concat([_filas(['2026-10-01', '2026-10-01'], ['2026-04-01', '2026-12-01'], '2026-10-01', v=1.0),
                   _filas(['2026-10-01'], ['2026-12-01'], '2026-11-01', v=2.0)], ignore_index=True)
    v = frontera.vigente_local(d).sort_values('cosecha_grupo')
    assert v.v.tolist() == [1.0, 2.0] and v.periodo_reporte.dt.month.tolist() == [10, 11]


def test_frontera_no_se_recorta_si_lo_provisional_es_por_cosecha_y_vence_con_hasta():
    pub = pd.DataFrame({'periodo': pd.to_datetime(['2026-09-01', '2026-10-01']), 'periodo_reporte': pd.to_datetime(['2026-09-01', '2026-10-01'])})
    assert frontera.calcular_frontera(pub, '2026-11-01', '2026-10-01', provisional_cosechas=['2026-12-01']) == T('2026-10-01')
    assert frontera.calcular_frontera(pub, '2026-11-01', '2026-10-01') == T('2026-09-01')            # sin cosechas: todo el programa
    assert frontera.provisional_activo('2027-02-01', '2026-10-01', '2027-02-01')
    assert not frontera.provisional_activo('2027-03-01', '2026-10-01', '2027-02-01')                 # vence: queda fijo


# ── verificación de pares: el G04 (provisional) no se verifica contra lo publicado ──
def test_los_pares_de_la_cosecha_provisional_no_se_verifican(tmp_path):
    from mediciones import pares
    pub = pd.DataFrame({'periodo': T('2026-10-01'), 'cosecha': [T('2026-04-01'), T('2026-12-01')], 'categoria': 'Galletas',
                        'accion': ['0000000001', '0000000002'], 'control': ['0000000011', '0000000012'], 'nivel_match': 1})
    ruta = tmp_path / 'df_matches.csv'
    # estado local: G01 igual; el G04 guardado con su ancla preliminar (octubre) -> sin excluir, la verificación lo daría por faltante
    pd.DataFrame({'cosecha': ['2026-04-01', '2026-10-01'], 'des_categoria': 'Galletas', 'cod_accion': ['1', '2'],
                  'cod_control': ['11', '99']}).to_csv(ruta, index=False)
    assert pares.verificar('cosechas', ruta, pub)['estado'] == 'incompleto'
    assert pares.verificar('cosechas', ruta, pares.excluir_cosechas(pub, ['2026-12-01']))['estado'] == 'ok'


def test_el_runner_excluye_las_cosechas_provisionales_antes_de_verificar():
    src = (RAIZ / 'src' / 'mediciones' / 'runner.py').read_text(encoding='utf-8')
    i = src.index('pares.excluir_cosechas(pub_pares, prog.provisional_cosechas)')
    assert src.index('frontera.provisional_activo(periodo_reporte, prog.provisional_desde, prog.provisional_hasta)') < i < src.index("v = pares.verificar(prog.pares['tipo'], ruta, pub_pares)")
