"""Exportables desde lo publicado: toneladas, brechas (par-mes y celda), espejos con vigencia; vigente / histórico; rango; Excel."""
import pandas as pd
import pytest

from mediciones import salidas
from mediciones.esquema import T_BRE, T_ESP, T_TON


def _tec(df, rep, version=1):
    return df.assign(periodo_reporte=pd.Timestamp(rep), version=version, corrida_id=f'c{rep}', config_hash='h')


def _publicado():
    """CPB (mensual): ago-sep sembrados en el cierre 2026-09, oct en el cierre 2026-10; el espejo de A cambia en octubre.
       Surtido (cosechas): G04 provisional desde octubre, republicado en el cierre 2026-11 (versión preliminar de octubre reemplazada)."""
    ton, bre, esp = [], [], []
    for per, rep, esp_a in (('2026-08-01', '2026-09-01', 'E1'), ('2026-09-01', '2026-09-01', 'E1'), ('2026-10-01', '2026-10-01', 'E2')):
        ton.append(_tec(pd.DataFrame({'programa': 'cpb', 'cod_cliente': ['A', 'B'], 'des_categoria': 'Galletas', 'periodo': pd.Timestamp(per),
                                      'cosecha_grupo': None, 'valor_atribuible': [1.0, 2.0], 'ub_atribuible_soles': [10.0, 20.0],
                                      'flg_estado_yoy': 'antiguo'}), rep))
        bre.append(_tec(pd.DataFrame({'programa': 'cpb', 'par_id': ['p1', 'p2'], 'periodo': pd.Timestamp(per), 'des_categoria': 'Galletas',
                                      'cosecha_grupo': None, 'did_kg': [100.0, 300.0], 'brecha_pct_celda': 5.0, 'reportable': True,
                                      'ic_pct_inf_80_celda': 2.0, 'ic_pct_sup_80_celda': 8.0}), rep))
        esp.append(_tec(pd.DataFrame({'programa': 'cpb', 'cod_cliente_accion': ['A', 'B'], 'des_categoria': 'Galletas', 'periodo': pd.Timestamp(per),
                                      'cosecha_grupo': None, 'cod_cliente_espejo': [esp_a, 'F1'], 'flg_estimado_retroactivo': False}), rep))
    for rep, v in (('2026-10-01', 11.0), ('2026-11-01', 9.0)):                       # surtido G04 octubre, republicado en noviembre
        ton.append(_tec(pd.DataFrame({'programa': 'surtido_motor', 'cod_cliente': ['S1'], 'des_categoria': 'Cereales',
                                      'periodo': pd.Timestamp('2026-10-01'), 'cosecha_grupo': pd.Timestamp('2026-12-01'),
                                      'valor_atribuible': [v / 10], 'ic_pct_inf_80': 1.0, 'ic_pct_sup_80': v + 5}), rep))
        bre.append(_tec(pd.DataFrame({'programa': 'surtido_motor', 'par_id': ['q1'], 'periodo': pd.Timestamp('2026-10-01'),
                                      'des_categoria': 'Cereales', 'cosecha_grupo': pd.Timestamp('2026-12-01'), 'did_kg': [50.0],
                                      'brecha_pct_celda': v, 'reportable': True}), rep))
    return {T_TON: pd.concat(ton, ignore_index=True), T_BRE: pd.concat(bre, ignore_index=True), T_ESP: pd.concat(esp, ignore_index=True)}


def _leer(pub):
    def leer(tabla, p, desde, hasta, modo):
        d = pub[tabla][pub[tabla].programa == p]
        d = d[(d.periodo >= desde) & ((d.periodo <= hasta) if hasta is not None else True)]
        return salidas.vigente_local(d) if modo == 'vigente' else d
    return leer


class _Prog:
    def __init__(self, prov=None, cosechas=None, hasta=None):
        self.provisional_desde, self.provisional_cosechas, self.provisional_hasta = prov, cosechas, hasta


REG = {'cpb': _Prog(), 'surtido_motor': _Prog('2026-10-01')}


def test_bases_vigentes_con_fecha_de_reporte_y_estado(tmp_path):
    out = salidas.exportar(['cpb', 'surtido_motor'], desde='2026-01-01', modo='vigente', raiz=tmp_path, leer=_leer(_publicado()),
                           registro=REG, verbose=False)
    xl = pd.read_excel(out['cpb']['excel'], sheet_name=None)
    assert set(xl) == {'Toneladas', 'Brechas_par_mes', 'Brechas_celda', 'Espejos_vigencia', 'Diccionario'}
    ton = xl['Toneladas']
    assert {'periodo', 'periodo_reporte', 'estado_publicacion', 'valor_atribuible', 'ub_atribuible_soles'} <= set(ton.columns)
    rep = ton.groupby(pd.to_datetime(ton.periodo).dt.strftime('%Y-%m'))['periodo_reporte'].first().map(lambda x: pd.Timestamp(x).strftime('%Y-%m'))
    assert rep.to_dict() == {'2026-08': '2026-09', '2026-09': '2026-09', '2026-10': '2026-10'}       # fecha de reporte por mes
    assert set(ton.estado_publicacion) == {'definitivo'}
    sur = pd.read_excel(out['surtido_motor']['excel'], sheet_name='Brechas_celda')
    assert sur.estado_publicacion.tolist() == ['preliminar'] and sur.brecha_pct.tolist() == [9.0]       # el G04 vigente = el del cierre de noviembre (preliminar: antes de su cosecha)


def test_brechas_por_celda_resume_pares_brecha_ic_toneladas_y_ub():
    pub = _publicado()
    b = salidas.armar_bases('cpb', pub[T_TON][pub[T_TON].programa == 'cpb'], pub[T_BRE][pub[T_BRE].programa == 'cpb'], None)['Brechas_celda']
    sep = b[b.periodo == pd.Timestamp('2026-09-01')].iloc[0]
    assert (sep.pares, sep.brecha_pct, sep.ic80_inf, sep.ic80_sup, sep.did_t, sep.ton_atribuible, sep.ub_atribuible_soles) == (2, 5.0, 2.0, 8.0, 0.4, 3.0, 30.0)
    assert sep.cosecha_grupo == '—'
    s = salidas.armar_bases('surtido_motor', pub[T_TON][pub[T_TON].programa == 'surtido_motor'].iloc[[1]],
                            pub[T_BRE][pub[T_BRE].programa == 'surtido_motor'].iloc[[1]], None)['Brechas_celda'].iloc[0]
    assert (s.ic80_inf, s.ic80_sup) == (1.0, 14.0)                 # en cosechas la banda sale de la base de toneladas


def test_vigencia_de_espejos_arma_tramos_desde_hasta():
    pub = _publicado()
    v = salidas.armar_bases('cpb', None, None, pub[T_ESP])['Espejos_vigencia']
    a = v[v.cod_cliente_accion == 'A'][['cod_cliente_espejo', 'desde', 'hasta', 'meses', 'primer_cierre', 'ultimo_cierre']]
    assert a.cod_cliente_espejo.tolist() == ['E1', 'E2'] and a.meses.tolist() == [2, 1]
    assert a.desde.dt.strftime('%Y-%m').tolist() == ['2026-08', '2026-10'] and a.hasta.dt.strftime('%Y-%m').tolist() == ['2026-09', '2026-10']
    b = v[v.cod_cliente_accion == 'B']
    assert len(b) == 1 and b.meses.iloc[0] == 3                    # mismo espejo los 3 meses: un solo tramo


def test_rango_desde_hasta_filtra_por_mes_medido(tmp_path):
    out = salidas.exportar(['cpb'], desde='2026-09-01', hasta='2026-09-01', raiz=tmp_path, leer=_leer(_publicado()), registro=REG, verbose=False)
    ton = pd.read_excel(out['cpb']['excel'], sheet_name='Toneladas')
    assert set(pd.to_datetime(ton.periodo).dt.strftime('%Y-%m')) == {'2026-09'}
    assert out['cpb']['excel'].endswith('cpb_vigente_202609_202609.xlsx')
    with pytest.raises(ValueError, match='anterior'):
        salidas.exportar(['cpb'], desde='2026-10-01', hasta='2026-09-01', raiz=tmp_path, leer=_leer(_publicado()), registro=REG, verbose=False)


def test_historico_trae_todas_las_publicaciones_y_marca_la_vigente():
    pub = _publicado()
    b = salidas.armar_bases('surtido_motor', pub[T_TON][pub[T_TON].programa == 'surtido_motor'], pub[T_BRE][pub[T_BRE].programa == 'surtido_motor'],
                            None, modo='historico', provisional_desde='2026-10-01')
    cel = b['Brechas_celda'].sort_values('periodo_reporte')
    assert cel.brecha_pct.tolist() == [11.0, 9.0] and cel.vigente.tolist() == [False, True]
    assert b['Toneladas'].vigente.tolist() == [False, True]


def test_rango_por_defecto_es_el_anio_en_curso():
    d, h = salidas.rango_por_defecto(hoy='2026-10-24')
    assert d == pd.Timestamp('2026-01-01') and h is None


def test_excel_parte_hojas_que_superan_el_limite(tmp_path, monkeypatch):
    monkeypatch.setattr(salidas, 'LIMITE_FILAS_EXCEL', 3)
    ruta = salidas._a_excel({'Toneladas': pd.DataFrame({'x': range(7)})}, tmp_path / 'x.xlsx')
    hojas = pd.read_excel(ruta, sheet_name=None)
    assert [k for k in hojas if k.startswith('Toneladas')] == ['Toneladas_1', 'Toneladas_2', 'Toneladas_3']
    assert sum(len(hojas[k]) for k in hojas if k.startswith('Toneladas')) == 7


def test_la_vista_de_vigencia_de_espejos_esta_en_el_ddl():
    from conftest import RAIZ
    ddl = (RAIZ / 'sql' / 'ddl_mediciones.sql').read_text(encoding='utf-8')
    i = ddl.index('CREATE OR REPLACE VIEW `acpe-dev-uc-sandbox-aa.mediciones.vw_espejos_vigencia`')
    bloque = ddl[i:ddl.index(';', i)]
    for trozo in ('vw_base_espejos_vigente', 'ROW_NUMBER() OVER', 'MIN(periodo) AS desde', 'MAX(periodo) AS hasta', 'primer_cierre'):
        assert trozo in bloque


def test_estado_preliminar_provisional_y_definitivo_por_cosecha():
    d = pd.DataFrame({'periodo': pd.to_datetime(['2026-10-01', '2026-10-01', '2026-12-01', '2026-09-01']),
                      'cosecha_grupo': pd.to_datetime(['2026-12-01', '2026-04-01', '2026-12-01', '2026-12-01']),
                      'periodo_reporte': pd.to_datetime(['2026-11-01', '2026-10-01', '2026-12-01', '2026-09-01'])})
    e = salidas.estado_publicacion(d, '2026-10-01', ['2026-12-01'], '2027-02-01')
    assert e.tolist() == ['preliminar', 'definitivo', 'provisional', 'definitivo']    # G04 oct (antes de dic) | G01 | G04 dic | antes de oct
    vencido = d.assign(periodo_reporte=pd.Timestamp('2027-03-01'))
    assert set(salidas.estado_publicacion(vencido, '2026-10-01', ['2026-12-01'], '2027-02-01')) == {'definitivo'}


def test_vista_previa_agrega_el_mes_nuevo_y_reemplaza_lo_provisional_sin_cargar(tmp_path):
    from mediciones.runner import Corrida
    pub = _publicado()
    nuevo_ton = pd.DataFrame({'programa': 'surtido_motor', 'cod_cliente': ['S1', 'S2'], 'des_categoria': 'Cereales',
                              'periodo': pd.to_datetime(['2026-10-01', '2026-11-01']), 'cosecha_grupo': pd.Timestamp('2026-12-01'),
                              'valor_atribuible': [0.5, 0.7], 'ub_atribuible_soles': [5.0, 7.0], 'corrida_id': 'c', 'config_hash': 'h'})
    nuevo_bre = pd.DataFrame({'programa': 'surtido_motor', 'par_id': ['q1', 'q1'], 'periodo': pd.to_datetime(['2026-10-01', '2026-11-01']),
                              'des_categoria': 'Cereales', 'cosecha_grupo': pd.Timestamp('2026-12-01'), 'did_kg': [60.0, 70.0],
                              'brecha_pct_celda': [12.0, 13.0], 'reportable': True})
    co = Corrida('202611_x', pd.Timestamp('2026-11-01'), 'dry_run',
                 {'surtido_motor': {'cargar_ton': nuevo_ton, 'cargar_bre': nuevo_bre, 'cargar_esp': None}, 'cpb': {'error': 'falló'}})
    out = salidas.vista_previa(co, desde='2026-01-01', raiz=tmp_path, leer=_leer_historico(pub), registro=REG, verbose=False)
    assert 'cpb' not in out and out['surtido_motor']['meses_que_publicaria'] == ['2026-10', '2026-11']
    cel = pd.read_excel(out['surtido_motor']['excel'], sheet_name='Brechas_celda').sort_values('periodo')
    assert cel.brecha_pct.tolist() == [12.0, 13.0]                     # octubre del G04 reemplazado por esta corrida (antes 9.0)
    assert (pd.to_datetime(cel.periodo_reporte).dt.strftime('%Y-%m') == '2026-11').all()
    assert 'vista_previa_202611' in out['surtido_motor']['excel']
    # lo publicado no se tocó
    assert len(pub[T_BRE][pub[T_BRE].programa == 'surtido_motor']) == 2


def _leer_historico(pub):
    def leer(tabla, p, desde, hasta, modo):
        d = pub[tabla][pub[tabla].programa == p]
        return d[(d.periodo >= desde) & ((d.periodo <= hasta) if hasta is not None else True)].copy()
    return leer
