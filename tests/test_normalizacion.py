import datetime as dt

import numpy as np
import pandas as pd
import pytest

from mediciones import normalizacion as nz
from mediciones.esquema import COLS_BRE, COLS_ESP, COLS_TON
from conftest import construir_crudas, construir_espejos

REP = dt.date(2026, 9, 1)


def _fila(ant, act, val):
    return pd.DataFrame([dict(cod_cliente='0000000001', cod_cliente_dex='0000000001', des_categoria='Galletas', des_oficina='X',
                              des_segmento_transaccional='1 Oro+', nom_compania='D', flg_cliente_perfecto=np.nan, cosecha_grupo=np.nan,
                              periodo=pd.Timestamp('2026-09-01'), mes_rel=np.nan, toneladas_ant=ant, toneladas_actual=act,
                              yoy_pct_raw=np.nan, ic_pct_inf_80=np.nan, ic_pct_sup_80=np.nan, flg_estado_yoy='x',
                              crec_atrib_80_pct=np.nan, ton_atrib_80=np.nan, flg_estimado_retroactivo=False, valor_atribuible=val)])


def test_estado_binario():
    est = nz.asignar_estado_yoy(pd.Series([1.0, 0.0, 0.0, np.nan, np.nan]), pd.Series([0.5, 0.3, 0.0, 0.2, 0.0]))
    assert est.fillna('nulo').tolist() == ['antiguo', 'nuevo', 'nulo', 'nuevo', 'nulo']


def test_regla_estandar_reactivado_pasa_a_nuevo_con_todo_su_volumen():
    out = nz.preparar_base_ton(_fila(0.0, 0.7, 0.0), 'cpb', REP, 'c')       # regla estándar (CPB/CPM); surtido/Champion usan la del programa
    assert out.flg_estado_yoy.iloc[0] == 'nuevo' and out.valor_atribuible.iloc[0] == 0.7
    assert out.kg_atribuibles.iloc[0] == 700.0


def test_inka_diferencial_con_signo_se_conserva():
    out = nz.preparar_base_ton(_fila(0.5, 0.2, -0.3), 'surtido', REP, 'c')       # compró menos que el año pasado
    assert out.flg_estado_yoy.iloc[0] == 'antiguo' and out.valor_atribuible.iloc[0] == -0.3
    out = nz.preparar_base_ton(_fila(0.5, 0.0, -0.5), 'surtido', REP, 'c')       # dejó de comprar
    assert out.valor_atribuible.iloc[0] == -0.5


def test_esquema_final_y_ids():
    ton, bre = construir_crudas(['2026-08-01', '2026-09-01'], n_clientes=5)
    ton['cod_cliente'] = ton['cod_cliente'].str.lstrip('0')                       # ids que perdieron los ceros
    n = nz.preparar_base_ton(ton, 'cpb', REP, 'c')
    assert list(n.columns) == COLS_TON and n.cod_cliente.str.len().eq(10).all()
    b = nz.preparar_base_brechas(bre, 'cpb', REP, 'c')
    assert list(b.columns) == COLS_BRE


def test_validar_detecta_problemas():
    ton, bre = construir_crudas(['2026-08-01', '2026-09-01'], n_clientes=5)
    n, b = nz.preparar_base_ton(ton, 'cpb', REP, 'c'), nz.preparar_base_brechas(bre, 'cpb', REP, 'c')
    err, res = nz.validar(n, b, REP)
    assert err == [] and res['clientes_ton'] == 5
    err, _ = nz.validar(pd.concat([n, n.head(3)]), b, REP)
    assert any('duplicadas' in e for e in err)
    err, _ = nz.validar(n, b, dt.date(2026, 8, 1))
    assert any('posteriores' in e for e in err)
    b2 = b.copy(); b2.loc[b2.index[0], 'cod_cliente_accion_dex'] = '9999999999'
    err, _ = nz.validar(n, b2, REP)
    assert any('sin fila' in e for e in err)


def test_id_del_control_se_normaliza_y_los_datos_de_celda_son_opcionales():
    ton, bre = construir_crudas(['2026-08-01'], n_clientes=3)
    sin = nz.preparar_base_brechas(bre.drop(columns='cod_cliente_control'), 'surtido', REP, 'c')   # el piloto no trae control ni datos de celda
    assert sin.cod_cliente_control.isna().all() and sin.n_pares_celda.isna().all()
    bre['cod_cliente_control'] = ['77', '88', np.nan, '99', '10', '11'][:len(bre)]
    con = nz.preparar_base_brechas(bre, 'cpb', REP, 'c')
    assert con.cod_cliente_control.dropna().str.len().eq(10).all() and con.cod_cliente_control.isna().sum() == 1


def _tres_bases():
    ton, bre = construir_crudas(['2026-08-01', '2026-09-01'], n_clientes=8, sin_espejo_frac=0.25)
    n, b = nz.preparar_base_ton(ton, 'cpb', REP, 'c'), nz.preparar_base_brechas(bre, 'cpb', REP, 'c')
    e = nz.preparar_base_espejos(construir_espejos(ton, bre), 'cpb', REP, 'c')
    return n, b, e


def test_espejos_esquema_estado_y_consistencia():
    n, b, e = _tres_bases()
    assert list(e.columns) == COLS_ESP and len(e) == len(n)                           # mismo grano que la base de toneladas
    assert set(e.estado_espejo) == {'con_espejo', 'sin_espejo'}
    assert e.loc[e.estado_espejo == 'con_espejo', 'motivo_sin_espejo'].isna().all()
    assert e.loc[e.estado_espejo == 'sin_espejo', 'motivo_sin_espejo'].notna().all()
    assert e.cod_cliente_espejo.dropna().str.len().eq(10).all() and str(e.nivel_match.dtype) == 'Int64'
    err, res = nz.validar(n, b, REP, esp=e)
    assert err == [] and 0 < res['pct_con_espejo'] < 100


def test_validar_detecta_espejos_incoherentes():
    n, b, e = _tres_bases()
    f = b.iloc[0]
    sin = e[~((e.cod_cliente_accion == f.cod_cliente_accion_dex) & (e.des_categoria == f.des_categoria) & (e.periodo == f.periodo))]
    err, _ = nz.validar(n, b, REP, esp=sin)
    assert any('sin espejo registrado' in x for x in err)
    otro = e.copy()
    k = (otro.cod_cliente_accion == f.cod_cliente_accion_dex) & (otro.des_categoria == f.des_categoria) & (otro.periodo == f.periodo)
    otro.loc[k, 'cod_cliente_espejo'] = '0000000042'
    err, _ = nz.validar(n, b, REP, esp=otro)
    assert any('difiere del control' in x for x in err)
    err, _ = nz.validar(n, b, REP, esp=pd.concat([e, e.head(2)]))
    assert any('duplicadas' in x for x in err)


def _fila_estado(flg, ant, act, val):
    f = _fila(ant, act, val)
    f['flg_estado_yoy'] = flg
    return f


def test_regla_programa_respeta_el_estado_y_el_atribuible_del_notebook():
    """Surtido/Champion: no_existia_periodo_pasado = todo; reactivado = 0 (existía, compró 0 ese mes del año pasado); normal = capado por el IC."""
    df = pd.concat([_fila_estado('no_existia_periodo_pasado', 0.0, 0.4, 0.4),
                    _fila_estado('reactivado', 0.0, 0.7, 0.0),
                    _fila_estado('normal', 1.0, 1.5, 0.2),
                    _fila_estado('sin_compra_ambos', 0.0, 0.0, 0.0)], ignore_index=True)
    out = nz.preparar_base_ton(df, 'surtido', REP, 'c', regla_estado='programa')
    assert out.flg_estado_yoy.tolist() == ['no_existia_periodo_pasado', 'reactivado', 'normal', 'sin_compra_ambos']
    assert out.valor_atribuible.tolist() == [0.4, 0.0, 0.2, 0.0]
    assert out.kg_atribuibles.tolist() == [400.0, 0.0, 200.0, 0.0]


def test_regla_estandar_sigue_atribuyendo_todo_a_los_nuevos():
    out = nz.preparar_base_ton(_fila_estado('reactivado', 0.0, 0.7, 0.0), 'cpb', REP, 'c')               # por defecto: estándar
    assert out.flg_estado_yoy.iloc[0] == 'nuevo' and out.valor_atribuible.iloc[0] == 0.7


def test_regla_programa_exige_que_el_programa_entregue_estado_y_atribuible():
    df = _fila(0.0, 0.7, 0.0).drop(columns=['flg_estado_yoy'])
    with pytest.raises(ValueError, match='flg_estado_yoy'):
        nz.preparar_base_ton(df, 'surtido', REP, 'c', regla_estado='programa')
    with pytest.raises(ValueError, match='regla_estado'):
        nz.preparar_base_ton(_fila(0.0, 0.7, 0.0), 'surtido', REP, 'c', regla_estado='otra')
