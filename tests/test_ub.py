import datetime as dt

import numpy as np
import pandas as pd

from mediciones import normalizacion as nz
from mediciones import ub
from mediciones.esquema import COLS_TON


def _ton():
    f = lambda **k: dict(dict(cod_cliente='0000000001', cod_cliente_dex='0000000001', des_categoria='Galletas', des_oficina='X',
                              des_segmento_transaccional='1', nom_compania='D', flg_cliente_perfecto=np.nan, cosecha_grupo=np.nan,
                              periodo=pd.Timestamp('2026-09-01'), mes_rel=np.nan, yoy_pct_raw=np.nan, ic_pct_inf_80=np.nan,
                              ic_pct_sup_80=np.nan, flg_estado_yoy='x', crec_atrib_80_pct=np.nan, ton_atrib_80=np.nan,
                              flg_estimado_retroactivo=False), **k)
    return pd.DataFrame([
        f(toneladas_ant=1.0, toneladas_actual=2.0, valor_atribuible=0.5),                                  # antiguo: prorrata
        f(cod_cliente='0000000002', cod_cliente_dex='0000000002', toneladas_ant=0.0, toneladas_actual=1.0, valor_atribuible=0.0),   # nuevo
        f(cod_cliente='0000000003', cod_cliente_dex='0000000003', toneladas_ant=1.0, toneladas_actual=0.0, valor_atribuible=0.0),   # sin compra
        f(cod_cliente='000C000993', cod_cliente_dex='000C000993', toneladas_ant=1.0, toneladas_actual=1.0, valor_atribuible=3.0),   # atribuible > compra, id con letras
    ])


def _ub(base):
    return pd.DataFrame({'periodo': base.periodo, 'cod_cliente': base.cod_cliente, 'des_categoria': base.des_categoria,
                         'venta_actual_soles': base.toneladas_actual * 1000.0, 'venta_sin_ratio_soles': 0.0,
                         'ub_actual_soles': base.toneladas_actual * 300.0, 'ton_detalle': base.toneladas_actual})


def test_reglas_de_ub():
    ton = nz.preparar_base_ton(_ton(), 'surtido', dt.date(2026, 9, 1), 'c')
    out, cob = ub.aplicar_ub(ton, _ub(ton))
    assert list(out.columns) == COLS_TON
    assert out.ub_atribuible_soles.iloc[0] == 600.0 * 0.5 / 2.0          # prorrata: ub_actual x valor / actual
    assert out.ub_atribuible_soles.iloc[1] == out.ub_actual_soles.iloc[1]   # nuevo: UB completa
    assert out.ub_atribuible_soles.iloc[2] == 0.0                         # sin compra y sin atribuible
    assert out.ub_atribuible_soles.iloc[3] == 300.0 * 3.0                 # sin tope: puede superar la UB actual
    assert cob['pct_cobertura_ub'] == 100.0 and cob['ton_detalle_vs_base'] == 1.0


def test_sql_valido_con_y_sin_negocio():
    sqlglot = __import__('pytest').importorskip('sqlglot')
    for neg in (False, True):
        sqlglot.parse_one(ub.sql_ub(neg), read='bigquery')


# ══ Método 'notebook' (CPB, CPM): UB base por tonelada, como el notebook ══
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

LEGADO = Path(__file__).parent / 'legado'
REP = dt.date(2026, 9, 1)


def _ton_cpb(filas):
    """Base de toneladas normalizada (regla estándar de CPB/CPM) a partir de tuplas (cod, categoría, periodo, ant, act, valor)."""
    f = lambda cod, cat, per, ant, act, val: dict(
        cod_cliente=cod, cod_cliente_dex=cod, des_categoria=cat, des_oficina='X', des_segmento_transaccional='1', nom_compania='D', flg_cliente_perfecto=np.nan,
        cosecha_grupo=np.nan, periodo=pd.Timestamp(per), mes_rel=np.nan, yoy_pct_raw=np.nan, ic_pct_inf_80=np.nan, ic_pct_sup_80=np.nan, flg_estado_yoy='x',
        crec_atrib_80_pct=np.nan, ton_atrib_80=np.nan, flg_estimado_retroactivo=False, toneladas_ant=ant, toneladas_actual=act, valor_atribuible=val)
    return nz.preparar_base_ton(pd.DataFrame([f(*x) for x in filas]), 'cpb', REP, 'c')


def _ub_con_base(ton, ub_base):
    """Lo que devolvería `consulta_ub(meses_atras=12)`: la UB del mes actual y la de hace un año (que es la UB base del mes actual)."""
    actual = pd.DataFrame({'periodo': ton.periodo, 'cod_cliente': ton.cod_cliente, 'des_categoria': ton.des_categoria, 'venta_actual_soles': 1.0, 'venta_sin_ratio_soles': 0.0,
                           'ub_actual_soles': ton.toneladas_actual * 777.0, 'ton_detalle': ton.toneladas_actual})
    hace_un_ano = pd.DataFrame({'periodo': (pd.to_datetime(ton.periodo) - pd.DateOffset(years=1)).dt.date, 'cod_cliente': ton.cod_cliente, 'des_categoria': ton.des_categoria,
                                'venta_actual_soles': 1.0, 'venta_sin_ratio_soles': 0.0, 'ub_actual_soles': np.asarray(ub_base), 'ton_detalle': 0.0})
    return pd.concat([actual, hace_un_ano], ignore_index=True)


def _filas_aleatorias(seed):
    rng = np.random.default_rng(seed)
    filas = []
    for per in ('2026-07-01', '2026-08-01', '2026-09-01'):
        for cat in ('Aceites', 'Galletas'):
            for i in range(40):
                nuevo = rng.random() < 0.2
                ant = 0.0 if nuevo else float(rng.uniform(0.001, 0.05))
                act = float(rng.uniform(0.001, 0.05)) if (nuevo or rng.random() > 0.2) else 0.0
                if nuevo:
                    valor = act
                elif act == 0:
                    valor = 0.0
                else:
                    valor = np.nan if rng.random() < 0.1 else ant * float(rng.uniform(-0.06, 0.03))      # sin banda: el atribuible queda vacío
                filas.append((f'{i + 1:010d}', cat, per, ant, act, valor))
    return filas


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_ub_notebook_identica_a_la_formula_del_notebook(seed):
    filas = _filas_aleatorias(seed)
    ton = _ton_cpb(filas)
    rng = np.random.default_rng(seed + 100)
    ub_base = np.where(rng.random(len(ton)) < 0.25, 0.0, rng.uniform(-3, 600, len(ton)) * (ton.toneladas_ant.to_numpy() + 0.01))     # algunos sin UB base (se rellenan con 0)
    out, cob = ub.aplicar_ub(ton, _ub_con_base(ton, ub_base), metodo='notebook')
    accion = pd.DataFrame({'periodo': pd.to_datetime(ton.periodo), 'des_categoria': ton.des_categoria, 'cod_cliente': ton.cod_cliente, 'flg_estado_yoy': ton.flg_estado_yoy,
                           'toneladas_ant': ton.toneladas_ant, 'toneladas_actual': ton.toneladas_actual, 'valor_atribuible': ton.valor_atribuible, 'ub_base_soles': ub_base})
    ns = {'accion': accion, 'np': np, 'pd': pd}
    exec(compile((LEGADO / 'ub_notebook.py').read_text(encoding='utf-8'), 'ub_notebook', 'exec'), ns)
    old = ns['accion'].reset_index(drop=True)
    pd.testing.assert_series_equal(old['ub_atrib_80_soles'], out['ub_atribuible_soles'].reset_index(drop=True), check_names=False, check_exact=True)
    pd.testing.assert_series_equal(old['ub_por_ton_ref'], out['ub_por_ton_ref'].reset_index(drop=True), check_names=False, check_exact=True)
    pd.testing.assert_series_equal(accion['ub_base_soles'].reset_index(drop=True), out['ub_base_soles'].reset_index(drop=True), check_names=False, check_exact=True)
    assert set(ton.flg_estado_yoy) == {'antiguo', 'nuevo'} and out['ub_atribuible_soles'].isna().any() and (out['ub_atribuible_soles'] == 0).any()   # el dato ejercita todas las ramas


def test_ub_notebook_reproduce_las_filas_reportadas_de_cpb():
    """Filas reales del Excel reportado (2026-01, Aceites): ub_atribuible = ub_base x ton_atribuible / toneladas_ant."""
    filas = [('0000012625', 'Aceites', '2026-01-01', 0.004944, 0.009888, 1.58208E-05),
             ('0000016351', 'Aceites', '2026-01-01', 0.015384, 0.003572, -0.000936886),
             ('0000016352', 'Aceites', '2026-01-01', 0.01044, 0.000458, -0.000635796),
             ('0000012372', 'Aceites', '2026-01-01', 0.009888, 0.010712, 3.16416E-05)]
    ton = _ton_cpb(filas)
    base = [-0.934851656, -1.146035549, -1.134531368, 0.0]
    out, _ = ub.aplicar_ub(ton, _ub_con_base(ton, base), metodo='notebook')
    assert np.allclose(out.ub_atribuible_soles.to_numpy(), [-0.002991525, 0.069793565, 0.06909296, 0.0], rtol=1e-6, atol=1e-9)    # las entradas reportadas traen ~9 dígitos
    assert list(out.columns) == COLS_TON and np.allclose(out.ub_base_soles.to_numpy(), base)


def test_ub_notebook_nuevos_usan_la_ub_por_tonelada_de_su_celda_y_sin_antiguos_es_cero():
    ton = _ton_cpb([('0000000001', 'Aceites', '2026-01-01', 0.010, 0.020, 0.0002), ('0000000002', 'Aceites', '2026-01-01', 0.020, 0.010, -0.0004),
                    ('0000000003', 'Aceites', '2026-01-01', 0.0, 0.030, 0.030), ('0000000004', 'Galletas', '2026-01-01', 0.0, 0.050, 0.050)])
    out, _ = ub.aplicar_ub(ton, _ub_con_base(ton, [10.0, 50.0, 0.0, 0.0]), metodo='notebook')
    ref = (10.0 + 50.0) / (0.010 + 0.020)                                           # S/ por TN de los antiguos de la celda mes x categoría
    assert np.isclose(out.ub_por_ton_ref.iloc[0], ref) and np.isclose(out.ub_atribuible_soles.iloc[2], 0.030 * ref)
    assert np.isnan(out.ub_por_ton_ref.iloc[3]) and out.ub_atribuible_soles.iloc[3] == 0.0       # celda sin antiguos: el nuevo no recibe UB (se rellena con 0)


def test_metodo_ub_desconocido_falla_claro():
    ton = _ton_cpb([('0000000001', 'Aceites', '2026-01-01', 0.01, 0.02, 0.001)])
    with pytest.raises(ValueError, match='metodo_ub'):
        ub.aplicar_ub(ton, _ub_con_base(ton, [1.0]), metodo='otro')


def test_consulta_ub_pide_los_meses_de_hace_un_ano_solo_cuando_se_le_indica():
    pytest.importorskip('google.cloud.bigquery')
    visto = {}

    class Cli:
        def query(self, q, job_config=None):
            visto['desde'] = {p.name: p.value for p in job_config.query_parameters if hasattr(p, 'value')}['desde']
            return self

        def to_dataframe(self):
            return pd.DataFrame({'periodo': [], 'cod_cliente': [], 'des_categoria': []})

    ton = _ton_cpb([('0000000001', 'Aceites', '2026-03-01', 0.01, 0.02, 0.001)])
    ub.consulta_ub(Cli(), ton, ['CP1'])
    assert visto['desde'] == dt.date(2026, 3, 1)
    ub.consulta_ub(Cli(), ton, ['CP1'], meses_atras=12)
    assert visto['desde'] == dt.date(2025, 3, 1)


def test_metodo_notebook_con_estados_de_cosechas():
    """Surtido/Champion no usan 'antiguo'/'nuevo': el método notebook decide por si el cliente tenía toneladas el año anterior."""
    import pandas as pd
    from mediciones import ub
    from mediciones.esquema import COLS_TON
    per = pd.Timestamp('2026-09-01').date()
    ton = pd.DataFrame({'programa': 'surtido_motor', 'cod_cliente': ['1', '2', '3', '4'], 'des_categoria': 'Galletas', 'periodo': per,
                        'toneladas_ant': [2.0, 0.0, 0.0, 0.0], 'toneladas_actual': [3.0, 1.0, 0.0, 2.0], 'valor_atribuible': [0.5, 1.0, 0.0, 0.0],
                        'flg_estado_yoy': ['normal', 'no_existia_periodo_pasado', 'sin_compra_ambos', 'reactivado']}).reindex(columns=COLS_TON)
    base = pd.Timestamp('2025-09-01').date()
    u = pd.DataFrame({'cod_cliente': ['1', '1', '2'], 'des_categoria': 'Galletas', 'periodo': [base, per, per],
                      'venta_actual_soles': 1.0, 'venta_sin_ratio_soles': 0.0, 'ub_actual_soles': [40.0, 70.0, 25.0], 'ton_detalle': 1.0})
    con, _ = ub.aplicar_ub(ton, u, metodo='notebook')
    v = con.set_index('cod_cliente')['ub_atribuible_soles'].round(6).to_dict()
    # 1: tenía año anterior -> 0.5 t x (40 / 2.0 S/ por TN base) = 10 · 2: sin año anterior -> 1.0 t x referencia (40/2 = 20) = 20 · 3: sin compra -> 0
    # 4: reactivado (compra 2 t pero se le atribuyen 0 t): 0 de UB atribuible. Antes se calculaba sobre toneladas_actual (2 t x 20 = 40) y quedaba inflada.
    assert v == {'0000000001': 10.0, '0000000002': 20.0, '0000000003': 0.0, '0000000004': 0.0} or v == {'1': 10.0, '2': 20.0, '3': 0.0, '4': 0.0}
    assert _cobertura_ok(con)


def _cobertura_ok(con):
    return int(((con['valor_atribuible'].abs() < 1e-12) & (con['ub_atribuible_soles'].abs() > 1e-9)).sum()) == 0
