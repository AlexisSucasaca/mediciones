"""
Pruebas DIFERENCIALES de metodologia/atribucion.py: el código anterior (copia literal en tests/legado/) y la función nueva deben dar exactamente lo mismo,
con datos aleatorios que incluyen los casos límite (sin banda, fecha de creación desconocida, ceros, reactivados con YoY infinito), y con la referencia real de surtido.
"""
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from mediciones.metodologia.atribucion import atribuir, estado_yoy

LEGADO = Path(__file__).parent / 'legado'
REFERENCIA = Path(__file__).parent.parent / 'referencias' / 'surtido_motor_202609_ton.parquet'


def _legado(nombre, **ns):
    """Ejecuta la copia literal del código anterior y devuelve su espacio de nombres."""
    ns.update(np=np, pd=pd)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile((LEGADO / f'{nombre}.py').read_text(encoding='utf-8'), nombre, 'exec'), ns)
    return ns


def _igual(a, b, nombre=''):
    assert_series_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_names=False, check_exact=True, obj=nombre)


def _banda(rng, n, p_sin=0.15):
    inf = pd.Series(rng.uniform(-60, 5, n))
    sup = pd.Series(rng.uniform(0, 90, n))
    inf[rng.random(n) < p_sin] = np.nan
    sup[rng.random(n) < p_sin] = np.nan
    return inf, sup


def _kg(rng, n, p_cero):
    return pd.Series(np.where(rng.random(n) < p_cero, 0.0, rng.uniform(1, 5000, n)))


def _panel_cosechas(seed, n=6000):
    rng = np.random.default_rng(seed)
    peso_ant, peso = _kg(rng, n, .35), _kg(rng, n, .30)
    periodo = pd.Timestamp('2026-06-01')
    periodo_ant = pd.Series(pd.Timestamp('2025-06-01'), index=range(n))
    fec = periodo_ant + pd.to_timedelta(rng.integers(-900, 300, n), unit='D')
    fec[rng.random(n) < .10] = pd.NaT                                                   # fecha de creación desconocida
    return peso_ant, peso, periodo_ant, fec, periodo


# ── cosechas: panel principal (banda IC sin sufijo) ──
@pytest.mark.parametrize('seed', [1, 2, 3])
def test_cosechas_panel_principal(seed):
    peso_ant, peso, periodo_ant, fec, _ = _panel_cosechas(seed)
    inf, sup = _banda(np.random.default_rng(seed + 10), len(peso))
    panel = pd.DataFrame({'peso_ant': peso_ant, 'peso': peso, 'toneladas_ant': peso_ant / 1000, 'toneladas_actual': peso / 1000,
                          'periodo_ant': periodo_ant, 'fec_creacion': fec, 'ic_pct_inf': inf, 'ic_pct_sup': sup})
    panel['toneladas_variacion'] = panel['toneladas_actual'] - panel['toneladas_ant']
    ns = _legado('cosechas_principal', panel=panel.copy(), sin_brecha=inf.isna() | sup.isna())
    old = ns['panel']
    new = atribuir('cliente_creado', panel['toneladas_ant'], panel['toneladas_actual'], inf, sup, fec_creacion=fec, periodo_ant=periodo_ant)
    _igual(old['flg_estado_yoy'], new['flg_estado_yoy'], 'estado')
    _igual(old['yoy_pct_raw'], new['yoy_pct_raw'], 'yoy')
    _igual(old['crecimiento_atribuible_pct'], new['crec_atrib_pct'], 'crecimiento')
    _igual(old['toneladas_atribuibles'], new['ton_atrib'], 'ton_atrib')
    assert set(new['flg_estado_yoy']) == {'normal', 'no_existia_periodo_pasado', 'reactivado', 'sin_compra_ambos'}      # el dato sí ejercita las 4 ramas


# ── cosechas: hueco retroactivo (banda IC80 promedio de mes_rel 0-2) ──
@pytest.mark.parametrize('seed', [4, 5])
def test_cosechas_retroactivo(seed):
    peso_ant, peso, periodo_ant, fec, _ = _panel_cosechas(seed, 4000)
    rng = np.random.default_rng(seed + 20)
    n = len(peso)
    retro = pd.DataFrame({'peso_ant_r': peso_ant, 'peso_actual_r': peso, 'toneladas_ant': peso_ant / 1000, 'toneladas_actual': peso / 1000,
                          'periodo_ant': periodo_ant, 'fec_creacion': fec, 'cosecha_grupo': rng.choice(pd.to_datetime(['2026-04-01', '2026-06-01', '2026-08-01']), n),
                          'des_categoria': rng.choice(['Galletas', 'Cereales'], n)})
    retro['toneladas_variacion'] = retro['toneladas_actual'] - retro['toneladas_ant']
    brech = pd.DataFrame([{'cosecha_grupo': pd.Timestamp(c), 'des_categoria': k, 'mes_rel': m, 'ic_pct_inf_80': rng.uniform(-40, 0), 'ic_pct_sup_80': rng.uniform(0, 70)}
                          for c in ('2026-04-01', '2026-06-01') for k in ('Galletas', 'Cereales') for m in (0, 1, 2)])        # la cosecha 08 queda sin banda
    ns = _legado('cosechas_retro', retro_pob=retro.copy(), df_brechas=brech)
    old = ns['retro_pob']
    new = atribuir('cliente_creado', old['toneladas_ant'], old['toneladas_actual'], old['ic_pct_inf_80'], old['ic_pct_sup_80'], fec_creacion=old['fec_creacion'], periodo_ant=old['periodo_ant'])
    _igual(old['flg_estado_yoy'], new['flg_estado_yoy'], 'estado')
    _igual(old['yoy_pct_raw'], new['yoy_pct_raw'], 'yoy')
    _igual(old['crecimiento_atribuible_pct'], new['crec_atrib_pct'], 'crecimiento')
    _igual(old['toneladas_atribuibles_80'], new['ton_atrib'], 'ton_atrib')


# ── cosechas: vistas (estado y YoY ya vienen de la base completa) + valor_atribuible final ──
@pytest.mark.parametrize('seed', [6, 7])
def test_cosechas_vistas_y_valor_final(seed):
    peso_ant, peso, periodo_ant, fec, _ = _panel_cosechas(seed)
    inf, sup = _banda(np.random.default_rng(seed + 30), len(peso))
    base = pd.DataFrame({'toneladas_ant': peso_ant / 1000, 'toneladas_actual': peso / 1000})
    est = estado_yoy('cliente_creado', base['toneladas_ant'], base['toneladas_actual'], fec, periodo_ant)
    yoy = pd.Series(np.where(base['toneladas_ant'] > 0, 100 * (base['toneladas_actual'] - base['toneladas_ant']) / base['toneladas_ant'],
                             np.where(est == 'reactivado', np.inf, np.where(est == 'no_existia_periodo_pasado', np.nan, 0.0))))
    vistas = base.assign(flg_estado_yoy=est, yoy_pct_raw=yoy, ic_pct_inf_80=inf, ic_pct_sup_80=sup)
    ns = _legado('cosechas_vistas', base_vistas=vistas.copy())
    old = ns['base_vistas']
    new = atribuir('cliente_creado', base['toneladas_ant'], base['toneladas_actual'], inf, sup, estado=est, yoy_pct_raw=yoy)
    _igual(old['crec_atrib_80'], new['crec_atrib_pct'], 'crec_80')
    _igual(old['ton_atrib_80'], new['ton_atrib'], 'ton_atrib_80')
    final = _legado('cosechas_valor', base_calculo_toneladas=old[['flg_estado_yoy', 'toneladas_actual', 'ton_atrib_80']].copy())['base_calculo_toneladas']
    _igual(final['valor_atribuible'], new['valor_atribuible'], 'valor_atribuible')


# ── mensual (CPB, CPM): nuevo = toneladas_ant == 0 ──
@pytest.mark.parametrize('seed', [8, 9, 10])
def test_mensual_accion(seed):
    rng = np.random.default_rng(seed)
    n = 5000
    accion = pd.DataFrame({'periodo': rng.choice(pd.to_datetime(['2026-07-01', '2026-08-01', '2026-09-01']), n), 'des_categoria': rng.choice(['Galletas', 'Cereales'], n),
                           'toneladas_ant': _kg(rng, n, .3) / 1000, 'toneladas_actual': _kg(rng, n, .3) / 1000})
    brech = pd.DataFrame([{'periodo': pd.Timestamp(p), 'des_categoria': c, 'ic_pct_inf_80': rng.uniform(-40, 0), 'ic_pct_sup_80': rng.uniform(0, 70)}
                          for p in ('2026-07-01', '2026-08-01') for c in ('Galletas', 'Cereales')])        # septiembre queda sin banda
    old = _legado('mensual_accion', accion=accion.copy(), df_brechas=brech)['accion']
    new = atribuir('sin_compra_mes_anterior', old['toneladas_ant'], old['toneladas_actual'], old['ic_pct_inf_80'], old['ic_pct_sup_80'])
    _igual(old['flg_estado_yoy'], new['flg_estado_yoy'], 'estado')
    _igual(old['yoy_pct_raw'], new['yoy_pct_raw'], 'yoy')
    _igual(old['crec_atrib_80_pct'], new['crec_atrib_pct'], 'crecimiento')
    _igual(old['ton_atrib_80'], new['ton_atrib'], 'ton_atrib')
    _igual(old['flg_sin_banda_ic'], new['flg_sin_banda_ic'], 'sin_banda')
    _igual(old['valor_atribuible'], new['valor_atribuible'], 'valor')


# ── la diferencia entre reglas, en un caso mínimo ──
def test_las_dos_reglas_difieren_solo_en_quien_no_compro_el_mes_anterior():
    ant, act = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0]), pd.Series([0.7, 0.4, 1.5, 0.0, 0.0])
    fec = pd.Series(pd.to_datetime(['2024-01-01', '2026-01-01', '2024-01-01', '2024-01-01', '2024-01-01']))      # el 2º cliente se creó después del año anterior
    per_ant = pd.Series(pd.Timestamp('2025-09-01'), index=range(5))
    banda = (pd.Series([-10.0] * 5), pd.Series([30.0] * 5))
    a = atribuir('cliente_creado', ant, act, *banda, fec_creacion=fec, periodo_ant=per_ant)
    b = atribuir('sin_compra_mes_anterior', ant, act, *banda)
    assert a.flg_estado_yoy.tolist() == ['reactivado', 'no_existia_periodo_pasado', 'normal', 'normal', 'sin_compra_ambos']
    assert b.flg_estado_yoy.tolist() == ['nuevo', 'nuevo', 'antiguo', 'antiguo', 'nuevo']
    assert a.valor_atribuible.tolist() == [0.0, 0.4, pytest.approx(0.3), 0.0, 0.0]          # reactivado = 0; no existía = todo
    assert b.valor_atribuible.tolist() == [0.7, 0.4, pytest.approx(0.3), 0.0, 0.0]          # ambos nuevos = todo
    assert a.valor_atribuible.sum() != b.valor_atribuible.sum() and (a.valor_atribuible - b.valor_atribuible).abs().sum() == pytest.approx(0.7)


def test_regla_desconocida_y_datos_faltantes():
    s = pd.Series([0.0])
    with pytest.raises(ValueError, match='regla_nuevo'):
        atribuir('otra', s, s, s, s)
    with pytest.raises(ValueError, match='fec_creacion'):
        atribuir('cliente_creado', s, s, s, s)


# ── datos REALES: la función reproduce lo que el notebook exportó en septiembre (Galletas y Cereales) ──
@pytest.mark.referencias
@pytest.mark.skipif(not REFERENCIA.exists(), reason='falta la referencia de surtido')
def test_reproduce_la_atribucion_real_del_notebook():
    r = pd.read_parquet(REFERENCIA)
    r = r[r.des_categoria.isin(['Galletas', 'Cereales'])].reset_index(drop=True)
    new = atribuir('cliente_creado', r['toneladas_ant'], r['toneladas_actual'], r['ic_pct_inf_80'], r['ic_pct_sup_80'], estado=r['flg_estado_yoy'])
    for col_ref, col_new in (('valor_atribuible', 'valor_atribuible'), ('ton_atrib_80', 'ton_atrib'), ('crec_atrib_80_pct', 'crec_atrib_pct')):
        ok = np.isclose(r[col_ref].fillna(-999), new[col_new].fillna(-999), rtol=0, atol=1e-9)
        assert ok.all(), f'{col_ref}: coincide {ok.mean():.4%} (filas distintas: {(~ok).sum():,})'
    assert abs(new['valor_atribuible'].sum() - r['valor_atribuible'].sum()) < 1e-6


# ══ PEGAMENTO: las regiones marcadas `# [atribucion:inicio X]` de los scripts REALES dan lo mismo que el código anterior ══
from mediciones.metodologia.atribucion import valor_atribuible  # noqa: E402

FAMILIAS_DIR = Path(__file__).parent.parent / 'src' / 'mediciones' / 'familias'


def _region(familia, nombre):
    L = (FAMILIAS_DIR / familia / 'metodologia.py').read_text(encoding='utf-8').split('\n')
    i = [k for k, l in enumerate(L) if l.startswith(f'# [atribucion:inicio {nombre}]')]
    j = [k for k, l in enumerate(L) if l.startswith(f'# [atribucion:fin {nombre}]')]
    assert len(i) == 1 and len(j) == 1, f'región {nombre!r} de {familia}: marcas {i}/{j}'
    return '\n'.join(L[i[0]:j[0] + 1])


def _correr(familia, nombre, ns):
    ns.update(np=np, pd=pd, atribuir=atribuir, estado_yoy=estado_yoy, valor_atribuible=valor_atribuible)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_region(familia, nombre), f'{familia}:{nombre}', 'exec'), ns)
    return ns


@pytest.mark.parametrize('seed', [11, 12])
def test_pegamento_cosechas_panel_principal(seed):
    peso_ant, peso, periodo_ant, fec, _ = _panel_cosechas(seed)
    inf, sup = _banda(np.random.default_rng(seed + 10), len(peso))
    panel = pd.DataFrame({'peso_ant': peso_ant, 'peso': peso, 'toneladas_ant': peso_ant / 1000, 'toneladas_actual': peso / 1000,
                          'periodo_ant': periodo_ant, 'fec_creacion': fec, 'ic_pct_inf': inf, 'ic_pct_sup': sup})
    panel['toneladas_variacion'] = panel['toneladas_actual'] - panel['toneladas_ant']
    old = _legado('cosechas_principal', panel=panel.copy(), sin_brecha=inf.isna() | sup.isna())['panel']
    ns = {'panel': panel.copy(), 'REGLA_NUEVO': 'cliente_creado'}
    _correr('cosechas', 'principal', ns)
    _correr('cosechas', 'principal_b', ns)          # entre ambas están las banderas, que no se tocaron
    new = ns['panel']
    for col in ('flg_estado_yoy', 'yoy_pct_raw', 'crecimiento_atribuible_pct', 'toneladas_atribuibles'):
        _igual(old[col], new[col], col)


@pytest.mark.parametrize('seed', [13, 14])
def test_pegamento_cosechas_retroactivo(seed):
    peso_ant, peso, periodo_ant, fec, _ = _panel_cosechas(seed, 4000)
    rng = np.random.default_rng(seed + 20)
    n = len(peso)
    retro = pd.DataFrame({'peso_ant_r': peso_ant, 'peso_actual_r': peso, 'toneladas_ant': peso_ant / 1000, 'toneladas_actual': peso / 1000, 'periodo_ant': periodo_ant,
                          'fec_creacion': fec, 'cosecha_grupo': rng.choice(pd.to_datetime(['2026-04-01', '2026-06-01', '2026-08-01']), n),
                          'des_categoria': rng.choice(['Galletas', 'Cereales'], n)})
    retro['toneladas_variacion'] = retro['toneladas_actual'] - retro['toneladas_ant']
    brech = pd.DataFrame([{'cosecha_grupo': pd.Timestamp(c), 'des_categoria': k, 'mes_rel': m, 'ic_pct_inf_80': rng.uniform(-40, 0), 'ic_pct_sup_80': rng.uniform(0, 70)}
                          for c in ('2026-04-01', '2026-06-01') for k in ('Galletas', 'Cereales') for m in (0, 1, 2)])
    old = _legado('cosechas_retro', retro_pob=retro.copy(), df_brechas=brech)['retro_pob']
    ns = {'retro_pob': retro.copy(), 'REGLA_NUEVO': 'cliente_creado'}
    _correr('cosechas', 'retro', ns)
    ns['retro_pob']['ic_pct_inf_80'] = old['ic_pct_inf_80'].to_numpy()       # la banda se une entre las dos regiones (código que no cambió)
    ns['retro_pob']['ic_pct_sup_80'] = old['ic_pct_sup_80'].to_numpy()
    _correr('cosechas', 'retro_b', ns)
    new = ns['retro_pob']
    for col in ('flg_estado_yoy', 'yoy_pct_raw', 'crecimiento_atribuible_pct', 'toneladas_atribuibles_80'):
        _igual(old[col], new[col], col)


def test_pegamento_cosechas_vistas_y_valor_final():
    peso_ant, peso, periodo_ant, fec, _ = _panel_cosechas(15)
    inf, sup = _banda(np.random.default_rng(45), len(peso))
    base = pd.DataFrame({'toneladas_ant': peso_ant / 1000, 'toneladas_actual': peso / 1000})
    est = estado_yoy('cliente_creado', base['toneladas_ant'], base['toneladas_actual'], fec, periodo_ant)
    yoy = pd.Series(np.where(base['toneladas_ant'] > 0, 100 * (base['toneladas_actual'] - base['toneladas_ant']) / base['toneladas_ant'],
                             np.where(est == 'reactivado', np.inf, np.where(est == 'no_existia_periodo_pasado', np.nan, 0.0))))
    vistas = base.assign(flg_estado_yoy=est, yoy_pct_raw=yoy, ic_pct_inf_80=inf, ic_pct_sup_80=sup)
    old = _legado('cosechas_vistas', base_vistas=vistas.copy())['base_vistas']
    old_val = _legado('cosechas_valor', base_calculo_toneladas=old[['flg_estado_yoy', 'toneladas_actual', 'ton_atrib_80']].copy())['base_calculo_toneladas']
    ns = {'base_vistas': vistas.copy(), 'REGLA_NUEVO': 'cliente_creado'}
    _correr('cosechas', 'vistas', ns)
    new = ns['base_vistas']
    _igual(old['crec_atrib_80'], new['crec_atrib_80'], 'crec_atrib_80')
    _igual(old['ton_atrib_80'], new['ton_atrib_80'], 'ton_atrib_80')
    ns2 = {'base_calculo_toneladas': new[['flg_estado_yoy', 'toneladas_actual', 'ton_atrib_80']].copy(), 'REGLA_NUEVO': 'cliente_creado'}
    _correr('cosechas', 'valor', ns2)
    _igual(old_val['valor_atribuible'], ns2['base_calculo_toneladas']['valor_atribuible'], 'valor_atribuible')


@pytest.mark.parametrize('seed', [16, 17])
def test_pegamento_mensual_accion(seed):
    rng = np.random.default_rng(seed)
    n = 5000
    accion = pd.DataFrame({'periodo': rng.choice(pd.to_datetime(['2026-07-01', '2026-08-01', '2026-09-01']), n), 'des_categoria': rng.choice(['Galletas', 'Cereales'], n),
                           'toneladas_ant': _kg(rng, n, .3) / 1000, 'toneladas_actual': _kg(rng, n, .3) / 1000})
    brech = pd.DataFrame([{'periodo': pd.Timestamp(p), 'des_categoria': c, 'ic_pct_inf_80': rng.uniform(-40, 0), 'ic_pct_sup_80': rng.uniform(0, 70)}
                          for p in ('2026-07-01', '2026-08-01') for c in ('Galletas', 'Cereales')])
    old = _legado('mensual_accion', accion=accion.copy(), df_brechas=brech)['accion']
    previa = accion.assign(toneladas_variacion=accion['toneladas_actual'] - accion['toneladas_ant']).merge(brech, on=['periodo', 'des_categoria'], how='left')   # lo que precede a la región
    ns = {'accion': previa, 'REGLA_NUEVO': 'sin_compra_mes_anterior'}
    _correr('mensual', 'accion', ns)
    new = ns['accion']
    for col in ('flg_estado_yoy', 'yoy_pct_raw', 'crec_atrib_80_pct', 'ton_atrib_80', 'flg_sin_banda_ic', 'valor_atribuible'):
        _igual(old[col], new[col], col)


def test_la_regla_sale_de_la_configuracion_y_se_inyecta_en_los_scripts(tmp_path):
    from mediciones.ejecucion import ejecutar_familia
    from mediciones.registro import cargar_registro
    familias, _ = cargar_registro(Path(__file__).parent.parent / 'config' / 'programas.yaml')
    assert familias['cosechas']['regla_nuevo'] == 'cliente_creado' and familias['mensual']['regla_nuevo'] == 'sin_compra_mes_anterior'
    (tmp_path / 'p.py').write_text("import pandas as pd\nton = pd.DataFrame({'regla': [REGLA_NUEVO]}); bre = pd.DataFrame()\n", encoding='utf-8')
    cfg = {'f': {'scripts': [], 'variable_mes': 'MES_X', 'salida_ton': 'ton', 'salida_brecha': 'bre', 'regla_nuevo': 'cliente_creado'}}
    r = ejecutar_familia('f', tmp_path / 'p.py', 'p', pd.Timestamp('2026-09-01'), tmp_path / 'wd', cfg, insumos='desactivado')
    assert r.ton['regla'].iloc[0] == 'cliente_creado'
