"""
Pruebas DIFERENCIALES de metodologia/inferencia.py: las funciones anteriores (copia literal en tests/legado/) y las movidas dan EXACTAMENTE lo mismo,
con el mismo generador aleatorio, para los dos métodos (por par: cosechas; por cliente: mensual).
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mediciones.metodologia import inferencia as inf
from mediciones.parametros import N_BOOT, NIVELES_IC, SEED_BOOT

LEGADO = Path(__file__).parent / 'legado'


def _legado(familia):
    ns = dict(np=np, pd=pd, zlib=zlib, N_BOOT=N_BOOT, SEED_BOOT=SEED_BOOT, NIVELES_IC=NIVELES_IC, N_MIN_PARES=30)
    exec(compile((LEGADO / f'inferencia_{familia}.py').read_text(encoding='utf-8'), f'inferencia_{familia}', 'exec'), ns)
    return ns


def _panel(seed):
    """Celdas cosecha × categoría × mes_rel de tamaños variados (algunas < 30 pares, que se descartan) y clientes que se repiten entre meses."""
    rng = np.random.default_rng(seed)
    filas = []
    for cos in ('2026-04-01', '2026-06-01'):
        for cat in ('Galletas', 'Cereales'):
            for mes_rel in (0, 1, 2, 3):
                n = int(rng.choice([12, 29, 30, 45, 120, 300]))
                clientes = rng.choice(np.arange(1, 400), n, replace=n > 399)
                pre = rng.uniform(0, 3000, n)
                pre[rng.random(n) < .05] = 0.0
                filas.append(pd.DataFrame({'cosecha_grupo': cos, 'des_categoria': cat, 'mes_rel': mes_rel, 'periodo': pd.Timestamp(cos) + pd.DateOffset(months=mes_rel),
                                           'cod_accion': [f'{c:010d}' for c in clientes], 'did_kg': rng.normal(25, 400, n), 'pre_accion': pre}))
    p = pd.concat(filas, ignore_index=True)
    p['periodo_str'] = p['periodo'].dt.strftime('%Y-%m')
    return p


@pytest.mark.parametrize('seed', [1, 2])
@pytest.mark.parametrize('cols', [['cosecha_grupo', 'des_categoria', 'mes_rel'], ['cosecha_grupo', 'des_categoria']])
def test_bootstrap_por_par_identico_al_codigo_anterior(seed, cols):
    p = _panel(seed)
    old = _legado('cosechas')['calcular_brechas_bootstrap'](p, cols, n_boot=300)
    new = inf.bootstrap_por_par(p, cols, n_boot=300)
    assert len(new) > 0
    assert_frame_equal(old, new, check_exact=True)
    n_min_dict = {('2026-04-01', 'Galletas'): 55}
    assert_frame_equal(_legado('cosechas')['calcular_brechas_bootstrap'](p, cols, n_boot=300, n_min_dict=n_min_dict), inf.bootstrap_por_par(p, cols, n_boot=300, n_min_dict=n_min_dict), check_exact=True)


@pytest.mark.parametrize('seed', [3, 4])
def test_bootstrap_por_cliente_identico_al_codigo_anterior(seed):
    p = _panel(seed)
    for cols in (['des_categoria', 'periodo_str'], ['cosecha_grupo', 'des_categoria', 'mes_rel'], []):               # [] = vista global
        old = _legado('mensual')['calcular_brechas_bootstrap'](p, cols, n_boot=300)
        new = inf.bootstrap_por_cliente(p, cols, n_boot=300)
        assert len(new) > 0
        assert_frame_equal(old, new, check_exact=True)


@pytest.mark.parametrize('seed', [5, 6])
def test_distribuciones_identicas_al_codigo_anterior(seed):
    p = _panel(seed)
    cols = ['cosecha_grupo', 'des_categoria', 'mes_rel']
    assert_frame_equal(_legado('cosechas')['calcular_distribucion_brechas'](p, cols), inf.distribucion_brechas_cosechas(p, cols), check_exact=True)
    assert_frame_equal(_legado('mensual')['calcular_distribucion_brechas'](p, cols), inf.distribucion_brechas_mensual(p, cols), check_exact=True)


def test_los_dos_metodos_siguen_siendo_distintos_y_lo_documentan():
    """Hoy NO se unifican: por par comparte un generador entre celdas (el IC depende de las demás celdas de la llamada); por cliente fija la semilla por celda."""
    p = _panel(7)
    cols = ['cosecha_grupo', 'des_categoria', 'mes_rel']
    tam = p.groupby(cols).size().reset_index(name='n')
    elegible = tam[tam.n >= 30].sort_values(cols).iloc[1:]                       # que no sea la primera celda: antes de ella el generador ya consumió sorteos
    cos, cat, mes = elegible.sort_values('n').iloc[-1][cols]
    una_sola = p[(p.cosecha_grupo == cos) & (p.des_categoria == cat) & (p.mes_rel == mes)]
    sel = lambda d: d[(d.cosecha_grupo == cos) & (d.des_categoria == cat) & (d.mes_rel == mes)].ic_pct_inf_80.iloc[0]
    assert sel(inf.bootstrap_por_par(p, cols, n_boot=300)) != sel(inf.bootstrap_por_par(una_sola, cols, n_boot=300))             # por par: depende de las otras celdas
    assert sel(inf.bootstrap_por_cliente(p, cols, n_boot=300)) == sel(inf.bootstrap_por_cliente(una_sola, cols, n_boot=300))     # por cliente: igual en cualquier vista
