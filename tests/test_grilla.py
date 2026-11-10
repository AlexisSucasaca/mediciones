"""
Pruebas DIFERENCIALES de metodologia/ids.py y grilla.py: el código anterior (copia literal en tests/legado/) y el nuevo deben dar exactamente lo mismo
(datos, texto impreso y CSV de caché), con la BigQuery simulada y con los envoltorios REALES de programas/surtido.py y programas/champion.py.
"""
import ast
import contextlib
import decimal
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mediciones.grilla import ejecutar_grilla as _grilla
from mediciones.metodologia import ids

LEGADO = Path(__file__).parent / 'legado'
PROGRAMAS = Path(__file__).parent.parent / 'src' / 'mediciones' / 'programas'


def _funcion(path, nombre, ns):
    """Ejecuta SOLO la definición `nombre` de un archivo (en `ns`) y la devuelve."""
    src = Path(path).read_text(encoding='utf-8')
    nodo = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == nombre)
    exec(compile(ast.Module([nodo], []), str(path), 'exec'), ns)
    return ns[nombre]


def _consulta():
    return pd.DataFrame({'cod_cliente': [123.0, np.nan, 45.0, 6.0], 'cod_cliente_dex': ['0000000123 ', None, '0000000045', 'nan'],
                         'periodo': ['2026-01-01', '2026-02-01', None, '2026-04-01'], 'fec_creacion': ['2020-05-01', '2021-01-01', '2022-03-04', None],
                         'flg_cliente_roma': [True, False, True, False], 'peso': [decimal.Decimal('10.50'), decimal.Decimal('3'), None, decimal.Decimal('1.25')],
                         'categoria': ['Galletas', 'Cereales', 'Galletas', 'Cereales']})


# ── ids ──
def test_ids_identicos_al_codigo_anterior():
    ns = {'pd': pd}
    exec(compile((LEGADO / 'ids.py').read_text(encoding='utf-8'), 'ids', 'exec'), ns)
    for df in (_consulta(), pd.DataFrame({'cod_cliente': ['001', ' 002 ', 'None', '<NA>'], 'otro': [1, 2, 3, 4]}), pd.DataFrame({'cod_cliente': [1, 2, 3]})):
        assert_frame_equal(ns['_normalizar_ids'](df.copy()), ids.normalizar_ids(df.copy()), check_exact=True)
    assert [ns['_es_columna_id_cliente'](c) for c in ('cod_cliente', 'COD_CLIENTE_DEX', 'flg_cliente_roma', 'cliente')] == \
           [ids.es_columna_id_cliente(c) for c in ('cod_cliente', 'COD_CLIENTE_DEX', 'flg_cliente_roma', 'cliente')] == [True, True, False, False]


# ── ejecutar_grilla: surtido y champion ──
CASOS = {'surtido': dict(sufijo='202609'), 'champion': dict(sufijo=None)}


def _ejecutar(f, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df = f('SELECT 1', 'consulta', **kw)
    return df, buf.getvalue()


def _mocks(monkeypatch, df=None, error=None):
    def fake(query, project_id=None, dialect=None, progress_bar_type=None):
        if error:
            raise error
        return df.copy()
    monkeypatch.setattr(pd, 'read_gbq', fake, raising=False)


@pytest.mark.parametrize('prog', ['surtido', 'champion'])
def test_ejecutar_grilla_identica_por_bigquery_y_por_cache(prog, tmp_path, monkeypatch):
    sufijo = CASOS[prog]['sufijo']
    comun = dict(os=os, pd=pd, decimal=decimal, SUFIJO_CACHE=sufijo, _PATRONES_FECHA=['periodo', 'fecha', 'fec_', 'cosecha'])
    # ── código anterior (copia literal) ──
    d_old = tmp_path / 'old'
    d_old.mkdir()
    ns_old = dict(comun, CACHE_DIR=str(d_old))
    exec(compile((LEGADO / 'ids.py').read_text(encoding='utf-8'), 'ids', 'exec'), ns_old)
    exec(compile((LEGADO / f'grilla_{prog}.py').read_text(encoding='utf-8'), f'grilla_{prog}', 'exec'), ns_old)
    f_old = ns_old['ejecutar_grilla']
    # ── envoltorio REAL del script del programa ──
    d_new = tmp_path / 'new'
    d_new.mkdir()
    ns_new = dict(comun, CACHE_DIR=str(d_new), _grilla=_grilla)
    f_new = _funcion(PROGRAMAS / f'{prog}.py', 'ejecutar_grilla', ns_new)

    _mocks(monkeypatch, _consulta())
    (a, ta), (b, tb) = _ejecutar(f_old), _ejecutar(f_new)                                  # ruta BigQuery (escribe el caché)
    assert_frame_equal(a, b, check_exact=True)
    assert ta == tb.replace(str(d_new), str(d_old))
    nombre = f'consulta_{sufijo}.csv' if sufijo else 'consulta.csv'
    assert (d_old / nombre).read_bytes() == (d_new / nombre).read_bytes()                  # mismo CSV de caché
    (a, ta), (b, tb) = _ejecutar(f_old, forzar_descarga=False), _ejecutar(f_new, forzar_descarga=False)       # ruta caché
    assert_frame_equal(a, b, check_exact=True)
    assert ta == tb.replace(str(d_new), str(d_old)) and 'CACHE' in ta
    _mocks(monkeypatch, error=RuntimeError('sin permisos'))                                 # error de BigQuery: imprime y devuelve vacío
    (a, ta), (b, tb) = _ejecutar(f_old), _ejecutar(f_new)
    assert a.empty and b.empty and ta == tb and 'sin permisos' in ta


def test_las_diferencias_entre_surtido_y_champion_siguen_siendo_las_de_antes(monkeypatch, tmp_path):
    """No se unificó el comportamiento: surtido convierte fechas y Decimal en la ruta de BigQuery; Champion no."""
    _mocks(monkeypatch, _consulta())
    s, _ = _ejecutar(lambda q, n, **kw: _grilla(q, n, cache_dir=str(tmp_path), sufijo_cache='202609', **kw))
    c, _ = _ejecutar(lambda q, n, **kw: _grilla(q, n, cache_dir=str(tmp_path), fechas_en_consulta=False, decimales_en_consulta=False, **kw))
    assert pd.api.types.is_datetime64_any_dtype(s['periodo']) and pd.api.types.is_float_dtype(s['peso'])
    assert not pd.api.types.is_datetime64_any_dtype(c['periodo']) and c['peso'].map(lambda v: isinstance(v, decimal.Decimal)).any()
