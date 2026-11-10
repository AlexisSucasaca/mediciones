"""Exclusión de surtido de bodega congelada por cierre: la lista solo cuenta los periodos <= el cierre."""
import json
import re

import pytest

from conftest import RAIZ
from mediciones.ejecucion import DIR_PAQUETE
from mediciones.metodologia import bodega

NOTEBOOK = RAIZ / 'motores_notebook' / '1_brechas_piloto_cmp_mensual.ipynb'


def _sin_espacios(sql):
    return re.sub(r'\s+', '', sql)


def test_sin_override_la_lista_se_congela_al_cierre():
    assert bodega.periodo_max_bodega('2026-09-01') == '2026-09-01'
    assert bodega.periodo_max_bodega('2026-10-01') == '2026-10-01'            # al avanzar el cierre, avanza el corte (los periodos nuevos entran en SU cierre)


def test_un_override_solo_puede_ser_anterior_o_igual_al_cierre_y_dia_1():
    assert bodega.periodo_max_bodega('2026-09-01', '2026-08-01') == '2026-08-01'
    assert bodega.periodo_max_bodega('2026-09-01', '2026-09-01') == '2026-09-01'
    with pytest.raises(ValueError, match='posterior al cierre'):
        bodega.periodo_max_bodega('2026-09-01', '2026-10-01')                 # no se mira el futuro: justo el defecto que se corrige
    with pytest.raises(ValueError, match='día 1'):
        bodega.periodo_max_bodega('2026-09-01', '2026-08-15')
    with pytest.raises(ValueError, match='quedaría vacía'):
        bodega.periodo_max_bodega('2026-09-01', '2026-03-01')


def test_el_sql_lleva_la_cota_superior_y_conserva_el_resto():
    sql = bodega.sql_surtido_bodeg('2026-09-01')
    assert "periodo >='2026-04-01'" in sql and "periodo <='2026-09-01'" in sql
    assert all(c in sql for c in bodega.COMPANIAS_BODEG) and 'GROUP BY ALL' in sql and bodega.TABLA_BODEG in sql
    assert "periodo <='2026-08-01'" in bodega.sql_surtido_bodeg(bodega.periodo_max_bodega('2026-09-01', '2026-08-01'))
    sqlglot = pytest.importorskip('sqlglot')
    assert len(sqlglot.parse(sql, read='bigquery')) == 1


def test_el_sql_es_el_del_notebook_original_salvo_la_cota():
    """Diferencial: quitando la cota, queda EXACTAMENTE la consulta del notebook (mismas compañías, mismo periodo inicial, mismo agrupado)."""
    if not NOTEBOOK.exists():
        pytest.skip('falta el notebook original')
    nb = json.load(open(NOTEBOOK, encoding='utf-8'))
    original = next(re.search(r'sql_ps\s*=\s*"""(.*?)"""', ''.join(c['source']), re.S).group(1) for c in nb['cells'] if re.search(r'sql_ps\s*=\s*"""', ''.join(c['source'])))
    nuevo = bodega.sql_surtido_bodeg('2026-09-01')
    assert _sin_espacios(nuevo).replace("ANDperiodo<='2026-09-01'", '') == _sin_espacios(original)


def test_el_script_de_surtido_usa_la_lista_congelada_y_no_conserva_la_consulta_sin_tope():
    src = (DIR_PAQUETE / 'programas' / 'surtido.py').read_text(encoding='utf-8')
    assert bodega.TABLA_BODEG not in src and 'FROM `' + bodega.TABLA_BODEG not in src                  # la consulta vive solo en metodologia/bodega.py
    assert 'sql_surtido_bodeg(PERIODO_MAX_BODEG_USADO)' in src and 'periodo_max_bodega(MES_CIERRE, PERIODO_MAX_BODEG)' in src
    assert re.search(r'^PERIODO_MAX_BODEG = None\b', src, re.M)                                         # por defecto: congelada al cierre

