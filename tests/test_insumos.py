import numpy as np
import pandas as pd
import pytest

from mediciones import insumos as ins
from mediciones.ejecucion import ejecutar_familia


def _df(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'cod_cliente': [f'{i:010d}' for i in range(n)], 'periodo': pd.date_range('2026-01-01', periods=n, freq='MS'),
                         'peso': rng.uniform(1, 9, n), 'n': np.arange(n), 'flag': [True, False] * (n // 2), 'cat': ['G', 'C'] * (n // 2)})


class FalsoBQ:
    """Reemplaza al read_gbq real y cuenta cuántas veces se consultó."""
    def __init__(self, df): self.df, self.llamadas = df, 0
    def __call__(self, query, *a, **k): self.llamadas += 1; return self.df.copy()


def test_hash_no_depende_del_orden_y_si_del_contenido():
    d = _df()
    assert ins.hash_df(d) == ins.hash_df(d.sample(frac=1, random_state=3).reset_index(drop=True))
    d2 = d.copy(); d2.loc[2, 'peso'] += 1e-6
    assert ins.hash_df(d) != ins.hash_df(d2)
    assert ins.hash_df(d) != ins.hash_df(d.iloc[:-1])
    assert ins.hash_df(d.iloc[0:0]) != ins.hash_df(d)


def test_un_cambio_de_sql_es_otra_consulta_pero_los_espacios_no():
    assert ins.clave_consulta('SELECT  a\nFROM t', 'p') == ins.clave_consulta('SELECT a FROM   t', 'p')
    assert ins.clave_consulta("SELECT a FROM t WHERE x = 'CMP'", 'p') != ins.clave_consulta("SELECT a FROM t WHERE x = 'CMP Impulso'", 'p')
    assert ins.clave_consulta('SELECT 1', 'p1') != ins.clave_consulta('SELECT 1', 'p2')


def test_consultar_guarda_snapshot_y_reutilizar_no_vuelve_a_consultar(tmp_path):
    bq = FalsoBQ(_df())
    capa = ins.CapaInsumos('consultar', tmp_path)
    leer = capa.envolver(bq)
    a = leer('SELECT 1', project_id='p')
    assert bq.llamadas == 1 and capa.entradas[0]['origen'] == 'bigquery' and list(tmp_path.glob('*.parquet'))
    capa2 = ins.CapaInsumos('reutilizar', tmp_path)
    b = capa2.envolver(bq)('SELECT 1', project_id='p')
    assert bq.llamadas == 1 and capa2.entradas[0]['origen'] == 'snapshot'            # no consultó
    pd.testing.assert_frame_equal(a, b)                                                # y los tipos se conservan (fechas, ids con ceros, bool, float...)
    assert capa2.entradas[0]['content_hash'] == capa.entradas[0]['content_hash']


def test_si_cambia_la_consulta_no_se_reutiliza_el_snapshot_viejo(tmp_path):
    bq = FalsoBQ(_df())
    ins.CapaInsumos('consultar', tmp_path).envolver(bq)("SELECT * FROM t WHERE canal = 'CMP'", project_id='p')
    capa = ins.CapaInsumos('reutilizar', tmp_path)
    capa.envolver(bq)("SELECT * FROM t WHERE canal = 'CMP Impulso'", project_id='p')
    assert bq.llamadas == 2 and capa.entradas[0]['origen'] == 'bigquery'


def test_reproducir_exige_snapshot_y_usa_el_contenido_pedido(tmp_path):
    bq = FalsoBQ(_df(seed=1))
    capa = ins.CapaInsumos('consultar', tmp_path)
    capa.envolver(bq)('SELECT 1', project_id='p')
    h1 = capa.entradas[0]['content_hash']
    bq.df = _df(seed=2)                                                                # BigQuery cambió retroactivamente
    capa.envolver(bq)('SELECT 1', project_id='p')
    h2 = capa.entradas[1]['content_hash']
    assert h1 != h2 and len(list(tmp_path.glob('*.parquet'))) == 2                     # los dos quedan guardados
    key = capa.entradas[0]['key']
    rep = ins.CapaInsumos('reproducir', tmp_path, contenidos={key: h1})
    out = rep.envolver(None)('SELECT 1', project_id='p')                               # sin acceso a BigQuery
    assert rep.entradas[0]['content_hash'] == h1 and ins.hash_df(out) == h1
    with pytest.raises(ins.InsumoNoDisponible):
        ins.CapaInsumos('reproducir', tmp_path).envolver(None)('SELECT otra_consulta', project_id='p')
    with pytest.raises(ins.InsumoNoDisponible):
        ins.CapaInsumos('reproducir', tmp_path, contenidos={key: 'inexistente'}).envolver(None)('SELECT 1', project_id='p')


def test_tipos_mixtos_caen_a_pickle_y_se_recuperan(tmp_path):
    d = pd.DataFrame({'a': [1, 'x', 3.5], 'b': [1, 2, 3]})
    capa = ins.CapaInsumos('consultar', tmp_path)
    capa.envolver(lambda q, **k: d.copy())('SELECT mixto', project_id='p')
    out = ins.CapaInsumos('reutilizar', tmp_path).envolver(None)('SELECT mixto', project_id='p')
    pd.testing.assert_frame_equal(d, out)


def test_la_capa_reemplaza_y_restaura_pd_read_gbq(tmp_path):
    original = getattr(pd, 'read_gbq', None)
    with ins.capa_insumos('reutilizar', tmp_path):
        assert pd.read_gbq is not original
    assert getattr(pd, 'read_gbq', None) is original
    with ins.capa_insumos('desactivado', tmp_path):
        assert getattr(pd, 'read_gbq', None) is original


def test_ejecutar_familia_registra_los_insumos_de_sus_consultas(tmp_path, monkeypatch):
    bq = FalsoBQ(_df())
    monkeypatch.setattr(pd, 'read_gbq', bq, raising=False)
    prog = tmp_path / 'p.py'
    prog.write_text("import pandas as pd\ndf = pd.read_gbq('SELECT x FROM t', project_id='proy')\nbase_calculo_toneladas = df\nbase_calculo_brecha = df\n")
    fam = tmp_path / 'f.py'
    fam.write_text('pass\n')
    cfg = {'mensual': {'scripts': ['f'], 'variable_mes': 'MES_X', 'salida_ton': 'base_calculo_toneladas', 'salida_brecha': 'base_calculo_brecha'}}
    r = ejecutar_familia('mensual', prog, 'prueba', pd.Timestamp('2026-09-01'), tmp_path / 'wd', cfg, scripts_familia=[fam])
    assert len(r.insumos) == 1 and r.insumos[0]['origen'] == 'bigquery' and r.insumos[0]['filas'] == 6
    assert list((tmp_path / 'wd' / 'insumos' / '202609').glob('*.parquet'))
    r2 = ejecutar_familia('mensual', prog, 'prueba', pd.Timestamp('2026-09-01'), tmp_path / 'wd', cfg, insumos='reproducir', scripts_familia=[fam])
    assert r2.insumos[0]['origen'] == 'snapshot' and bq.llamadas == 1


def test_sin_pd_read_gbq_se_consulta_con_pandas_gbq_sin_recursion(tmp_path, monkeypatch):
    import sys
    import types
    bq = FalsoBQ(_df())
    modulo = types.ModuleType('pandas_gbq')
    modulo.read_gbq = bq
    monkeypatch.setitem(sys.modules, 'pandas_gbq', modulo)
    monkeypatch.delattr(pd, 'read_gbq', raising=False)
    with ins.capa_insumos('consultar', tmp_path) as capa:
        out = pd.read_gbq('SELECT 1', project_id='p')
    assert bq.llamadas == 1 and len(out) == 6 and capa.entradas[0]['origen'] == 'bigquery' and not hasattr(pd, 'read_gbq')
