import json
from pathlib import Path

import nbformat

from mediciones import trazabilidad as tz
from mediciones.registro import Programa, cargar_registro
from conftest import RAIZ

FAMILIAS, PROGRAMAS = cargar_registro(RAIZ / 'config' / 'programas.yaml')
MES = __import__('pandas').Timestamp('2026-09-01')


def test_manifiesto_completo_y_estable():
    m1 = tz.construir_manifiesto(PROGRAMAS['cpb'], FAMILIAS, RAIZ, 'c1', MES)
    m2 = tz.construir_manifiesto(PROGRAMAS['cpb'], FAMILIAS, RAIZ, 'c2', MES)
    assert m1['config_hash'] == m2['config_hash']                                   # no depende de la corrida
    for k in ('hash_codigo', 'hash_config', 'hash_entorno', 'versiones', 'detalle_codigo'):
        assert m1[k]
    assert {'pandas', 'numpy', 'scipy', 'statsmodels', 'python'} <= set(m1['versiones'])
    assert {'metodologia.py', 'cpb.py', 'parametros.py', 'matching.py'} <= set(m1['detalle_codigo'])
    assert tz.construir_manifiesto(PROGRAMAS['cpm'], FAMILIAS, RAIZ, 'c', MES)['config_hash'] != m1['config_hash']


def test_el_hash_cambia_si_cambia_el_codigo_o_la_configuracion(tmp_path):
    s = tmp_path / 'p.py'
    s.write_text('x = 1\n')
    prog = Programa('t', 'mensual', 'familia', script=str(s))
    h1 = tz.construir_manifiesto(prog, FAMILIAS, RAIZ, 'c', MES)['config_hash']
    s.write_text('x = 2\n')
    assert tz.construir_manifiesto(prog, FAMILIAS, RAIZ, 'c', MES)['config_hash'] != h1
    s.write_text('x = 1\n')
    assert tz.construir_manifiesto(prog, FAMILIAS, RAIZ, 'c', MES)['config_hash'] == h1
    prog2 = Programa('t', 'mensual', 'familia', script=str(s), negocios_ub=['CP2'])
    assert tz.construir_manifiesto(prog2, FAMILIAS, RAIZ, 'c', MES)['config_hash'] != h1


def test_motores_notebook_hashean_solo_el_codigo(tmp_path):
    def nb(salida):
        n = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('a = 1', outputs=[nbformat.v4.new_output('stream', text=salida)])])
        r = tmp_path / 'n.ipynb'
        nbformat.write(n, str(r))
        return r
    prog = Programa('t', 'cosechas', 'notebook', notebook='n.ipynb', adaptador='estandar')
    nb('uno')
    h1 = tz.hash_codigo(prog, FAMILIAS, tmp_path)[0]
    nb('otra salida distinta')
    assert tz.hash_codigo(prog, FAMILIAS, tmp_path)[0] == h1                        # las salidas del notebook no cuentan


def test_insumos_en_el_manifiesto_y_guardado(tmp_path):
    m = tz.construir_manifiesto(PROGRAMAS['cpb'], FAMILIAS, RAIZ, 'c9', MES)
    tz.completar_con_insumos(m, [{'key': 'a', 'content_hash': '1'}, {'key': 'b', 'content_hash': '2'}])
    h = m['hash_insumos']
    tz.completar_con_insumos(m, [{'key': 'b', 'content_hash': '2'}, {'key': 'a', 'content_hash': '1'}])
    assert m['hash_insumos'] == h                                                    # no depende del orden de las consultas
    tz.completar_con_insumos(m, [{'key': 'a', 'content_hash': '1'}, {'key': 'b', 'content_hash': 'CAMBIO'}])
    assert m['hash_insumos'] != h
    ruta = tz.guardar_manifiesto(tmp_path, m)
    assert json.loads(ruta.read_text(encoding='utf-8'))['config_hash'] == m['config_hash'] and tz.cargar_manifiesto(tmp_path, 'c9', 'cpb')['corrida_id'] == 'c9'


def test_el_config_hash_cubre_el_nucleo_compartido_la_normalizacion_y_la_ub():
    """Si cambia la atribución, los ids, la normalización o la UB, cambian los números: el config_hash debe enterarse."""
    nombres = {a.name for a in tz.archivos_de_codigo(PROGRAMAS['cpb'], FAMILIAS)}
    assert {'atribucion.py', 'ids.py', 'grilla.py', 'normalizacion.py', 'ub.py', 'parametros.py', 'matching.py'} <= nombres
