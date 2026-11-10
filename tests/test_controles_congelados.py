"""Surtido: los controles ya emparejados (estado congelado) conservan su serie aunque hoy los saque un filtro de la base (territorio, compañía, bodega)."""
import os
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1] / 'src' / 'mediciones'


# ── Surtido: el bloque real del programa, ejecutado con datos sintéticos ──
def _src_surtido():
    return (RAIZ / 'programas' / 'surtido.py').read_text(encoding='utf-8')


def _bloque_congelados_surtido():
    src = _src_surtido()
    return src[src.index('# [controles_congelados:inicio]'):src.index('# [controles_congelados:fin]')]


def _bloque_bodega_surtido():
    src = _src_surtido()
    ini = src.index('# eliminamos surtido_bodeg')
    fin = src.index('# Definir cosechas y sus rangos', ini)
    return src[ini:fin]


def _bloque_filtro_base_surtido():
    src = _src_surtido()
    ini = src.index('# base de clientes que tienen surtido o impulsos')
    fin = src.index('# Opción 1: con boolean indexing', ini)
    return src[ini:fin]


def _ejecutar_bloque(tmp_path, conservar, rehacer=False, con_estado=True):
    carpeta = tmp_path / 'outputs'
    carpeta.mkdir(exist_ok=True)
    if con_estado:
        pd.DataFrame({'cosecha': ['2026-04-01'] * 2, 'cod_accion': ['0000000010', '0000000011'],
                      'cod_control': ['4246795', '0000000099']}).to_csv(carpeta / 'df_matches.csv', index=False)   # un control sin ceros y otro con ceros
    base = pd.DataFrame({'cod_cliente': ['0004246795', '0000000099', '0000000555', '0000000777'],      # 1) control congelado en bodega  2) control congelado fuera  3) otro en bodega  4) otro fuera
                         'surtido_bodeg': [True, False, True, False], 'v': [1, 2, 3, 4]})
    ns = {'os': os, 'pd': pd, 'base_seg_limpio': base, 'RUTA_SALIDA_MATCH': str(carpeta), 'REHACER_MATCHING': rehacer, 'CONSERVAR_CONTROLES_CONGELADOS': conservar}
    exec(_bloque_congelados_surtido(), ns)
    exec(_bloque_bodega_surtido(), ns)
    return set(ns['base_seg_limpio']['cod_cliente'])


def test_surtido_conserva_controles_congelados_y_excluye_el_resto_de_bodega(tmp_path):
    assert _ejecutar_bloque(tmp_path, conservar=True) == {'0004246795', '0000000099', '0000000777'}


def test_surtido_sin_conservar_excluye_toda_la_lista_como_antes(tmp_path):
    assert _ejecutar_bloque(tmp_path, conservar=False) == {'0000000099', '0000000777'}


def test_surtido_con_rehacer_matching_o_sin_estado_excluye_toda_la_lista(tmp_path):
    assert _ejecutar_bloque(tmp_path, conservar=True, rehacer=True) == {'0000000099', '0000000777'}
    sin_estado = tmp_path / 'sin_estado'
    sin_estado.mkdir()
    assert _ejecutar_bloque(sin_estado, conservar=True, con_estado=False) == {'0000000099', '0000000777'}


def test_el_bloque_de_surtido_no_deja_nombres_huerfanos():
    bloque = _bloque_congelados_surtido()
    for nombre in ('CONSERVAR_CONTROLES_CONGELADOS', 'RUTA_SALIDA_MATCH', 'REHACER_MATCHING'):
        assert nombre in bloque
    src = _src_surtido()
    assert re.search(r'^CONSERVAR_CONTROLES_CONGELADOS\s*=\s*True', src, re.M)
    i_carga = src.index('# [controles_congelados:inicio]')
    assert re.search(r'^RUTA_SALIDA_MATCH\s*=', src, re.M).start() < i_carga, 'RUTA_SALIDA_MATCH debe estar definida antes de usarse'
    assert i_carga < src.index('# base de clientes que tienen surtido o impulsos') < src.index('# eliminamos surtido_bodeg'), \
        'los controles congelados se leen antes de los dos filtros que los usan'


def test_surtido_controles_congelados_conservan_su_serie_aunque_salgan_del_territorio_de_impulso(tmp_path):
    carpeta = tmp_path / 'outputs'
    carpeta.mkdir()
    pd.DataFrame({'cosecha': ['2026-04-01'], 'cod_accion': ['0000000010'], 'cod_control': ['1634208']}).to_csv(carpeta / 'df_matches.csv', index=False)
    base = pd.DataFrame({'cod_cliente': ['0000000010', '0001634208', '0000000555', '0000000777', '0000000888'],
                         # acción | control congelado que salió de impulso | control nuevo fuera de impulso | en impulso | compañía excluida sin surtido
                         'tiene_surtido': [True, False, False, False, False],
                         'nom_compania': ['A', 'A', 'A', 'A', 'DEX X']})
    for conservar, esperado in ((True, {'0000000010', '0001634208', '0000000777'}), (False, {'0000000010', '0000000777'})):
        ns = {'os': os, 'pd': pd, 'sell_out_cosecha_roma_seg': base, 'RUTA_SALIDA_MATCH': str(carpeta), 'REHACER_MATCHING': False,
              'CONSERVAR_CONTROLES_CONGELADOS': conservar, 'dex': ['DEX X'], 'clientes_impulso': ['0000000777']}
        exec(_bloque_congelados_surtido(), ns)
        exec(_bloque_filtro_base_surtido(), ns)
        assert set(ns['base_seg_limpio']['cod_cliente']) == esperado
