"""Reproducibilidad en el runner: config_hash, manifiesto, diagnóstico y umbral de deriva, pares contra lo publicado, insumos exactos."""
import json

import pandas as pd

from mediciones import pares
from mediciones.esquema import T_BRE, T_ESP, T_TON
from conftest import ServiciosMemoria
from test_runner_memoria import AGO, SEP, _meses, correr, motor  # noqa: F401  (motor es un fixture)


def _hashes(svc):
    return {t: set(svc.t[t].config_hash.unique()) for t in (T_TON, T_BRE, T_ESP)}


def test_config_hash_en_las_tres_bases_y_manifiesto_por_corrida(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    c = correr(raiz_tmp, AGO, svc, modo='cargar')
    ch = c.res['cpb']['config_hash']
    assert all(h == {ch} for h in _hashes(svc).values())                                 # cada fila dice con qué versión se produjo
    man = json.loads((raiz_tmp / 'runs' / c.corrida_id / 'manifiesto_cpb.json').read_text(encoding='utf-8'))
    assert man['config_hash'] == ch and man['hash_insumos'] and len(man['insumos']) == 1
    claves = {f['metrica'] for f in svc.log if f['etapa'] == 'manifiesto'}
    assert {'config_hash', 'hash_codigo', 'hash_config', 'hash_entorno', 'hash_insumos'} <= claves
    assert c.resumen.config_hash.tolist() == [ch]


def test_el_diagnostico_de_la_deriva_distingue_insumos_de_codigo(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    c = correr(raiz_tmp, SEP, svc, modo='dry_run')                                      # mismo código: lo que cambió son los datos
    diag = [f['valor_txt'] for f in c.log if f['etapa'] == 'deriva' and f['metrica'].endswith('diagnostico')]
    assert diag and all('INSUMOS' in d for d in diag)
    for t in (T_TON, T_BRE, T_ESP):                                                      # ahora lo publicado salió de otro código/configuración
        svc.t[t]['config_hash'] = 'otra_version'
    c = correr(raiz_tmp, SEP, svc, modo='dry_run')
    diag = [f['valor_txt'] for f in c.log if f['etapa'] == 'deriva' and f['metrica'].endswith('diagnostico')]
    assert diag and all('distintos' in d for d in diag)


def test_la_deriva_sobre_el_umbral_bloquea_y_el_umbral_es_por_tabla(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    c = correr(raiz_tmp, SEP, svc, modo='cargar', umbral_deriva={T_ESP: 100})            # espejos tolerados, toneladas no
    assert c.resumen.estado.tolist() == ['bloqueado'] and 'base_ton_atribuibles' in c.res['cpb']['error'] and 'base_espejos' not in c.res['cpb']['error']
    assert svc.t[T_TON].periodo_reporte.nunique() == 1                                    # no se cargó nada de setiembre
    c = correr(raiz_tmp, SEP, svc, modo='cargar', umbral_deriva={T_ESP: 100, T_TON: 100, T_BRE: 100},
               tolerancia_celda={'ton_pct': 100})                   # tolerancia por celda relajada a propósito
    assert c.resumen.estado.tolist() == ['ok'] and svc.t[T_TON].periodo_reporte.nunique() == 2


def test_el_bloqueo_por_defecto_es_por_celda_y_un_umbral_por_filas_solo_si_se_pide(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    c = correr(raiz_tmp, SEP, svc, modo='dry_run')
    cel = c.res['cpb']['celdas_ton']
    assert int(cel['fuera'].sum()) == 2 and set(cel.columns) >= {'periodo', 'des_categoria', 'publicado', 'recalculado', 'dif', 'dif_pct', 'fuera'}
    c = correr(raiz_tmp, SEP, svc, modo='dry_run', tolerancia_celda={'ton_pct': 50})          # con una tolerancia mayor, nada fuera
    assert not c.resumen.alertas.iloc[0] or 'celdas' not in c.resumen.alertas.iloc[0]
    c = correr(raiz_tmp, SEP, svc, modo='dry_run', tolerancia_celda={'ton_pct': 50}, umbral_deriva={T_ESP: 0.0})
    assert 'base_espejos' in c.resumen.alertas.iloc[0]                                          # el umbral por filas, solo si se pide


def test_en_dry_run_la_deriva_alerta_pero_no_bloquea(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    c = correr(raiz_tmp, SEP, svc, modo='dry_run')
    assert c.resumen.estado.tolist() == ['ok'] and 'base_ton_atribuibles' in c.resumen.alertas.iloc[0]


def test_aceptar_deriva_exige_motivo(raiz_tmp, motor):
    import pytest
    with pytest.raises(ValueError):
        correr(raiz_tmp, SEP, ServiciosMemoria(), aceptar_deriva=True)


def test_los_pares_locales_se_restauran_desde_lo_publicado(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    ruta = raiz_tmp / 'data' / 'cpb' / 'cpb_pares_fijos.csv'
    ruta.unlink()                                                                          # se perdió el estado local
    c = correr(raiz_tmp, AGO, svc, modo='dry_run')
    assert c.res['cpb']['pares']['estado'] == 'restaurado' and ruta.exists()
    assert pares.verificar('mensual', ruta, pares.publicados_normalizados(svc.leer_pares('cpb')))['estado'] == 'ok'


def test_pares_distintos_a_lo_publicado_bloquean_antes_de_ejecutar(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    ruta = raiz_tmp / 'data' / 'cpb' / 'cpb_pares_fijos.csv'
    d = pd.read_csv(ruta, dtype=str)
    d.loc[d.index[:5], 'cod_control'] = '0000000042'                                      # alguien re-emparejó
    d.to_csv(ruta, index=False)
    motor['llamadas'].clear()
    c = correr(raiz_tmp, SEP, svc, modo='cargar')
    assert c.resumen.estado.tolist() == ['bloqueado'] and c.res['cpb']['error'].startswith('PARES') and motor['llamadas'] == []   # ni siquiera corrió el motor
    c = correr(raiz_tmp, SEP, svc, modo='dry_run')                                        # en dry_run solo avisa
    assert 'error' not in c.res['cpb'] and motor['llamadas'] == [('cpb', '2026-09')]
    c = correr(raiz_tmp, SEP, svc, modo='cargar', ignorar_pares=True, aceptar_deriva=True, motivo_deriva='re-emparejado validado')
    assert c.resumen.estado.tolist() == ['ok']


def test_reproducir_usa_los_insumos_exactos_de_la_corrida_anterior(raiz_tmp, motor):
    _meses(motor)
    c1 = correr(raiz_tmp, AGO, ServiciosMemoria(), modo='dry_run')
    c2 = correr(raiz_tmp, AGO, ServiciosMemoria(), modo='dry_run', insumos='reproducir', reproducir_de=c1.corrida_id)
    assert motor['kw']['insumos'] == 'reproducir'
    assert motor['kw']['contenidos'] == {e['key']: e['content_hash'] for e in c1.res['cpb']['manifiesto']['insumos']} and c2.resumen.estado.tolist() == ['ok']


def test_un_fallo_dentro_de_un_script_del_motor_se_ubica_en_su_celda_y_linea(tmp_path):
    from mediciones.runner import guardar_traceback, ubicar_fallo
    d = tmp_path / 'programas'
    d.mkdir()
    script = d / 'portada.py'
    script.write_text("x = 1\n# ══════════════ celda 7 del notebook ══════════════\ny = {'a': 1}\nz = y['cosecha_agrupada']\n", encoding='utf-8')
    try:
        exec(compile(script.read_text(encoding='utf-8'), str(script), 'exec'), {})
    except KeyError as e:
        donde = ubicar_fallo(e)
        assert 'programas/portada.py' in donde and 'línea 4' in donde and 'celda 7 del notebook' in donde and "y['cosecha_agrupada']" in donde
        ruta = guardar_traceback(e, tmp_path, '202609_x', 'champion_motor')
        assert ruta.exists() and 'KeyError' in ruta.read_text(encoding='utf-8')
    try:
        {}['x']
    except KeyError as e2:
        assert ubicar_fallo(e2) == ''                                 # un error fuera de los scripts del motor no inventa ubicación
