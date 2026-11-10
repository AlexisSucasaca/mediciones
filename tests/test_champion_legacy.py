"""Champion: elección del archivo legacy por mes, lectura del formato de octubre y entrada de las altas que solo vienen en la legacy."""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1] / 'src' / 'mediciones'
T = pd.Timestamp


def _src():
    return (RAIZ / 'programas' / 'champion.py').read_text(encoding='utf-8')


def _bloque(nombre):
    s = _src()
    return s[s.index(f'# [{nombre}:inicio]'):s.index(f'# [{nombre}:fin]')]


def _tocar(ruta, antes=0):
    ruta.write_bytes(b'x')
    t = time.time() - antes
    os.utime(ruta, (t, t))


def _elegir(tmp_path, monkeypatch, mes, archivos):
    monkeypatch.chdir(tmp_path)
    for nombre, antes in archivos:
        _tocar(tmp_path / nombre, antes)
    ns = {'os': os, 'MES_CIERRE': T(mes), 'ARCHIVO_LEGACY': 'BD_Clientes_Perfectos_Snack_-_Setiembre.xlsx'}
    exec(_bloque('legacy_archivo'), ns)
    return ns['ARCHIVO_LEGACY']


def test_toma_el_archivo_del_mes_de_cierre_aunque_haya_otro_mas_reciente(tmp_path, monkeypatch):
    # el de setiembre fue tocado DESPUÉS (más reciente): igual gana el de octubre cuando se cierra octubre, incluso con el error de tipeo 'Perdectos'
    arch = [('BD_Clientes_Perdectos_Snack_-_Octubre.xlsx', 5000), ('BD_Clientes_Perfectos_Snack_-_Setiembre.xlsx', 10)]
    assert _elegir(tmp_path, monkeypatch, '2026-10-01', arch) == 'BD_Clientes_Perdectos_Snack_-_Octubre.xlsx'
    assert _elegir(tmp_path, monkeypatch, '2026-09-01', arch) == 'BD_Clientes_Perfectos_Snack_-_Setiembre.xlsx'     # re-correr setiembre usa el de setiembre


def test_sin_archivo_del_mes_toma_el_mas_reciente(tmp_path, monkeypatch):
    arch = [('BD_Clientes_Perfectos_Snack_-_Octubre.xlsx', 5000), ('BD_Clientes_Perfectos_Snack_-_Setiembre.xlsx', 10)]
    assert _elegir(tmp_path, monkeypatch, '2026-12-01', arch) == 'BD_Clientes_Perfectos_Snack_-_Setiembre.xlsx'      # el modificado más recientemente
    assert _elegir(tmp_path, monkeypatch, '2026-12-01', [('BD_Clientes_Perfectos_Snack_-_Noviembre.xlsx', 1)]) == 'BD_Clientes_Perfectos_Snack_-_Noviembre.xlsx'


def test_lee_el_formato_de_octubre_con_cliente_dex_cod_y_codigo_antiguo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({'Región': 'LIMA', 'Segmento': 'Oro', 'Código antiguo ': [np.nan, 1573235.0, np.nan],        # el espacio final y los flotantes vienen así
                  'Cliente DEX Cod': [1450622, 4268289, 4153536], 'CHAMPIONS': [2025, 2025, 2026],
                  'Region': [np.nan] * 3}).to_excel('BD_Clientes_Perdectos_Snack_-_Octubre.xlsx', index=False)
    ns = {'pd': pd, 'np': np, 'ARCHIVO_LEGACY': 'BD_Clientes_Perdectos_Snack_-_Octubre.xlsx',
          '_z10': lambda s: s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(10)}
    exec(_bloque('legacy_lectura'), ns)
    leg = ns['legacy']
    assert leg['cod_cliente_dex'].tolist() == ['0001450622', '0004268289', '0004153536']
    assert leg['cod_antiguo'].dropna().tolist() == ['0001573235'] and leg.loc[1, 'cod_antiguo'] == '0001573235'


def test_el_formato_anterior_con_cod_cliente_dex_sigue_funcionando(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({'cod_cliente_dex': [1450622, 4268289]}).to_excel('legacy.xlsx', index=False)
    ns = {'pd': pd, 'np': np, 'ARCHIVO_LEGACY': 'legacy.xlsx', '_z10': lambda s: s.astype(str).str.zfill(10)}
    exec(_bloque('legacy_lectura'), ns)
    assert ns['legacy']['cod_cliente_dex'].tolist() == ['0001450622', '0004268289'] and ns['legacy']['cod_antiguo'].isna().all()


def _entradas(tmp_path, monkeypatch, solo_leg, incremental=True, mes='2026-10-01', publicadas=None, mapa=None, previos=None):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'out').mkdir(exist_ok=True)
    # estado congelado de setiembre: 'A' tiene par (entró en setiembre), 'B' quedó sin pareja (entró en setiembre)
    pd.DataFrame({'cod_accion': ['A'], 'cod_control': ['C'], 'cosecha': ['2026-03-01'], 'min_per_ac': ['2026-09-01']}).to_csv(tmp_path / 'out' / 'df_matches.csv', index=False)
    pd.DataFrame({'cod_accion': ['B'], 'cosecha': ['2026-03-01'], 'min_periodo_cosecha': ['2026-09-01']}).to_csv(tmp_path / 'out' / 'df_sin_pareja.csv', index=False)
    ns = {'pd': pd, 'os': os, 'solo_leg': set(solo_leg), 'MODO_INCREMENTAL': incremental, 'LEGACY_ENTRADA_SIN_MES': T(mes),
          'ENTRADAS_PUBLICADAS': publicadas or {}, 'mapa_homologacion': mapa or {}, 'ids_previos': previos or set(), 'MES_CIERRE': T(mes),
          'RUTA_PARES_CONGELADOS': 'out/df_matches.csv', 'RUTA_SALIDA': 'out',
          'df_cosecha': pd.DataFrame({'cod_cliente_dex': ['X'], 'min_periodo_cosecha': [T('2026-04-01')]}), 'ids_tratados': {'X'}}
    exec(_bloque('solo_leg_entrada'), ns)
    return dict(zip(ns['df_cosecha']['cod_cliente_dex'], ns['df_cosecha']['min_periodo_cosecha'])), ns


def test_las_altas_de_setiembre_conservan_su_entrada_y_las_de_octubre_entran_en_octubre(tmp_path, monkeypatch):
    ent, ns = _entradas(tmp_path, monkeypatch, ['A', 'B', 'NUEVO'])
    assert ent['A'] == T('2026-09-01') and ent['B'] == T('2026-09-01')          # con par y sin pareja: siguen siendo de setiembre
    assert ent['NUEVO'] == T('2026-10-01') and ent['X'] == T('2026-04-01')       # la alta de octubre entra en octubre; las hojas no se tocan
    assert ns['ids_tratados'] == {'X', 'A', 'B', 'NUEVO'}


def test_sin_estado_congelado_todas_entran_en_el_mes_de_cierre(tmp_path, monkeypatch):
    ent, _ = _entradas(tmp_path, monkeypatch, ['A', 'NUEVO'], incremental=False)
    assert ent['A'] == T('2026-10-01') and ent['NUEVO'] == T('2026-10-01')


def test_una_entrada_guardada_posterior_al_cierre_no_se_adelanta(tmp_path, monkeypatch):
    ent, _ = _entradas(tmp_path, monkeypatch, ['A'], mes='2026-08-01')             # re-correr agosto: no puede entrar en setiembre
    assert ent['A'] == T('2026-08-01')


def test_un_cliente_que_no_esta_en_pares_ni_en_sin_pareja_conserva_su_entrada_publicada(tmp_path, monkeypatch):
    """El caso de 0004047610: está en la lista de clientes, fuera de la muestra (ni par ni sin pareja). Su entrada sale de lo ya publicado."""
    ent, _ = _entradas(tmp_path, monkeypatch, ['A', 'FUERA_DE_MUESTRA', 'NUEVO'], publicadas={'FUERA_DE_MUESTRA': T('2026-09-01')})
    assert ent['FUERA_DE_MUESTRA'] == T('2026-09-01')          # sin lo publicado habría pasado a octubre y perdido su setiembre
    assert ent['A'] == T('2026-09-01') and ent['NUEVO'] == T('2026-10-01')


def test_lo_publicado_manda_sobre_el_estado_congelado(tmp_path, monkeypatch):
    ent, _ = _entradas(tmp_path, monkeypatch, ['A'], publicadas={'A': T('2026-08-01')})
    assert ent['A'] == T('2026-08-01')


def test_entradas_publicadas_desde_lo_publicado():
    from mediciones import frontera
    pub = pd.DataFrame({'cod_cliente': ['1', '0000000001', '2', '2', 2.0], 'des_categoria': 'Galletas',
                        'periodo': pd.to_datetime(['2026-10-01', '2026-09-01', '2026-10-01', '2026-09-15', '2026-11-01'])})
    # '1' y '0000000001' son el mismo cliente (se normaliza a 10 dígitos); gana el primer mes publicado
    assert frontera.entradas_publicadas(pub) == {'0000000001': T('2026-09-01'), '0000000002': T('2026-09-01')}
    assert frontera.entradas_publicadas(None) == {} and frontera.entradas_publicadas(pub.iloc[0:0]) == {}


def test_el_runner_pasa_las_entradas_publicadas_solo_a_champion():
    src = (RAIZ / 'runner.py').read_text(encoding='utf-8')
    i = src.index("if prog.script == 'champion':")
    assert "params['ENTRADAS_PUBLICADAS'] = frontera.entradas_publicadas(_publicado(T_TON, p))" in src[i:i + 400]
    assert src.index("params['CALENDARIO_COSECHAS']") < i < src.index('r = ejecutar_familia(prog.familia')


# ── Segmentos que la base de minimayoristas no trae ──
def _ns_seg(tmp_path, monkeypatch, manual, mapa=None):
    monkeypatch.chdir(tmp_path)
    if manual is not None:
        pd.DataFrame(manual).to_csv('segmentos_manual.csv', index=False)
    seg = pd.DataFrame({'customer_id': ['0000000001', '0000000002'], 'Segmento': ['Plus', 'Base']})
    ns = {'pd': pd, 'os': os, 'seg': seg, 'mapa_homologacion': mapa or {}, 'ARCHIVO_SEGMENTOS_MANUAL': 'segmentos_manual.csv',
          '_z10': lambda s: s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(10)}
    exec(_bloque('segmentos_manual'), ns)
    return ns['seg'].set_index('customer_id')['Segmento'].to_dict(), ns


def test_el_archivo_manual_completa_solo_a_quienes_la_base_no_trae(tmp_path, monkeypatch, capsys):
    seg, _ = _ns_seg(tmp_path, monkeypatch, {'customer_id': ['1', '4243522', '4263159'], 'Segmento': ['Premium', 'Plus', 'Base'], 'observacion': ['', 'alta oct', '']})
    assert seg['0000000001'] == 'Plus'                                           # la base ya lo trae: manda la base
    assert seg['0004243522'] == 'Plus' and seg['0004263159'] == 'Base'           # los que faltaban, completados
    assert 'completados desde segmentos_manual.csv: 2' in capsys.readouterr().out


def test_el_manual_acepta_el_codigo_nuevo_o_el_antiguo(tmp_path, monkeypatch):
    seg, _ = _ns_seg(tmp_path, monkeypatch, {'customer_id': ['4256110'], 'Segmento': ['Premium']}, mapa={'0004256110': '0001358967'})
    assert seg['0001358967'] == 'Premium' and '0004256110' not in seg            # se unifica al código antiguo, como el resto


def test_el_manual_avisa_un_segmento_mal_escrito_y_falla_si_faltan_columnas(tmp_path, monkeypatch, capsys):
    _ns_seg(tmp_path, monkeypatch, {'customer_id': ['4243522'], 'Segmento': ['Premiun']})
    assert 'no existen en la base' in capsys.readouterr().out
    import pytest
    with pytest.raises(ValueError, match='faltan las columnas'):
        _ns_seg(tmp_path, monkeypatch, {'cliente': ['4243522'], 'seg': ['Plus']})


def test_sin_archivo_manual_no_cambia_nada(tmp_path, monkeypatch):
    seg, _ = _ns_seg(tmp_path, monkeypatch, None)
    assert seg == {'0000000001': 'Plus', '0000000002': 'Base'}


# ── Conciliación mensual de clientes ──
def _conciliar(tmp_path, monkeypatch, con_mes_anterior=True):
    monkeypatch.chdir(tmp_path)
    cols = lambda cod, ant, nom: pd.DataFrame({'cod_cliente_dex': cod, 'Código antiguo ': ant, 'Cliente DEX': nom})
    if con_mes_anterior:                                    # setiembre: A,B,C,D (D es el código antiguo de E), y baja real: F
        cols([1, 2, 3, 4, 6], [np.nan] * 5, ['A', 'B', 'C', 'D', 'F']).to_excel('BD_Clientes_Perfectos_Snack_-_Setiembre.xlsx', index=False)
    # octubre: formato nuevo; E reemplaza a D (cambio de código); G es alta real; F ya no está
    pd.DataFrame({'Cliente DEX Cod': [1, 2, 3, 5, 7], 'Código antiguo ': [np.nan, np.nan, np.nan, 4.0, np.nan], 'Cliente DEX': ['A', 'B', 'C', 'E', 'G']}).to_excel(
        'BD_Clientes_Perdectos_Snack_-_Octubre.xlsx', index=False)
    ns = {'pd': pd, 'np': np, 'os': os, 'MES_CIERRE': T('2026-10-01'), 'ARCHIVO_LEGACY': 'BD_Clientes_Perfectos_Snack_-_Setiembre.xlsx',
          'ARCHIVO_SEGMENTOS_MANUAL': 'segmentos_manual.csv', 'mapa_homologacion': {},
          '_z10': lambda s: s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(10)}
    exec(_bloque('legacy_archivo'), ns)                     # elige el de octubre y arma _cands_leg
    exec(_bloque('legacy_lectura'), ns)
    ns['mapa_homologacion'] = {'0000000005': '0000000004'}  # lo que arma el motor con el 'Código antiguo'
    ns['legacy']['cod_original'] = ns['legacy']['cod_cliente_dex']
    ns['legacy']['cod_cliente_dex'] = ns['legacy']['cod_cliente_dex'].replace(ns['mapa_homologacion'])
    ns['ids_leg'] = set(ns['legacy']['cod_cliente_dex'])
    ns['seg'] = pd.DataFrame({'customer_id': ['0000000001', '0000000002', '0000000003', '0000000004'], 'Segmento': 'Plus'})   # faltan G (alta) y... E ya está por D
    exec(_bloque('conciliacion_mensual'), ns)
    return ns


def test_conciliacion_separa_altas_bajas_y_cambios_de_codigo(tmp_path, monkeypatch, capsys):
    ns = _conciliar(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert ns['_altas_reales'] == {'0000000007'} and ns['_bajas_reales'] == {'0000000006'}      # el cambio de código D->E no cuenta como alta ni baja
    assert len(ns['_cambios_nuevos']) == 1 and 'Altas: 1 | Bajas: 1 | Códigos antiguos nuevos este mes: 1' in out
    csv = pd.read_csv(tmp_path / 'conciliacion' / 'clientes_202610.csv', dtype=str)
    assert sorted(csv.tipo) == ['alta', 'baja', 'cambio_de_codigo', 'sin_segmento']
    sin = csv[csv.tipo == 'sin_segmento'].iloc[0]
    assert sin.cod_cliente_dex == '0000000007' and 'alta de este mes' in sin.detalle and 'segmentos_manual.csv' in sin.detalle


def test_conciliacion_sin_archivo_del_mes_anterior_solo_informa_los_sin_segmento(tmp_path, monkeypatch, capsys):
    ns = _conciliar(tmp_path, monkeypatch, con_mes_anterior=False)
    assert 'no hay archivo de clientes de 2026-09' in capsys.readouterr().out
    csv = pd.read_csv(tmp_path / 'conciliacion' / 'clientes_202610.csv', dtype=str)
    assert csv.tipo.tolist() == ['sin_segmento']


def test_una_baja_publicada_que_vino_solo_por_la_lista_sigue_siendo_accion(tmp_path, monkeypatch, capsys):
    """El caso de 0004047610: publicado en setiembre, ya no está en la lista ni en las hojas: no debe desaparecer del universo."""
    ent, ns = _entradas(tmp_path, monkeypatch, ['NUEVO'], publicadas={'BAJA': T('2026-09-01')})
    assert ent['BAJA'] == T('2026-09-01') and 'BAJA' in ns['ids_tratados']
    assert '[BAJAS] 1 clientes publicados' in capsys.readouterr().out


def test_no_se_mantiene_como_baja_a_quien_hoy_esta_en_la_accion_o_se_unifica_a_otro_codigo(tmp_path, monkeypatch):
    ent, ns = _entradas(tmp_path, monkeypatch, [], publicadas={'X': T('2026-09-01'), 'VIEJO': T('2026-09-01'), 'FUTURO': T('2026-11-01'), 'PREVIO': T('2026-02-01')},
                        mapa={'VIEJO': 'X'}, previos={'PREVIO'})
    assert set(ent) == {'X'}                                  # X ya es acción por las hojas; VIEJO se unifica a X; FUTURO entró después del cierre; PREVIO no es acción


def test_sin_publicado_no_hay_bajas_que_mantener(tmp_path, monkeypatch):
    ent, _ = _entradas(tmp_path, monkeypatch, ['A'], publicadas={})
    assert set(ent) == {'X', 'A'}


def test_avisa_si_el_runner_esta_desactualizado(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ns = {'pd': pd, 'os': os, 'solo_leg': set(), 'MODO_INCREMENTAL': False, 'LEGACY_ENTRADA_SIN_MES': T('2026-10-01'), 'MES_CIERRE': T('2026-10-01'),
          'RUTA_PARES_CONGELADOS': 'x.csv', 'RUTA_SALIDA': '.', 'mapa_homologacion': {}, 'ids_previos': set(),
          'df_cosecha': pd.DataFrame({'cod_cliente_dex': ['X'], 'min_periodo_cosecha': [T('2026-04-01')]}), 'ids_tratados': {'X'}}
    exec(_bloque('solo_leg_entrada'), ns)                    # sin ENTRADAS_PUBLICADAS
    assert 'DESACTUALIZADOS' in capsys.readouterr().out
    ns['ENTRADAS_PUBLICADAS'] = {}
    exec(_bloque('solo_leg_entrada'), ns)
    assert 'DESACTUALIZADOS' not in capsys.readouterr().out
