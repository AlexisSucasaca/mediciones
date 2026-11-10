import datetime as dt

from mediciones import normalizacion as nz
from mediciones import paridad
from conftest import construir_crudas


def test_paridad_detecta_alteraciones_y_faltantes():
    crudo, _ = construir_crudas(['2026-08-01', '2026-09-01'], n_clientes=10)
    ton = nz.preparar_base_ton(crudo, 'cpb', dt.date(2026, 9, 1), 'c')
    assert paridad.paridad_con_legacy(ton, crudo)['coincide']
    alt = crudo.copy()
    alt.loc[alt.index[:3], 'toneladas_actual'] += 0.5
    r = paridad.paridad_con_legacy(ton, alt)
    assert not r['coincide'] and r['max_dif_toneladas_actual'] == 0.5
    r = paridad.paridad_con_legacy(ton, crudo.iloc[:-4])
    assert not r['coincide'] and r['solo_en_motor'] == 4


def test_paridad_acepta_la_base_de_calculo_de_cpb_cpm():
    crudo, _ = construir_crudas(['2026-08-01', '2026-09-01'], n_clientes=6)
    ton = nz.preparar_base_ton(crudo, 'cpb', dt.date(2026, 9, 1), 'c')
    ref = crudo.rename(columns={'valor_atribuible': 'ton_atribuible'}).assign(rol='accion_matcheado')
    ctrl = ref.head(5).assign(rol='control_espejo', ton_atribuible=float('nan'), cod_cliente='0000099999')       # controles en la misma hoja
    assert paridad.paridad_con_legacy(ton, __import__('pandas').concat([ref, ctrl]))['coincide']


def test_paridad_suma_las_filas_repetidas_de_la_referencia_como_lo_hace_el_motor():
    """Inka Chips: la referencia trae varias filas por cliente-mes (4 categorías reales bajo una etiqueta); el motor ya las entrega sumadas."""
    import pandas as pd
    crudo, _ = construir_crudas(['2026-08-01', '2026-09-01'], n_clientes=6)
    ik = crudo[crudo['des_categoria'] == 'Galletas'].copy().assign(des_categoria='Inkachips')
    partes = []
    for ant, act in [(0.0, 0.002), (0.001, 0.0004), (0.002, 0.005)]:
        partes.append(ik.assign(toneladas_ant=ant, toneladas_actual=act, valor_atribuible=act - ant))
    legacy = pd.concat([crudo[crudo['des_categoria'] != 'Inkachips']] + partes, ignore_index=True)           # con llaves repetidas
    suma = ik.assign(toneladas_ant=0.003, toneladas_actual=0.0074, valor_atribuible=0.0044)
    motor = nz.preparar_base_ton(pd.concat([crudo[crudo['des_categoria'] != 'Inkachips'], suma], ignore_index=True), 'surtido_motor', dt.date(2026, 9, 1), 'c', regla_estado='programa')
    r = paridad.paridad_con_legacy(motor, legacy, regla_estado='programa')
    assert r['coincide'], r
    assert r['filas_legacy_agregadas'] == 2 * len(ik)                    # 3 filas por llave -> 1 (se agregaron 2 de cada 3)


def test_paridad_con_regla_del_programa_compara_contra_lo_reportado_sin_reaplicar_reglas():
    import pandas as pd
    crudo, _ = construir_crudas(['2026-09-01'], n_clientes=4)
    crudo = crudo.copy()
    crudo.loc[crudo.index[:2], ['toneladas_ant', 'toneladas_actual', 'valor_atribuible', 'flg_estado_yoy']] = [0.0, 0.7, 0.0, 'reactivado']
    motor = nz.preparar_base_ton(crudo, 'surtido_motor', dt.date(2026, 9, 1), 'c', regla_estado='programa')
    ok = paridad.paridad_con_legacy(motor, crudo, regla_estado='programa')
    assert ok['coincide'] and ok['suma_valor_atribuible_motor'] == ok['suma_valor_atribuible_legacy']
    # con la regla estándar la referencia se transformaría (reactivado -> todo) y la diferencia con el motor SÍ se vería
    assert not paridad.paridad_con_legacy(motor, crudo)['coincide']
