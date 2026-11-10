"""El bloque de espejos de la familia de cosechas se ejecuta aquí con un espacio de nombres simulado (el motor completo necesita tus datos)."""
import datetime as dt

import numpy as np
import pytest
import pandas as pd

from mediciones import normalizacion as nz
from mediciones.ejecucion import DIR_PAQUETE

COS = pd.Timestamp('2026-08-01')
REP = dt.date(2026, 9, 1)


def _ns():
    filas, pares = [], []
    for cl in ('0000000001', '0000000002', '0000000003', '0000000004'):
        for ct in ('Galletas', 'Cereales', 'Inkachips'):
            for per, rel, retro in (('2026-07-01', -1, True), ('2026-08-01', 0, False), ('2026-09-01', 1, False)):
                filas.append(dict(cod_cliente=cl, cod_cliente_dex=cl, des_categoria=ct, des_oficina='Sur', des_segmento_transaccional='2 Oro',
                                  nom_compania='DEX', flg_cliente_perfecto=np.nan, cosecha_grupo=COS, periodo=pd.Timestamp(per), mes_rel=rel,
                                  toneladas_ant=1.0, toneladas_actual=1.5, yoy_pct_raw=50., ic_pct_inf_80=5., ic_pct_sup_80=15.,
                                  flg_estado_yoy='x', crec_atrib_80_pct=10., ton_atrib_80=.1, flg_estimado_retroactivo=retro, valor_atribuible=.1))
                if ct != 'Inkachips' and cl != '0000000004':          # el cliente 4 queda sin pareja
                    pares.append(dict(par_id=f'{COS.date()}|{ct}|{cl}', cod_cliente_accion_dex=cl, nom_compania_accion='DEX',
                                      flg_cliente_perfecto=np.nan, cosecha_grupo=COS, cosecha_label=None, des_categoria=ct, des_oficina='Sur',
                                      des_segmento_transaccional='2 Oro', periodo=pd.Timestamp(per), mes_rel=rel, pre_accion=1000., post_accion=1500.,
                                      pre_control=900., post_control=950., did_kg=450., flg_estimado_retroactivo=retro, brecha_pct_celda=12., reportable=True))
    ton = pd.DataFrame(filas)
    matches = pd.DataFrame([dict(cosecha=COS, des_categoria=ct, cod_accion=cl, cod_control=f'{900 + int(cl)}',                  # ids SIN ceros: se normalizan
                                 ofi_ct='Norte', comp_ct='DEX2', seg_ct='3 Plata')
                            for cl in ('0000000001', '0000000002', '0000000003') for ct in ('Galletas', 'Cereales')])
    sin = pd.DataFrame([dict(cosecha=COS, des_categoria=ct, cod_accion='0000000004') for ct in ('Galletas',)])
    return dict(base_calculo_toneladas=ton, base_calculo_brecha=pd.DataFrame(pares), df_matches=matches, df_sin_pareja=sin,
                CATEGORIAS=['Galletas', 'Cereales'], GRUPO_COSECHA={COS: 'GRUPO 03'}, __name__='__main__')


def test_espejos_de_la_familia_de_cosechas():
    ns = _ns()
    exec(compile((DIR_PAQUETE / 'familias' / 'cosechas' / 'exportar.py').read_text(encoding='utf-8'), 'exportar.py', 'exec'), ns)
    esp = ns['base_calculo_espejos']
    assert set(esp.des_categoria) == {'Galletas', 'Cereales'}                         # Inka Chips no tiene espejo
    assert len(esp) == 4 * 2 * 3                                                       # cada cliente de acción x categoría x mes (incluye el retroactivo)
    con, sin = esp[esp.cod_cliente_espejo.notna()], esp[esp.cod_cliente_espejo.isna()]
    assert len(con) == 3 * 2 * 3 and set(con.cod_cliente_espejo) == {'0000000901', '0000000902', '0000000903'}     # espejos normalizados a 10 caracteres
    assert set(sin.cod_cliente_accion) == {'0000000004'}
    assert set(sin[sin.des_categoria == 'Galletas'].motivo_sin_espejo) == {'sin_pareja'}
    assert set(sin[sin.des_categoria == 'Cereales'].motivo_sin_espejo) == {'sin_match_registrado'}
    assert con.par_id.str.startswith('2026-08-01|').all() and esp.flg_estimado_retroactivo.sum() == 4 * 2         # los meses retroactivos conservan su marca
    # coherente con las otras dos bases del mismo motor
    ton = nz.preparar_base_ton(ns['base_calculo_toneladas'], 'surtido_motor', REP, 'c', regla_estado='programa')
    bre = nz.preparar_base_brechas(ns['base_calculo_brecha'], 'surtido_motor', REP, 'c')
    e = nz.preparar_base_espejos(esp, 'surtido_motor', REP, 'c')
    err, res = nz.validar(ton, bre, REP, esp=e)
    assert err == [] and res['filas_espejos'] == len(esp)
    assert list(ns['base_calculo_brecha'].columns[-4:]) == ['cod_cliente_control', 'n_pares_celda', 'ic_pct_inf_80_celda', 'ic_pct_sup_80_celda']


def _con_inka_duplicado():
    """Patrón real del notebook: 4 categorías reales recodificadas a 'Inkachips' sin agregar -> varias filas por cliente-mes."""
    ns = _ns()
    ton = ns['base_calculo_toneladas']
    ik = ton[ton.des_categoria == 'Inkachips'].copy()
    subcat = []
    for i, (ant, act) in enumerate([(0.0, 0.002), (0.001, 0.0004), (0.0, 0.0), (0.002, 0.005)]):
        d = ik.copy()
        d['toneladas_ant'], d['toneladas_actual'], d['valor_atribuible'] = ant, act, act - ant
        d['crec_atrib_80_pct'] = (act - ant) / ant * 100 if ant > 0 else float('nan')
        subcat.append(d)
    ns['base_calculo_toneladas'] = pd.concat([ton[ton.des_categoria != 'Inkachips']] + subcat, ignore_index=True)
    return ns


def test_inka_chips_con_filas_repetidas_se_suma_al_grano_cliente_mes():
    ns = _con_inka_duplicado()
    n_ik, v_ik = 4 * 3, 4 * 3 * (0.002 + (0.0004 - 0.001) + 0.0 + (0.005 - 0.002))
    exec(compile((DIR_PAQUETE / 'familias' / 'cosechas' / 'exportar.py').read_text(encoding='utf-8'), 'exportar.py', 'exec'), ns)
    ton = ns['base_calculo_toneladas']
    assert not ton.duplicated(['cod_cliente', 'des_categoria', 'periodo']).any()
    ik = ton[ton.des_categoria == 'Inkachips']
    assert len(ik) == n_ik and abs(ik.valor_atribuible.sum() - v_ik) < 1e-12              # el total no cambia al agregar
    assert ((ik.valor_atribuible - (ik.toneladas_actual - ik.toneladas_ant)).abs() < 1e-12).all()
    assert (ik.toneladas_ant.round(6) == 0.003).all() and (ik.toneladas_actual.round(6) == 0.0074).all()
    assert (ik.crec_atrib_80_pct.round(4) == round((0.0074 - 0.003) / 0.003 * 100, 4)).all()       # % recalculado sobre la suma
    assert len(ton[ton.des_categoria != 'Inkachips']) == 4 * 2 * 3                          # Galletas y Cereales intactas
    err, _ = nz.validar(nz.preparar_base_ton(ton, 'surtido_motor', REP, 'c', regla_estado='programa'),
                        nz.preparar_base_brechas(ns['base_calculo_brecha'], 'surtido_motor', REP, 'c'), REP)
    assert err == []                                                                       # y la validación de carga ya no la rechaza


def test_llave_duplicada_en_galletas_sigue_siendo_error():
    ns = _ns()
    ton = ns['base_calculo_toneladas']
    ns['base_calculo_toneladas'] = pd.concat([ton, ton[ton.des_categoria == 'Galletas'].head(1)], ignore_index=True)
    with pytest.raises(AssertionError, match='llaves duplicadas'):
        exec(compile((DIR_PAQUETE / 'familias' / 'cosechas' / 'exportar.py').read_text(encoding='utf-8'), 'exportar.py', 'exec'), ns)


def _ejecutar_export(ns):
    exec(compile((DIR_PAQUETE / 'familias' / 'cosechas' / 'exportar.py').read_text(encoding='utf-8'), 'exportar.py', 'exec'), ns)
    ton = nz.preparar_base_ton(ns['base_calculo_toneladas'], 'surtido_motor', REP, 'c', regla_estado='programa')
    bre = nz.preparar_base_brechas(ns['base_calculo_brecha'], 'surtido_motor', REP, 'c')
    esp = nz.preparar_base_espejos(ns['base_calculo_espejos'], 'surtido_motor', REP, 'c')
    return ns, nz.validar(ton, bre, REP, esp=esp)


def _sin_toneladas(ns, cliente, periodo):
    ton = ns['base_calculo_toneladas']
    ns['base_calculo_toneladas'] = ton[~((ton.cod_cliente == cliente) & (ton.periodo == pd.Timestamp(periodo)) & ton.des_categoria.isin(['Galletas', 'Cereales']))]
    return ns


def test_par_mes_retroactivo_de_la_brecha_sin_toneladas_tiene_espejo_y_no_bloquea():
    """La brecha retroactiva se replica a todos los pares de la cosecha, aunque el cliente aún no apareciera ese mes en el sell-out."""
    ns, (err, res) = _ejecutar_export(_sin_toneladas(_ns(), '0000000002', '2026-07-01'))        # julio = mes retroactivo
    esp = ns['base_calculo_espejos']
    ret = esp[(esp.cod_cliente_accion == '0000000002') & (esp.periodo == pd.Timestamp('2026-07-01'))]
    assert set(ret.des_categoria) == {'Galletas', 'Cereales'} and ret.flg_estimado_retroactivo.all()
    assert (ret.cod_cliente_espejo == '0000000902').all()                                       # el espejo del par se conserva
    assert len(esp) == 4 * 2 * 3                                                                # no se pierde ninguna fila de espejo
    assert err == [] and res['espejos_retroactivos_sin_toneladas'] == 2


def test_mes_no_retroactivo_sin_toneladas_sigue_siendo_error():
    ns, (err, _) = _ejecutar_export(_sin_toneladas(_ns(), '0000000002', '2026-08-01'))          # agosto = mes de la cosecha, no retroactivo
    assert any('sin fila en la base de toneladas' in e for e in err)
