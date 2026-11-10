"""
Salida de Champion hacia las 3 bases estándar: el pie de la familia (familias/champion/salida.py) + el exportador de cosechas, con datos con la forma que deja el
notebook de Champion (SIN cod_cliente_dex, flg_cliente_perfecto ni flg_estimado_retroactivo; con las columnas propias mes_exposicion, ton_con_crecimiento...).
"""
import numpy as np
import pandas as pd

from mediciones import normalizacion as nz
from mediciones.ejecucion import DIR_PAQUETE
from test_exportar_cosechas import REP, _ns

FAM = DIR_PAQUETE / 'familias'
NO_CALCULA = ['cod_cliente_dex', 'flg_cliente_perfecto', 'flg_estimado_retroactivo']


def _como_champion():
    ns = _ns()
    ton = ns['base_calculo_toneladas'].drop(columns=NO_CALCULA).copy()
    ton['mes_exposicion'] = ton['mes_rel'] + 1
    for c in ('ton_con_crecimiento', 'ton_sin_crecimiento', 'ton_nuevos', 'ton_neto_sin_nuevos'):
        ton[c] = 0.1
    ton['flg_sin_banda'] = False
    ton['reportable'] = True
    bre = ns['base_calculo_brecha'].rename(columns={'cod_cliente_accion_dex': 'cod_cliente_accion'}).drop(columns=['flg_cliente_perfecto', 'flg_estimado_retroactivo']).copy()
    bre['mes_exposicion'] = bre['mes_rel'] + 1
    ns.update(base_toneladas_export=ton, base_calculo_toneladas=ton[ton.des_categoria != 'Inkachips'], base_calculo_brecha=bre)
    return ns


def _correr(ns):
    for rel in ('champion/salida.py', 'cosechas/exportar.py'):
        exec(compile((FAM / rel).read_text(encoding='utf-8'), rel, 'exec'), ns)
    return ns


def test_champion_llega_a_las_tres_bases_y_valida_sin_errores():
    ns = _correr(_como_champion())
    ton = nz.preparar_base_ton(ns['base_calculo_toneladas'], 'champion_motor', REP, 'c', regla_estado='programa')
    bre = nz.preparar_base_brechas(ns['base_calculo_brecha'], 'champion_motor', REP, 'c')
    esp = nz.preparar_base_espejos(ns['base_calculo_espejos'], 'champion_motor', REP, 'c')
    err, res = nz.validar(ton, bre, REP, esp=esp)
    assert err == [], err
    assert set(ton.des_categoria) == {'Galletas', 'Cereales', 'Inkachips'}                   # Inka Chips viaja en la base de toneladas, como en la hoja del notebook
    assert (ton.cod_cliente_dex == ton.cod_cliente).all()
    assert not ton.flg_estimado_retroactivo.any() and ton.flg_cliente_perfecto.isna().all()   # lo que Champion no calcula queda declarado, no inventado
    assert len(esp) > 0 and esp.cod_cliente_espejo.notna().any()


def test_el_pie_no_pisa_lo_que_el_notebook_ya_calcula():
    ns = _como_champion()
    ton_antes = ns['base_toneladas_export'].copy()
    _correr(ns)
    out = ns['base_calculo_toneladas']
    assert out['valor_atribuible'].sum() == ton_antes['valor_atribuible'].sum() or abs(out['valor_atribuible'].sum() - ton_antes['valor_atribuible'].sum()) < 1e-9   # Inka suma pero conserva el total
