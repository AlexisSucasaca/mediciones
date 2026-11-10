import datetime as dt

import pandas as pd

from mediciones import incremental as inc
from mediciones import normalizacion as nz
from mediciones.esquema import CLAVE, T_ESP, T_TON
from conftest import construir_crudas, construir_espejos

AGO, SEP = dt.date(2026, 8, 1), dt.date(2026, 9, 1)


def _base(periodos, **kw):
    ton, bre = construir_crudas(periodos, **kw)
    return nz.preparar_base_ton(ton, 'surtido', SEP, 'c')


def _publicado(ton, hasta):
    p = ton[ton.periodo <= hasta][CLAVE[T_TON] + ['valor_atribuible']].copy()
    p['periodo_reporte'] = hasta
    p['version'] = 1
    return p


def test_solo_entra_el_mes_nuevo():
    ton = _base(['2026-07-01', '2026-08-01', '2026-09-01'])
    nuevas, r, deriva = inc.comparar_con_publicado(ton, _publicado(ton, AGO), T_TON)
    assert len(nuevas) == (ton.periodo == SEP).sum() and r['filas_con_deriva'] == 0 and deriva.empty
    assert set(nuevas.periodo) == {SEP}


def test_primera_carga_entra_todo():
    ton = _base(['2026-08-01', '2026-09-01'])
    nuevas, r, _ = inc.comparar_con_publicado(ton, None, T_TON)
    assert len(nuevas) == len(ton) and r['filas_ya_publicadas'] == 0


def test_deriva_se_reporta_y_no_se_aplica():
    ton = _base(['2026-07-01', '2026-08-01', '2026-09-01'])
    pub = _publicado(ton, AGO)
    idx = pub.sample(7, random_state=1).index
    pub.loc[idx, 'valor_atribuible'] += 0.5
    nuevas, r, deriva = inc.comparar_con_publicado(ton, pub, T_TON)
    assert r['filas_con_deriva'] == 7 and abs(r['suma_abs_deriva'] - 3.5) < 1e-9
    assert len(nuevas) == (ton.periodo == SEP).sum()           # lo cambiado NO entra como nuevo


def test_llaves_publicadas_ausentes_hoy():
    ton = _base(['2026-08-01', '2026-09-01'])
    pub = _publicado(ton, AGO)
    extra = pd.DataFrame({'cod_cliente': ['9999999999'] * 2, 'des_categoria': ['Galletas'] * 2, 'periodo': [AGO] * 2,
                          'valor_atribuible': [1.0] * 2, 'periodo_reporte': [AGO] * 2, 'version': [1] * 2})
    _, r, _ = inc.comparar_con_publicado(ton, pd.concat([pub, extra]), T_TON)
    assert r['filas_publicadas_ausentes_hoy'] == 2


def test_entrante_retroactivo_entra_como_llave_nueva_con_periodo_viejo():
    ton = _base(['2026-07-01', '2026-08-01', '2026-09-01'])
    pub = _publicado(ton, AGO)
    quitar = ton[ton.periodo <= AGO].head(10)[CLAVE[T_TON]]
    pub = pub.merge(quitar, on=CLAVE[T_TON], how='left', indicator=True)
    pub = pub[pub._merge == 'left_only'].drop(columns='_merge')
    nuevas, _, _ = inc.comparar_con_publicado(ton, pub, T_TON)
    assert (nuevas.periodo <= AGO).sum() == 10


def test_reproceso_republica_el_bloque_completo():
    ton = _base(['2026-07-01', '2026-08-01', '2026-09-01'])
    pub = _publicado(ton, AGO)
    rep = inc.seleccionar_para_reproceso(ton, pub, T_TON, AGO)
    assert len(rep) == len(ton)                                  # bloque de agosto + lo nuevo de setiembre


def test_deriva_en_el_espejo_publicado_valor_de_texto():
    ton, bre = construir_crudas(['2026-07-01', '2026-08-01', '2026-09-01'], sin_espejo_frac=0.2)
    esp = nz.preparar_base_espejos(construir_espejos(ton, bre), 'cpb', SEP, 'c')
    pub = esp[esp.periodo <= AGO][CLAVE[T_ESP] + ['cod_cliente_espejo']].copy()
    pub['periodo_reporte'], pub['version'] = AGO, 1
    con = pub[pub.cod_cliente_espejo.notna()].index[:4]
    pub.loc[con, 'cod_cliente_espejo'] = '0000000042'                      # el espejo publicado era otro
    nuevas, r, deriva = inc.comparar_con_publicado(esp, pub, T_ESP)
    assert r['filas_con_deriva'] == 4 and len(deriva) == 4
    assert len(nuevas) == (esp.periodo == SEP).sum()                       # lo cambiado no entra como nuevo
    _, r0, _ = inc.comparar_con_publicado(esp, _pub_igual(esp), T_ESP)     # sin espejo vs sin espejo NO es deriva
    assert r0['filas_con_deriva'] == 0


def _pub_igual(esp):
    p = esp[esp.periodo <= AGO][CLAVE[T_ESP] + ['cod_cliente_espejo']].copy()
    p['periodo_reporte'], p['version'] = AGO, 1
    return p
