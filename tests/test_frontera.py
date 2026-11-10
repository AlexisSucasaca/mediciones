"""Frontera de meses cerrados: lo publicado no cambia; cada cierre solo publica meses nuevos (salvo provisionales o reapertura explícita)."""
import pandas as pd
import pytest
import yaml

from mediciones import frontera
from mediciones.esquema import T_BRE, T_ESP, T_TON
from conftest import ServiciosMemoria
from test_runner_memoria import AGO, JUL, SEP, _crudas, _meses, correr, motor  # noqa: F401  (motor es un fixture)

TODO = {T_TON: 100, T_BRE: 100, T_ESP: 100}      # la deriva de los meses cerrados se mide, pero en estas pruebas no bloquea
SIN_TOPE = {'brecha_pp': 1e9, 'ton_pct': 1e9, 'ton_abs': 1e9}


def _pub(filas):
    return pd.DataFrame(filas, columns=['periodo', 'periodo_reporte'])


# ── Unidad ──
def test_frontera_es_el_ultimo_mes_publicado_por_cierres_anteriores():
    base = _pub([('2026-01-01', '2026-09-01'), ('2026-09-01', '2026-09-01')])         # línea base sembrada: ene-sep en el cierre de septiembre
    assert frontera.calcular_frontera(None, '2026-10-01') is None                     # nada publicado: primera carga
    assert frontera.calcular_frontera(base, '2026-10-01') == pd.Timestamp('2026-09-01')
    assert frontera.calcular_frontera(base, '2026-09-01') is None                     # el propio cierre no cuenta
    assert frontera.rebanada_existente(base, '2026-09-01') and not frontera.rebanada_existente(base, '2026-10-01')


def test_provisionales_y_reapertura_recortan_la_frontera():
    pub = _pub([('2026-09-01', '2026-09-01'), ('2026-10-01', '2026-10-01')])
    assert frontera.calcular_frontera(pub, '2026-11-01') == pd.Timestamp('2026-10-01')
    assert frontera.calcular_frontera(pub, '2026-11-01', provisional_desde='2026-10-01') == pd.Timestamp('2026-09-01')
    assert frontera.calcular_frontera(pub, '2026-11-01', reabrir_desde='2026-08-01') == pd.Timestamp('2026-07-01')


def test_separar_publica_solo_lo_posterior_a_la_frontera():
    df = pd.DataFrame({'periodo': pd.to_datetime(['2026-08-01', '2026-09-01', '2026-10-01'])})
    pub, cer = frontera.separar(df, pd.Timestamp('2026-09-01'))
    assert pub['periodo'].dt.month.tolist() == [10] and cer['periodo'].dt.month.tolist() == [8, 9]
    pub, cer = frontera.separar(df, None)
    assert len(pub) == 3 and len(cer) == 0


def test_vigente_toma_por_mes_la_rebanada_mas_reciente_y_su_ultima_version():
    d = pd.DataFrame({'programa': 'x', 'cod_cliente': ['a', 'b', 'a', 'a', 'a'],
                      'periodo': pd.to_datetime(['2026-10-01', '2026-10-01', '2026-10-01', '2026-09-01', '2026-09-01']),
                      'periodo_reporte': pd.to_datetime(['2026-10-01', '2026-10-01', '2026-11-01', '2026-09-01', '2026-09-01']),
                      'version': [1, 1, 1, 1, 2], 'v': [1, 2, 3, 4, 5]})
    v = frontera.vigente_local(d).sort_values(['periodo', 'cod_cliente'])
    # sep: versión 2 del cierre de sep; oct: la rebanada de nov reemplaza COMPLETO a la de oct (b ya no sale)
    assert v['v'].tolist() == [5, 3]


# ── Runner de punta a punta ──
def _con_provisional(raiz, programa, desde):
    ruta = raiz / 'config' / 'programas.yaml'
    cfg = yaml.safe_load(ruta.read_text(encoding='utf-8'))
    cfg['programas'][programa]['provisional_desde'] = desde
    ruta.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')


def _vigente(svc, tabla, programa='cpb'):
    return frontera.vigente_local(svc.t[tabla][svc.t[tabla].programa == programa])


def _agregar_alta_tardia(motor, mes, en_periodo):
    """El recálculo de `mes` trae un cliente nuevo en un periodo ya cerrado (como los 8 de CPB)."""
    ton, bre, esp = motor['por_mes'][mes]
    fila = ton[ton.periodo == pd.Timestamp(en_periodo)].head(1).copy()
    fila['cod_cliente'] = '0000888888'
    motor['por_mes'][mes] = (pd.concat([ton, fila], ignore_index=True), bre, esp)


def test_el_cierre_nuevo_publica_solo_su_mes_y_lo_cerrado_no_cambia(raiz_tmp, motor):
    _meses(motor)                                            # setiembre recalcula agosto distinto (5 toneladas y 3 espejos)
    _agregar_alta_tardia(motor, SEP, AGO)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')                # "línea base": jul-ago
    antes = _vigente(svc, T_TON).copy()
    c = correr(raiz_tmp, SEP, svc, modo='cargar', umbral_deriva=TODO, tolerancia_celda=SIN_TOPE)
    r = c.res['cpb']['incr_ton']
    assert r['frontera'] == AGO and r['meses_a_publicar'] == ['2026-09']
    assert r['filas_cerradas_con_deriva'] == 5 and r['filas_cerradas_que_no_estaban_publicadas'] == 1   # se miden...
    cargado_sep = svc.t[T_TON][svc.t[T_TON].periodo_reporte == pd.Timestamp(SEP).date()]
    assert set(pd.to_datetime(cargado_sep.periodo).dt.strftime('%Y-%m')) == {'2026-09'}                # ...pero no se publican
    v = _vigente(svc, T_TON)
    cerrado = v[pd.to_datetime(v.periodo) <= pd.Timestamp(AGO)].sort_values(['cod_cliente', 'des_categoria', 'periodo'])
    assert cerrado[['cod_cliente', 'des_categoria', 'periodo', 'valor_atribuible']].reset_index(drop=True).equals(
        antes.sort_values(['cod_cliente', 'des_categoria', 'periodo'])[['cod_cliente', 'des_categoria', 'periodo', 'valor_atribuible']].reset_index(drop=True))
    assert '0000888888' not in set(v.cod_cliente)            # el alta tardía no entra a un mes cerrado
    esp_v = _vigente(svc, T_ESP)
    assert (esp_v.cod_cliente_espejo == '0000999999').sum() == 0     # los espejos publicados de agosto no se pisan


def test_volver_a_cargar_un_cierre_publicado_no_carga_nada_y_reproceso_sube_version(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    n = len(svc.t[T_TON])
    c = correr(raiz_tmp, AGO, svc, modo='cargar')
    assert len(svc.t[T_TON]) == n and c.res['cpb']['incr_ton']['filas_nuevas'] == 0
    with pytest.raises(ValueError, match='motivo'):
        correr(raiz_tmp, AGO, svc, modo='cargar', reproceso=True)
    correr(raiz_tmp, AGO, svc, modo='cargar', reproceso=True, motivo='corrección acordada', umbral_deriva=TODO, tolerancia_celda=SIN_TOPE)
    assert svc.t[T_TON].version.max() == 2 and len(_vigente(svc, T_TON)) == n


def test_los_meses_provisionales_se_reemplazan_completos_en_cada_cierre(raiz_tmp, motor):
    _con_provisional(raiz_tmp, 'cpb', AGO)                   # como Impulso: desde agosto los meses se pueden revisar
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    c = correr(raiz_tmp, SEP, svc, modo='cargar', umbral_deriva=TODO, tolerancia_celda=SIN_TOPE)
    assert c.res['cpb']['incr_ton']['frontera'] == JUL and c.res['cpb']['incr_ton']['meses_a_publicar'] == ['2026-08', '2026-09']
    v = _vigente(svc, T_TON)
    ago = v[pd.to_datetime(v.periodo) == pd.Timestamp(AGO)]
    assert set(pd.to_datetime(ago.periodo_reporte).dt.strftime('%Y-%m')) == {'2026-09'}         # agosto vigente = el del cierre de septiembre
    jul = v[pd.to_datetime(v.periodo) == pd.Timestamp(JUL)]
    assert set(pd.to_datetime(jul.periodo_reporte).dt.strftime('%Y-%m')) == {'2026-08'}         # julio sigue fijo


def test_reabrir_meses_cerrados_exige_motivo_y_los_republica(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    with pytest.raises(ValueError, match='motivo'):
        correr(raiz_tmp, SEP, svc, modo='cargar', reabrir_desde=AGO)
    c = correr(raiz_tmp, SEP, svc, modo='cargar', reabrir_desde=AGO, motivo='nuevo insumo acordado', umbral_deriva=TODO, tolerancia_celda=SIN_TOPE)
    assert c.res['cpb']['incr_ton']['meses_a_publicar'] == ['2026-08', '2026-09']
    v = _vigente(svc, T_TON)
    assert set(pd.to_datetime(v[pd.to_datetime(v.periodo) == pd.Timestamp(AGO)].periodo_reporte).dt.strftime('%Y-%m')) == {'2026-09'}


def test_un_cierre_anterior_al_ultimo_publicado_se_bloquea(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, SEP, svc, modo='cargar')                # se publica jul-sep de una vez (primera carga)
    n = len(svc.t[T_TON])
    c = correr(raiz_tmp, AGO, svc, modo='cargar')            # alguien corre agosto después
    assert c.resumen.estado.tolist() == ['bloqueado'] and 'CIERRE' in c.res['cpb']['error']
    assert len(svc.t[T_TON]) == n                            # no se cargó nada
    c = correr(raiz_tmp, AGO, svc, modo='dry_run')           # en vista previa solo avisa
    assert c.resumen.estado.tolist() == ['ok']


def test_si_una_tabla_no_admite_la_carga_no_se_carga_ninguna(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()

    def verificar(df, tabla):
        if tabla == T_TON:
            raise ValueError("columnas obligatorias que esta carga deja vacías: ['rol']")
    svc.verificar_carga = verificar
    c = correr(raiz_tmp, AGO, svc, modo='cargar')
    assert c.resumen.estado.tolist() == ['bloqueado'] and 'rol' in c.res['cpb']['error']
    assert all(svc.t[t].empty for t in (T_TON, T_BRE, T_ESP))       # ni brechas ni espejos quedaron cargados a medias
