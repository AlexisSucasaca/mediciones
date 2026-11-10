"""Runner de punta a punta con servicios en memoria (sin BigQuery) y el motor simulado: varios meses consecutivos."""
from pathlib import Path

import pandas as pd
import pytest

from mediciones import runner
from mediciones.ejecucion import ResultadoFamilia
from mediciones.esquema import T_BRE, T_ESP, T_TON
from conftest import ServiciosMemoria, construir_crudas, construir_espejos

JUL, AGO, SEP = '2026-07-01', '2026-08-01', '2026-09-01'


def escribir_estado_local(workdir, programa, esp):
    """Imita al motor mensual real: guarda los pares congelados (conserva los ya fijados y agrega solo los nuevos)."""
    ruta = workdir / f'{programa}_pares_fijos.csv'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = esp[esp.cod_cliente_espejo.notna()][['periodo', 'des_categoria', 'cod_cliente_accion', 'cod_cliente_espejo']].rename(
        columns={'des_categoria': 'categoria', 'cod_cliente_accion': 'cod_accion', 'cod_cliente_espejo': 'cod_control'})
    nuevo['periodo'] = pd.to_datetime(nuevo['periodo']).dt.strftime('%Y-%m-%d')
    if ruta.exists():
        viejo = pd.read_csv(ruta, dtype=str)
        k = ['periodo', 'categoria', 'cod_accion']
        nuevo = nuevo[~nuevo.set_index(k).index.isin(viejo.set_index(k).index)]
        nuevo = pd.concat([viejo, nuevo], ignore_index=True)
    nuevo.to_csv(ruta, index=False)


@pytest.fixture
def motor(monkeypatch):
    """Motor falso: devuelve crudas por mes; `fallar` permite romper un programa."""
    estado = {'por_mes': {}, 'fallar': set(), 'llamadas': []}

    def falso(familia, script, programa, mes, workdir, familias, devolver_namespace=False, **kw):
        estado['kw'] = kw
        estado['llamadas'].append((programa, pd.Timestamp(mes).strftime('%Y-%m')))
        if programa in estado['fallar']:
            raise RuntimeError('BigQuery no respondió')
        ton, bre, esp = estado['por_mes'][pd.Timestamp(mes).strftime('%Y-%m-%d')]
        if esp is not None:
            escribir_estado_local(Path(workdir), programa, esp)
        ins = [{'key': 'k' + programa, 'content_hash': 'c' + pd.Timestamp(mes).strftime('%m'), 'origen': 'bigquery', 'filas': 10, 'columnas': 3,
                'sql_preview': 'SELECT 1', 'proyecto': None, 'archivo': 'x.parquet', 'segundos': 0.1}]
        return ResultadoFamilia(ton.copy(), bre.copy(), 3.0, None if esp is None else esp.copy(), None, ins)
    monkeypatch.setattr(runner, 'ejecutar_familia', falso)
    return estado


def _crudas(periodos, **kw):
    ton, bre = construir_crudas(periodos, seed=1, sin_espejo_frac=0.2, **kw)
    return ton, bre, construir_espejos(ton, bre)


def _meses(motor):
    motor['por_mes'][AGO] = _crudas([JUL, AGO])
    # setiembre: mismos datos de jul-ago (con 5 filas de agosto recalculadas distinto) + el mes nuevo
    ton, bre, esp = _crudas([JUL, AGO, SEP])
    # solo clientes antiguos con compra: en los 'nuevos' el atribuible siempre es lo comprado y la normalización lo fija
    cambio = ton[(ton.periodo == pd.Timestamp(AGO)) & (ton.toneladas_ant > 0) & (ton.toneladas_actual > 0)].head(5)
    for _, r in cambio.iterrows():
        ton.loc[(ton.cod_cliente == r.cod_cliente) & (ton.des_categoria == r.des_categoria) & (ton.periodo == r.periodo), 'valor_atribuible'] += 0.25
    # y 3 clientes de agosto cambian de espejo al recalcular (no debe pisarse lo ya publicado)
    quien = esp[(esp.periodo == pd.Timestamp(AGO)) & esp.cod_cliente_espejo.notna()].head(3).index
    esp.loc[quien, 'cod_cliente_espejo'] = '0000999999'
    bre.loc[bre.set_index(['cod_cliente_accion_dex', 'des_categoria', 'periodo']).index.isin(
        esp.loc[quien].set_index(['cod_cliente_accion', 'des_categoria', 'periodo']).index), 'cod_cliente_control'] = '0000999999'
    motor['por_mes'][SEP] = (ton, bre, esp)


def correr(raiz, mes, svc, **kw):
    kw.setdefault('programas', ['cpb'])
    return runner.correr_mes(mes, kw.pop('programas'), raiz=raiz, servicios=svc, verbose=False, exportar_excel=False, **kw)


def test_dos_cierres_consecutivos_solo_agregan_lo_nuevo(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    c1 = correr(raiz_tmp, AGO, svc, modo='cargar')
    assert c1.resumen.estado.tolist() == ['ok']
    n_ago = len(svc.t[T_TON])
    assert n_ago > 0 and [c[0] for c in svc.cargas[:3]] == [T_BRE, T_ESP, T_TON]          # brechas, espejos y al final toneladas
    assert svc.t[T_TON].ub_atribuible_soles.notna().any()                                 # UB calculada vía servicios
    antes = svc.t[T_TON].copy()

    bloqueado = correr(raiz_tmp, SEP, svc, modo='cargar')              # +0.25 t en 5 clientes de agosto: 2 celdas mes x categoría fuera de 1 % -> bloquea
    assert bloqueado.resumen.estado.tolist() == ['bloqueado'] and len(svc.t[T_TON]) == n_ago
    assert 'DERIVA' in bloqueado.res['cpb']['error'] and 'base_ton_atribuibles' in bloqueado.res['cpb']['error'] and 'celdas' in bloqueado.res['cpb']['error']
    assert 'base_espejos' not in bloqueado.res['cpb']['error']        # los espejos que cambian se informan, no bloquean por defecto
    c2 = correr(raiz_tmp, SEP, svc, modo='cargar', aceptar_deriva=True, motivo_deriva='recalculo validado con negocio')
    assert c2.resumen.estado.tolist() == ['ok'] and any(f['metrica'] == 'aceptada' for f in c2.log)
    r = c2.res['cpb']
    assert r['incr_ton']['filas_nuevas'] == (r['ton'].periodo == pd.Timestamp(SEP).date()).sum()
    assert r['incr_ton']['filas_con_deriva'] == 5                                        # se reporta…
    assert r['incr_esp']['filas_con_deriva'] == 3 and r['incr_esp']['filas_nuevas'] == (r['esp'].periodo == pd.Timestamp(SEP).date()).sum()
    pub = svc.t[T_TON]
    assert len(pub) == n_ago + r['incr_ton']['filas_nuevas']
    pd.testing.assert_frame_equal(pub.iloc[:n_ago].reset_index(drop=True), antes.reset_index(drop=True))   # …y NO se pisa
    assert (svc.t[T_ESP].cod_cliente_espejo != '0000999999').all()                         # tampoco el espejo ya publicado
    assert set(pub[pub.periodo_reporte == pd.Timestamp(SEP).date()].periodo) == {pd.Timestamp(SEP).date()}


def test_repetir_el_mismo_cierre_no_duplica(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    n = len(svc.t[T_TON])
    c = correr(raiz_tmp, AGO, svc, modo='cargar')
    assert len(svc.t[T_TON]) == n and c.res['cpb']['incr_ton']['filas_nuevas'] == 0


def test_dry_run_no_escribe(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    c = correr(raiz_tmp, AGO, svc, modo='dry_run')
    assert svc.cargas == [] and c.res['cpb']['incr_ton']['filas_nuevas'] > 0


def test_un_programa_que_falla_no_detiene_a_los_demas(raiz_tmp, motor):
    _meses(motor)
    motor['fallar'].add('cpm')
    motor['por_mes'][AGO] = _crudas([JUL, AGO])
    c = correr(raiz_tmp, AGO, ServiciosMemoria(), modo='dry_run', programas=['cpm', 'cpb'])
    estados = dict(zip(c.resumen.programa, c.resumen.estado))
    assert estados == {'cpm': 'falló', 'cpb': 'ok'} and 'BigQuery no respondió' in c.res['cpm']['error']


def test_reproceso_publica_una_version_nueva_del_bloque(raiz_tmp, motor):
    _meses(motor)
    svc = ServiciosMemoria()
    correr(raiz_tmp, AGO, svc, modo='cargar')
    correr(raiz_tmp, SEP, svc, modo='cargar', aceptar_deriva=True, motivo_deriva='recalculo validado')
    correr(raiz_tmp, SEP, svc, modo='cargar', reproceso=True, motivo='corrección de insumo')
    assert svc.ultima_version(T_TON, 'cpb', pd.Timestamp(SEP).date()) == 2
    vig = svc._vigente(T_TON, 'cpb')
    assert set(vig[vig.periodo_reporte == pd.Timestamp(SEP).date()].version) == {2}      # la vista vigente toma la v2
    with pytest.raises(ValueError):
        correr(raiz_tmp, SEP, svc, modo='cargar', reproceso=True)                        # sin motivo


def test_validacion_con_errores_bloquea_la_carga(raiz_tmp, motor):
    ton, bre, esp = _crudas([JUL, AGO])
    motor['por_mes'][AGO] = (pd.concat([ton, ton.head(4)]), bre, esp)                    # llaves duplicadas
    svc = ServiciosMemoria()
    c = correr(raiz_tmp, AGO, svc, modo='cargar')
    assert c.resumen.estado.tolist() == ['con errores'] and svc.cargas == []


def test_ejecutar_false_reutiliza_la_corrida_guardada(raiz_tmp, motor):
    _meses(motor)
    correr(raiz_tmp, AGO, ServiciosMemoria(), modo='dry_run')
    motor['llamadas'].clear()
    c = correr(raiz_tmp, AGO, ServiciosMemoria(), modo='dry_run', ejecutar=False)
    assert motor['llamadas'] == [] and c.resumen.estado.tolist() == ['ok']


def test_entradas_invalidas(raiz_tmp, motor):
    with pytest.raises(ValueError):
        correr(raiz_tmp, '2026-09-15', ServiciosMemoria())
    with pytest.raises(ValueError):
        correr(raiz_tmp, SEP, ServiciosMemoria(), programas=['no_existe'])
    with pytest.raises(ValueError):
        correr(raiz_tmp, SEP, ServiciosMemoria(), modo='otro')


def test_un_motor_sin_espejos_carga_solo_toneladas_y_brechas(raiz_tmp, motor):
    ton, bre, _ = _crudas([JUL, AGO])
    motor['por_mes'][AGO] = (ton, bre, None)              # p. ej. los motores que todavía son notebooks no emiten espejos
    svc = ServiciosMemoria()
    c = correr(raiz_tmp, AGO, svc, modo='cargar')
    assert c.resumen.estado.tolist() == ['ok'] and [x[0] for x in svc.cargas] == [T_BRE, T_TON] and svc.t[T_ESP].empty


def test_espejos_inconsistentes_bloquean_la_carga(raiz_tmp, motor):
    ton, bre, esp = _crudas([JUL, AGO])
    esp = esp.copy()
    huerfano = esp.iloc[[0]].copy(); huerfano['cod_cliente_accion'] = '0000000777'      # cliente de acción que no está en la base de toneladas
    motor['por_mes'][AGO] = (ton, bre, pd.concat([esp, huerfano]))
    svc = ServiciosMemoria()
    c = correr(raiz_tmp, AGO, svc, modo='cargar')
    assert c.resumen.estado.tolist() == ['con errores'] and svc.cargas == []
    assert any('sin fila en la base de toneladas' in e for e in c.res['cpb']['errores'])


def test_cpb_calcula_la_ub_con_el_metodo_del_notebook(raiz_tmp, motor):
    """CPB pide a la consulta de UB los 12 meses anteriores (UB base) y la base queda con ub_base_soles y ub_por_ton_ref."""
    _meses(motor)
    svc = ServiciosMemoria()
    c1 = correr(raiz_tmp, AGO, svc, modo='cargar')
    assert svc.meses_atras_ub == 12
    ton = c1.res['cpb']['cargar_ton']
    assert ton['ub_base_soles'].notna().any() and ton['ub_por_ton_ref'].notna().any() and ton['ub_atribuible_soles'].notna().any()
