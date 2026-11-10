"""Motor mensual de punta a punta, EN PROCESO, con un programa sintético (cumple el mismo contrato de datos que cpb/cpm)."""
import datetime as dt
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from mediciones import normalizacion as nz
from mediciones.esquema import T_BRE, T_ESP, T_TON
from mediciones import runner
from mediciones.ejecucion import ejecutar_familia
from mediciones.registro import cargar_registro
from conftest import RAIZ, ServiciosMemoria

SINTETICO = Path(__file__).parent / 'programas_sinteticos' / 'sintetico.py'
FAMILIAS, _ = cargar_registro(RAIZ / 'config' / 'programas.yaml')
MES = pd.Timestamp('2026-09-01')


@pytest.mark.lento
def test_motor_mensual_en_proceso(tmp_path):
    wd = tmp_path / 'wd'
    r = ejecutar_familia('mensual', SINTETICO, 'sintetico', MES, wd, FAMILIAS)
    assert len(r.ton) > 500 and len(r.bre) > 300
    ton = nz.preparar_base_ton(r.ton, 'cpb', dt.date(2026, 9, 1), 't')
    bre = nz.preparar_base_brechas(r.bre, 'cpb', dt.date(2026, 9, 1), 't')
    assert r.esp is not None
    esp = nz.preparar_base_espejos(r.esp, 'cpb', dt.date(2026, 9, 1), 't')
    err, res = nz.validar(ton, bre, dt.date(2026, 9, 1), esp=esp)      # incluye la coherencia espejos <-> brechas <-> toneladas
    assert err == []
    assert len(esp) == len(ton) and (esp.estado_espejo == 'con_espejo').sum() >= len(bre)
    assert esp.loc[esp.estado_espejo == 'sin_espejo', 'motivo_sin_espejo'].notna().all()
    # el programa sintético siembra +18% en los clientes Perfecto: el DiD debe recuperar un efecto positivo del orden esperado
    brecha = bre.groupby(['des_categoria', 'periodo']).brecha_pct_celda.first()
    assert 5 < brecha.mean() < 35
    # la base de brechas lleva el control del par y los datos de la celda, como la base de cálculo de los notebooks CPB/CPM
    assert bre.cod_cliente_control.notna().all() and bre.cod_cliente_control.str.len().eq(10).all()
    assert bre.n_pares_celda.notna().all() and bre.brecha_pct_celda.notna().all()
    assert bre.ic_pct_inf_80_celda.notna().any() and (bre.ic_pct_inf_80_celda.dropna() < bre.ic_pct_sup_80_celda.dropna()).all()
    assert (wd / 'sintetico_pares_fijos.csv').exists()            # los pares quedan congelados para el mes siguiente
    # segunda corrida: reutiliza los pares fijos y reproduce exactamente las mismas bases
    r2 = ejecutar_familia('mensual', SINTETICO, 'sintetico', MES, wd, FAMILIAS)
    pd.testing.assert_frame_equal(r.ton.reset_index(drop=True), r2.ton.reset_index(drop=True))
    assert r2.segundos < r.segundos


@pytest.mark.lento
def test_runner_completo_con_motor_en_proceso(tmp_path):
    cfg = (RAIZ / 'config' / 'programas.yaml').read_text(encoding='utf-8')
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'programas.yaml').write_text(
        cfg + f"\n  sintetico:\n    familia: mensual\n    motor: familia\n    script: {SINTETICO.as_posix()}\n", encoding='utf-8')
    svc = ServiciosMemoria()
    co = runner.correr_mes(MES, ['sintetico'], modo='cargar', raiz=tmp_path, servicios=svc, verbose=False, exportar_excel=False)
    assert co.resumen.estado.tolist() == ['ok'] and [c[0] for c in svc.cargas] == [T_BRE, T_ESP, T_TON]
    n = len(svc.t['base_ton_atribuibles'])
    co2 = runner.correr_mes(MES, ['sintetico'], modo='cargar', raiz=tmp_path, servicios=svc, verbose=False, exportar_excel=False)
    assert len(svc.t['base_ton_atribuibles']) == n and co2.res['sintetico']['incr_ton']['filas_nuevas'] == 0


@pytest.mark.lento
def test_determinismo_desde_cero_y_con_distinto_numero_de_hilos(tmp_path):
    """Dos corridas DESDE CERO (sin pares fijos previos), una con otro número de hilos, deben dar exactamente las mismas bases y los mismos pares congelados."""
    a = ejecutar_familia('mensual', SINTETICO, 'sintetico', MES, tmp_path / 'a', FAMILIAS)
    with mock.patch('os.cpu_count', return_value=4):
        b = ejecutar_familia('mensual', SINTETICO, 'sintetico', MES, tmp_path / 'b', FAMILIAS)
    for nombre in ('ton', 'bre', 'esp'):
        pd.testing.assert_frame_equal(getattr(a, nombre).reset_index(drop=True), getattr(b, nombre).reset_index(drop=True), check_exact=True)
    pa, pb = (pd.read_csv(tmp_path / x / 'sintetico_pares_fijos.csv') for x in ('a', 'b'))
    k = list(pa.columns)
    assert pa.sort_values(k).reset_index(drop=True).equals(pb.sort_values(k).reset_index(drop=True))


@pytest.mark.lento
def test_calcular_solo_el_mes_nuevo_pasa_el_qa_y_da_lo_mismo(tmp_path):
    """CPB/CPM en un cierre normal calculan solo el mes nuevo (RECALCULAR_DESDE): el QA debe pasar y el mes debe salir igual que calculando todo."""
    wd = tmp_path / 'wd'
    completo = ejecutar_familia('mensual', SINTETICO, 'sintetico', MES, wd, FAMILIAS)
    ultimo = pd.to_datetime(completo.ton['periodo']).max()
    solo = ejecutar_familia('mensual', SINTETICO, 'sintetico', MES, wd, FAMILIAS, parametros={'RECALCULAR_DESDE': ultimo.strftime('%Y-%m-%d')})
    assert set(pd.to_datetime(solo.ton['periodo'])) == {ultimo}                     # solo el mes nuevo
    k = ['cod_cliente', 'des_categoria', 'periodo']
    a = completo.ton[pd.to_datetime(completo.ton['periodo']) == ultimo].sort_values(k).reset_index(drop=True)
    b = solo.ton.sort_values(k).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False)              # mismo resultado para ese mes
    assert solo.segundos < completo.segundos
