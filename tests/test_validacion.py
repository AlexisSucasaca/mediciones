"""validar_corrida: una corrida limpia no tiene ✘; cada defecto inyectado se detecta; la comparación con referencia diagnostica la causa de la diferencia."""
import numpy as np
import pandas as pd
import pytest

from conftest import RAIZ
from mediciones.ejecucion import DIR_PAQUETE
from mediciones.validacion import MAL, OK, validar_corrida
from test_exportar_cosechas import _ns

CONFIG = RAIZ / 'config' / 'programas.yaml'
ESTADO = {'0000000001': 'normal', '0000000002': 'reactivado', '0000000003': 'no_existia_periodo_pasado', '0000000004': 'sin_compra_ambos'}


def _bases():
    """Salida estándar del exportador de cosechas con estados coherentes con las reglas de atribución de surtido."""
    ns = _ns()
    t = ns['base_calculo_toneladas']
    gc = ~t.des_categoria.eq('Inkachips')
    t['flg_estado_yoy'] = np.where(gc, t.cod_cliente.map(ESTADO), None)
    act = t['toneladas_actual']
    t['valor_atribuible'] = np.select([t.flg_estado_yoy.eq('normal'), t.flg_estado_yoy.eq('no_existia_periodo_pasado')], [t['toneladas_ant'] * 0.10, act], default=0.0)
    t.loc[~gc, 'valor_atribuible'] = 0.1                                                          # Inka Chips: diferencial YoY, sin estado
    exec(compile((DIR_PAQUETE / 'familias' / 'cosechas' / 'exportar.py').read_text(encoding='utf-8'), 'exportar.py', 'exec'), ns)
    return ns['base_calculo_toneladas'].copy(), ns['base_calculo_brecha'].copy(), ns['base_calculo_espejos'].copy()


def _corrida(tmp_path, ton, bre, esp, nombre='surtido_motor'):
    d = tmp_path / 'runs' / '202609_20261001T000000_abc'
    d.mkdir(parents=True, exist_ok=True)
    ton.to_parquet(d / f'{nombre}_ton.parquet', index=False)
    bre.to_parquet(d / f'{nombre}_brecha.parquet', index=False)
    esp.to_parquet(d / f'{nombre}_espejos.parquet', index=False)
    return d


def _validar(tmp_path, **kw):
    return validar_corrida(tmp_path, config=CONFIG, verbose=False, **kw)


def _estado(out, texto):
    f = out[out.chequeo.str.contains(texto, regex=False)]
    assert len(f) >= 1, f'no hay chequeo con {texto!r}: {list(out.chequeo)}'
    return f.iloc[0].resultado


def test_una_corrida_limpia_no_tiene_ningun_error(tmp_path):
    _corrida(tmp_path, *_bases())
    out = _validar(tmp_path)
    assert not (out.resultado == MAL).any(), out[out.resultado == MAL][['chequeo', 'detalle']].to_string()
    assert _estado(out, 'DiD =') == OK and _estado(out, 'Estado reactivado') == OK and _estado(out, 'banda IC80') == OK
    assert (tmp_path / 'runs' / '202609_20261001T000000_abc' / 'validacion_202609.xlsx').exists()


@pytest.mark.parametrize('chequeo, defecto', [
    ('Llave única en toneladas', lambda t, b, e: (pd.concat([t, t.iloc[[0]]], ignore_index=True), b, e)),
    ('Estado reactivado', lambda t, b, e: (t.assign(valor_atribuible=np.where(t.flg_estado_yoy.eq('reactivado'), 0.5, t.valor_atribuible)), b, e)),
    ('Estado no_existia_periodo_pasado', lambda t, b, e: (t.assign(valor_atribuible=np.where(t.flg_estado_yoy.eq('no_existia_periodo_pasado'), 0.2, t.valor_atribuible)), b, e)),
    ('banda IC80', lambda t, b, e: (t.assign(valor_atribuible=np.where(t.flg_estado_yoy.eq('normal'), t.toneladas_ant * 0.90, t.valor_atribuible)), b, e)),   # 90 % > tope 15 %
    ('DiD =', lambda t, b, e: (t, b.assign(did_kg=b.did_kg + 7), e)),
    ('Validación estándar', lambda t, b, e: (t, b, e.iloc[: len(e) // 2])),                       # faltan espejos de pares reportados
    ('último periodo', lambda t, b, e: (t[t.periodo < pd.Timestamp('2026-09-01')], b, e)),        # sin el mes de cierre
])
def test_cada_defecto_inyectado_se_detecta(tmp_path, chequeo, defecto):
    t, b, e = defecto(*_bases())
    _corrida(tmp_path, t, b, e)
    out = _validar(tmp_path)
    assert _estado(out, chequeo) == MAL, out[['chequeo', 'resultado', 'detalle']].to_string()


def test_comparacion_con_referencia_diagnostica_la_causa(tmp_path):
    t, b, e = _bases()
    _corrida(tmp_path, t, b, e)
    ref = tmp_path / 'ref.parquet'

    def causa(ref_df):
        ref_df.to_parquet(ref, index=False)
        out = _validar(tmp_path, referencias={'surtido_motor': str(ref)})
        return _estado(out, 'Paridad fila a fila'), out[out.chequeo == 'Causa más probable de la diferencia'].iloc[0].detalle

    ok, c = causa(t.copy())
    assert ok == OK and c == 'sin diferencias'
    r = t.copy()
    r.loc[r.index[:5], 'toneladas_actual'] += 1.0                                              # datos de entrada distintos
    ok, c = causa(r)
    assert ok == MAL and 'TONELADAS de entrada' in c
    r = t.copy()
    r['ic_pct_sup_80'] = r['ic_pct_sup_80'] + 3.0                                              # solo cambió la banda IC
    r.loc[r.flg_estado_yoy.eq('normal'), 'valor_atribuible'] += 0.01
    ok, c = causa(r)
    assert ok == MAL and 'BANDA IC' in c
    r = t.copy()
    r.loc[r.flg_estado_yoy.eq('normal'), 'valor_atribuible'] += 0.01                           # mismas entradas y bandas, distinto atribuible
    ok, c = causa(r)
    assert ok == MAL and 'REGLA' in c
    r = t.iloc[:-6].copy()                                                                     # la población cambió
    ok, c = causa(r)
    assert ok == MAL and 'población' in c
    with pd.ExcelFile(tmp_path / 'runs' / '202609_20261001T000000_abc' / 'validacion_202609.xlsx') as x:
        assert {'Chequeos', 'surtido_motor totales', 'surtido_motor vs ref', 'surtido_motor top difs'} <= set(x.sheet_names)


def test_un_programa_que_no_se_puede_validar_no_tumba_a_los_demas(tmp_path):
    t, b, e = _bases()
    d = _corrida(tmp_path, t, b, e)
    pd.DataFrame({'x': [1]}).to_parquet(d / 'cpb_ton.parquet')                                  # base corrupta de otro programa
    pd.DataFrame({'x': [1]}).to_parquet(d / 'cpb_brecha.parquet')
    out = _validar(tmp_path)
    assert (out[out.programa == 'cpb'].resultado == MAL).any() and not (out[out.programa == 'surtido_motor'].resultado == MAL).any()


def test_el_estado_de_inka_chips_no_desactiva_las_reglas_de_los_demas(tmp_path):
    t, b, e = _bases()
    t.loc[t.des_categoria.eq('Inkachips'), 'flg_estado_yoy'] = 'cien_pct_atribuible_sin_control'
    _corrida(tmp_path, t, b, e)
    out = _validar(tmp_path)
    assert _estado(out, 'Estado reactivado') == OK and not (out.resultado == MAL).any()


def test_la_referencia_con_filas_repetidas_de_inka_se_suma_antes_de_comparar(tmp_path):
    t, b, e = _bases()
    _corrida(tmp_path, t, b, e)
    ik = t[t.des_categoria.eq('Inkachips')]
    mitad = ik.assign(toneladas_ant=ik.toneladas_ant / 2, toneladas_actual=ik.toneladas_actual / 2, valor_atribuible=ik.valor_atribuible / 2)
    ref = pd.concat([t[~t.des_categoria.eq('Inkachips')], mitad, mitad], ignore_index=True)           # 2 filas por llave que suman la fila del motor
    ref.to_parquet(tmp_path / 'ref.parquet', index=False)
    out = _validar(tmp_path, referencias={'surtido_motor': str(tmp_path / 'ref.parquet')})
    assert _estado(out, 'Paridad fila a fila') == OK
    assert 'valor_atribuible: 0' in out[out.chequeo == 'Filas comunes que difieren, por columna'].iloc[0].detalle


def test_la_comparacion_incluye_la_ub_y_la_marca_cuando_difiere(tmp_path):
    t, b, e = _bases()
    t['ub_atribuible_soles'] = t['valor_atribuible'] * 100.0
    t['ub_base_soles'] = 5.0
    _corrida(tmp_path, t, b, e)
    ref = t.copy()
    ref.to_parquet(tmp_path / 'ref.parquet', index=False)
    out = _validar(tmp_path, referencias={'surtido_motor': str(tmp_path / 'ref.parquet')})
    assert _estado(out, 'UB atribuible (motor vs referencia)') == OK
    ref.loc[ref.index[:7], 'ub_atribuible_soles'] += 1.0
    ref.to_parquet(tmp_path / 'ref.parquet', index=False)
    out = _validar(tmp_path, referencias={'surtido_motor': str(tmp_path / 'ref.parquet')})
    f = out[out.chequeo == 'UB atribuible (motor vs referencia)'].iloc[0]
    assert f.resultado == '⚠' and 'difieren: 7' in f.detalle
