"""customer_feat de la metodología de cosechas: sirve a surtido (con las columnas de segmentación estratégica) y a Champion (que no las tiene)."""
from pathlib import Path

import pandas as pd
import pytest

RUTA = Path(__file__).parent.parent / 'src' / 'mediciones' / 'familias' / 'cosechas' / 'metodologia.py'
OPCIONALES = ['flg_cliente_roma', 'cosecha_roma', 'es_roma', 'des_segmento_estrategica', 'flg_potential']
BASE = ['cod_cliente', 'des_oficina', 'nom_compania', 'des_segmento_transaccional']


def _region():
    L = RUTA.read_text(encoding='utf-8').split('\n')
    i = [k for k, l in enumerate(L) if l.startswith('# [customer_feat:inicio]')]
    j = [k for k, l in enumerate(L) if l.startswith('# [customer_feat:fin]')]
    assert len(i) == 1 and len(j) == 1
    return '\n'.join(L[i[0]:j[0] + 1])


def _frame(columnas):
    n = 6
    d = {c: [f'{c}{k % 3}' for k in range(n)] for c in columnas}
    d['cod_cliente'] = [f'{k % 3:010d}' for k in range(n)]                  # duplicados a propósito: customer_feat los colapsa
    return pd.DataFrame(d)


def _ejecutar(df):
    ns = {'base_seg_limpio': df}
    exec(compile(_region(), 'customer_feat', 'exec'), ns)
    return ns['customer_feat']


def test_surtido_conserva_exactamente_las_nueve_columnas_en_su_orden():
    orden = ['cod_cliente', 'des_oficina', 'nom_compania', 'flg_cliente_roma', 'cosecha_roma', 'es_roma', 'des_segmento_transaccional', 'des_segmento_estrategica', 'flg_potential']
    df = _frame(orden + ['peso', 'periodo'])
    assert _ejecutar(df).columns.tolist() == orden


def test_champion_sin_las_columnas_de_surtido_no_se_cae():
    cf = _ejecutar(_frame(BASE + ['peso', 'des_zona', 'ffvv_impulso']))
    assert cf.columns.tolist() == BASE and len(cf) == 3
    assert not set(OPCIONALES) & set(cf.columns)


@pytest.mark.parametrize('falta', BASE[1:])
def test_si_falta_una_columna_que_se_usa_rio_abajo_falla_con_un_mensaje_claro(falta):
    with pytest.raises(AssertionError, match=falta):
        _ejecutar(_frame([c for c in BASE if c != falta]))
