import re

import nbformat
import pytest

from mediciones.ejecucion import DIR_PAQUETE, inyectar_mes
from mediciones.registro import cargar_registro
from conftest import RAIZ

FAMILIAS, PROGRAMAS = cargar_registro(RAIZ / 'config' / 'programas.yaml')


def test_registro_completo():
    assert set(PROGRAMAS) == {'cpb', 'cpm', 'surtido_motor', 'champion_motor'}            # los motores-notebook se retiraron
    assert all(p.motor == 'familia' for p in PROGRAMAS.values())
    for p in PROGRAMAS.values():
        assert p.familia in FAMILIAS


@pytest.mark.parametrize('nombre', [n for n, p in PROGRAMAS.items() if p.motor == 'familia'])
def test_motores_familia_tienen_scripts(nombre):
    p = PROGRAMAS[nombre]
    assert (DIR_PAQUETE / 'programas' / f'{p.script}.py').exists()
    for s in FAMILIAS[p.familia]['scripts']:
        assert (DIR_PAQUETE / 'familias' / p.familia / f'{s}.py').exists()


@pytest.mark.parametrize('nombre', [n for n, p in PROGRAMAS.items() if p.motor == 'notebook'])
def test_inyeccion_del_mes_en_notebooks(nombre):
    import pandas as pd
    p = PROGRAMAS[nombre]
    ruta = RAIZ / p.notebook
    if not ruta.exists():
        pytest.skip('notebook no incluido')
    nb = nbformat.read(str(ruta), as_version=4)
    assert inyectar_mes(nb, p.patrones_mes, pd.Timestamp('2026-10-01'), nombre) == [1]
    assert re.search(r"MES_CIERRE\s*=\s*pd\.Timestamp\('2026-10-01'\)", ''.join(c.source for c in nb.cells if c.cell_type == 'code'))


def test_registro_rechaza_familia_desconocida(tmp_path):
    f = tmp_path / 'x.yaml'
    f.write_text("familias: {a: {}}\nprogramas:\n  p: {familia: b, motor: familia}\n")
    with pytest.raises(ValueError):
        cargar_registro(f)


def test_regla_de_estado_por_familia():
    familias, _ = cargar_registro(RAIZ / 'config' / 'programas.yaml')
    assert familias['cosechas']['estado_yoy'] == 'programa' and familias['mensual']['estado_yoy'] == 'estandar'


def test_metodo_de_ub_por_programa():
    """CPB y CPM usan la UB del notebook; surtido y Champion no tienen UB en sus notebooks y conservan la del motor."""
    _, programas = cargar_registro(RAIZ / 'config' / 'programas.yaml')
    assert {n: p.metodo_ub for n, p in programas.items() if p.motor == 'familia'} == {'cpb': 'notebook', 'cpm': 'notebook', 'surtido_motor': 'notebook', 'champion_motor': 'notebook'}   # mismo cálculo de UB en los 4
