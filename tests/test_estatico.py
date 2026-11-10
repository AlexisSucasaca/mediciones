"""Guardas estáticas: los scripts compuestos no deben tener nombres sin definir ni duplicar el kernel del núcleo."""
import ast
import subprocess
import sys

import pytest

from mediciones.ejecucion import DIR_PAQUETE
from mediciones.registro import cargar_registro
from conftest import RAIZ

FAMILIAS, PROGRAMAS = cargar_registro(RAIZ / 'config' / 'programas.yaml')
MOTORES = [n for n, p in PROGRAMAS.items() if p.motor == 'familia']


@pytest.mark.parametrize('nombre', MOTORES)
def test_scripts_compuestos_sin_nombres_indefinidos(nombre, tmp_path):
    pytest.importorskip('pyflakes')
    p = PROGRAMAS[nombre]
    archivos = [DIR_PAQUETE / 'programas' / f'{p.script}.py'] + [DIR_PAQUETE / 'familias' / p.familia / f'{s}.py' for s in FAMILIAS[p.familia]['scripts']]
    f = tmp_path / 'compuesto.py'
    f.write_text('\n'.join(a.read_text(encoding='utf-8') for a in archivos), encoding='utf-8')
    out = subprocess.run([sys.executable, '-m', 'pyflakes', str(f)], capture_output=True, text=True).stdout
    indef = sorted({l.split("'")[1] for l in out.splitlines() if 'undefined name' in l} - {'PROGRAMA', 'display', 'REGLA_NUEVO'})      # nombres que inyecta ejecutar_familia
    assert indef == [], f'nombres sin definir en {nombre}: {indef}'


@pytest.mark.parametrize('nombre', [n for n, p in PROGRAMAS.items() if p.motor == 'familia'])
def test_el_kernel_vive_solo_en_el_nucleo(nombre):
    """En cada programa en proceso (script del programa + scripts de su familia) el kernel del matching se importa del núcleo y no se redefine."""
    p = PROGRAMAS[nombre]
    archivos = [DIR_PAQUETE / 'programas' / f'{p.script}.py'] + [DIR_PAQUETE / 'familias' / p.familia / f'{s}.py' for s in FAMILIAS[p.familia]['scripts']]
    textos = [a.read_text(encoding='utf-8') for a in archivos]
    for t in textos:
        assert '_correr_semilla' not in {n.name for n in ast.parse(t).body if isinstance(n, ast.FunctionDef)}
    assert any('from mediciones.nucleo.matching import _correr_semilla' in t for t in textos)


def test_la_logica_compartida_vive_solo_en_los_modulos_compartidos():
    """Ningún script vuelve a definir lo que ya está en metodologia/ o grilla.py (ids y atribución); solo pueden definir el envoltorio de ejecutar_grilla."""
    prohibidas = {'_es_columna_id_cliente', '_normalizar_ids', 'estado_yoy', 'atribuir', 'valor_atribuible', 'calcular_brechas_bootstrap', 'calcular_distribucion_brechas'}
    for ruta in list((DIR_PAQUETE / 'familias').glob('*/*.py')) + list((DIR_PAQUETE / 'programas').glob('*.py')):
        defs = {n.name for n in ast.parse(ruta.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)}
        assert not (defs & prohibidas), f'{ruta.name} redefine lógica compartida: {sorted(defs & prohibidas)}'
    for prog in ('surtido', 'champion'):
        src = (DIR_PAQUETE / 'programas' / f'{prog}.py').read_text(encoding='utf-8')
        assert 'from mediciones.grilla import ejecutar_grilla as _grilla' in src
