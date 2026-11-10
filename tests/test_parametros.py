"""Los parámetros metodológicos comunes viven en un solo lugar y los scripts no los redefinen ni los modifican en sitio."""
import ast
import re

import numpy as np
import pytest

from mediciones import parametros as P
from mediciones.ejecucion import DIR_PAQUETE

ESPERADO = {'VENTANA_MESES': 12, 'OUTLIER_METODO': 'iqr_log', 'OUTLIER_K': 1.5, 'OUTLIER_NIVEL': ['des_oficina'], 'USAR_LOG': True,
            'CALIPER_REL': 0.60, 'CALIPER_ULTIMO': 0.90, 'NIVEL_MAX': 3, 'CORR_UMBRAL': 0.55, 'N_SEMILLAS_DEFAULT': 50,
            'MAX_SEMILLAS_PROBAR_EQUIVALENCIA': 10, 'MARGEN_ERROR_OBJETIVO': 0.05, 'HORIZONTE_MESES_REPORTE': 5, 'ALPHA_TOST': 0.05,
            'NIVELES_IC': [80, 90, 95], 'N_BOOT': 2000, 'SEED_BOOT': 42}
SCRIPTS = [DIR_PAQUETE / r for r in ('familias/mensual/metodologia.py', 'familias/cosechas/metodologia.py',
                                      'programas/cpb.py', 'programas/cpm.py', 'programas/surtido.py', 'programas/champion.py')]


def test_valores_congelados():
    for k, v in ESPERADO.items():
        assert getattr(P, k) == v, k
    assert np.isclose(P.MARGEN_EQUIVALENCIA_PEND, np.log1p(0.05) / 5)


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.name)
def test_los_scripts_no_redefinen_ni_mutan_los_parametros(script):
    src = script.read_text(encoding='utf-8')
    comunes = set(ESPERADO) | {'MARGEN_EQUIVALENCIA_PEND'}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign):
            assert not {t.id for t in n.targets if isinstance(t, ast.Name)} & comunes, f'{script.name} redefine un parámetro común'
    assert not re.search(r'\b(NIVELES_IC|OUTLIER_NIVEL)\s*(\+=|\.(append|extend|remove|insert|pop|sort|clear))', src)
    assert 'from mediciones.parametros import' in src
