# flake8: noqa
# ruff: noqa
# Script de FAMILIA: se ejecuta con exec() en un espacio de nombres compartido con el script del programa
# (mediciones.ejecucion.ejecutar_familia). No es un módulo importable: usa variables globales definidas por el notebook portado (programas/champion.py).
# =============================================================================
# SALIDA DE CHAMPION hacia las bases estándar. El notebook exporta (celda 49) 'Base toneladas (cli-mes)' = base_toneladas_export (Cereales/Galletas + Inka Chips)
# y 'Base brecha (par-mes)' = base_calculo_brecha. Aquí se les da la misma conversión que hacía el adaptador `adaptar_champion` (código DEX = código de cliente;
# banderas que Champion no calcula) y luego corre el exportador estándar de la familia de cosechas (../cosechas/exportar.py), que arma las tres bases:
# toneladas, brechas y espejos (el par fijo de cada cliente de acción, mes a mes).
# Lo que Champion no calcula queda declarado, no inventado: flg_cliente_perfecto en blanco, flg_estimado_retroactivo = False (no estima meses retroactivos).
# =============================================================================
import numpy as np
import pandas as pd

base_calculo_toneladas = base_toneladas_export.copy()
base_calculo_toneladas['cod_cliente_dex'] = base_calculo_toneladas['cod_cliente']
base_calculo_brecha = base_calculo_brecha.rename(columns={'cod_cliente_accion': 'cod_cliente_accion_dex'})
for _d in (base_calculo_toneladas, base_calculo_brecha):
    for _c, _v in (('flg_estimado_retroactivo', False), ('flg_cliente_perfecto', np.nan)):
        if _c not in _d.columns:
            _d[_c] = _v
