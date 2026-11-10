"""Parámetros metodológicos COMUNES a las dos familias (mismo valor en CPB, CPM, surtido y Champion).

Lo propio de cada programa (categorías, MIN_MESES por categoría, columnas de matching, overrides) sigue en programas/<script>.py.
"""
import numpy as np

VENTANA_MESES = 12
OUTLIER_METODO = "iqr_log"
OUTLIER_K = 1.5
OUTLIER_NIVEL = ["des_oficina"]
USAR_LOG = True
CALIPER_REL = 0.60
CALIPER_ULTIMO = 0.90
NIVEL_MAX = 3
CORR_UMBRAL = 0.55
N_SEMILLAS_DEFAULT = 50
MAX_SEMILLAS_PROBAR_EQUIVALENCIA = 10
MARGEN_ERROR_OBJETIVO = 0.05
HORIZONTE_MESES_REPORTE = 5
MARGEN_EQUIVALENCIA_PEND = np.log1p(MARGEN_ERROR_OBJETIVO) / HORIZONTE_MESES_REPORTE
ALPHA_TOST = 0.05
NIVELES_IC = [80, 90, 95]
N_BOOT = 2000
SEED_BOOT = 42
